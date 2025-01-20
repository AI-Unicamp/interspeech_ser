import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import sys
import os
import random
import numpy as np
sys.path.append(os.getcwd())
from benchmark import utils
import logging
import time
from tqdm import tqdm
from src.losses import loss
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModel
from torch import nn
import torch.nn.functional as F
import csv
import math
from torch.optim.lr_scheduler import _LRScheduler


def set_deterministic(seed: int = 42):
    """
    Configure PyTorch and other libraries for deterministic behavior.

    Args:
        seed (int): Seed value for random number generators.
    """
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set seeds for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--config_path", type=str, default="./configs/config_cat.json")

args = parser.parse_args()

set_deterministic(seed=args.seed)

import json
from collections import defaultdict
config_path = args.config_path
with open(config_path, "r") as f:
    config = json.load(f)
base_path = config['wav_dir']
lazy_path1 = config['lazy_dir1']
lazy_path2 = config['lazy_dir2']
text_path = config["txt_dir"]
label_path = config["label_path"]

feature1_dim = config['feat1_dim']
feature2_dim = config['feat2_dim']

BATCH_SIZE = config['batch_size']
ACCUMULATION_STEP = config['accum_step']
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS= config['epochs']
LR=config['lr']
MODEL_PATH = config['model_path']
os.makedirs(MODEL_PATH, exist_ok=True)
# HEAD_DIM = config['head_dim']
# WC = config["weight_decay"]
# DROPOUT = config["dropout_head"]



# Start logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODEL_PATH, '%s-%d.log' % ('loggingtxt', time.time()))),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()



# print(config["use_balanced_batch"])
try:
    balanced_batch = config["use_balanced_batch"]
except:
    balanced_batch = False

try:
    use_focalloss = config["use_focalloss"]
except:
    use_focalloss = False

import pandas as pd
import numpy as np

# Load the CSV file
test_df = pd.read_csv('./test/Categorical_test.csv')
# text_df = pd.read_csv(text_path)
# df = label_df.merge(text_df, on = 'FileName', how = 'left')
# Filter out only 'Train' samples

# 1111111_df = df[df['Split_Set'] == 'Development']



# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
classes_ = ['A', 'S', 'H', 'U', 'F', 'D', 'C', 'N']

map_argmax = dict()
for i, c in enumerate(classes_):
    map_argmax[i] = c


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function for padding feature tensors in a batch.

    Args:
        batch (list of dict): Each dict contains 'feat' (tensor) and 'label' (tensor).

    Returns:
        dict: Batched and padded features and labels.
    """
    # Extract features and labels from the batch
    feats1 = [item['feat1'] for item in batch]
    feats2 = [item['feat2'] for item in batch]
    utt = [item['utt'] for item in batch]

    # Pad features to have the same length
    padded_feats1 = pad_sequence(feats1, batch_first=True)  # (batch_size, max_seq_len, feat_dim)
    padded_feats2 = pad_sequence(feats2, batch_first=True)  # (batch_size, max_seq_len, feat_dim)

    return {
        "feat1": padded_feats1,
        "feat2": padded_feats2,
        "utt": utt
    }

class MultiLabelAudioDataset(Dataset):
    def __init__(self, wav_files, lazy_path1, lazy_path2):
        self.wav_paths = wav_files
        self.lazy_path1 = lazy_path1
        self.lazy_path2 = lazy_path2
        self.verbose_one = True
    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        feat_name1 = os.path.join(self.lazy_path1, self.wav_paths[idx].replace('.wav','.pt'))
        feat_name2 = os.path.join(self.lazy_path2, self.wav_paths[idx].replace('.wav','.pt'))
        
        if(self.verbose_one):
            print(feat_name1, feat_name2)
            self.verbose_one = False
        feat1 = torch.load(feat_name1)
        feat2 = torch.load(feat_name2)
        utt = self.wav_paths[idx]
        return {
            "feat1": feat1,
            "feat2": feat2,
             "utt": utt
        }

class MultiModalEmotionClassifier(nn.Module):
    def __init__(
        self,
        features1_dim=1024,  
        features2_dim=768, 
        fusion_hidden_dim=512,
        num_emotions=8,
        dropout=0.5
    ):
        super().__init__()
        
        # Separate modality processing
        self.speech_projection = nn.Linear(features1_dim, fusion_hidden_dim)
        self.text_projection = nn.Linear(features2_dim, fusion_hidden_dim)
        
        self.speech_norm = nn.LayerNorm(fusion_hidden_dim)
        self.text_norm = nn.LayerNorm(fusion_hidden_dim)

        # GRU layers
        self.speech_gru = nn.GRU(
            fusion_hidden_dim,
            fusion_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.text_gru = nn.GRU(
            fusion_hidden_dim,
            fusion_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Cross-modal attention
        self.speech_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 1, dropout=dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 1, dropout=dropout, batch_first=True)
        
        # Simple attention pooling
        self.speech_attn = nn.Linear(fusion_hidden_dim * 2, 1)
        self.text_attn = nn.Linear(fusion_hidden_dim * 2, 1)
        
        # Classifier (input is concatenated representations)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 4, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_emotions)
        )
        
        self.layer_norm = nn.LayerNorm(fusion_hidden_dim * 4)
        
    def attention_pool(self, features, attention_layer):
        # features: [batch, seq_len, hidden]
        
        # Calculate attention scores
        attn_weights = attention_layer(features)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention
        weighted_features = features * attn_weights
        pooled = weighted_features.sum(dim=1)  # [batch, hidden]
        
        return pooled
        
    def forward(self, features1, features2):
        # Project both modalities to same dimension
        speech_proj = self.speech_projection(features1)  # [batch, seq_len, fusion_hidden_dim]
        text_proj = self.text_projection(features2)    # [batch, seq_len, fusion_hidden_dim]
        
        speech_proj = self.speech_norm(speech_proj)
        text_proj = self.text_norm(text_proj)

        # Pass through GRUs
        speech_hidden, _ = self.speech_gru(speech_proj)  # [batch, seq_len, fusion_hidden_dim*2]
        text_hidden, _ = self.text_gru(text_proj)        # [batch, seq_len, fusion_hidden_dim*2]
        
        # Cross-modal attention
        speech_attended, _ = self.speech_attention(
            speech_hidden, text_hidden, text_hidden
        )
        text_attended, _ = self.text_attention(
            text_hidden, speech_hidden, speech_hidden
        )
        
        # Combine attended features
        speech_final = speech_hidden + speech_attended
        text_final = text_hidden + text_attended
        
        # Simple attention pooling
        speech_pooled = self.attention_pool(speech_final, self.speech_attn)
        text_pooled = self.attention_pool(text_final, self.text_attn)
        
        # Concatenate pooled representations
        concatenated = torch.cat([speech_pooled, text_pooled], dim=-1)  # [batch, fusion_hidden_dim*4]
        
        # Layer norm and classify
        normalized = self.layer_norm(concatenated)
        logits = self.classifier(normalized)
        
        return logits


val_dataset = MultiLabelAudioDataset(test_df['FileName'].tolist(), lazy_path1, lazy_path2)

val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ser_model = MultiModalEmotionClassifier(
        features1_dim=feature1_dim,  
        features2_dim=feature2_dim, 
        fusion_hidden_dim=512,
        num_emotions=8,
        dropout=0.5
    )
ser_model.to(device)
ser_model.load_state_dict(torch.load(MODEL_PATH+"/multimodal_ser.pt"))
ser_model.eval()


logger.info("Starting scoring test samples...")

total_pred = [] 
total_utt = []
for batch in tqdm(val_loader):
    inputs1 = batch["feat1"].to(device)
    inputs2 = batch["feat2"].to(device)
    utt = batch['utt']

    with torch.no_grad():

        outputs = ser_model(inputs1,inputs2)
        logits = outputs

        total_pred.append(logits)
        total_utt.extend(utt)

total_pred = torch.cat(total_pred, 0)

# data = []
# for pred, utt in zip(total_pred, total_utt):
#     pred_values = map_argmax[np.argmax(pred.cpu().numpy().flatten())]
#     data.append([utt, pred_values])

dtype = 'test'


def save_predictions_with_probs(total_pred, total_utt, map_argmax, model_path, dtype='dev'):
    data = []
    
    # Get number of classes from the first prediction tensor
    num_classes = len(total_pred[0].cpu().numpy().flatten())
    
    for pred, utt in zip(total_pred, total_utt):
        # Convert prediction tensor to numpy array and flatten
        pred_probs = pred.cpu().numpy().flatten()
        
        # Get the predicted class
        pred_class = map_argmax[np.argmax(pred_probs)]
        
        # Create row with filename, prediction, and all class probabilities
        row = [utt, pred_class]
        for i, prob in enumerate(pred_probs):
            row.append(f'{prob:.4f}')
        
        data.append(row)
    
    # Create headers for the CSV
    headers = ['FileName', 'Prediction']
    for i in range(num_classes):
        headers.append(f'class_{i}_prob')
    
    # Writing to CSV file
    os.makedirs(os.path.join(model_path, 'results'), exist_ok=True)
    csv_filename = os.path.join(model_path, 'results', f'{dtype}.csv')
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
    
    return csv_filename

csv = save_predictions_with_probs(total_pred, total_utt, map_argmax, MODEL_PATH, dtype=dtype)









# # Writing to CSV file
# os.makedirs(os.path.join(MODEL_PATH, 'results'), exist_ok=True)
# csv_filename = os.path.join(MODEL_PATH, 'results', f'{dtype}.csv')

# data = np.array(data)
# # os.makedirs(MODEL_PATH + '/results', exist_ok=True) 
# df = pd.DataFrame({'FileName': data[:,0], 'EmoClass': data[:,1]})
# df = df.sort_values(by='FileName').reset_index(drop = True)
# df.to_csv(csv_filename, index=False)

# # Logging
# logger.info(f"Predictions saved at {csv_filename}")
