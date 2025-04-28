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

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """
        Cosine annealing scheduler with warm restarts
        
        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

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

logger.info(f"Starting an Lazy OwnSermodel wavlm-based experiment in model path = {MODEL_PATH}")
logger.info(f"Using LR = {LR} Epochs = {EPOCHS} Batch size = {BATCH_SIZE} Accum steps = {ACCUMULATION_STEP}")
logger.info(f"Using balanced batch = {balanced_batch}")
logger.info(f"Using focalloss = {use_focalloss}")


import pandas as pd
import numpy as np

# Load the CSV file
label_df = pd.read_csv(label_path)
text_df = pd.read_csv(text_path)
df = label_df.merge(text_df, on = 'FileName', how = 'left')
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']
val_df = df[df['Split_Set'] == 'Development']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()
# Total number of samples
total_samples = len(train_df)
# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
print(class_weights)
# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]
# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
# Print or return the tensor
print(class_weights_tensor)


# Calculate class frequencies
class_frequencies_val = val_df[classes].sum().to_dict()
# Total number of samples
total_samples_val = len(val_df)
# Calculate class weights
class_weights_val = {cls: total_samples_val / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies_val.items()}
# print(class_weights)
# Convert to list in the order of classes
weights_list_val = [class_weights_val[cls] for cls in classes]
# Convert to PyTorch tensor
class_weights_tensor_val = torch.tensor(weights_list_val, device='cuda', dtype=torch.float)


logger.info(f"Class weights: {class_weights_tensor}")

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
    labels = [item['label'] for item in batch]

    # Pad features to have the same length
    padded_feats1 = pad_sequence(feats1, batch_first=True)  # (batch_size, max_seq_len, feat_dim)
    padded_feats2 = pad_sequence(feats2, batch_first=True)  # (batch_size, max_seq_len, feat_dim)

    # Stack labels into a single tensor
    stacked_labels = torch.stack(labels)  # (batch_size, num_labels)

    return {
        "feat1": padded_feats1,
        "feat2": padded_feats2,
        "label": stacked_labels,
    }

class MultiLabelAudioDataset(Dataset):
    def __init__(self, wav_files, labels, lazy_path1, lazy_path2):
        self.wav_paths = wav_files
        self.labels = labels
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
        label = self.labels[idx]
        feat1 = torch.load(feat_name1)
        feat2 = torch.load(feat_name2)

        return {
            "feat1": feat1,
            "feat2": feat2,
            "label": torch.tensor(label, dtype=torch.float),  # Use float for BCEWithLogitsLoss
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
        self.speech_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 8, dropout=dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(fusion_hidden_dim * 2, 8, dropout=dropout, batch_first=True)
        
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

train_dataset = MultiLabelAudioDataset(train_df['FileName'].tolist(), train_df[classes].values, lazy_path1, lazy_path2)
val_dataset = MultiLabelAudioDataset(val_df['FileName'].tolist(), val_df[classes].values, lazy_path1, lazy_path2)


if(balanced_batch):
    logger.info(f'Using balanced batch. Computing sample weights...')
    class_frequencies = train_df[classes].sum().to_dict()
    # Calculate inverse frequency weights
    class_weights = {cls: 1/freq if freq != 0 else 0 for cls, freq in class_frequencies.items()}

    # Normalize weights
    factor = len(class_weights) / sum(class_weights.values())
    class_weights = {cls: w * factor for cls, w in class_weights.items()}

    

    # Create per-sample weights based on their class
    sample_weights = [class_weights[train_df[classes].iloc[i].idxmax()] for i in range(len(train_df))]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,               
        num_samples=len(train_dataset),       
        replacement=True                 
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
# 4. Create DataLoaders
else:

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ser_model = MultiModalEmotionClassifier(
        features1_dim=feature1_dim,  
        features2_dim=feature2_dim, 
        fusion_hidden_dim=512,
        num_emotions=8,
        dropout=0.5
    )
ser_model.to(device)

optimizer = torch.optim.AdamW(ser_model.parameters(), lr=LR, weight_decay=1e-6)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # For multi-label classification

# Create scheduler
batch_size = BATCH_SIZE
dataset_size = train_df.shape[0]
steps_per_epoch = math.ceil(dataset_size / batch_size)  # ≈ 313 steps
total_iterations = EPOCHS * steps_per_epoch  # 100 * 313 = 31,300
scheduler = CosineAnnealingScheduler(
    optimizer,
    T_max=EPOCHS,  # Total epochs or iterations
    eta_min=1e-6  # Minimum learning rate
)

min_epoch=0
min_loss=1e10
max_f1 = 0

logger.info("Starting training...")
focal_loss = loss.FocalLoss(alpha=1, gamma=2, reduction='mean', dynamic_alpha=True)

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    ser_model.train()
    batch_cnt = 0

    for batch in tqdm(train_loader):
        inputs1 = batch["feat1"].to(device)
        inputs2 = batch["feat2"].to(device)

        # print(inputs1.shape, inputs2.shape)
        # labels = batch["label"].to(device)
        y = batch['label']; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()

        optimizer.zero_grad()
        outputs = ser_model(inputs1, inputs2)
        logits = outputs
        if(balanced_batch):
            loss = utils.CE_weight_category(logits, y, None)
        else:
            loss = utils.CE_weight_category(logits, y, class_weights_tensor)

        if(use_focalloss):
            
            floss = focal_loss(logits, y)

            total_loss = floss / ACCUMULATION_STEP
        else:
            total_loss = loss / ACCUMULATION_STEP
            
        total_loss.backward()

        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(train_loader):
            optimizer.step()
            
        batch_cnt += 1
        current_lr = scheduler.get_last_lr()[0]
        # Logging
        if((batch_cnt+1)%200 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss = {loss} current lr = {current_lr}")
    
    scheduler.step()
    ser_model.eval()
    total_pred = [] 
    total_y = []

    val_preds, val_labels = [], []
    for batch in tqdm(val_loader):
        inputs1 = batch["feat1"].to(device)
        inputs2 = batch["feat2"].to(device)
        labels = batch["label"].to(device)
        y = batch['label']; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()


        with torch.no_grad():

            outputs = ser_model(inputs1,inputs2)
            logits = outputs

            total_pred.append(logits)
            total_y.append(labels)

            preds = torch.argmax(logits, dim=1)  # Get index of max logit per row
            preds_one_hot = torch.zeros_like(logits).scatter_(1, preds.unsqueeze(1), 1)  # Convert to one-hot
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

        # Calculate accuracy based on one-hot predictions
        # val_preds_binary = (torch.tensor(val_preds)).numpy()
        # val_labels_binary = (torch.tensor(val_labels)).numpy()
        # f1 = f1_score(val_labels, val_preds, average='macro')

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(total_pred, total_y, class_weights_tensor_val)
    f1 = f1_score(val_labels, val_preds, average='macro')

    dev_loss = loss
    # Logging
    logger.info(f"|VALIDATION| Epoch ({epoch+1}/{EPOCHS}): eval_loss = {loss} eval f1 = {f1}")

    if max_f1 < f1:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = dev_loss
        max_f1 = f1

        print("Save",min_epoch)
        print("Loss",min_loss)

        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "multimodal_ser.pt"))

