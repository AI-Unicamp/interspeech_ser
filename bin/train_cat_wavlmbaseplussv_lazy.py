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
audio_lazy_path = config['audio_lazy_dir']
text_path = config["txt_dir"]
label_path = config["label_path"]

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
    feats = [item['feat'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad features to have the same length
    padded_feats = pad_sequence(feats, batch_first=True)  # (batch_size, max_seq_len, feat_dim)

    # Stack labels into a single tensor
    stacked_labels = torch.stack(labels)  # (batch_size, num_labels)

    return {
        "feat": padded_feats,
        "label": stacked_labels,
    }

class MultiLabelAudioDataset(Dataset):
    def __init__(self, wav_files, labels, lazy_path):
        self.wav_paths = wav_files
        self.labels = labels
        self.lazy_path = lazy_path

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        feat_name = os.path.join(self.lazy_path, self.wav_paths[idx].replace('.wav','.pt'))
        label = self.labels[idx]
        feat = torch.load(feat_name)
        return {
            "feat": feat,
            "label": torch.tensor(label, dtype=torch.float),  # Use float for BCEWithLogitsLoss
        }

class WavLMSERClassifier(nn.Module):
    def __init__(self, wavlm_dim=1024, hidden_dim=512, num_categories=8, num_attention_heads=4):
        super().__init__()

        # Initial dropout for input features
        self.wav_dropout = nn.Dropout(0.5)

        # Linear projection to hidden dimension
        self.wav_proj = nn.Linear(wavlm_dim, hidden_dim)

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.5,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(hidden_dim)

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=32, stride=32)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_categories)
        )

    def forward(self, wavlm_output):
        # Apply initial dropout to inputs
        wav_x = self.wav_dropout(wavlm_output)

        # Project to hidden dimension
        wav_x = self.wav_proj(wav_x)

        # Process with multi-head attention
        attn_output, _ = self.multihead_attn(wav_x, wav_x, wav_x)
        wav_x = self.attn_norm(attn_output + wav_x)  # Residual connection with layer norm

        # Convolutional processing
        wav_x = wav_x.transpose(1, 2)  # Change to (batch_size, hidden_dim, seq_len) for Conv1D
        conv_x = self.conv1d(wav_x)
        conv_x = self.conv_norm(conv_x.transpose(1, 2))  # Back to (batch_size, seq_len, hidden_dim) after LayerNorm

        # Max pooling
        pooled_x = self.max_pool(conv_x.transpose(1, 2)).transpose(1, 2)  # Back to (batch_size, pooled_len, hidden_dim)

        # Flatten for classifier
        flattened_x = torch.mean(pooled_x, dim=1)  # Global average pooling over sequence length

        # Pass through classifier
        output = self.classifier(flattened_x)

        return output

train_dataset = MultiLabelAudioDataset(train_df['FileName'].tolist(), train_df[classes].values, audio_lazy_path)
val_dataset = MultiLabelAudioDataset(val_df['FileName'].tolist(), val_df[classes].values, audio_lazy_path)


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
ser_model = WavLMSERClassifier(wavlm_dim=768, hidden_dim=512, num_categories=8, num_attention_heads=1)
ser_model.to(device)

optimizer = torch.optim.AdamW(ser_model.parameters(), lr=LR, weight_decay=1e-6)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # For multi-label classification


min_epoch=0
min_loss=1e10

logger.info("Starting training...")
# focal_loss = loss.FocalLoss(alpha=1, gamma=3, reduction='mean', dynamic_alpha=True)

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    ser_model.train()
    batch_cnt = 0

    for batch in tqdm(train_loader):
        inputs = batch["feat"].to(device)
        # labels = batch["label"].to(device)
        y = batch['label']; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()

        optimizer.zero_grad()
        outputs = ser_model(inputs)
        logits = outputs

        loss = utils.CE_weight_category(logits, y, None)

        if(use_focalloss):
            
            floss = focal_loss(logits, y)

            total_loss = (loss+floss) / ACCUMULATION_STEP
        else:
            total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()

        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(train_loader):
            optimizer.step()
        batch_cnt += 1

        # Logging
        if((batch_cnt+1)%200 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss = {loss}")

    ser_model.eval()
    total_pred = [] 
    total_y = []

    val_preds, val_labels = [], []
    for batch in tqdm(val_loader):
        inputs = batch["feat"].to(device)
        labels = batch["label"].to(device)
        y = batch['label']; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()


        with torch.no_grad():

            outputs = ser_model(inputs)
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

    if min_loss > dev_loss:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)

        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "ser.pt"))

