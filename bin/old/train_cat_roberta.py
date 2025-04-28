import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import sys
import os
sys.path.append(os.getcwd())
from benchmark import utils
import logging
import time
from tqdm import tqdm
from src.losses import loss
from torch.utils.data import WeightedRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--config_path", type=str, default="./configs/config_cat.json")

args = parser.parse_args()


import json
from collections import defaultdict
config_path = args.config_path
with open(config_path, "r") as f:
    config = json.load(f)
base_path = config['wav_dir']
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

logger.info(f"Starting an Text-based experiment in model path = {MODEL_PATH}")
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

logger.info(f"Class weights: {class_weights_tensor}")


class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),  # Use float for BCEWithLogitsLoss
        }

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_dataset = MultiLabelTextDataset(train_df['transcription'].tolist(), train_df[classes].values, tokenizer)
val_dataset = MultiLabelTextDataset(val_df['transcription'].tolist(), val_df[classes].values, tokenizer)


if(balanced_batch):
    class_frequencies = train_df[classes].sum().to_dict()
    # Calculate inverse frequency weights
    class_weights = {cls: 1/freq if freq != 0 else 0 for cls, freq in class_frequencies.items()}

    # Normalize weights
    factor = len(class_weights) / sum(class_weights.values())
    class_weights = {cls: w * factor for cls, w in class_weights.items()}

    val_df =  df[df['Split_Set'] == 'Development']

    # Create per-sample weights based on their class
    sample_weights = [class_weights[train_df[classes].iloc[i].idxmax()] for i in range(len(train_df))]
    logger.info(f'Using balanced batch.')
    sampler = WeightedRandomSampler(
        weights=sample_weights,               
        num_samples=len(train_dataset),       
        replacement=True                 
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1)
# 4. Create DataLoaders
else:

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE//4)

print("Loading pre-trained Roberta model...")

text_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)

optimizer = torch.optim.AdamW(text_model.parameters(), lr=LR, weight_decay=1e-1)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # For multi-label classification

min_epoch=0
min_loss=1e10

logger.info("Starting training...")
focal_loss = loss.FocalLoss(alpha=1, gamma=3, reduction='mean', dynamic_alpha=True)

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    text_model.train()
    batch_cnt = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = text_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = utils.CE_weight_category(logits, labels, class_weights_tensor)
        if(use_focalloss):
            y = batch['label']; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
            floss = focal_loss(logits, y)

            total_loss = (loss+floss) / ACCUMULATION_STEP
        else:
            total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()

        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(train_loader):
            optimizer.step()
        batch_cnt += 1

        # Logging
        if((batch_cnt+1)%1000 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss = {loss}")

    text_model.eval()
    total_pred = [] 
    total_y = []

    val_preds, val_labels = [], []
    for batch in tqdm(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():

            outputs = text_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            total_pred.append(logits)
            total_y.append(labels)

            preds = torch.argmax(logits, dim=1)  # Get index of max logit per row
            preds_one_hot = torch.zeros_like(logits).scatter_(1, preds.unsqueeze(1), 1)  # Convert to one-hot
            
            val_preds.extend(preds_one_hot.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        # Calculate accuracy based on one-hot predictions
        # val_preds_binary = (torch.tensor(val_preds)).numpy()
        # val_labels_binary = (torch.tensor(val_labels)).numpy()
        acc = accuracy_score(val_labels, val_preds)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(total_pred, total_y, class_weights_tensor)

    # Calculate accuracy based on one-hot predictions
    # val_preds_binary = (torch.tensor(val_preds)).numpy()
    # val_labels_binary = (torch.tensor(val_labels)).numpy()
    # acc = accuracy_score(val_labels_binary, val_preds_binary)
    # print(f"Validation Accuracy: {acc}")

    dev_loss = loss
    # Logging
    logger.info(f"|VALIDATION| Epoch ({epoch+1}/{EPOCHS}): eval_loss = {loss} eval acc = {acc}")

    if min_loss > dev_loss:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)

        torch.save(text_model.state_dict(), \
            os.path.join(MODEL_PATH,  "text_ser.pt"))
        tokenizer.save_pretrained(MODEL_PATH)

