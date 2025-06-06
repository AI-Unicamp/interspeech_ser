# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import logging
import time 

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from transformers import AutoModel
import importlib
# Self-Written Modules
sys.path.append(os.getcwd())
from benchmark import net
from benchmark import utils
from torch.utils.data import WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel
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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
# parser.add_argument("--ssl_type", type=str, default="wavlm-large")
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--accumulation_steps", type=int, default=1)
# parser.add_argument("--epochs", type=int, default=10)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--config_path", type=str, default="./configs/config_cat.json")
# parser.add_argument("--head_dim", type=int, default=1024)

# parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling")
args = parser.parse_args()

import json
from collections import defaultdict
# config_path = "configs/config_cat.json"
config_path = args.config_path
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
text_path = config["txt_dir"]
label_path = config["label_path"]

SSL_TYPE = utils.get_ssl_type(config['ssl_type'])
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = config['batch_size']
ACCUMULATION_STEP = config['accum_step']
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS= config['epochs']
LR=config['lr']
MODEL_PATH = config['model_path']
os.makedirs(MODEL_PATH, exist_ok=True)
HEAD_DIM = config['head_dim']
POOLING_TYPE = config['pooling_type']
WC = config["weight_decay"]
DROPOUT = config["dropout_head"]
USE_TIMBRE_PERTURB = config['use_timbre_perturb']
TP_PROB = config['tp_prob']
# utils.set_deterministic(args.seed)
# SSL_TYPE = utils.get_ssl_type(args.ssl_type)
# assert SSL_TYPE != None, print("Invalid SSL type!")
# BATCH_SIZE = args.batch_size
# ACCUMULATION_STEP = args.accumulation_steps
# assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
# EPOCHS=args.epochs
# LR=args.lr
# MODEL_PATH = args.model_path
# os.makedirs(MODEL_PATH, exist_ok=True)


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
    normalize_wav = config["normalize_wav"]
except:
    normalize_wav = True

logger.info(f"Starting an experimento in model path = {MODEL_PATH}")
logger.info(f"Using ssl = {SSL_TYPE} LR = {LR} Epochs = {EPOCHS} Batch size = {BATCH_SIZE} Accum steps = {ACCUMULATION_STEP}")
logger.info(f"Using balanced batch = {balanced_batch}")
logger.info(f"Using normalize wav = {normalize_wav}")
logger.info(f"Using Timbre Perturbation = {USE_TIMBRE_PERTURB}")


import pandas as pd
import numpy as np

# Load the CSV file
label_df = pd.read_csv(label_path)
text_df = pd.read_csv(text_path)
df = label_df.merge(text_df, on = 'FileName', how = 'left')
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

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


val_df =  df[df['Split_Set'] == 'Development']

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
# Print or return the tensor
# print(class_weights_tensor)



logger.info(f"Class weights: {class_weights_tensor}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
text_model = RobertaModel.from_pretrained("roberta-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.eval(); text_model.to(device)

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs, normalize_wav=normalize_wav, use_tp = USE_TIMBRE_PERTURB, tp_prob= TP_PROB)
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_txt_set = utils.TxtSet(df[df["Split_Set"] == 'Train'].transcription.tolist(), tokenizer)
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std, normalize_wav=normalize_wav)
        cur_txt_set = utils.TxtSet(df[df["Split_Set"] == 'Development'].transcription.tolist(), tokenizer)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.CAT_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts, cur_txt_set])

    if((balanced_batch) & (dtype == "train")):
        logger.info('Using balanced batch')
        class_frequencies = train_df[classes].sum().to_dict()
        # Calculate inverse frequency weights
        class_weights = {cls: 1/freq if freq != 0 else 0 for cls, freq in class_frequencies.items()}

        # Normalize weights
        factor = len(class_weights) / sum(class_weights.values())
        class_weights = {cls: w * factor for cls, w in class_weights.items()}

        

        # Create per-sample weights based on their class
        sample_weights = [class_weights[train_df[classes].iloc[i].idxmax()] for i in range(len(train_df))]
        logger.info(f'Using balanced batch.')
        sampler = WeightedRandomSampler(
            weights=sample_weights,               
            num_samples=len(total_dataset[dtype]),       
            replacement=True                 
        )
        total_dataloader[dtype] = DataLoader(
            total_dataset[dtype], batch_size=cur_bs, sampler=sampler, 
            pin_memory=True, num_workers=4,
            collate_fn=utils.collate_fn_txt_wav_lab_mask
        )
    else:
        total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_txt_wav_lab_mask
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.eval(); ssl_model.cuda()

########## Implement pooling method ##########
wav_feat_dim = ssl_model.config.hidden_size
txt_feat_dim = text_model.config.hidden_size

class MultimodalSERClassifier(nn.Module):
    def __init__(self, wavlm_dim=1024, roberta_dim=768, hidden_dim=512, num_categories=8, num_transformer_layers=2):
        super().__init__()
        
        # Initial dropouts for input features
        self.wav_dropout = nn.Dropout(0.5)
        self.rob_dropout = nn.Dropout(0.5)
        
        # Linear projections to match hidden dimension
        self.wav_proj = nn.Linear(wavlm_dim, hidden_dim)
        self.rob_proj = nn.Linear(roberta_dim, hidden_dim)
        
        # Transformer layers for WavLM
        self.wav_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dim_feedforward=hidden_dim * 4,
                dropout=0.5,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        
        # Transformer layers for RoBERTa
        self.rob_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dim_feedforward=hidden_dim * 4,
                dropout=0.5,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        
        # Gating mechanisms (as shown in FIONA framework)
        self.wav_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.rob_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_categories)
        )
        
    def forward(self, wavlm_output, roberta_output):
        # Apply initial dropout to inputs
        wav_x = self.wav_dropout(wavlm_output)
        rob_x = self.rob_dropout(roberta_output)
        
        # Project to hidden dimension
        wav_x = self.wav_proj(wav_x)
        rob_x = self.rob_proj(rob_x)
        
        # Process with transformer layers
        wav_x = self.wav_transformer(wav_x)
        rob_x = self.rob_transformer(rob_x)
        
        # Mean pooling over sequence length dimension
        wav_x = torch.mean(wav_x, dim=1)
        rob_x = torch.mean(rob_x, dim=1)
        
        # Apply gating mechanism (sigmoid multiplication as in FIONA)
        wav_gate = self.wav_gate(wav_x)
        rob_gate = self.rob_gate(rob_x)
        
        # Element-wise multiplication with gates
        wav_x = wav_x * wav_gate
        rob_x = rob_x * rob_gate
        
        # Concatenate gated features
        combined = torch.cat([wav_x, rob_x], dim=1)
        
        # Pass through classifier
        output = self.classifier(combined)
        
        return output, wav_x, rob_x  # Return gated features for CKA loss

ser_model = MultimodalSERClassifier(wavlm_dim=wav_feat_dim, roberta_dim=txt_feat_dim, hidden_dim=HEAD_DIM, num_categories=8)
##############################################
ser_model.eval(); ser_model.cuda()

# ssl_opt = torch.optim.AdamW(ssl_model.parameters(), LR, weight_decay=WC)
ser_opt = torch.optim.AdamW(ser_model.parameters(), LR, weight_decay=WC)

# scaler = GradScaler()
# ssl_opt.zero_grad(set_to_none=True)
ser_opt.zero_grad(set_to_none=True)

from src.losses import loss
focal_loss = loss.FocalLoss(alpha=1, gamma=3, reduction='mean', dynamic_alpha=True)
cka_loss = loss.CKALoss()

# Create scheduler
batch_size = BATCH_SIZE
dataset_size = train_df.shape[0]
steps_per_epoch = math.ceil(dataset_size / batch_size)  # ≈ 313 steps
total_iterations = EPOCHS * steps_per_epoch  # 100 * 313 = 31,300
scheduler = CosineAnnealingScheduler(
    ser_opt,
    T_max=total_iterations,  # Total epochs or iterations
    eta_min=1e-6  # Minimum learning rate
)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_loss"])
lm.alloc_stat_type_list(["dev_loss"])

min_epoch=0
min_loss=1e10

logger.info("Starting training...")

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    lm.init_stat()
    # ssl_model.train()
    # pool_model.train()
    ser_model.train()    
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        txt_ids = xy_pair[4]; txt_ids=txt_ids.cuda(non_blocking=True).long()
        txt_ids_mask = xy_pair[5]; txt_ids_mask=txt_ids_mask.cuda(non_blocking=True).float()

        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state # (B, T, 1024)
            # ssl = pool_model(ssl, mask)
            txt = text_model(txt_ids, attention_mask = txt_ids_mask).last_hidden_state
        
        emo_pred, wav_x, rob_x = ser_model(ssl, txt)

        loss_cka = 1-cka_loss(wav_x, rob_x)

        loss = utils.CE_weight_category(emo_pred, y, None)
        # loss = focal_loss(emo_pred, y)

        total_loss = (loss+loss_cka) / ACCUMULATION_STEP
        total_loss.backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):

            ser_opt.step()
            ser_opt.zero_grad(set_to_none=True)
            # Update learning rate
            scheduler.step()
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_loss", loss)
        current_lr = scheduler.get_last_lr()[0]
        if((batch_cnt+1)%200 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss = {loss} cka_loss {loss_cka} current lr = {current_lr}")

    ssl_model.eval()
    text_model.eval()
    ser_model.eval() 
    total_pred = [] 
    total_y = []
    total_wav = [] 
    total_rob = [] 
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        txt_ids = xy_pair[4]; txt_ids=txt_ids.cuda(non_blocking=True).long()
        txt_ids_mask = xy_pair[5]; txt_ids_mask=txt_ids_mask.cuda(non_blocking=True).float()

        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state # (B, T, 1024)
            # ssl = pool_model(ssl, mask)
            txt = text_model(txt_ids, attention_mask = txt_ids_mask).last_hidden_state
            
            emo_pred, wv, rb = ser_model(ssl, txt)

            total_pred.append(emo_pred)
            total_y.append(y)
            total_wav.append(wv)
            total_rob.append(rb)
    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    total_wav = torch.cat(total_wav, 0)
    total_rob = torch.cat(total_rob, 0)
    loss = utils.CE_weight_category(total_pred, total_y, class_weights_tensor_val)
    loss_cka_eval = 1-cka_loss(total_wav, total_rob)
    
    # Logging
    lm.add_torch_stat("dev_loss", loss)
    logger.info(f"|VALIDATION| Epoch ({epoch+1}/{EPOCHS}): eval_loss = {loss} eval_cka = {loss_cka_eval}")

    # Save model
    lm.print_stat()

        
    dev_loss = lm.get_stat("dev_loss")
    if min_loss > dev_loss:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)
        save_model_list = ["ser", "ssl"]


        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "final_ser.pt"))
 

