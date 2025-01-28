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


import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel,  AutoProcessor, AutoFeatureExtractor
from peft import LoraConfig, get_peft_model
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
label_path = config["label_path"]

# SSL_TYPE = config['ssl_type']
# assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = config['batch_size']
ACCUMULATION_STEP = config['accum_step']
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS= config['epochs']
LR=config['lr']
MODEL_PATH = config['model_path']
os.makedirs(MODEL_PATH, exist_ok=True)
# HEAD_DIM = config['head_dim']
# POOLING_TYPE = config['pooling_type']
WC = config["weight_decay"]
# DROPOUT = config["dropout_head"]
USE_TIMBRE_PERTURB = False
TP_PROB = 0
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


normalize_wav = False


logger.info(f"Starting an experimento in model path = {MODEL_PATH}")
# logger.info(f"Using ssl = {SSL_TYPE} LR = {LR} Epochs = {EPOCHS} Batch size = {BATCH_SIZE} Accum steps = {ACCUMULATION_STEP}")
# logger.info(f"Using balanced batch = {balanced_batch}")
# logger.info(f"Using normalize wav = {normalize_wav}")
# logger.info(f"Using Timbre Perturbation = {USE_TIMBRE_PERTURB}")


import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)
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

logger.info(f"Class weights: {class_weights_tensor}")

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-large')

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
    # cur_utts = cur_utts[:100]
    # cur_labs = cur_labs[:100]
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    # cur_wavs = cur_wavs[:100]

    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs, normalize_wav=normalize_wav, use_tp = USE_TIMBRE_PERTURB, tp_prob= TP_PROB, processor = feature_extractor, type_processor="wavlm")
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std, normalize_wav=normalize_wav, processor = feature_extractor, type_processor="wavlm")
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.CAT_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])

    if((balanced_batch) & (dtype == "train")):
        logger.info('Using balanced batch')
        class_frequencies = train_df[classes].sum().to_dict()
        total_samples = len(train_df)
        class_weights_ = {cls: 1/np.sqrt(freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
        weights_list_ = [class_weights_[cls] for cls in classes]
        # Convert to PyTorch tensor
        class_weights_tensor_ = torch.tensor(weights_list_, device='cuda', dtype=torch.float)
        logger.info(f'Using balanced batch. Weights = {class_weights_tensor_}')
        sampler = WeightedRandomSampler(
            weights=class_weights_tensor_,               
            num_samples=len(total_dataset[dtype]),       
            replacement=True                 
        )
        total_dataloader[dtype] = DataLoader(
            total_dataset[dtype], batch_size=cur_bs, sampler=sampler, 
            pin_memory=True, num_workers=4,
            collate_fn=utils.collate_fn_wav_lab_mask
        )
    else:
        total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )

# print("Loading pre-trained ", SSL_TYPE, " model...")






class WavLMClassifier(nn.Module):
    def __init__(self, wavlm_model_name='microsoft/wavlm-large', num_emotions=8):
        super().__init__()
        # Load Whisper model and feature extractor
        # self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        # self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.wavlm = AutoModel.from_pretrained(wavlm_model_name)
        # Freeze original Whisper parameters
        # for param in self.whisper.parameters():
        #     param.requires_grad = False
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.2,
            bias='none',
            task_type='FEATURE_EXTRACTION'
        )
        
        # Apply LoRA
        self.wavlm = get_peft_model(self.wavlm, lora_config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_emotions)
        )
    
    def forward(self, audio, attention_mask):
        # Convert torch tensor to numpy if needed
        # if torch.is_tensor(audio):
            # audio = audio.numpy()

        # print(audio.shape)

        INPUTS = {'input_values':audio, 
            'attention_mask':attention_mask,
            'output_hidden_states':True}
        # print(self.wavlm.forward.__code__.co_varnames)
        # Extract features from Whisper
        wavlm_outputs = self.wavlm.model.forward(**INPUTS)
        
        # feats = whisper_outputs['hidden_states'][-1
        # print(whisper_outputs["hidden_states"].shape)
        # print(feats.shape)
        # Pool features
        pooled_features = wavlm_outputs.last_hidden_state.mean(dim=1)
        
        # Classify emotions
        return self.classifier(pooled_features)


ser_model = WavLMClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ser_model.to(device)
ser_model.eval()

ser_opt = torch.optim.AdamW(ser_model.parameters(), LR, weight_decay=WC)

ser_opt.zero_grad(set_to_none=True)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_loss"])
lm.alloc_stat_type_list(["dev_loss"])

min_epoch=0
min_loss=1e10

logger.info("Starting training...")

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    lm.init_stat()
    ser_model.train()    
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        
        # print(type(x))
        # print(x)
        emo_pred = ser_model(x, attention_mask=mask)

        loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)

        total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):

            ser_opt.step()

            ser_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_loss", loss)
        if((batch_cnt+1)%1000 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss = {loss}")

            torch.save(ser_model.state_dict(), os.path.join(MODEL_PATH,  "whisper_lora_ser_step{batch_cnt}.pt"))

    ser_model.eval() 
    total_pred = [] 
    total_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
        
        with torch.no_grad():
            emo_pred = ser_model(x, attention_mask=mask)

            total_pred.append(emo_pred)
            total_y.append(y)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)
    # Logging
    lm.add_torch_stat("dev_loss", loss)
    logger.info(f"|VALIDATION| Epoch ({epoch+1}/{EPOCHS}): eval_loss = {loss}")

    # Save model
    lm.print_stat()

        
    dev_loss = lm.get_stat("dev_loss")
    if min_loss > dev_loss:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)

        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "whisper_lora_ser.pt"))


