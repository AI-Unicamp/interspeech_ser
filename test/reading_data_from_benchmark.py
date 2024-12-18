import sys
import os

sys.path.append(os.getcwd())

from torch.utils.data import ConcatDataset, DataLoader
from benchmark import utils
import json
from collections import defaultdict
config_path = "benchmark/configs/config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']


MODEL_PATH = 'test/cat'

BATCH_SIZE = 2
ACCUMULATION_STEP = 1

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    if dtype == "train":
        if(not os.path.isfile(MODEL_PATH+"/train_norm_stat.pkl")):
            cur_wav_set = utils.WavSet(cur_wavs)
            cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        else:
            print('Reading training statistics from previous run')
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
            cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.CAT_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )


print('Starting dimensional dataloader')


config_path = "benchmark/configs/config_dim.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]


# Load the CSV file
label_path = label_path
df = pd.read_csv(label_path)
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']


MODEL_PATH = 'test/dim'

total_dataset_dim=dict()
total_dataloader_dim=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    if dtype == "train":
        if(not os.path.isfile(MODEL_PATH+"/train_norm_stat.pkl")):
            cur_wav_set = utils.WavSet(cur_wavs)
            cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        else:
            print('Reading training statistics from previous run')
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
            cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    else:
        if dtype == "dev":
            wav_mean = total_dataset_dim["train"].datasets[0].wav_mean
            wav_std = total_dataset_dim["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    total_dataset_dim[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
    total_dataloader_dim[dtype] = DataLoader(
        total_dataset_dim[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )

if __name__ == "__main__":
    print("Iterating categorical dataloader...")
    batch = next(iter(total_dataloader['train']))

    print("Printing labels and utt information")
    print(batch[1])
    print(batch[3])

    print("#" * 10) 
    print("#" * 10) 
    print("#" * 10) 

    print("Iterating dimensional dataloader...")
    batch = next(iter(total_dataloader_dim['train']))

    print("Printing labels and utt information")
    print(batch[1])
    print(batch[3])