# -*- coding: UTF-8 -*-
import os
import sys
import argparse
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import AutoModel,  AutoFeatureExtractor
from concurrent.futures import ThreadPoolExecutor

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--df_path", type=str, default="./")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--wav_dir", type=str, default="./")
parser.add_argument("--num_workers", type=int, default=4)  # Number of parallel workers
args = parser.parse_args()

# Set global variables
SSL_TYPE = args.ssl_type
DF_PATH = args.df_path
AUDIOS_PATH = args.wav_dir
SAVE_PATH = args.save_path
NUM_WORKERS = args.num_workers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device = {device}')

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)
N = len(os.listdir(SAVE_PATH))
print(f"Save path = {SAVE_PATH} created. It has {N} files in it.")

# Function: Extract and save features
def extract_and_save(wav_path, model, processor, save_path):
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model(**inputs).last_hidden_state.squeeze(0)

        pt_name = os.path.splitext(os.path.basename(wav_path))[0]
        file_path = os.path.join(save_path, f"{pt_name}.pt")
        torch.save(feats, file_path)
    except Exception as e:
        print(f"Failed to process {wav_path}: {e}")

# Parallel processing wrapper
def process_file(wav_file):
    wpath = os.path.join(AUDIOS_PATH, wav_file)
    if os.path.isfile(wpath):
        extract_and_save(wpath, ssl_model, ssl_processor, SAVE_PATH)
    else:
        print(f"File {wpath} not found.")

# Main workflow
everything_ok = True

# Load dataframe
print(f"Reading dataframe {DF_PATH}")
try:
    df = pd.read_csv(DF_PATH)
except Exception as e:
    print(f"Error reading dataframe from {DF_PATH}: {e}")
    everything_ok = False

# Check audio files
if everything_ok:
    print(f"Checking files in {AUDIOS_PATH}")
    missing_files = []
    for w in df.FileName.values:
        wpath = os.path.join(AUDIOS_PATH, w)
        if not os.path.isfile(wpath):
            missing_files.append(wpath)
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f" - {file}")
        everything_ok = False

# Load SSL model
if everything_ok:
    print(f"Extracting features using {SSL_TYPE}")

    try:
        ssl_processor = AutoFeatureExtractor.from_pretrained(SSL_TYPE)
        ssl_model = AutoModel.from_pretrained(SSL_TYPE)
        ssl_model.eval()
        ssl_model.to(device)
    except OSError as e:
        print(f"Error: No pretrained model found with the name {SSL_TYPE}")
        everything_ok = False

# Feature extraction in parallel
if everything_ok:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_file, df.FileName.values), total=len(df.FileName.values), desc="Extracting features"))
else:
    print("Something went wrong, make sure everything is correct before running again!")