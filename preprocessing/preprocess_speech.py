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
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--wav_dir", type=str, default="./")
parser.add_argument("--num_workers", type=int, default=4)  # Number of parallel workers
parser.add_argument("--n_layer", type=int, default=-1)
parser.add_argument("--use_average", type=str, default='n')  

args = parser.parse_args()

# Set global variables
SSL_TYPE = args.ssl_type
AUDIOS_PATH = args.wav_dir
SAVE_PATH = args.save_path
NUM_WORKERS = args.num_workers
LAYER = args.n_layer


AVERAGE = True if args.use_average == 'y' else False

print(f"Using average = {AVERAGE}")

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
        inputs['output_hidden_states'] = True

        if(AVERAGE):
            with torch.no_grad():
                outputs = model(**inputs)
            # Get all hidden states
            hidden_states = outputs.hidden_states
            
            # Get last 4 hidden states and stack them
            last_four_states = torch.stack(hidden_states[-4:])
            
            # Calculate mean across the last 4 layers
            # Shape: [sequence_length, hidden_size]
            feats = torch.mean(last_four_states, dim=0).squeeze(0)
        else:
            with torch.no_grad():
                feats = model(**inputs)
            feats = feats['hidden_states'][N].squeeze(0)

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


wav_paths = os.listdir(AUDIOS_PATH)
len_files = len(wav_paths)
print(f"{len_files} file are going to be processed...")
df = pd.DataFrame({'FileName': wav_paths})

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