# -*- coding: UTF-8 -*-
import os
import sys
import argparse
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from concurrent.futures import ThreadPoolExecutor

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--roberta_type", type=str, default="roberta")
parser.add_argument("--df_path", type=str, default="./")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--num_workers", type=int, default=4)  # Number of parallel workers
parser.add_argument("--max_len", type=int, default=80)  

args = parser.parse_args()

# Set global variables
SSL_TYPE = args.roberta_type
DF_PATH = args.df_path
SAVE_PATH = args.save_path
NUM_WORKERS = args.num_workers
MAX_LEN = args.max_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device = {device}')

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)
N = len(os.listdir(SAVE_PATH))
print(f"Save path = {SAVE_PATH} created. It has {N} files in it.")

# Function: Extract and save features
def extract_and_save(text, wav_name, model, save_path):
    
    try:
        encoding = tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            feats = model(**encoding).last_hidden_state.squeeze(0)
        pt_name = os.path.splitext(os.path.basename(wav_name))[0] ## PRECISO ASSOCIAR COM O WAVPATH MESMO
        file_path = os.path.join(save_path, f"{pt_name}.pt")
        torch.save(feats, file_path)
    except Exception as e:
        print(f"Failed to process {wav_path}: {e}")

# Parallel processing wrapper
def process_file(text, wav_name):
    # wpath = os.path.join(AUDIOS_PATH, wav_file)
    # if os.path.isfile(wpath):
    extract_and_save(text, wav_name, text_model, SAVE_PATH)
    # else:
        # print(f"File {wpath} not found.")

# Main workflow
everything_ok = True

# Load dataframe
print(f"Reading dataframe {DF_PATH}")
try:
    df = pd.read_csv(DF_PATH)
except Exception as e:
    print(f"Error reading dataframe from {DF_PATH}: {e}")
    everything_ok = False


# Load txt model
if everything_ok:
    print(f"Extracting features using {SSL_TYPE}")

    try:
        tokenizer = RobertaTokenizer.from_pretrained(SSL_TYPE)
        text_model = RobertaModel.from_pretrained(SSL_TYPE)
        text_model.eval()
        text_model.to(device)
    except OSError as e:
        print(f"Error: No pretrained model found with the name {SSL_TYPE}")
        everything_ok = False

# Feature extraction in parallel
if everything_ok:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_file, df.transcription.values, df.FileName.values), total=len(df.transcription.values), desc="Extracting features"))
else:
    print("Something went wrong, make sure everything is correct before running again!")