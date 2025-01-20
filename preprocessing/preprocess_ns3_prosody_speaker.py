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
import sys
sys.path.append(os.getcwd())
from src.ns3 import FACodecEncoderV2, FACodecDecoderV2
import numpy as np

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--wav_dir", type=str, default="./")
parser.add_argument("--num_workers", type=int, default=4)  # Number of parallel workers

args = parser.parse_args()

# Set global variables
AUDIOS_PATH = args.wav_dir
SAVE_PATH = args.save_path
NUM_WORKERS = args.num_workers


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f'Using device = {device}')

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)
N = len(os.listdir(SAVE_PATH))
print(f"Save path = {SAVE_PATH} created. It has {N} files in it.")

# Function: Extract and save features
def extract_and_save(wav_path, model, save_path):
    try:
        y, sr = librosa.load(wav_path, sr=16000)

        y = np.pad(y, (0, 200 - len(y) % 200))
        y = torch.tensor(y).float().to(device).unsqueeze(0).unsqueeze(0)

        prosody_codes = ssl_processor.get_prosody_feature(y)
        encoded = ssl_processor(y)

        out, qs, _, _, result = ssl_model.get_processed_style_speaker_embedding(encoded, prosody_codes)

        
        feats = result.squeeze(0).permute(1,0)
        # feats = qs[:ssl_model.vq_num_q_p].squeeze(0).permute(1,0)
        # print(feats.shape)

        pt_name = os.path.splitext(os.path.basename(wav_path))[0]
        file_path = os.path.join(save_path, f"{pt_name}.pt")
        torch.save(feats, file_path)
    except Exception as e:
        print(f"Failed to process {wav_path}: {e}")

# Parallel processing wrapper
def process_file(wav_file):
    wpath = os.path.join(AUDIOS_PATH, wav_file)
    if os.path.isfile(wpath):
        extract_and_save(wpath, ssl_model, SAVE_PATH)
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
    print(f"Extracting prosody codes from NaturalSpeech3")

    try:
        ssl_processor = FACodecEncoderV2(
                            ngf=32,
                            up_ratios=[2, 4, 5, 5],
                            out_channels=256,
                        )

        ssl_model = FACodecDecoderV2(
                            in_channels=256,
                            upsample_initial_channel=1024,
                            ngf=32,
                            up_ratios=[5, 5, 4, 2],
                            vq_num_q_c=2,
                            vq_num_q_p=1,
                            vq_num_q_r=3,
                            vq_dim=256,
                            codebook_dim=8,
                            codebook_size_prosody=10,
                            codebook_size_content=10,
                            codebook_size_residual=10,
                            use_gr_x_timbre=True,
                            use_gr_residual_f0=True,
                            use_gr_residual_phone=True,
                        )
        
        encoder_v2_ckpt = "./pretrained_models/ns3/ns3_facodec_encoder_v2.bin"
        ssl_processor.load_state_dict(torch.load(encoder_v2_ckpt, map_location='cpu'))
        ssl_processor.eval()
        ssl_processor.to(device)
        
        decoder_v2_ckpt = "./pretrained_models/ns3/ns3_facodec_decoder_v2.bin"
        ssl_model.load_state_dict(torch.load(decoder_v2_ckpt, map_location='cpu'))
        ssl_model.eval()
        ssl_model.to(device)
    except OSError as e:
        print(f"Error: No pretrained model found with the name")
        everything_ok = False

# Feature extraction in parallel
if everything_ok:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_file, df.FileName.values), total=len(df.FileName.values), desc="Extracting features"))
else:
    print("Something went wrong, make sure everything is correct before running again!")