# -*- coding: UTF-8 -*-
import os
import sys
import argparse
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from transformers import AutoModel,  AutoProcessor
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        audio_length = len(y)
        n_frames = int(np.ceil(audio_length / 320))

        # inputs = {k: v.to(device) for k, v in inputs.items()}
        input_features = inputs["input_features"].to(device)

        if(AVERAGE):
            with torch.no_grad():
                outputs = model.encoder(input_features, output_hidden_states=True)

            
            # Get all hidden states
            hidden_states = outputs.hidden_states

            # Get last 4 hidden states and stack them
            last_four_states = torch.stack(hidden_states[-4:])
            
            # Calculate mean across the last 4 layers
            # Shape: [sequence_length, hidden_size]
            feats = torch.mean(last_four_states, dim=0).squeeze(0)
        else:
            with torch.no_grad():
                feats = model.encoder(input_features, output_hidden_states=True)
            
            feats = feats['hidden_states'][LAYER].squeeze(0)

        actual_frames = min(n_frames, feats.shape[1])
        feats = feats[:actual_frames, :]

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

from torch import nn
from peft import LoraConfig, get_peft_model
class WhisperAudioClassifier(nn.Module):
    def __init__(self, whisper_model_name='openai/whisper-large-v3', num_emotions=8):
        super().__init__()
        # Load Whisper model and feature extractor
        # self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        # self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.whisper = AutoModel.from_pretrained(whisper_model_name)
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
        self.whisper = get_peft_model(self.whisper, lora_config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.whisper.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_emotions)
        )
    
    def forward(self, audio, attention_mask):
        # Convert torch tensor to numpy if needed
        # if torch.is_tensor(audio):
            # audio = audio.numpy()

    
        # Extract features from Whisper
        whisper_outputs = self.whisper.encoder(
            input_features=audio, 
            attention_mask=attention_mask,
            output_hidden_states=True)
        
        # feats = whisper_outputs['hidden_states'][-1
        # print(whisper_outputs["hidden_states"].shape)
        # print(feats.shape)
        # Pool features
        pooled_features = whisper_outputs.last_hidden_state.mean(dim=1)
        
        # Classify emotions
        return self.classifier(pooled_features)




# Load SSL model
if everything_ok:
    print(f"Extracting features using {SSL_TYPE}")

    try:
        ssl_processor = AutoProcessor.from_pretrained(SSL_TYPE)
        # ssl_model = AutoModel.from_pretrained(SSL_TYPE)
        ssl_model = WhisperAudioClassifier()
        checkpoint = torch.load("/workspace/lucas.ueda/interspeech_ser/experiments/LORA_WHISPER_LARGE_V3/whisper_lora_ser.pt", map_location = "cpu")
        ssl_model.load_state_dict(checkpoint)
        ssl_model = ssl_model.whisper
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