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
df = pd.read_csv('benchmark/' + label_path)
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

if __name__ == "__main__":
    print('hello world')