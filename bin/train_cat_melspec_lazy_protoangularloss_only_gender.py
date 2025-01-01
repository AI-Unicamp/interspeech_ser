import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import argparse
import sys
import os
import random
import librosa
import numpy as np
sys.path.append(os.getcwd())
from src.information_encoder.samplers import PerfectBatchSampler
from src.information_encoder.losses import AngleProtoLoss
from benchmark import utils
import logging
import time
from tqdm import tqdm
from src.losses import loss
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModel
from torch import nn
import torch.nn.functional as F
from benchmark.utils.dataset.dataset import fixed_timbre_perturb

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

def set_deterministic(seed: int = 42):
    """
    Configure PyTorch and other libraries for deterministic behavior.

    Args:
        seed (int): Seed value for random number generators.
    """
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set seeds for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--config_path", type=str, default="./configs/config_cat.json")

args = parser.parse_args()

set_deterministic(seed=args.seed)

import json
from collections import defaultdict
config_path = args.config_path
with open(config_path, "r") as f:
    config = json.load(f)
base_path = config['wav_dir']
audio_lazy_path = config['audio_lazy_dir']
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

logger.info(f"Starting an Lazy OwnSermodel wavlm-based experiment in model path = {MODEL_PATH}")
logger.info(f"Using LR = {LR} Epochs = {EPOCHS} Batch size = {BATCH_SIZE} Accum steps = {ACCUMULATION_STEP}")
logger.info(f"Using balanced batch = {balanced_batch}")
logger.info(f"Using focalloss = {use_focalloss}")


import pandas as pd
import numpy as np

# Load the CSV file
label_df = pd.read_csv(label_path)
text_df = pd.read_csv(text_path)

gender_labels = pd.read_csv('/workspace/lucas.ueda/interspeech_ser/data/Labels/labels_consensus.csv')
df = label_df.merge(text_df, on = 'FileName', how = 'left')
df = label_df.merge(gender_labels[["FileName","Gender"]], on = 'FileName', how = 'left')

# Classes (emotions)
classes = ['Female', 'Male']

#Collapsing target
# df["target"] = df[classes].idxmax(axis=1)
label_mapping = {name: idx for idx, name in enumerate(classes)}
df["target"] = df["Gender"].map(label_mapping)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']
val_df = df[df['Split_Set'] == 'Development']



# logger.info(f"Class weights: {class_weights_tensor}")

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function for padding feature tensors in a batch.

    Args:
        batch (list of dict): Each dict contains 'feat' (tensor) and 'label' (tensor).

    Returns:
        dict: Batched and padded features and labels.
    """
    # Extract features and labels from the batch
    feats = [item['feat'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad features to have the same length
    padded_feats = pad_sequence(feats, batch_first=True)  # (batch_size, max_seq_len, feat_dim)

    # Stack labels into a single tensor
    stacked_labels = torch.stack(labels)  # (batch_size, num_labels)

    return {
        "feat": padded_feats,
        "label": stacked_labels,
    }




import torchaudio
import torchaudio.transforms as T

import random
class EncoderDataset(Dataset):
    def __init__(
        self,
        df,
        audio_path,
        num_classes_in_batch=8,
        num_utter_per_class=10,
        verbose=True
    ):
        """
        Args:
            ap (TTS.tts.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            seq_len (int): voice segment length in seconds.
            verbose (bool): print diagnostic information.
        """
        super().__init__()
        self.items = df
        self.audio_path = audio_path
        self.num_utter_per_class = num_utter_per_class
        self.classes, self.items = self.__parse_items()

        # Create the mel spectrogram transform
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=1600,
            n_fft=400 * 2,
            win_length=400,
            hop_length=160,
            n_mels=80
        )

        self.classname_to_classid = {key: i for i, key in enumerate(self.classes)}
        self.verbose = verbose
        # Data Augmentation
        # self.augmentator = None
        # self.gaussian_augmentation_config = None
        # if augmentation_config:
        #     self.data_augmentation_p = augmentation_config["p"]
        #     if self.data_augmentation_p and ("additive" in augmentation_config or "rir" in augmentation_config):
        #         self.augmentator = AugmentWAV(ap, augmentation_config)

        #     if "gaussian" in augmentation_config.keys():
        #         self.gaussian_augmentation_config = augmentation_config["gaussian"]

        if self.verbose:
            print("\n > DataLoader initialization")
            print(f" | > Classes per Batch: {num_classes_in_batch}")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Num Classes: {len(self.classes)}")
            print(f" | > Classes: {self.classes}")

    def load_and_extract_melspectrogram(self,filepath):
        """
        Loads a WAV file and extracts a mel spectrogram.
        
        Args:
            filepath (str): Path to the WAV file.
            sample_rate (int): Target sample rate for resampling. Default is 16kHz.
            n_mels (int): Number of mel filter banks. Default is 80.
            win_length (int): Window size in samples. Default is 400 (25ms at 16kHz).
            hop_length (int): Hop size in samples. Default is 160 (10ms at 16kHz).

        Returns:
            torch.Tensor: Mel spectrogram of shape (length, n_mels).
        """
        # Load the WAV file
        waveform, original_sample_rate = librosa.load(filepath, sr = 16000)
        # r = random.random()
        # if(r<0.5):
            # waveform = fixed_timbre_perturb(waveform, sr = 16000, segment_size= 16000//2, formant_rate=1.4, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname='null')

        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Compute the mel spectrogram
        mel_spectrogram = self.mel_spectrogram_transform(waveform)
        
        # Convert to log scale (optional, commonly used for speech processing)
        mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
        
        # Transpose to the shape (length, n_mels)
        mel_spectrogram = mel_spectrogram.squeeze(0).transpose(0, 1)
        
        return mel_spectrogram

    def __parse_items(self):
        class_to_utters = {}
        for i in range(self.items.shape[0]):
            path_ = self.items["FileName"].values[i]
            class_name = self.items['target'].values[i]
            if class_name in class_to_utters.keys():
                class_to_utters[class_name].append(path_)
            else:
                class_to_utters[class_name] = [
                    path_,
                ]

        # skip classes with number of samples >= self.num_utter_per_class
        class_to_utters = {k: v for (k, v) in class_to_utters.items() if len(v) >= self.num_utter_per_class}

        classes = list(class_to_utters.keys())
        classes.sort()

        new_items = []
        for i in range(self.items.shape[0]):
            path_ = self.items["FileName"].values[i]
            class_name = self.items['target'].values[i]
            # ignore filtered classes
            if class_name not in classes:
                continue

            new_items.append({"FileName": path_, "class_name": class_name})

        # print(len(new_items))
        # print(new_items[:5])

        return classes, new_items

    def __len__(self):
        return len(self.items)

    def get_num_classes(self):
        return len(self.classes)

    def get_class_list(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes
        self.classname_to_classid = {key: i for i, key in enumerate(self.classes)}

    def get_map_classid_to_classname(self):
        return dict((c_id, c_n) for c_n, c_id in self.classname_to_classid.items())

    def __getitem__(self, idx):
        # print(idx)
        return self.items[idx]

    def collate_fn(self, batch):
        # get the batch class_ids
        labels = []
        feats = []
        for item in batch:
            utter_path = item["FileName"]
            class_name = item["class_name"]

            # get classid
            class_id = self.classname_to_classid[class_name]

            feat_name = os.path.join(self.audio_path, utter_path)
            feat = self.load_and_extract_melspectrogram(feat_name)

            # load wav file
            # wav = self.load_wav(utter_path)
            # offset = random.randint(0, wav.shape[0] - self.seq_len)
            # wav = wav[offset : offset + self.seq_len]


            # if(self.use_timbre_perturb):
            #     # wav = finegrained_timbre_perturb(np.asarray(wav, dtype=np.float32), 5, self.sample_rate , self.sample_rate//2, 1.4, 0.01, 75,600)
            #     wav = fixed_timbre_perturb(wav, sr = self.sample_rate, segment_size= self.sample_rate//2, formant_rate=1.4, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname='null')
            # # if self.augmentator is not None and self.data_augmentation_p:
            # #     if random.random() < self.data_augmentation_p:
            # #         wav = self.augmentator.apply_one(wav)

            # if not self.use_torch_spec:
            #     mel = self.ap.melspectrogram(wav)
            #     feats.append(torch.FloatTensor(mel))
            # else:
            #     feats.append(torch.FloatTensor(wav))
            feats.append(feat.float())
            labels.append(class_id)

        # feats = torch.stack(feats)
        feats = pad_sequence(feats, batch_first=True) 
        labels = torch.LongTensor(labels)

        return feats, labels


class MultiLabelAudioDataset(Dataset):
    def __init__(self, wav_files, labels, lazy_path):
        self.wav_paths = wav_files
        self.labels = labels
        self.lazy_path = lazy_path

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        feat_name = os.path.join(self.lazy_path, self.wav_paths[idx].replace('.wav','.pt'))
        label = self.labels[idx]
        feat = torch.load(feat_name)
        return {
            "feat": feat,
            "label": torch.tensor(label, dtype=torch.float),  # Use float for BCEWithLogitsLoss
        }


class BidirectionalReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim, use_nonlinear_proj = False):

        super().__init__()
        self.num_mel = num_mel
        self.embedding_dim = embedding_dim
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim//2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.5)
    
        self.use_nonlinear_proj = use_nonlinear_proj

        if(self.use_nonlinear_proj):
            self.proj = nn.Linear(embedding_dim, embedding_dim)
            nn.init.xavier_normal_(self.proj.weight) # Good init for projection
            # self.proj.bias.data.zero_() # Not random bias to "move" z

    def forward(self, inputs):
        batch_size = inputs.size(0)
        # x = inputs.view(batch_size, 1, -1, self.num_mel)
        x = inputs.unsqueeze(1)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==2, batch_size, encoding_size=384]
        out = torch.cat([out[0,:,:], out[1,:,:]], dim = 1)
        # out: 2D tensor [batch_size, encoding_size = 384]

        # print(out.shape)
        if(self.use_nonlinear_proj):
            out = torch.tanh(self.proj(out))
            out = self.dropout(out)
            
        return out

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height

class GenderRepresentationModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, num_categories=8, num_attention_heads=4):
        super().__init__()

        # Initial dropout for input features
        self.wav_dropout = nn.Dropout(0.5)

        # Linear projection to hidden dimension
        self.wav_proj = nn.Linear(wavlm_dim, hidden_dim)

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.5,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(hidden_dim)

        # Attention pooling layer
        self.attn_pooling = nn.Linear(hidden_dim, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_categories)
        )

    def forward(self, wavlm_output):
        # Apply initial dropout to inputs
        wav_x = self.wav_dropout(wavlm_output)

        # Project to hidden dimension
        wav_x = self.wav_proj(wav_x)

        # Process with multi-head attention
        attn_output, _ = self.multihead_attn(wav_x, wav_x, wav_x)
        wav_x = self.attn_norm(attn_output + wav_x)  # Residual connection with layer norm

        # Convolutional processing
        wav_x = wav_x.transpose(1, 2)  # Change to (batch_size, hidden_dim, seq_len) for Conv1D
        conv_x = self.conv1d(wav_x)
        conv_x = self.conv_norm(conv_x.transpose(1, 2))  # Back to (batch_size, seq_len, hidden_dim) after LayerNorm

        # Attention pooling
        attn_weights = F.softmax(self.attn_pooling(conv_x), dim=1)  # Shape: (batch_size, seq_len, 1)
        embeddings = torch.sum(conv_x * attn_weights, dim=1)  # Weighted sum over sequence length

        # Pass through classifier
        output = self.classifier(embeddings)

        return embeddings, output


num_classes_in_batch = 2
num_utter_per_class = 32
num_utter_per_class_val = 32


train_dataset = EncoderDataset(train_df, base_path, num_classes_in_batch,num_utter_per_class)
val_dataset = EncoderDataset(val_df, base_path, num_classes_in_batch,num_utter_per_class_val)


logger.info(f'Using Perfect Sampler Batch. Computing items...')
# class_frequencies = train_df[classes].sum().to_dict()



classes_ = train_dataset.get_class_list()

train_sampler = PerfectBatchSampler(
    train_dataset.items,
    classes_,
    batch_size=num_classes_in_batch * num_utter_per_class,  # total batch size
    num_classes_in_batch=num_classes_in_batch,
    num_gpus=1,
    shuffle=True,
    drop_last=True,
)

val_sampler = PerfectBatchSampler(
    val_dataset.items,
    classes_,
    batch_size=num_classes_in_batch * num_utter_per_class_val,  # total batch size
    num_classes_in_batch=num_classes_in_batch,
    num_gpus=1,
    shuffle=False,
    drop_last=True,
)


train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=val_dataset.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ser_model = BidirectionalReferenceEncoder(num_mel=80, embedding_dim=256, use_nonlinear_proj = False)
ser_model.to(device)

# optimizer = torch.optim.AdamW(ser_model.parameters(), lr=LR, weight_decay=1e-6)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = torch.optim.RAdam(ser_model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)

# Create scheduler
batch_size = num_classes_in_batch * num_utter_per_class
dataset_size = train_df.shape[0]
steps_per_epoch = math.ceil(dataset_size / batch_size)  # ≈ 313 steps
total_iterations = EPOCHS * steps_per_epoch  # 100 * 313 = 31,300
scheduler = CosineAnnealingScheduler(
    optimizer,
    T_max=total_iterations,  # Total epochs or iterations
    eta_min=1e-6  # Minimum learning rate
)

min_epoch=0
min_loss=1e10

logger.info("Starting training...")
# focal_loss = loss.FocalLoss(alpha=1, gamma=2, reduction='mean', dynamic_alpha=True)

angleloss = AngleProtoLoss()

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    ser_model.train()
    batch_cnt = 0

    for batch in tqdm(train_loader):
        inputs, labels = batch

        labels=labels.cuda(non_blocking=True).long()
        inputs=inputs.cuda(non_blocking=True)

        # agroup samples of each class in the batch. perfect sampler produces [3,2,1,3,2,1] we need [3,3,2,2,1,1]
        labels = torch.transpose(
            labels.view(num_utter_per_class, num_classes_in_batch), 0, 1
        ).reshape(labels.shape)
        inputs = torch.transpose(
            inputs.view(num_utter_per_class, num_classes_in_batch, -1), 0, 1
        ).reshape(inputs.shape)


        optimizer.zero_grad()
        embeddings  = ser_model(inputs)

        # loss_ce = utils.CE_weight_category(logits, labels, None)
        loss_angle = angleloss(
                        embeddings.view(num_classes_in_batch, embeddings.shape[0] // num_classes_in_batch, -1), labels
                    )

        total_loss = loss_angle

        total_loss.backward()

        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(train_loader):
            optimizer.step()
            scheduler.step()
        batch_cnt += 1
        current_lr = scheduler.get_last_lr()[0]
        # Logging
        if((batch_cnt+1)%50 == 0):
            logger.info(f"Epoch ({epoch+1}/{EPOCHS})| step = {batch_cnt}: loss angle = {loss_angle} current lr = {current_lr}")

    ser_model.eval()

    losses_angles = []
    for batch in tqdm(val_loader):
        inputs, labels = batch

        labels=labels.cuda(non_blocking=True).long()
        inputs=inputs.cuda(non_blocking=True)

                # agroup samples of each class in the batch. perfect sampler produces [3,2,1,3,2,1] we need [3,3,2,2,1,1]
        labels = torch.transpose(
            labels.view(num_utter_per_class_val, num_classes_in_batch), 0, 1
        ).reshape(labels.shape)
        inputs = torch.transpose(
            inputs.view(num_utter_per_class_val, num_classes_in_batch, -1), 0, 1
        ).reshape(inputs.shape)

        with torch.no_grad():

            embeddings = ser_model(inputs)

            # loss_ce = utils.CE_weight_category(logits, labels, None)
            loss_angle = angleloss(
                            embeddings.view(num_classes_in_batch, embeddings.shape[0] // num_classes_in_batch, -1), labels
                        )

            losses_angles.append(loss_angle.item())

        # Calculate accuracy based on one-hot predictions
        # val_preds_binary = (torch.tensor(val_preds)).numpy()
        # val_labels_binary = (torch.tensor(val_labels)).numpy()
        # f1 = f1_score(val_labels, val_preds, average='macro')

    angle_loss_total = np.mean(losses_angles)

    # Logging
    logger.info(f"|VALIDATION| Epoch ({epoch+1}/{EPOCHS}): eval angle loss = {angle_loss_total}")

    if min_loss > angle_loss_total:
        logger.info(f"New best model at epoch {epoch+1}")
        min_epoch = epoch
        min_loss = angle_loss_total

        print("Save",min_epoch)
        print("Loss",min_loss)

        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "angle_gender.pt"))

