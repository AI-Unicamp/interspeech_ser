import numpy as np
import pickle as pk
import torch.utils as torch_utils
from . import normalizer
import random 

# from torch import nn
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
import parselmouth
def sampler(ratio):
  shifts = np.random.rand((1)) * (ratio - 1.) + 1.

  # print(shifts)
  # flip
  flip = np.random.rand((1)) < 0.5

  # print(flip)

  shifts[flip] = shifts[flip] ** -1
  return shifts[0]

def sliced_timbre_perturb(wav, sr = 22050, segment_size= 22050//2, formant_rate=1.4, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname='null'):
  out = np.array([])
  # print(segment_size)
  for i in range((len(wav)//segment_size) + 1):
    formant_shift = sampler(formant_rate)
    # print(segment_size*i, segment_size*(i+1))
    out_tmp = timbre_perturb(wav[segment_size*i:segment_size*(i+1)], sr, formant_shift, pitch_steps, pitch_floor, pitch_ceil, fname)

    out = np.concatenate((out,out_tmp))

  return out

def timbre_perturb(wav, sr, formant_shift=1.0, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname = 'null'):
  snd = parselmouth.Sound(wav, sampling_frequency=sr)
  # pitch_steps: float = 0.01
  # pitch_floor: float = 75
  # pitch_ceil: float = 600
  # pitch, pitch_median = get_pitch_median(snd, None)
  # pitch_floor = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic")

  ## Customize
  formant_shift= formant_shift
  pitch_shift = 1.
  pitch_range = 1.
  duration_factor = 1.

  # fname = "/content/ex_bia.wav"
#   snd, sr  =  librosa.load(wav, sr = sr)
#   snd = wav

  try:
    pitch = parselmouth.praat.call(
        snd, 'To Pitch', pitch_steps, pitch_floor, pitch_ceil)
  except:
    if(fname != 'null'):
      print(fname)
    # GLOBAL_COUNT+=1
    return snd
  ndpit = pitch.selected_array['frequency']
  # if all unvoiced
  nonzero = ndpit > 1e-5
  if nonzero.sum() == 0:
      return snd
  # if voiced
  # print(ndpit[nonzero].min())
  median, minp = np.median(ndpit[nonzero]).item(), ndpit[nonzero].min().item()
  # scale
  updated = median * pitch_shift
  scaled = updated + (minp * pitch_shift - updated) * pitch_range
  # for preventing infinite loop of `Change gender`
  # ref:https://github.com/praat/praat/issues/1926
  if scaled < 0.:
      pitch_range = 1.
  out, = parselmouth.praat.call(
      (snd, pitch), 'Change gender',
      formant_shift,
      median * pitch_shift,
      pitch_range,
      duration_factor).values

  return out

def fixed_timbre_perturb(wav, sr = 22050, segment_size= 22050//2, formant_rate=1.4, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname='null'):

  formant_shift = sampler(formant_rate)
#   print(formant_shift)
  # print(segment_size*i, segment_size*(i+1))
  out_tmp = timbre_perturb(wav, sr, formant_shift, pitch_steps, pitch_floor, pitch_ceil, fname)

  return out_tmp

"""
All dataset should have the same order based on the utt_list
"""
def load_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    return wav_mean, wav_std


class CombinedSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CombinedSet, self).__init__()
        self.datasets = kwargs.get("datasets", args[0]) 
        self.data_len = len(self.datasets[0])
        for cur_dataset in self.datasets:
            assert len(cur_dataset) == self.data_len, "All dataset should have the same order based on the utt_list"
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        result = []
        for cur_dataset in self.datasets:
            result.append(cur_dataset[idx])
        return result


class TxtSet(torch_utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )

        return (encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0))

class WavSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        self.normalize_wav = kwargs.get("normalize_wav", True)
        self.use_tp = kwargs.get("use_tp", False)
        self.tp_prob = kwargs.get("tp_prob", 1)
        self.type_processor = kwargs.get("type_processor", "whisper")
        self.processor = kwargs.get("processor", None)
        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])

        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = normalizer. get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        
        if(self.use_tp):
            r = random.random()
            if(r<self.tp_prob):
                cur_wav = fixed_timbre_perturb(cur_wav, sr = 16000, segment_size= 16000//2, formant_rate=1.4, pitch_steps = 0.01, pitch_floor=75, pitch_ceil=600, fname='null')

        if(self.normalize_wav):
            cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)

        # print(cur_wav.shape)
        if(self.processor is not None):
            if(self.type_processor == "whisper"):
                cur_wav = self.processor(
                cur_wav, 
                sampling_rate=16000, 
                return_tensors='pt'
                )
                # print(cur_wav)
                cur_wav = cur_wav["input_features"].squeeze(0)
            else:
                cur_wav = self.processor(
                cur_wav, 
                sampling_rate=16000, 
                return_tensors='pt',
                padding=True
                )["input_values"].squeeze(0)
                # print(cur_wav.shape)

        # print(cur_wav.shape)

        result = (cur_wav, cur_dur)
        return result

class ADV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = (cur_lab - self.min_score) / (self.max_score-self.min_score)
        result = cur_lab
        return result
    
class CAT_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CAT_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        result = cur_lab
        return result

class SpkSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SpkSet, self).__init__()
        self.spk_list = kwargs.get("spk_list", args[0])
    
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        cur_lab = self.spk_list[idx]
        result = cur_lab
        return result    

class UttSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(UttSet, self).__init__()
        self.utt_list = kwargs.get("utt_list", args[0])
    
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        cur_lab = self.utt_list[idx]
        result = cur_lab
        return result
