# src/dataset.py

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio  # still used for resampling only
from torch.utils.data import Dataset


class ASVspoof2019Dataset(Dataset):
    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    PROTOCOL_FILES = {
        "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    }

    AUDIO_DIRS = {
        "train": "ASVspoof2019_LA_train/flac",
        "dev":   "ASVspoof2019_LA_dev/flac",
        "eval":  "ASVspoof2019_LA_eval/flac",
    }

    def __init__(self, root_dir: str, split: str, target_sr: int = 16000, max_len_sec: float = 4.0):
        assert split in self.PROTOCOL_FILES, f"split must be one of {list(self.PROTOCOL_FILES)}"
        self.root_dir = root_dir
        self.split = split
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_len_sec)

        protocol_path = os.path.join(root_dir, self.PROTOCOL_FILES[split])
        self.audio_dir = os.path.join(root_dir, self.AUDIO_DIRS[split])

        self.entries = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[1]
                label_str = parts[-1]
                label = self.LABEL_MAP[label_str]
                self.entries.append((filename, label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, label = self.entries[idx]
        filepath = os.path.join(self.audio_dir, filename + ".flac")

        audio_np, sr = sf.read(filepath, dtype="float32")  # shape: (num_samples,) since mono
        waveform = torch.from_numpy(audio_np)

        if waveform.ndim > 1:  # safety: collapse to mono if somehow stereo
            waveform = waveform.mean(dim=-1)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        if waveform.shape[0] > self.max_len:
            waveform = waveform[: self.max_len]
        else:
            pad_len = self.max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform, label, filename