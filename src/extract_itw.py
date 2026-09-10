# src/extract_itw.py
#
# Fully self-contained: extracts Wav2Vec2 embeddings for the In-The-Wild
# dataset and caches them, in the exact same format as your existing
# train/dev/eval embeddings. Requires no edits to any other file.
#
# Run with:  python3 extract_itw.py

import os
import csv
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import Wav2Vec2Classifier


# ---------------------------------------------------------------------
# 1. Dataset class for In-The-Wild (mirrors ASVspoof2019Dataset's
#    __getitem__ contract: returns waveform, label, filename)
# ---------------------------------------------------------------------
class ITWDataset(Dataset):
    LABEL_MAP = {"bona-fide": 0, "spoof": 1}  # note: hyphenated, unlike "bonafide"
    META_FILE = "meta.csv"

    def __init__(self, root_dir: str, target_sr: int = 16000, max_len_sec: float = 4.0):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_len_sec)

        meta_path = os.path.join(root_dir, self.META_FILE)
        self.entries = []
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f)  # header: file,speaker,label
            for row in reader:
                filename = row["file"]
                label = self.LABEL_MAP[row["label"].strip()]
                self.entries.append((filename, label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, label = self.entries[idx]
        filepath = os.path.join(self.root_dir, filename)

        audio_np, sr = sf.read(filepath, dtype="float32")
        waveform = torch.from_numpy(audio_np)

        if waveform.ndim > 1:
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


# ---------------------------------------------------------------------
# 2. Extraction routine (same logic as your extract_embeddings.py)
# ---------------------------------------------------------------------
def extract_and_cache_itw(root_dir, output_dir, batch_size=8):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Extracting ITW embeddings on device={device}")

    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, "itw_embeddings.npy")
    labels_path = os.path.join(output_dir, "itw_labels.npy")
    filenames_path = os.path.join(output_dir, "itw_filenames.npy")

    if os.path.exists(embeddings_path):
        print(f"Already cached at {embeddings_path}, skipping.")
        return

    ds = ITWDataset(root_dir, max_len_sec=4.0)
    print(f"Loaded {len(ds)} ITW samples")

    model = Wav2Vec2Classifier(freeze_backbone=True).to(device)
    model.backbone.eval()

    def collate(batch):
        waveforms, labels, filenames = zip(*batch)
        waveforms_np = [w.numpy() for w in waveforms]
        inputs = model.feature_extractor(
            waveforms_np, sampling_rate=16000, return_tensors="pt", padding=True
        )
        return inputs.input_values, torch.tensor(labels), filenames

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_embeddings, all_labels, all_filenames = [], [], []

    with torch.no_grad():
        for input_values, labels, filenames in tqdm(loader, desc="Extracting ITW"):
            input_values = input_values.to(device)
            outputs = model.backbone(input_values)
            pooled = outputs.last_hidden_state.mean(dim=1)  # (batch, 768)

            all_embeddings.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend(filenames)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(embeddings_path, all_embeddings)
    np.save(labels_path, all_labels)
    np.save(filenames_path, np.array(all_filenames))

    print(f"Saved {all_embeddings.shape[0]} embeddings to {embeddings_path}")
    print(f"Embedding shape: {all_embeddings.shape}")


if __name__ == "__main__":
    root_dir = "../data/in_the_wild/release_in_the_wild"
    output_dir = "../data/cached_embeddings"
    extract_and_cache_itw(root_dir, output_dir, batch_size=8)
