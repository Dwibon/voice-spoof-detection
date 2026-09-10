# src/extract_embeddings.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ASVspoof2019Dataset
from model import Wav2Vec2Classifier


def extract_and_cache(split, root_dir, output_dir, batch_size=8):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Extracting embeddings for split='{split}' on device={device}")

    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, f"{split}_embeddings.npy")
    labels_path = os.path.join(output_dir, f"{split}_labels.npy")
    filenames_path = os.path.join(output_dir, f"{split}_filenames.npy")

    if os.path.exists(embeddings_path):
        print(f"Already cached at {embeddings_path}, skipping.")
        return

    ds = ASVspoof2019Dataset(root_dir, split=split, max_len_sec=4.0)

    # reuse the model just for its feature_extractor + frozen backbone
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
        for input_values, labels, filenames in tqdm(loader, desc=f"Extracting {split}"):
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
    root_dir = "../data/asvspoof2019/LA"
    output_dir = "../data/cached_embeddings"

    extract_and_cache("train", root_dir, output_dir, batch_size=8)
    extract_and_cache("dev", root_dir, output_dir, batch_size=8)
    extract_and_cache("eval", root_dir, output_dir, batch_size=8)