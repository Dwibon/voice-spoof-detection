# src/eval_itw.py

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from train_from_cache import ClassifierHead, compute_eer


def evaluate_itw():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    cache_dir = "../data/cached_embeddings"
    itw_x = np.load(os.path.join(cache_dir, "itw_embeddings.npy"))
    itw_y = np.load(os.path.join(cache_dir, "itw_labels.npy"))

    itw_x = torch.tensor(itw_x, dtype=torch.float32)
    itw_y = torch.tensor(itw_y, dtype=torch.float32)
    print("ITW set:", itw_x.shape)

    model = ClassifierHead(input_dim=768).to(device)
    model.load_state_dict(torch.load("../models/classifier_head_best.pt", map_location=device))
    model.eval()

    itw_ds = TensorDataset(itw_x, itw_y)
    itw_loader = DataLoader(itw_ds, batch_size=64, shuffle=False)

    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in itw_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    itw_acc = correct / total
    itw_eer, itw_thresh = compute_eer(np.array(all_labels), np.array(all_preds))

    print(f"\n=== In-The-Wild Cross-Dataset Eval Results ===")
    print(f"Accuracy: {itw_acc*100:.2f}%")
    print(f"EER:      {itw_eer*100:.2f}%")
    print(f"EER threshold: {itw_thresh:.4f}")


if __name__ == "__main__":
    evaluate_itw()