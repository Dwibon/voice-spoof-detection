# src/train_from_cache.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_cached(split, cache_dir="../data/cached_embeddings"):
    embeddings = np.load(f"{cache_dir}/{split}_embeddings.npy")
    labels = np.load(f"{cache_dir}/{split}_labels.npy")
    return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def compute_eer(labels, scores):
    """
    labels: 0=bonafide, 1=spoof (your convention)
    scores: predicted probability of being spoof (higher = more likely spoof)
    EER is computed treating 'spoof' as the positive class.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    eer_threshold = thresholds[eer_threshold_idx]
    return eer, eer_threshold


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_x, train_y = load_cached("train")
    dev_x, dev_y = load_cached("dev")
    print("Train:", train_x.shape, "Dev:", dev_x.shape)

    train_ds = TensorDataset(train_x, train_y)
    dev_ds = TensorDataset(dev_x, dev_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False)

    num_pos = train_y.sum().item()
    num_neg = len(train_y) - num_pos
    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)
    print(f"bonafide={int(num_neg)}, spoof={int(num_pos)}, pos_weight={pos_weight.item():.4f}")

    model = ClassifierHead(input_dim=768).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    best_eer = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dev_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        dev_acc = correct / total
        dev_eer, dev_eer_thresh = compute_eer(np.array(all_labels), np.array(all_preds))
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_loss:.4f} | dev_acc={dev_acc:.4f} | dev_EER={dev_eer*100:.2f}%")

        if dev_eer < best_eer:
            best_eer = dev_eer
            best_epoch = epoch + 1
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/classifier_head_best.pt")

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/classifier_head_baseline.pt")

    print(f"\nBest dev EER: {best_eer*100:.2f}% at epoch {best_epoch}")
    print("Best model saved to ../models/classifier_head_best.pt")

    return model, np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    train()