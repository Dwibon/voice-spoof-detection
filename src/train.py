# src/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import ASVspoof2019Dataset
from model import Wav2Vec2Classifier


def collate_fn(batch, feature_extractor):
    waveforms, labels, filenames = zip(*batch)
    waveforms_np = [w.numpy() for w in waveforms]

    inputs = feature_extractor(
        waveforms_np, sampling_rate=16000, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values  # (batch, num_samples)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return input_values, labels_tensor, filenames


def compute_pos_weight(dataset):
    """pos_weight for BCEWithLogitsLoss: ratio of negative to positive samples.
    Here 'positive' class = spoof (label=1), which is the majority class,
    so pos_weight will actually come out < 1 — that's correct, it downweights
    the already-abundant spoof class relative to bonafide."""
    labels = [label for _, label in dataset.entries]
    num_pos = sum(labels)          # spoof count
    num_neg = len(labels) - num_pos  # bonafide count
    pos_weight = num_neg / num_pos
    print(f"bonafide={num_neg}, spoof={num_pos}, pos_weight={pos_weight:.4f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    root_dir = "../data/asvspoof2019/LA"
    train_ds = ASVspoof2019Dataset(root_dir, split="train", max_len_sec=4.0)
    dev_ds = ASVspoof2019Dataset(root_dir, split="dev", max_len_sec=4.0)

    model = Wav2Vec2Classifier(freeze_backbone=True).to(device)

    collate = lambda batch: collate_fn(batch, model.feature_extractor)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # only the classifier head has requires_grad=True (backbone frozen)
    optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.classifier_head.train()
        total_loss = 0.0

        for step, (input_values, labels, filenames) in enumerate(train_loader):
            input_values = input_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, step {step}/{len(train_loader)}, loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} done. Avg train loss: {avg_loss:.4f}")

        # quick dev check
        model.classifier_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_values, labels, filenames in dev_loader:
                input_values = input_values.to(device)
                labels = labels.to(device)
                logits = model(input_values)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1} dev accuracy: {correct/total:.4f}")

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/wav2vec2_classifier_baseline.pt")
    print("Saved model to ../models/wav2vec2_classifier_baseline.pt")


if __name__ == "__main__":
    train()