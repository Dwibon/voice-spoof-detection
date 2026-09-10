# src/eval_from_cache.py

import numpy as np
import torch

from train_from_cache import ClassifierHead, load_cached, compute_eer


def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    eval_x, eval_y = load_cached("eval")
    print("Eval set:", eval_x.shape)

    model = ClassifierHead(input_dim=768).to(device)
    model.load_state_dict(torch.load("../models/classifier_head_best.pt", map_location=device))
    model.eval()

    from torch.utils.data import TensorDataset, DataLoader
    eval_ds = TensorDataset(eval_x, eval_y)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    eval_acc = correct / total
    eval_eer, eval_thresh = compute_eer(np.array(all_labels), np.array(all_preds))

    print(f"\n=== ASVspoof2019 LA Eval Results ===")
    print(f"Accuracy: {eval_acc*100:.2f}%")
    print(f"EER:      {eval_eer*100:.2f}%")
    print(f"EER threshold: {eval_thresh:.4f}")


if __name__ == "__main__":
    evaluate()