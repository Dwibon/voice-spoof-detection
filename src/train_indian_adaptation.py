from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

ASV_CACHE = ROOT / "data" / "cached_embeddings"
INDIAN_CACHE = ROOT / "data" / "cached_indian_adapt"
INDIAN_VAL_CACHE = ROOT / "data" / "cached_indian_val"

BASELINE = ROOT / "models" / "classifier_head_best.pt"
OUTPUT = ROOT / "models" / "classifier_head_indian_adapted.pt"

LR = 1e-4
EPOCHS = 15
BATCH_SIZE = 64
PATIENCE = 4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_cache(cache_dir, prefix):
    X = np.load(cache_dir / f"{prefix}_embeddings.npy")
    y = np.load(cache_dir / f"{prefix}_labels.npy")
    return X.astype(np.float32), y.astype(np.float32)


def load_model():
    model = ClassifierHead()

    state = torch.load(
        BASELINE,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    return model


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    return acc, auc


# ---------------------------------------------------------
# Load ASVspoof training data
# ---------------------------------------------------------

X_asv, y_asv = load_cache(ASV_CACHE, "train")

# ---------------------------------------------------------
# Load Indian adaptation data
# ---------------------------------------------------------

X_ind, y_ind = load_cache(INDIAN_CACHE, "train")

print("========================================")
print("INDIAN DOMAIN ADAPTATION")
print("========================================")
print(f"Device: {DEVICE}")
print(f"ASVspoof training: {X_asv.shape}")
print(f"Indian adaptation: {X_ind.shape}")
print(f"Indian bonafide: {(y_ind == 0).sum()}")
print(f"Indian spoof:    {(y_ind == 1).sum()}")

# ---------------------------------------------------------
# Combine datasets
# ---------------------------------------------------------

X = np.concatenate([X_asv, X_ind], axis=0)
y = np.concatenate([y_asv, y_ind], axis=0)

n_asv = len(X_asv)

# ---------------------------------------------------------
# Sampling strategy
#
# Target distribution:
#   60% ASVspoof
#   40% Indian
#
# Within each domain:
#   50% bonafide
#   50% spoof
# ---------------------------------------------------------

weights = np.zeros(len(y), dtype=np.float64)

asv_bona = (y_asv == 0).sum()
asv_spoof = (y_asv == 1).sum()

ind_bona = (y_ind == 0).sum()
ind_spoof = (y_ind == 1).sum()

# ASVspoof gets 60% of sampling probability.
weights[:n_asv][y_asv == 0] = 0.30 / asv_bona
weights[:n_asv][y_asv == 1] = 0.30 / asv_spoof

# Indian domain gets 40%.
weights[n_asv:][y_ind == 0] = 0.20 / ind_bona
weights[n_asv:][y_ind == 1] = 0.20 / ind_spoof

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(weights),
    num_samples=len(X_asv),
    replacement=True
)

dataset = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(y)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = load_model().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ---------------------------------------------------------
# Validation set
# ---------------------------------------------------------

X_val, y_val = load_cache(INDIAN_VAL_CACHE, "val")

best_acc = -1.0
best_auc = -1.0
best_state = None
patience_count = 0

print("\nStarting adaptation...")
print(f"Learning rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("Sampling: 60% ASVspoof / 40% Indian")
print("Class balance: 50% bonafide / 50% spoof")
print()

for epoch in range(1, EPOCHS + 1):

    model.train()

    losses = []

    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    val_acc, val_auc = evaluate(
        model,
        X_val,
        y_val
    )

    mean_loss = float(np.mean(losses))

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {mean_loss:.4f} | "
        f"Indian Val Accuracy {val_acc*100:.2f}% | "
        f"ROC-AUC {val_auc:.4f}"
    )

    # Select primarily by validation accuracy,
    # secondarily by ROC-AUC.
    improved = (
        val_acc > best_acc
        or (
            val_acc == best_acc
            and val_auc > best_auc
        )
    )

    if improved:
        best_acc = val_acc
        best_auc = val_auc
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= PATIENCE:
        print("\nEarly stopping.")
        break

# ---------------------------------------------------------
# Save best adapted model
# ---------------------------------------------------------

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(
    model.state_dict(),
    OUTPUT
)

print("\n========================================")
print("ADAPTATION COMPLETE")
print("========================================")
print(f"Best Indian validation accuracy: {best_acc*100:.2f}%")
print(f"Best Indian validation ROC-AUC:  {best_auc:.4f}")
print(f"Saved adapted model:")
print(OUTPUT)
