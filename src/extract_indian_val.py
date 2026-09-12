from pathlib import Path
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import soundfile as sf
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = PROJECT_ROOT / "data" / "indian_accent_eval"
OUTPUT = PROJECT_ROOT / "data" / "cached_indian_val"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Loading Wav2Vec2...")

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = False


def extract(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, sr, TARGET_SR
        )

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        output = model(
            inputs.input_values.to(device)
        )

    return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


embeddings = []
labels = []
filenames = []

# 0 = bona-fide, 1 = spoof
sources = [
    (
        BASE / "bonafide_wav",
        0,
        "val.csv"
    ),
    (
        BASE / "spoof_wav" / "val",
        1,
        "val.csv"
    ),
]

import pandas as pd

meta = pd.read_csv(BASE / "val.csv")

for label_dir, label, _ in sources:

    print("\nLabel:", "bonafide" if label == 0 else "spoof")

    for i, row in meta.iterrows():

        stem = f"{row['client_id']}_{row['sentence_id']}"

        if label == 0:
            wav = label_dir / f"{stem}.wav"
        else:
            wav = label_dir / f"{stem}_spoof.wav"

        if not wav.exists():
            raise FileNotFoundError(wav)

        emb = extract(wav)

        embeddings.append(emb)
        labels.append(label)
        filenames.append(str(wav.relative_to(PROJECT_ROOT)))

        print(f"  {i+1}/{len(meta)}", end="\r")

    print()

embeddings = np.stack(embeddings).astype(np.float32)
labels = np.asarray(labels, dtype=np.int64)
filenames = np.asarray(filenames)

np.save(OUTPUT / "val_embeddings.npy", embeddings)
np.save(OUTPUT / "val_labels.npy", labels)
np.save(OUTPUT / "val_filenames.npy", filenames)

print("\n========================================")
print("INDIAN VALIDATION EMBEDDINGS COMPLETE")
print("========================================")
print("Embeddings:", embeddings.shape)
print("Labels:", labels.shape)
print("Bonafide:", int((labels == 0).sum()))
print("Spoof:", int((labels == 1).sum()))
print("Saved to:", OUTPUT)
