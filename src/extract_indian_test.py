from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import soundfile as sf
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = PROJECT_ROOT / "data" / "indian_accent_eval"
OUTPUT = PROJECT_ROOT / "data" / "cached_indian_test"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"

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


meta = pd.read_csv(BASE / "test.csv")

embeddings = []
labels = []
filenames = []

for label, label_dir, suffix in [
    (0, BASE / "bonafide_wav", ""),
    (1, BASE / "spoof_wav" / "test", "_spoof"),
]:

    name = "bonafide" if label == 0 else "spoof"
    print(f"\n{name}: {len(meta)} files")

    for i, row in meta.iterrows():

        stem = f"{row['client_id']}_{row['sentence_id']}"
        wav = label_dir / f"{stem}{suffix}.wav"

        if not wav.exists():
            raise FileNotFoundError(wav)

        embeddings.append(extract(wav))
        labels.append(label)
        filenames.append(str(wav.relative_to(PROJECT_ROOT)))

        if (i + 1) % 8 == 0 or i + 1 == len(meta):
            print(f"  {i+1}/{len(meta)}")

embeddings = np.stack(embeddings).astype(np.float32)
labels = np.asarray(labels, dtype=np.int64)
filenames = np.asarray(filenames)

np.save(OUTPUT / "test_embeddings.npy", embeddings)
np.save(OUTPUT / "test_labels.npy", labels)
np.save(OUTPUT / "test_filenames.npy", filenames)

print("\n========================================")
print("INDIAN TEST EMBEDDINGS COMPLETE")
print("========================================")
print("Embeddings:", embeddings.shape)
print("Labels:", labels.shape)
print("Bonafide:", int((labels == 0).sum()))
print("Spoof:", int((labels == 1).sum()))
print("Saved to:", OUTPUT)
