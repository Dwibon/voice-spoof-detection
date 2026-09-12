from pathlib import Path
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import soundfile as sf
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "indian_accent_eval"
INPUT_ROOT = DATA_ROOT / "augmented" / "train"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "cached_indian_adapt"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

all_embeddings = []
all_labels = []
all_filenames = []

for label_name, label in [("bonafide", 0), ("spoof", 1)]:

    files = sorted((INPUT_ROOT / label_name).glob("*.wav"))

    print(f"\n{label_name}: {len(files)} files")

    for i, wav_path in enumerate(files, 1):

        audio, sr = sf.read(wav_path, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        waveform = torch.from_numpy(audio)

        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                TARGET_SR
            )

        inputs = processor(
            waveform.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            embedding = outputs.last_hidden_state.mean(dim=1)

        all_embeddings.append(embedding.squeeze(0).cpu().numpy())
        all_labels.append(label)
        all_filenames.append(str(wav_path.relative_to(PROJECT_ROOT)))

        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

embeddings = np.stack(all_embeddings).astype(np.float32)
labels = np.asarray(all_labels, dtype=np.int64)
filenames = np.asarray(all_filenames)

np.save(OUTPUT_ROOT / "train_embeddings.npy", embeddings)
np.save(OUTPUT_ROOT / "train_labels.npy", labels)
np.save(OUTPUT_ROOT / "train_filenames.npy", filenames)

print("\n========================================")
print("INDIAN ADAPTATION EMBEDDINGS COMPLETE")
print("========================================")
print("Embeddings:", embeddings.shape)
print("Labels:", labels.shape)
print("Bonafide:", int((labels == 0).sum()))
print("Spoof:", int((labels == 1).sum()))
print("Saved to:", OUTPUT_ROOT)
