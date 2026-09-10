import os
import random

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROTOCOL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "asvspoof2019",
    "LA",
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt",
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "asvspoof2019",
    "LA",
    "ASVspoof2019_LA_train",
    "flac",
)

records = []

with open(PROTOCOL_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 2:
            continue

        speaker = parts[0]
        file_id = parts[1]

        if not file_id.endswith(".flac"):
            file_id += ".flac"

        filepath = os.path.join(AUDIO_DIR, file_id)

        if os.path.exists(filepath):
            records.append((speaker, filepath))

# Find two recordings belonging to different speakers
random.seed(42)

speaker_a, file_a = random.choice(records)

different = [
    (speaker, filepath)
    for speaker, filepath in records
    if speaker != speaker_a
]

speaker_b, file_b = random.choice(different)

print("Different-speaker pair:")
print()
print("Speaker A:", speaker_a)
print("File A:", file_a)
print()
print("Speaker B:", speaker_b)
print("File B:", file_b)
