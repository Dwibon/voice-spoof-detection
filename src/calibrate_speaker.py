import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from speaker_check import SpeakerConsistencyChecker

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

NUM_PAIRS = 500
SEED = 42


def load_protocol():
    records = []

    with open(PROTOCOL_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            speaker_id = parts[0]
            file_id = parts[1]

            if not file_id.endswith(".flac"):
                file_id += ".flac"

            filepath = os.path.join(AUDIO_DIR, file_id)

            if os.path.exists(filepath):
                records.append(
                    {
                        "speaker": speaker_id,
                        "file": filepath,
                    }
                )

    return records


def build_pairs(records):
    random.seed(SEED)

    speakers = {}

    for record in records:
        speakers.setdefault(
            record["speaker"], []
        ).append(record)

    # Same-speaker pairs
    same_pairs = []

    for speaker, files in speakers.items():
        if len(files) < 2:
            continue

        shuffled = files.copy()
        random.shuffle(shuffled)

        for i in range(len(shuffled) - 1):
            same_pairs.append(
                (
                    shuffled[i],
                    shuffled[i + 1],
                )
            )

    random.shuffle(same_pairs)
    same_pairs = same_pairs[:NUM_PAIRS]

    # Different-speaker pairs
    speaker_list = list(speakers.keys())

    different_pairs = []

    while len(different_pairs) < NUM_PAIRS:
        speaker_a, speaker_b = random.sample(
            speaker_list,
            2,
        )

        record_a = random.choice(
            speakers[speaker_a]
        )

        record_b = random.choice(
            speakers[speaker_b]
        )

        different_pairs.append(
            (
                record_a,
                record_b,
            )
        )

    return same_pairs, different_pairs


def main():
    print("Loading protocol...")

    records = load_protocol()

    print(
        f"Available audio records: {len(records)}"
    )

    same_pairs, different_pairs = build_pairs(
        records
    )

    print(
        f"Same-speaker pairs: "
        f"{len(same_pairs)}"
    )

    print(
        f"Different-speaker pairs: "
        f"{len(different_pairs)}"
    )

    checker = SpeakerConsistencyChecker()

    # Cache embeddings so we don't repeatedly
    # process the same audio.
    embedding_cache = {}

    def get_embedding(filepath):
        if filepath not in embedding_cache:
            embedding_cache[filepath] = (
                checker.get_embedding(filepath)
            )

        return embedding_cache[filepath]

    def similarity(file_a, file_b):
        emb_a = get_embedding(file_a)
        emb_b = get_embedding(file_b)

        return torch.dot(
            F.normalize(emb_a, dim=0),
            F.normalize(emb_b, dim=0),
        ).item()

    same_scores = []
    different_scores = []

    print("\nCalculating same-speaker similarities...")

    for i, (a, b) in enumerate(same_pairs):
        score = similarity(
            a["file"],
            b["file"],
        )

        same_scores.append(score)

        print(
            f"Same {i + 1:02d}/"
            f"{len(same_pairs)}: "
            f"{score:.4f}"
        )

    print("\nCalculating different-speaker similarities...")

    for i, (a, b) in enumerate(different_pairs):
        score = similarity(
            a["file"],
            b["file"],
        )

        different_scores.append(score)

        print(
            f"Different {i + 1:02d}/"
            f"{len(different_pairs)}: "
            f"{score:.4f}"
        )

    same_scores = np.array(same_scores)
    different_scores = np.array(different_scores)

    print("\n==============================")
    print("SPEAKER SIMILARITY CALIBRATION")
    print("==============================")

    print("\nSame-speaker:")
    print(
        f"Mean:   {same_scores.mean():.4f}"
    )
    print(
        f"Median: {np.median(same_scores):.4f}"
    )
    print(
        f"Min:    {same_scores.min():.4f}"
    )
    print(
        f"Max:    {same_scores.max():.4f}"
    )

    print("\nDifferent-speaker:")
    print(
        f"Mean:   {different_scores.mean():.4f}"
    )
    print(
        f"Median: {np.median(different_scores):.4f}"
    )
    print(
        f"Min:    {different_scores.min():.4f}"
    )
    print(
        f"Max:    {different_scores.max():.4f}"
    )

    # Find threshold with minimum classification error.
    all_scores = np.concatenate(
        [same_scores, different_scores]
    )

    thresholds = np.linspace(
        all_scores.min(),
        all_scores.max(),
        500,
    )

    best_threshold = None
    best_accuracy = -1

    for threshold in thresholds:

        same_correct = (
            same_scores >= threshold
        ).sum()

        different_correct = (
            different_scores < threshold
        ).sum()

        accuracy = (
            same_correct + different_correct
        ) / (
            len(same_scores)
            + len(different_scores)
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print("\nCalibrated threshold:")
    print(
        f"Threshold: {best_threshold:.4f}"
    )
    print(
        f"Pair classification accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
