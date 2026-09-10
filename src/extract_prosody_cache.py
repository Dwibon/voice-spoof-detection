# src/extract_prosody_cache.py

import os
import numpy as np
from tqdm import tqdm

from dataset import ASVspoof2019Dataset
from prosody_features import extract_prosodic_features


FEATURE_NAMES = [
    "pitch_mean_hz",
    "pitch_variance_hz2",
    "voiced_ratio",
    "pause_ratio",
    "pause_count",
    "speech_rate_proxy",
]


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ROOT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "asvspoof2019",
    "LA",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cached_prosody",
)


def extract_prosody_cache(split):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features_path = os.path.join(
        OUTPUT_DIR,
        f"{split}_prosody.npy",
    )

    labels_path = os.path.join(
        OUTPUT_DIR,
        f"{split}_labels.npy",
    )

    filenames_path = os.path.join(
        OUTPUT_DIR,
        f"{split}_filenames.npy",
    )

    if (
        os.path.exists(features_path)
        and os.path.exists(labels_path)
        and os.path.exists(filenames_path)
    ):
        print(f"\nAlready cached: {split}")
        print(f"Features: {features_path}")
        return

    print(f"\n=== Extracting prosodic features: {split} ===")

    ds = ASVspoof2019Dataset(
        ROOT_DIR,
        split=split,
        max_len_sec=4.0,
    )

    print(f"Files in split: {len(ds)}")

    all_features = []
    all_labels = []
    all_filenames = []

    for idx in tqdm(
        range(len(ds)),
        desc=f"Prosody {split}",
    ):

        waveform, label, filename = ds[idx]

        # Protocol filenames do not contain ".flac".
        # Make sure the actual audio filename does.
        if not filename.endswith(".flac"):
            audio_filename = filename + ".flac"
        else:
            audio_filename = filename

        filepath = os.path.join(
            ds.audio_dir,
            audio_filename,
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Audio file not found:\n{filepath}"
            )

        features = extract_prosodic_features(filepath)

        feature_vector = [
            features["pitch_mean_hz"],
            features["pitch_variance_hz2"],
            features["voiced_ratio"],
            features["pause_ratio"],
            features["pause_count"],
            features["speech_rate_proxy"],
        ]

        all_features.append(feature_vector)
        all_labels.append(label)

        # Keep the original dataset filename convention
        # so we can compare ordering with the existing cache.
        all_filenames.append(filename)

    features_array = np.asarray(
        all_features,
        dtype=np.float32,
    )

    labels_array = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    filenames_array = np.asarray(
        all_filenames,
    )

    np.save(features_path, features_array)
    np.save(labels_path, labels_array)
    np.save(filenames_path, filenames_array)

    print("\nSaved:")
    print(f"  Features:  {features_path}")
    print(f"  Labels:    {labels_path}")
    print(f"  Filenames:  {filenames_path}")

    print(f"\nFeature shape: {features_array.shape}")
    print(f"Labels shape:  {labels_array.shape}")
    print(f"Files:         {len(filenames_array)}")

    print("\nFeature columns:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {i}: {name}")


if __name__ == "__main__":

    extract_prosody_cache("train")
    extract_prosody_cache("dev")
    extract_prosody_cache("eval")