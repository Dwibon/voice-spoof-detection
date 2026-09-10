import os
import csv
import numpy as np

from prosody_features import extract_prosodic_features


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = (
    "data/asvspoof2019/LA/"
    "ASVspoof2019_LA_eval/flac"
)

PROTOCOL_PATH = (
    "data/asvspoof2019/LA/"
    "ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.eval.trl.txt"
)

NUM_BONAFIDE = 50
NUM_SPOOF = 50

OUTPUT_CSV = "data/prosodic_analysis.csv"


# ============================================================
# LOAD LABELS FROM ASVSPOOF PROTOCOL
# ============================================================

def load_labels():

    if not os.path.exists(PROTOCOL_PATH):
        raise FileNotFoundError(
            f"Protocol file not found:\n{PROTOCOL_PATH}"
        )

    labels = {}

    with open(
        PROTOCOL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            # Example protocol line:
            #
            # LA_0039 LA_E_2834763 - A11 spoof
            #
            # parts[0] = LA_0039
            # parts[1] = LA_E_2834763
            # parts[-1] = spoof

            if len(parts) < 5:
                continue

            # IMPORTANT:
            # The second field is the actual audio filename ID.
            filename = parts[1]

            # Last field is the class label.
            label = parts[-1].lower()

            if label in ("bonafide", "spoof"):

                labels[filename] = label

    return labels


# ============================================================
# FIND AUDIO FILES
# ============================================================

def find_labeled_files(labels):

    available = {
        "bonafide": [],
        "spoof": []
    }

    for filename, label in labels.items():

        filepath = os.path.join(
            DATASET_ROOT,
            filename + ".flac"
        )

        if os.path.exists(filepath):

            available[label].append(
                filepath
            )

    return available


# ============================================================
# SELECT BALANCED DATASET
# ============================================================

def select_files(available):

    print(
        f"Available bonafide files: "
        f"{len(available['bonafide'])}"
    )

    print(
        f"Available spoof files: "
        f"{len(available['spoof'])}"
    )

    if len(available["bonafide"]) < NUM_BONAFIDE:

        raise ValueError(
            f"Only {len(available['bonafide'])} "
            f"bonafide files were found, "
            f"but {NUM_BONAFIDE} are required."
        )

    if len(available["spoof"]) < NUM_SPOOF:

        raise ValueError(
            f"Only {len(available['spoof'])} "
            f"spoof files were found, "
            f"but {NUM_SPOOF} are required."
        )

    # Sort to make the experiment reproducible.
    bonafide = sorted(
        available["bonafide"]
    )[:NUM_BONAFIDE]

    spoof = sorted(
        available["spoof"]
    )[:NUM_SPOOF]

    return bonafide, spoof


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_dataset(files, label):

    results = []

    total = len(files)

    for index, filepath in enumerate(files):

        filename = os.path.basename(
            filepath
        )

        print(
            f"[{index + 1:03d}/{total:03d}] "
            f"{label:8s} "
            f"{filename}"
        )

        try:

            features = extract_prosodic_features(
                filepath
            )

            row = {
                "filename": filename,
                "label": label,
                **features
            }

            results.append(row)

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_csv(results):

    output_dir = os.path.dirname(
        OUTPUT_CSV
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    fieldnames = [
        "filename",
        "label",
        "pitch_mean_hz",
        "pitch_variance_hz2",
        "voiced_ratio",
        "pause_ratio",
        "pause_count",
        "speech_rate_proxy"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nSaved results to:\n{OUTPUT_CSV}"
    )


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "pitch_mean_hz",
    "pitch_variance_hz2",
    "voiced_ratio",
    "pause_ratio",
    "pause_count",
    "speech_rate_proxy"
]


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(results):

    print(
        "\n=== PROSODIC FEATURE ANALYSIS ==="
    )

    bonafide = [
        row
        for row in results
        if row["label"] == "bonafide"
    ]

    spoof = [
        row
        for row in results
        if row["label"] == "spoof"
    ]

    print(
        f"\nBonafide samples analyzed: "
        f"{len(bonafide)}"
    )

    print(
        f"Spoof samples analyzed: "
        f"{len(spoof)}"
    )

    print(
        "\n"
        f"{'Feature':<24}"
        f"{'Bonafide Mean':>16}"
        f"{'Spoof Mean':>16}"
        f"{'Difference':>16}"
    )

    print("-" * 72)

    for feature in FEATURES:

        bonafide_values = np.array([
            float(row[feature])
            for row in bonafide
        ])

        spoof_values = np.array([
            float(row[feature])
            for row in spoof
        ])

        bonafide_mean = np.mean(
            bonafide_values
        )

        spoof_mean = np.mean(
            spoof_values
        )

        difference = (
            spoof_mean
            - bonafide_mean
        )

        print(
            f"{feature:<24}"
            f"{bonafide_mean:>16.4f}"
            f"{spoof_mean:>16.4f}"
            f"{difference:>16.4f}"
        )


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_effect_sizes(results):

    print(
        "\n=== STANDARDIZED CLASS DIFFERENCE ==="
    )

    print(
        "Larger absolute values indicate "
        "stronger separation between the two groups."
    )

    print(
        f"\n{'Feature':<24}"
        f"{'Effect Size':>16}"
    )

    print("-" * 42)

    bonafide = [
        row
        for row in results
        if row["label"] == "bonafide"
    ]

    spoof = [
        row
        for row in results
        if row["label"] == "spoof"
    ]

    for feature in FEATURES:

        bonafide_values = np.array([
            float(row[feature])
            for row in bonafide
        ])

        spoof_values = np.array([
            float(row[feature])
            for row in spoof
        ])

        mean_bonafide = np.mean(
            bonafide_values
        )

        mean_spoof = np.mean(
            spoof_values
        )

        std_bonafide = np.std(
            bonafide_values,
            ddof=1
        )

        std_spoof = np.std(
            spoof_values,
            ddof=1
        )

        pooled_std = np.sqrt(
            (
                std_bonafide ** 2
                + std_spoof ** 2
            ) / 2
        )

        if pooled_std > 0:

            effect_size = (
                mean_spoof
                - mean_bonafide
            ) / pooled_std

        else:

            effect_size = 0.0

        print(
            f"{feature:<24}"
            f"{effect_size:>16.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== O8 PROSODIC FEATURE ANALYSIS ==="
    )

    print(
        f"\nDataset:\n{DATASET_ROOT}"
    )

    print(
        f"\nProtocol:\n{PROTOCOL_PATH}"
    )

    print(
        f"\nTarget samples:"
        f"\n  Bonafide: {NUM_BONAFIDE}"
        f"\n  Spoof:    {NUM_SPOOF}"
    )

    # --------------------------------------------------------
    # Load protocol labels
    # --------------------------------------------------------

    labels = load_labels()

    print(
        f"\nLabels loaded: {len(labels)}"
    )

    # --------------------------------------------------------
    # Match labels to audio files
    # --------------------------------------------------------

    available = find_labeled_files(
        labels
    )

    # --------------------------------------------------------
    # Select balanced sample
    # --------------------------------------------------------

    bonafide, spoof = select_files(
        available
    )

    # --------------------------------------------------------
    # Bonafide
    # --------------------------------------------------------

    print(
        "\n=== EXTRACTING BONAFIDE FEATURES ==="
    )

    bonafide_results = extract_dataset(
        bonafide,
        "bonafide"
    )

    # --------------------------------------------------------
    # Spoof
    # --------------------------------------------------------

    print(
        "\n=== EXTRACTING SPOOF FEATURES ==="
    )

    spoof_results = extract_dataset(
        spoof,
        "spoof"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    results = (
        bonafide_results
        + spoof_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        results
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    calculate_statistics(
        results
    )

    calculate_effect_sizes(
        results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()