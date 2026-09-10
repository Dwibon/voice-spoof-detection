import os
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

from train_from_cache import ClassifierHead


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cached_embeddings"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classifier_head_best.pt"
)

CALIBRATION_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "models",
    "risk_calibration.npz"
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    print(
        f"Using device: {DEVICE}"
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Classifier model not found:\n{MODEL_PATH}"
        )

    model = ClassifierHead(
        input_dim=768
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD DEVELOPMENT CACHE
# ============================================================

def load_dev_cache():

    embeddings_path = os.path.join(
        CACHE_DIR,
        "dev_embeddings.npy"
    )

    labels_path = os.path.join(
        CACHE_DIR,
        "dev_labels.npy"
    )

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            "\nDevelopment embeddings were not found.\n"
            f"Expected:\n{embeddings_path}\n\n"
            "Generate the ASVspoof2019 LA development "
            "embeddings first."
        )

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            "\nDevelopment labels were not found.\n"
            f"Expected:\n{labels_path}\n\n"
            "Generate the ASVspoof2019 LA development "
            "cache first."
        )

    embeddings = np.load(
        embeddings_path
    )

    labels = np.load(
        labels_path
    )

    embeddings = torch.from_numpy(
        embeddings
    ).float()

    labels = torch.from_numpy(
        labels
    ).float()

    return embeddings, labels


# ============================================================
# GET RAW MODEL SCORES
# ============================================================

def get_raw_scores(model, embeddings):

    scores = []

    batch_size = 64

    with torch.no_grad():

        for start in range(
            0,
            len(embeddings),
            batch_size
        ):

            batch = embeddings[
                start:start + batch_size
            ].to(DEVICE)

            logits = model(
                batch
            )

            probabilities = torch.sigmoid(
                logits
            )

            scores.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        scores,
        dtype=np.float64
    )


# ============================================================
# MAIN CALIBRATION
# ============================================================

def calibrate():

    print(
        "\n=== O4 RISK CALIBRATION ==="
    )

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nCache directory:\n{CACHE_DIR}"
    )

    # --------------------------------------------------------
    # Load development set
    # --------------------------------------------------------

    print(
        "\nLoading ASVspoof2019 LA development cache..."
    )

    dev_x, dev_y = load_dev_cache()

    dev_labels = dev_y.numpy().astype(int)

    print(
        f"Development samples: "
        f"{len(dev_labels)}"
    )

    print(
        f"Bonafide samples: "
        f"{np.sum(dev_labels == 0)}"
    )

    print(
        f"Spoof samples: "
        f"{np.sum(dev_labels == 1)}"
    )

    print(
        f"Embedding shape: "
        f"{dev_x.shape}"
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    model = load_classifier()

    # --------------------------------------------------------
    # Generate raw probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating classifier scores..."
    )

    raw_scores = get_raw_scores(
        model,
        dev_x
    )

    # --------------------------------------------------------
    # Raw model metrics
    # --------------------------------------------------------

    raw_predictions = (
        raw_scores >= 0.5
    ).astype(int)

    raw_accuracy = accuracy_score(
        dev_labels,
        raw_predictions
    )

    raw_auc = roc_auc_score(
        dev_labels,
        raw_scores
    )

    raw_brier = brier_score_loss(
        dev_labels,
        raw_scores
    )

    print(
        "\n=== RAW MODEL ==="
    )

    print(
        f"Accuracy: "
        f"{raw_accuracy * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{raw_auc:.4f}"
    )

    print(
        f"Brier score: "
        f"{raw_brier:.4f}"
    )

    # --------------------------------------------------------
    # Platt scaling
    # --------------------------------------------------------
    #
    # Logistic regression learns a mapping:
    #
    # raw classifier probability
    #              ↓
    # calibrated probability
    #
    # This is fitted ONLY on the development set.
    #
    # The ASVspoof evaluation set remains untouched.
    # --------------------------------------------------------

    print(
        "\nFitting Platt probability calibration..."
    )

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    )

    calibrator.fit(
        raw_scores.reshape(-1, 1),
        dev_labels
    )

    calibrated_scores = (
        calibrator.predict_proba(
            raw_scores.reshape(-1, 1)
        )[:, 1]
    )

    # --------------------------------------------------------
    # Calibrated metrics
    # --------------------------------------------------------

    calibrated_predictions = (
        calibrated_scores >= 0.5
    ).astype(int)

    calibrated_accuracy = (
        accuracy_score(
            dev_labels,
            calibrated_predictions
        )
    )

    calibrated_auc = (
        roc_auc_score(
            dev_labels,
            calibrated_scores
        )
    )

    calibrated_brier = (
        brier_score_loss(
            dev_labels,
            calibrated_scores
        )
    )

    print(
        "\n=== CALIBRATED MODEL ==="
    )

    print(
        f"Accuracy: "
        f"{calibrated_accuracy * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{calibrated_auc:.4f}"
    )

    print(
        f"Brier score: "
        f"{calibrated_brier:.4f}"
    )

    # --------------------------------------------------------
    # Calibration improvement
    # --------------------------------------------------------

    brier_improvement = (
        raw_brier - calibrated_brier
    )

    print(
        "\n=== CALIBRATION IMPROVEMENT ==="
    )

    print(
        f"Raw Brier score: "
        f"{raw_brier:.4f}"
    )

    print(
        f"Calibrated Brier score: "
        f"{calibrated_brier:.4f}"
    )

    if brier_improvement > 0:

        print(
            f"Improvement: "
            f"{brier_improvement:.4f}"
        )

        print(
            "Calibration improved probability quality."
        )

    elif brier_improvement < 0:

        print(
            f"Change: "
            f"{brier_improvement:.4f}"
        )

        print(
            "Calibration did not improve the Brier score."
        )

    else:

        print(
            "No change in Brier score."
        )

    # --------------------------------------------------------
    # Calibration parameters
    # --------------------------------------------------------

    coefficient = float(
        calibrator.coef_[0][0]
    )

    intercept = float(
        calibrator.intercept_[0]
    )

    print(
        "\n=== CALIBRATION PARAMETERS ==="
    )

    print(
        f"Coefficient: "
        f"{coefficient:.8f}"
    )

    print(
        f"Intercept: "
        f"{intercept:.8f}"
    )

    # --------------------------------------------------------
    # Risk threshold analysis
    # --------------------------------------------------------

    print(
        "\n=== RISK THRESHOLD ANALYSIS ==="
    )

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    for threshold in thresholds:

        predictions = (
            calibrated_scores >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            dev_labels,
            predictions
        )

        print(
            f"Threshold {threshold:.2f} "
            f"→ Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    low_count = np.sum(
        calibrated_scores < 0.30
    )

    medium_count = np.sum(
        (calibrated_scores >= 0.30)
        &
        (calibrated_scores < 0.70)
    )

    high_count = np.sum(
        calibrated_scores >= 0.70
    )

    total = len(
        calibrated_scores
    )

    print(
        "\n=== CALIBRATED RISK DISTRIBUTION ==="
    )

    print(
        f"LOW:    {low_count:6d} "
        f"({low_count / total * 100:.2f}%)"
    )

    print(
        f"MEDIUM: {medium_count:6d} "
        f"({medium_count / total * 100:.2f}%)"
    )

    print(
        f"HIGH:   {high_count:6d} "
        f"({high_count / total * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Save calibration parameters
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            CALIBRATION_OUTPUT
        ),
        exist_ok=True
    )

    np.savez(
        CALIBRATION_OUTPUT,
        coefficient=coefficient,
        intercept=intercept
    )

    print(
        "\n=== CALIBRATION SAVED ==="
    )

    print(
        CALIBRATION_OUTPUT
    )

    # --------------------------------------------------------
    # Example conversions
    # --------------------------------------------------------

    example_scores = np.array(
        [
            0.01,
            0.10,
            0.30,
            0.50,
            0.70,
            0.90,
            0.99,
        ],
        dtype=np.float64
    )

    example_calibrated = (
        calibrator.predict_proba(
            example_scores.reshape(-1, 1)
        )[:, 1]
    )

    print(
        "\n=== EXAMPLE CALIBRATION ==="
    )

    print(
        "Raw probability "
        "→ Calibrated probability"
    )

    for raw, calibrated in zip(
        example_scores,
        example_calibrated
    ):

        print(
            f"{raw:.2f}"
            f" → "
            f"{calibrated:.4f}"
            f" "
            f"({calibrated * 100:.2f}%)"
        )

    print(
        "\n=== O4 CALIBRATION COMPLETE ==="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    calibrate()