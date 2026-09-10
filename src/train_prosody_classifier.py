import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROSODY_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cached_prosody"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "prosody_classifier.npz"
)

FEATURE_NAMES = [
    "pitch_mean_hz",
    "pitch_variance_hz2",
    "voiced_ratio",
    "pause_ratio",
    "pause_count",
    "speech_rate_proxy",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    feature_path = os.path.join(
        PROSODY_DIR,
        f"{split}_prosody.npy"
    )

    label_path = os.path.join(
        PROSODY_DIR,
        f"{split}_labels.npy"
    )

    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Missing feature file:\n{feature_path}"
        )

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Missing label file:\n{label_path}"
        )

    X = np.load(
        feature_path
    ).astype(np.float64)

    y = np.load(
        label_path
    ).astype(int)

    return X, y


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(name, X, y):

    print(
        f"\n=== {name.upper()} DATASET ==="
    )

    print(
        f"Samples: {len(y)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Bonafide: {np.sum(y == 0)}"
    )

    print(
        f"Spoof:    {np.sum(y == 1)}"
    )

    print(
        f"NaN values: {np.isnan(X).sum()}"
    )

    print(
        f"Inf values: {np.isinf(X).sum()}"
    )


# ============================================================
# MAIN
# ============================================================

def train():

    print(
        "\n================================================"
    )

    print(
        "       O8 PROSODIC CLASSIFIER"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    print(
        "\nLoading prosodic feature caches..."
    )

    X_train, y_train = load_split(
        "train"
    )

    X_dev, y_dev = load_split(
        "dev"
    )

    X_eval, y_eval = load_split(
        "eval"
    )

    print_dataset_summary(
        "train",
        X_train,
        y_train
    )

    print_dataset_summary(
        "dev",
        X_dev,
        y_dev
    )

    print_dataset_summary(
        "eval",
        X_eval,
        y_eval
    )

    # --------------------------------------------------------
    # Handle invalid numerical values
    # --------------------------------------------------------

    X_train = np.where(
        np.isfinite(X_train),
        X_train,
        np.nan
    )

    X_dev = np.where(
        np.isfinite(X_dev),
        X_dev,
        np.nan
    )

    X_eval = np.where(
        np.isfinite(X_eval),
        X_eval,
        np.nan
    )

    # --------------------------------------------------------
    # Build lightweight classifier
    # --------------------------------------------------------
    #
    # Imputer:
    #   handles missing pitch values etc.
    #
    # StandardScaler:
    #   puts all six features on comparable scales.
    #
    # LogisticRegression:
    #   lightweight and interpretable baseline.
    #
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            ),
        ]
    )

    print(
        "\nTraining prosodic classifier..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Development evaluation
    # --------------------------------------------------------

    dev_probabilities = model.predict_proba(
        X_dev
    )[:, 1]

    dev_predictions = (
        dev_probabilities >= 0.5
    ).astype(int)

    dev_accuracy = accuracy_score(
        y_dev,
        dev_predictions
    )

    dev_auc = roc_auc_score(
        y_dev,
        dev_probabilities
    )

    dev_brier = brier_score_loss(
        y_dev,
        dev_probabilities
    )

    dev_cm = confusion_matrix(
        y_dev,
        dev_predictions
    )

    print(
        "\n=== DEVELOPMENT RESULTS ==="
    )

    print(
        f"Accuracy: "
        f"{dev_accuracy * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{dev_auc:.4f}"
    )

    print(
        f"Brier score: "
        f"{dev_brier:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        dev_cm
    )

    # --------------------------------------------------------
    # Evaluation-set performance
    # --------------------------------------------------------
    #
    # The model was trained only on TRAIN.
    # DEV is used for development analysis.
    # EVAL is kept as the final held-out measurement.
    #
    # --------------------------------------------------------

    print(
        "\nGenerating evaluation predictions..."
    )

    eval_probabilities = model.predict_proba(
        X_eval
    )[:, 1]

    eval_predictions = (
        eval_probabilities >= 0.5
    ).astype(int)

    eval_accuracy = accuracy_score(
        y_eval,
        eval_predictions
    )

    eval_auc = roc_auc_score(
        y_eval,
        eval_probabilities
    )

    eval_brier = brier_score_loss(
        y_eval,
        eval_probabilities
    )

    eval_cm = confusion_matrix(
        y_eval,
        eval_predictions
    )

    print(
        "\n=== EVALUATION RESULTS ==="
    )

    print(
        f"Accuracy: "
        f"{eval_accuracy * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{eval_auc:.4f}"
    )

    print(
        f"Brier score: "
        f"{eval_brier:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        eval_cm
    )

    # --------------------------------------------------------
    # Feature coefficients
    # --------------------------------------------------------

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    print(
        "\n=== FEATURE IMPORTANCE ==="
    )

    feature_importance = sorted(
        zip(
            FEATURE_NAMES,
            coefficients
        ),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for name, coefficient in feature_importance:

        direction = (
            "spoof-associated"
            if coefficient > 0
            else "bonafide-associated"
        )

        print(
            f"{name:25s} "
            f"{coefficient:+.6f} "
            f"({direction})"
        )

    # --------------------------------------------------------
    # Save model parameters
    # --------------------------------------------------------
    #
    # Since sklearn Pipeline objects are not being serialized
    # here, save the fitted parameters needed for inference.
    #
    # --------------------------------------------------------

    imputer = model.named_steps[
        "imputer"
    ]

    scaler = model.named_steps[
        "scaler"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    np.savez(
        MODEL_PATH,
        imputer_statistics=imputer.statistics_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classifier_coef=classifier.coef_[0],
        classifier_intercept=classifier.intercept_[0],
        feature_names=np.array(
            FEATURE_NAMES
        )
    )

    print(
        "\n=== MODEL SAVED ==="
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n================================================"
    )

    print(
        "             O8 COMPLETE"
    )

    print(
        "================================================"
    )

    print(
        f"\nProsody evaluation accuracy: "
        f"{eval_accuracy * 100:.2f}%"
    )

    print(
        f"Prosody evaluation ROC-AUC: "
        f"{eval_auc:.4f}"
    )

    print(
        f"Prosody evaluation Brier score: "
        f"{eval_brier:.4f}"
    )

    print(
        "\nNext step: compare prosody against "
        "the Wav2Vec2 baseline and test feature fusion."
    )


if __name__ == "__main__":
    train()
