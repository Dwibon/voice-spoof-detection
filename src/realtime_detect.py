import os
import sys
import numpy as np
import torch
import soundfile as sf
import torchaudio

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

from train_from_cache import ClassifierHead
from risk_engine import calculate_risk
from explanation_engine import generate_explanations
from speaker_check import SpeakerConsistencyChecker

from privacy import (
    create_privacy_safe_result,
    validate_privacy_safe_result,
    get_privacy_policy,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SR = 16000

# Real-time simulation
WINDOW_SEC = 4
STEP_SEC = 1
ROLLING_WINDOW = 3

MODEL_NAME = "facebook/wav2vec2-base"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classifier_head_indian_adapted.pt"
)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path):
    """
    Load audio, convert to mono, and resample to 16 kHz.
    """

    waveform_np, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    waveform = torch.from_numpy(
        waveform_np
    )

    # Convert stereo/multi-channel audio to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(
            dim=-1
        )

    # Resample to target sampling rate
    if sample_rate != TARGET_SR:

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SR
        )

        waveform = resampler(
            waveform
        )

    return waveform


# ============================================================
# WINDOW GENERATION
# ============================================================

def make_windows(waveform):
    """
    Split audio into overlapping fixed-length windows.

    Window length: 4 seconds
    Step: 1 second

    The final short window is zero-padded.
    """

    window_samples = int(
        WINDOW_SEC * TARGET_SR
    )

    step_samples = int(
        STEP_SEC * TARGET_SR
    )

    total_samples = waveform.shape[0]

    windows = []
    start_positions = []

    start = 0

    while start < total_samples:

        end = start + window_samples

        window = waveform[start:end]

        # Zero-pad final short window
        if window.shape[0] < window_samples:

            padding = (
                window_samples
                - window.shape[0]
            )

            window = torch.nn.functional.pad(
                window,
                (0, padding)
            )

        windows.append(window)

        start_positions.append(start)

        # Stop after the complete audio has been covered
        if end >= total_samples:
            break

        start += step_samples

    return windows, start_positions


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(device):
    """
    Load pretrained Wav2Vec2 backbone and trained classifier.
    """

    print(
        "\nLoading Wav2Vec2 model..."
    )

    feature_extractor = (
        Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    backbone = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )

    backbone = backbone.to(device)

    backbone.eval()

    # Freeze Wav2Vec2 backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Load trained classifier head
    classifier = ClassifierHead(
        input_dim=768
    ).to(device)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{MODEL_PATH}"
        )

    classifier.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    classifier.eval()

    print(
        "Model loaded successfully."
    )

    return (
        feature_extractor,
        backbone,
        classifier
    )


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embedding(
    waveform,
    feature_extractor,
    backbone,
    device
):
    """
    Convert an audio window into a 768-dimensional
    Wav2Vec2 embedding.
    """

    waveform_np = (
        waveform
        .cpu()
        .numpy()
    )

    inputs = feature_extractor(
        waveform_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    input_values = (
        inputs.input_values.to(device)
    )

    with torch.no_grad():

        outputs = backbone(
            input_values
        )

        # Mean-pool over time
        embedding = (
            outputs.last_hidden_state
            .mean(dim=1)
        )

    return embedding


# ============================================================
# ROLLING AVERAGE
# ============================================================

def rolling_average(
    values,
    window_size
):
    """
    Calculate rolling average of window-level
    spoof probabilities.
    """

    result = []

    for i in range(len(values)):

        start = max(
            0,
            i - window_size + 1
        )

        current_window = (
            values[start:i + 1]
        )

        result.append(
            float(
                np.mean(
                    current_window
                )
            )
        )

    return result


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect(
    audio_path,
    reference_audio=None
):
    """
    Complete voice-risk detection pipeline.

    Implemented objectives:

    O3  - Real-time window-based detection
    O4  - Risk scoring
    O5  - Explanation generation
    O9  - Speaker consistency
    O10 - Privacy-by-design
    O11 - Recommendation mapping
    """

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # O10 — PRIVACY POLICY
    # ========================================================

    privacy_policy = (
        get_privacy_policy()
    )

    print(
        "\n=== PRIVACY-BY-DESIGN ==="
    )

    print(
        "Raw audio persistence: "
        f"{'DISABLED' if not privacy_policy['raw_audio_persistence'] else 'ENABLED'}"
    )

    print(
        "Raw audio logging: "
        f"{'DISABLED' if not privacy_policy['raw_audio_logging'] else 'ENABLED'}"
    )

    print(
        "Temporary audio cleanup: "
        f"{'ENABLED' if privacy_policy['temporary_audio_cleanup'] else 'DISABLED'}"
    )

    print(
        "Stored output: "
        "Derived detection information only"
    )

    # ========================================================
    # CHECK AUDIO
    # ========================================================

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{audio_path}"
        )

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    waveform = load_audio(
        audio_path
    )

    duration = (
        waveform.shape[0]
        / TARGET_SR
    )

    print(
        f"\nAudio duration: "
        f"{duration:.2f} seconds"
    )

    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    windows, start_positions = (
        make_windows(waveform)
    )

    print(
        f"Windows generated: "
        f"{len(windows)}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    (
        feature_extractor,
        backbone,
        classifier
    ) = load_model(device)

    # ========================================================
    # WINDOW PREDICTIONS
    # ========================================================

    probabilities = []

    print(
        "\n=== WINDOW PREDICTIONS ==="
    )

    for i, window in enumerate(
        windows
    ):

        embedding = extract_embedding(
            window,
            feature_extractor,
            backbone,
            device
        )

        with torch.no_grad():

            logits = classifier(
                embedding
            )

            probability = (
                torch.sigmoid(
                    logits
                ).item()
            )

        probabilities.append(
            probability
        )

    # ========================================================
    # ROLLING AGGREGATION
    # ========================================================

    rolling_probabilities = (
        rolling_average(
            probabilities,
            ROLLING_WINDOW
        )
    )

    for i, probability in enumerate(
        probabilities
    ):

        start_time = (
            start_positions[i]
            / TARGET_SR
        )

        end_time = (
            start_time
            + WINDOW_SEC
        )

        print(
            f"Window {i + 1:02d} "
            f"[{start_time:5.2f}s - "
            f"{end_time:5.2f}s] "
            f"spoof = "
            f"{probability * 100:.2f}% "
            f"| rolling = "
            f"{rolling_probabilities[i] * 100:.2f}%"
        )

    # ========================================================
    # FINAL PROBABILITY
    # ========================================================

    final_probability = (
        rolling_probabilities[-1]
    )

    # ========================================================
    # O4 — RISK ENGINE
    # ========================================================

    risk = calculate_risk(
        final_probability
    )

    # ========================================================
    # O5 — EXPLANATION ENGINE
    # ========================================================

    explanation = generate_explanations(
        final_probability,
        probabilities
    )

    # ========================================================
    # O9 — SPEAKER CONSISTENCY
    # ========================================================

    speaker_result = None

    if reference_audio is not None:

        if not os.path.exists(
            reference_audio
        ):

            raise FileNotFoundError(
                f"Reference audio not found:\n"
                f"{reference_audio}"
            )

        print(
            "\n=== SPEAKER CONSISTENCY CHECK ==="
        )

        speaker_checker = (
            SpeakerConsistencyChecker(
                device=str(device)
            )
        )

        speaker_checker.enroll(
            reference_audio
        )

        speaker_result = (
            speaker_checker.check(
                audio_path
            )
        )

        print(
            f"Cosine similarity: "
            f"{speaker_result['speaker_similarity']:.4f}"
        )

        print(
            f"Consistency: "
            f"{speaker_result['speaker_consistency']}"
        )

    # ========================================================
    # FINAL RISK ASSESSMENT
    # ========================================================

    print(
        "\n=== REAL-TIME VOICE RISK ASSESSMENT ==="
    )

    print(
        f"Windows analyzed: "
        f"{len(probabilities)}"
    )

    print(
        f"Rolling window size: "
        f"{min(len(probabilities), ROLLING_WINDOW)}"
    )

    print(
        f"Spoof probability: "
        f"{final_probability:.4f}"
    )

    print(
        f"Spoof percentage: "
        f"{final_probability * 100:.2f}%"
    )

    # ========================================================
    # RISK
    # ========================================================

    print(
        "\n=== RISK ASSESSMENT ==="
    )

    print(
        f"Risk Score: "
        f"{risk['risk_score']:.2f}/100"
    )

    print(
        f"Risk Level: "
        f"{risk['risk_level']}"
    )

    print(
        f"Confidence: "
        f"{risk['confidence']:.2f}%"
    )

    # ========================================================
    # O11 — RECOMMENDATION
    # ========================================================

    print(
        "\n=== RECOMMENDATION ==="
    )

    print(
        f"Action: "
        f"{risk['recommendation']['action']}"
    )

    print(
        f"Recommendation: "
        f"{risk['recommendation']['message']}"
    )

    # ========================================================
    # O9 — SPEAKER RESULT
    # ========================================================

    if speaker_result is not None:

        print(
            "\n=== SPEAKER CONSISTENCY ==="
        )

        print(
            f"Cosine similarity: "
            f"{speaker_result['speaker_similarity']:.4f}"
        )

        print(
            f"Speaker consistency: "
            f"{speaker_result['speaker_consistency']}"
        )

    # ========================================================
    # O5 — EXPLANATION
    # ========================================================

    print(
        "\n=== EXPLANATION ==="
    )

    print(
        explanation["summary"]
    )

    print(
        "Evidence:"
    )

    for evidence in explanation["evidence"]:

        print(
            f"• {evidence}"
        )

    # ========================================================
    # DERIVED RESULT
    # ========================================================

    result = {

        "spoof_probability": round(
            final_probability,
            4
        ),

        "spoof_percentage": round(
            final_probability * 100,
            2
        ),

        "risk_score": risk[
            "risk_score"
        ],

        "risk_level": risk[
            "risk_level"
        ],

        "confidence": risk[
            "confidence"
        ],

        # O11
        "recommendation": risk[
            "recommendation"
        ],

        # O5
        "explanation": explanation,

        # O3
        "windows_analyzed": len(
            probabilities
        ),

        "rolling_window_size": min(
            len(probabilities),
            ROLLING_WINDOW
        ),

        # O9
        "speaker_similarity": (
            speaker_result[
                "speaker_similarity"
            ]
            if speaker_result is not None
            else None
        ),

        "speaker_consistency": (
            speaker_result[
                "speaker_consistency"
            ]
            if speaker_result is not None
            else None
        ),
    }

    # ========================================================
    # O10 — PRIVACY-SAFE RESULT
    # ========================================================

    privacy_safe_result = (
        create_privacy_safe_result(
            result
        )
    )

    # Validate that raw-audio fields have not leaked
    # into the returned result.

    validate_privacy_safe_result(
        privacy_safe_result
    )

    print(
        "\nPrivacy validation: PASSED"
    )

    print(
        "Raw audio is not included in the "
        "returned detection result."
    )

    return privacy_safe_result


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) not in [2, 3]:

        print(
            "\nUsage:"
        )

        print(
            "python3 src/realtime_detect.py "
            "<audio_file>"
        )

        print(
            "\nWith speaker reference:"
        )

        print(
            "python3 src/realtime_detect.py "
            "<audio_file> "
            "<reference_audio>"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    reference_file = (
        sys.argv[2]
        if len(sys.argv) == 3
        else None
    )

    detect(
        audio_file,
        reference_file
    )