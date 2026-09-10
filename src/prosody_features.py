import numpy as np
import librosa


TARGET_SR = 16000


def extract_prosodic_features(audio_path):
    """
    Extract lightweight prosodic features from an audio file.

    Features:
        - pitch_mean_hz
        - pitch_variance_hz2
        - voiced_ratio
        - pause_ratio
        - pause_count
        - speech_rate_proxy

    Returns:
        Dictionary containing the extracted features.
    """

    # --------------------------------------------------
    # 1. LOAD AUDIO
    # --------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio file contains no samples.")

    # --------------------------------------------------
    # 2. PITCH / F0
    # --------------------------------------------------

    # Estimate fundamental frequency using PYIN.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=TARGET_SR
    )

    # Keep only valid voiced pitch values.
    valid_pitch = f0[~np.isnan(f0)]

    if len(valid_pitch) > 0:

        pitch_mean = float(
            np.mean(valid_pitch)
        )

        pitch_variance = float(
            np.var(valid_pitch)
        )

    else:

        pitch_mean = 0.0
        pitch_variance = 0.0

    # --------------------------------------------------
    # 3. VOICED RATIO
    # --------------------------------------------------

    voiced_frames = np.sum(
        voiced_flag
    )

    total_frames = len(
        voiced_flag
    )

    if total_frames > 0:

        voiced_ratio = float(
            voiced_frames / total_frames
        )

    else:

        voiced_ratio = 0.0

    # --------------------------------------------------
    # 4. PAUSE / SILENCE ANALYSIS
    # --------------------------------------------------

    # RMS energy for each frame.
    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=256
    )[0]

    # Convert to decibels.
    rms_db = librosa.amplitude_to_db(
        rms,
        ref=np.max
    )

    # Treat frames below -35 dB relative to the
    # maximum signal energy as potential pauses.
    pause_threshold_db = -35.0

    pause_frames = (
        rms_db < pause_threshold_db
    )

    pause_ratio = float(
        np.mean(pause_frames)
    )

    # --------------------------------------------------
    # 5. PAUSE COUNT
    # --------------------------------------------------

    # Count transitions from speech → pause.
    pause_starts = (
        pause_frames[1:]
        & ~pause_frames[:-1]
    )

    pause_count = int(
        np.sum(pause_starts)
    )

    # --------------------------------------------------
    # 6. SPEECH RATE PROXY
    # --------------------------------------------------

    duration_seconds = (
        len(y) / TARGET_SR
    )

    if duration_seconds > 0:

        speech_rate_proxy = float(
            voiced_frames / duration_seconds
        )

    else:

        speech_rate_proxy = 0.0

    # --------------------------------------------------
    # 7. RETURN FEATURES
    # --------------------------------------------------

    features = {

        "pitch_mean_hz":
            round(pitch_mean, 2),

        "pitch_variance_hz2":
            round(pitch_variance, 2),

        "voiced_ratio":
            round(voiced_ratio, 4),

        "pause_ratio":
            round(pause_ratio, 4),

        "pause_count":
            pause_count,

        "speech_rate_proxy":
            round(speech_rate_proxy, 2)
    }

    return features


def print_features(features):
    """
    Print extracted features in a readable format.
    """

    print("\n=== PROSODIC FEATURES ===")

    print(
        f"Mean pitch: "
        f"{features['pitch_mean_hz']:.2f} Hz"
    )

    print(
        f"Pitch variance: "
        f"{features['pitch_variance_hz2']:.2f} Hz²"
    )

    print(
        f"Voiced ratio: "
        f"{features['voiced_ratio'] * 100:.2f}%"
    )

    print(
        f"Pause ratio: "
        f"{features['pause_ratio'] * 100:.2f}%"
    )

    print(
        f"Pause count: "
        f"{features['pause_count']}"
    )

    print(
        f"Speech-rate proxy: "
        f"{features['speech_rate_proxy']:.2f} voiced frames/sec"
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python3 src/prosody_features.py "
            "<audio_file>"
        )

        sys.exit(1)

    audio_path = sys.argv[1]

    features = extract_prosodic_features(
        audio_path
    )

    print_features(
        features
    )
