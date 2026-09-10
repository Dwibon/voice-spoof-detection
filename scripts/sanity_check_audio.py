"""
Sanity check: load one bonafide and one spoof file, confirm sample rate,
duration, and basic waveform shape look right before building anything else.

Usage:
    python scripts/sanity_check_audio.py --file path/to/audio.flac
"""

import argparse
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def inspect_file(path: str):
    audio, sr = sf.read(path)
    duration = len(audio) / sr

    print(f"File:        {path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration:    {duration:.2f} s")
    print(f"Shape:       {audio.shape}")
    print(f"Channels:    {1 if audio.ndim == 1 else audio.shape[1]}")
    print(f"Min/Max amp: {audio.min():.4f} / {audio.max():.4f}")
    print(f"Any NaNs:    {np.isnan(audio).any()}")
    print("-" * 40)

    # ASVspoof2019 is 16kHz mono FLAC — flag anything unexpected
    if sr != 16000:
        print(f"⚠️  WARNING: expected 16000 Hz, got {sr} Hz. "
              f"You'll need to resample before feeding this to Wav2Vec2.")

    plt.figure(figsize=(10, 3))
    plt.plot(audio)
    plt.title(f"Waveform: {path}\nsr={sr}Hz, dur={duration:.2f}s")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    out_path = "outputs/waveform_check.png"
    plt.savefig(out_path)
    print(f"Saved waveform plot to {out_path}")

    return audio, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to an audio file")
    args = parser.parse_args()
    inspect_file(args.file)
