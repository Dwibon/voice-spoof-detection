# src/speaker_check.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio

from speechbrain.inference.speaker import SpeakerRecognition


TARGET_SR = 16000
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MATCH_THRESHOLD = 0.2132


class SpeakerConsistencyChecker:
    """
    Enroll a reference speaker and compare incoming audio
    against that enrolled speaker using cosine similarity.
    """

    def __init__(self, device=None):

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(f"Loading speaker model on {self.device}...")

        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "speechbrain",
            "spkrec-ecapa-voxceleb",
        )

        self.model = SpeakerRecognition.from_hparams(
            source=MODEL_SOURCE,
            savedir=cache_dir,
            run_opts={
                "device": str(self.device)
            },
        )

        self.reference_embedding = None

        print("Speaker model loaded successfully.")

    def load_audio(self, audio_path):
        """
        Load audio using soundfile, avoiding the TorchCodec
        dependency required by newer torchaudio versions.
        """

        waveform_np, sample_rate = sf.read(
            audio_path,
            dtype="float32",
        )

        waveform = torch.from_numpy(waveform_np)

        # Stereo -> mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        # Resample to 16 kHz
        if sample_rate != TARGET_SR:

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SR,
            )

            waveform = resampler(waveform)

        # SpeechBrain expects [batch, time]
        waveform = waveform.unsqueeze(0)

        return waveform

    def get_embedding(self, audio_path):

        waveform = self.load_audio(audio_path)

        waveform = waveform.to(self.device)

        with torch.no_grad():

            embedding = self.model.encode_batch(
                waveform
            )

        # Typical SpeechBrain output:
        # [batch, 1, embedding_dim]
        embedding = embedding.squeeze()

        # L2 normalization
        embedding = F.normalize(
            embedding,
            p=2,
            dim=0,
        )

        return embedding

    def enroll(self, reference_audio):

        print(
            f"\nEnrolling reference speaker:\n"
            f"{reference_audio}"
        )

        self.reference_embedding = self.get_embedding(
            reference_audio
        )

        print(
            f"Reference embedding shape: "
            f"{tuple(self.reference_embedding.shape)}"
        )

        print("Speaker enrollment complete.")

        return self.reference_embedding

    def compare(self, test_audio):

        if self.reference_embedding is None:
            raise RuntimeError(
                "No reference speaker enrolled. "
                "Call enroll() first."
            )

        test_embedding = self.get_embedding(
            test_audio
        )

        similarity = torch.dot(
            self.reference_embedding,
            test_embedding,
        ).item()

        similarity = max(
            -1.0,
            min(1.0, similarity),
        )

        return similarity

    def check(self, test_audio):

        similarity = self.compare(test_audio)

        # Initial thresholds only.
        # We will calibrate these using real comparisons.
        if similarity >= SPEAKER_MATCH_THRESHOLD:
            consistency = "CONSISTENT"
        else:
            consistency = "MISMATCH"

        return {
            "speaker_similarity": round(
                similarity,
                4,
            ),
            "speaker_consistency": consistency,
        }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python3 src/speaker_check.py "
            "<reference_audio> <test_audio>"
        )

        sys.exit(1)

    reference_audio = sys.argv[1]
    test_audio = sys.argv[2]

    if not os.path.exists(reference_audio):

        print(
            f"Reference audio not found:\n"
            f"{reference_audio}"
        )

        sys.exit(1)

    if not os.path.exists(test_audio):

        print(
            f"Test audio not found:\n"
            f"{test_audio}"
        )

        sys.exit(1)

    checker = SpeakerConsistencyChecker()

    checker.enroll(reference_audio)

    result = checker.check(test_audio)

    print("\n=== SPEAKER CONSISTENCY RESULT ===")

    print(
        f"Cosine similarity: "
        f"{result['speaker_similarity']:.4f}"
    )

    print(
        f"Consistency: "
        f"{result['speaker_consistency']}"
    )