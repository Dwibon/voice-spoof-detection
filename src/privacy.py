import os
import shutil
import tempfile
from contextlib import contextmanager


# ============================================================
# PRIVACY-BY-DESIGN POLICY
# ============================================================

RAW_AUDIO_PERSISTENCE = False
RAW_AUDIO_LOGGING = False
TEMPORARY_AUDIO_CLEANUP = True


# ============================================================
# PRIVACY POLICY
# ============================================================

def get_privacy_policy():
    """
    Return the privacy policy enforced by the detector.
    """

    return {
        "raw_audio_persistence": RAW_AUDIO_PERSISTENCE,
        "raw_audio_logging": RAW_AUDIO_LOGGING,
        "temporary_audio_cleanup": TEMPORARY_AUDIO_CLEANUP,
        "derived_features_allowed": True,
        "risk_scores_allowed": True,
        "speaker_similarity_allowed": True,
        "description": (
            "Raw audio is processed temporarily for inference "
            "and is not persisted by the detection pipeline. "
            "Only derived detection information may be retained."
        ),
    }


# ============================================================
# TEMPORARY AUDIO DIRECTORY
# ============================================================

@contextmanager
def temporary_audio_directory():
    """
    Create an isolated temporary directory for application
    processing.

    Everything created inside this directory is automatically
    removed when processing finishes.
    """

    temp_dir = tempfile.mkdtemp(
        prefix="voice_detector_"
    )

    try:
        yield temp_dir

    finally:

        if os.path.exists(temp_dir):

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# ============================================================
# TEMPORARY FILE CLEANUP
# ============================================================

def delete_temporary_file(file_path):
    """
    Delete a file created by the application.

    This function must only be used with temporary files
    created by the application.
    """

    if not file_path:
        return

    if os.path.exists(file_path):

        os.remove(file_path)


def cleanup_temporary_directory(directory):
    """
    Delete an application-created temporary directory.
    """

    if not directory:
        return

    if os.path.exists(directory):

        shutil.rmtree(
            directory,
            ignore_errors=True
        )


# ============================================================
# PRIVACY-SAFE RESULT
# ============================================================

def create_privacy_safe_result(result):
    """
    Return only derived detection information.

    Raw audio data, waveforms, and source audio paths are
    deliberately excluded.
    """

    safe_result = {

        # Detector output
        "spoof_probability": result.get(
            "spoof_probability"
        ),

        "spoof_percentage": result.get(
            "spoof_percentage"
        ),

        # Risk information
        "risk_score": result.get(
            "risk_score"
        ),

        "risk_level": result.get(
            "risk_level"
        ),

        "confidence": result.get(
            "confidence"
        ),

        # O11
        "recommendation": result.get(
            "recommendation"
        ),

        # O5
        "explanation": result.get(
            "explanation"
        ),

        # O3
        "windows_analyzed": result.get(
            "windows_analyzed"
        ),

        "rolling_window_size": result.get(
            "rolling_window_size"
        ),

        # O9
        "speaker_similarity": result.get(
            "speaker_similarity"
        ),

        "speaker_consistency": result.get(
            "speaker_consistency"
        ),
    }

    return safe_result


# ============================================================
# PRIVACY VALIDATION
# ============================================================

def validate_privacy_safe_result(result):
    """
    Verify that raw-audio-related fields are not present
    in the returned result.
    """

    forbidden_fields = {
        "audio",
        "audio_data",
        "audio_bytes",
        "waveform",
        "raw_audio",
        "raw_audio_data",
        "audio_path",
        "source_audio",
        "source_audio_path",
    }

    detected_forbidden_fields = (
        forbidden_fields.intersection(
            result.keys()
        )
    )

    if detected_forbidden_fields:

        raise ValueError(
            "Privacy validation failed. "
            "Raw-audio fields found: "
            f"{detected_forbidden_fields}"
        )

    return True


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=== PRIVACY MODULE TEST ==="
    )

    policy = get_privacy_policy()

    print(
        f"Raw audio persistence: "
        f"{policy['raw_audio_persistence']}"
    )

    print(
        f"Raw audio logging: "
        f"{policy['raw_audio_logging']}"
    )

    print(
        f"Temporary cleanup: "
        f"{policy['temporary_audio_cleanup']}"
    )

    print(
        f"Derived features allowed: "
        f"{policy['derived_features_allowed']}"
    )

    # --------------------------------------------------------
    # Test temporary directory
    # --------------------------------------------------------

    temp_dir = None

    with temporary_audio_directory() as directory:

        temp_dir = directory

        test_file = os.path.join(
            directory,
            "test_audio.tmp"
        )

        with open(
            test_file,
            "w"
        ) as file:

            file.write(
                "temporary test data"
            )

        print(
            f"\nTemporary directory created: "
            f"{directory}"
        )

        print(
            f"Temporary file exists: "
            f"{os.path.exists(test_file)}"
        )

    print(
        f"Temporary directory exists after "
        f"cleanup: "
        f"{os.path.exists(temp_dir)}"
    )

    # --------------------------------------------------------
    # Test privacy-safe result
    # --------------------------------------------------------

    test_result = {
        "spoof_probability": 0.85,
        "risk_score": 85.0,
        "risk_level": "HIGH",
        "confidence": 85.0,
        "recommendation": {
            "action": "ESCALATE"
        },
        "explanation": {
            "summary": "Test"
        },
        "windows_analyzed": 1,
        "rolling_window_size": 1,
        "speaker_similarity": None,
        "speaker_consistency": None,
    }

    safe_result = create_privacy_safe_result(
        test_result
    )

    validate_privacy_safe_result(
        safe_result
    )

    print(
        "\nPrivacy-safe result validation: PASSED"
    )

    print(
        "\nPrivacy module test complete."
    )