"""
User-friendly explanation engine for voice spoof detection.

This module converts model outputs into:
    1. A human-readable explanation
    2. Supporting technical evidence

The explanations are rule-based. They describe patterns in the
detector output and do not claim to represent the internal
reasoning of the neural network.
"""


# ============================================================
# Thresholds
# ============================================================

ELEVATED_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70

SUDDEN_INCREASE = 0.20
STRONG_INCREASE = 0.40


# ============================================================
# Main Explanation Function
# ============================================================

def generate_explanations(
    rolling_score,
    window_probabilities
):
    """
    Generate a user-friendly explanation and supporting
    technical evidence.

    Parameters
    ----------
    rolling_score : float
        Final rolling spoof score in the range [0.0, 1.0].

    window_probabilities : list
        Spoof scores for individual analyzed portions.

    Returns
    -------
    dict
        {
            "summary": str,
            "evidence": list[str]
        }
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not 0.0 <= rolling_score <= 1.0:
        raise ValueError(
            "rolling_score must be between 0.0 and 1.0"
        )

    if not window_probabilities:
        return {
            "summary": (
                "There was not enough audio information "
                "to make a reliable assessment."
            ),
            "evidence": [
                "No analysis windows were available."
            ]
        }

    # ========================================================
    # Basic statistics
    # ========================================================

    recent = window_probabilities[-3:]

    recent_average = sum(recent) / len(recent)

    maximum_score = max(window_probabilities)

    # --------------------------------------------------------
    # Early vs recent comparison
    # --------------------------------------------------------

    if len(window_probabilities) >= 4:

        early = window_probabilities[:-3]

        early_average = sum(early) / len(early)

    else:

        early_average = window_probabilities[0]

    recent_change = recent_average - early_average

    # --------------------------------------------------------
    # Latest detector-score change
    # --------------------------------------------------------

    if len(window_probabilities) >= 2:

        latest_change = (
            window_probabilities[-1]
            - window_probabilities[-2]
        )

    else:

        latest_change = 0.0

    # ========================================================
    # LOW RISK
    # ========================================================

    if rolling_score < ELEVATED_THRESHOLD:

        summary = (
            "The voice appears consistent with genuine speech. "
            "The detector found no strong evidence suggesting "
            "that the audio is AI-generated or manipulated."
        )

        evidence = [
            (
                f"Overall detector score: "
                f"{rolling_score * 100:.1f}%."
            )
        ]

        if all(
            probability < ELEVATED_THRESHOLD
            for probability in recent
        ):

            evidence.append(
                "The analyzed portions of the recording "
                "showed consistently low detector scores."
            )

        return {
            "summary": summary,
            "evidence": evidence
        }

    # ========================================================
    # HIGH RISK
    # ========================================================

    if rolling_score >= HIGH_THRESHOLD:

        # ----------------------------------------------------
        # Consistently high pattern
        # ----------------------------------------------------

        if all(
            probability >= HIGH_THRESHOLD
            for probability in recent
        ):

            summary = (
                "The audio shows strong signs that may be "
                "associated with AI-generated or manipulated "
                "speech. These signs remain consistently strong "
                "across the analyzed recording."
            )

            evidence = [
                (
                    f"Overall detector score: "
                    f"{rolling_score * 100:.1f}%."
                ),
                (
                    "Multiple analyzed portions received "
                    "high detector scores."
                )
            ]

        # ----------------------------------------------------
        # Strong increase
        # ----------------------------------------------------

        elif latest_change >= SUDDEN_INCREASE:

            summary = (
                "The detector found strong signs that may be "
                "associated with AI-generated or manipulated "
                "speech. The evidence became substantially "
                "stronger in the later part of the recording, "
                "so the voice should be independently verified."
            )

            evidence = [
                (
                    f"Overall detector score: "
                    f"{rolling_score * 100:.1f}%."
                ),
                (
                    f"The detector score increased by "
                    f"{latest_change * 100:.1f} percentage points "
                    f"between the final two analyzed portions."
                )
            ]

        # ----------------------------------------------------
        # General high-risk case
        # ----------------------------------------------------

        else:

            summary = (
                "The audio shows strong signs that may be "
                "associated with AI-generated or manipulated "
                "speech. The detector found a high level of "
                "suspicious evidence in the analyzed recording."
            )

            evidence = [
                (
                    f"Overall detector score: "
                    f"{rolling_score * 100:.1f}%."
                ),
                (
                    f"Highest individual detector score: "
                    f"{maximum_score * 100:.1f}%."
                )
            ]

        return {
            "summary": summary,
            "evidence": evidence
        }

    # ========================================================
    # MEDIUM RISK
    # ========================================================

    # --------------------------------------------------------
    # Strong change between earlier and later portions
    # --------------------------------------------------------

    if recent_change >= STRONG_INCREASE:

        summary = (
            "The detector found some signs that may be "
            "associated with AI-generated or manipulated speech. "
            "These signs were stronger in the later part of the "
            "recording, so additional verification is recommended."
        )

        evidence = [
            (
                f"Overall detector score: "
                f"{rolling_score * 100:.1f}%."
            ),
            (
                "The later portions of the recording received "
                "substantially higher detector scores than the "
                "earlier portions."
            )
        ]

    # --------------------------------------------------------
    # Recent increase
    # --------------------------------------------------------

    elif latest_change >= SUDDEN_INCREASE:

        summary = (
            "The detector found some signs that may be "
            "associated with AI-generated or manipulated speech. "
            "The detector response became noticeably stronger "
            "in the later part of the recording, so the result "
            "should be treated with caution."
        )

        evidence = [
            (
                f"Overall detector score: "
                f"{rolling_score * 100:.1f}%."
            ),
            (
                f"The detector score increased by "
                f"{latest_change * 100:.1f} percentage points "
                f"between the final two analyzed portions."
            )
        ]

    # --------------------------------------------------------
    # Consistently elevated
    # --------------------------------------------------------

    elif all(
        probability >= ELEVATED_THRESHOLD
        for probability in recent
    ):

        summary = (
            "The detector found some signs that may be "
            "associated with AI-generated or manipulated speech. "
            "These signs were present across the recent part of "
            "the recording, but the evidence is not strong enough "
            "for a high-risk classification."
        )

        evidence = [
            (
                f"Overall detector score: "
                f"{rolling_score * 100:.1f}%."
            ),
            (
                "Recent analyzed portions consistently produced "
                "elevated detector scores."
            )
        ]

    # --------------------------------------------------------
    # Mixed / uncertain
    # --------------------------------------------------------

    else:

        summary = (
            "The detector found some suspicious characteristics "
            "in the audio, but the evidence is mixed. The recording "
            "cannot be confidently classified as either genuine "
            "or AI-generated from the available signal alone."
        )

        evidence = [
            (
                f"Overall detector score: "
                f"{rolling_score * 100:.1f}%."
            ),
            (
                "Detector scores varied across different "
                "parts of the recording."
            )
        ]

    # ========================================================
    # Return
    # ========================================================

    return {
        "summary": summary,
        "evidence": evidence
    }


# ============================================================
# Test Cases
# ============================================================

if __name__ == "__main__":

    test_cases = [

        (
            "Low-risk example",
            0.10,
            [0.08, 0.12, 0.10]
        ),

        (
            "Medium-risk example",
            0.50,
            [0.20, 0.40, 0.55]
        ),

        (
            "High-risk example",
            0.85,
            [0.75, 0.82, 0.90]
        ),

        (
            "Sudden-increase example",
            0.65,
            [0.05, 0.08, 0.20, 0.70]
        ),

        (
            "Mixed example",
            0.55,
            [0.10, 0.80, 0.20, 0.65]
        )
    ]

    print("=== EXPLANATION ENGINE TEST ===")

    for name, score, probabilities in test_cases:

        print(f"\n{name}")

        print(
            f"Rolling score: "
            f"{score * 100:.2f}%"
        )

        result = generate_explanations(
            rolling_score=score,
            window_probabilities=probabilities
        )

        print("\nExplanation:")

        print(
            result["summary"]
        )

        print("\nEvidence:")

        for evidence_item in result["evidence"]:

            print(
                f"• {evidence_item}"
            )