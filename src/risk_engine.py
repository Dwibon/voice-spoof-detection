LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


def calculate_recommendation(risk_level):
    """
    Map risk level to recommended action.
    """

    recommendations = {
        "LOW": {
            "action": "PROCEED",
            "message": (
                "Voice appears low risk. "
                "No additional verification is required."
            ),
        },

        "MEDIUM": {
            "action": "VERIFY",
            "message": (
                "Exercise caution. "
                "Verify the caller using another communication channel."
            ),
        },

        "HIGH": {
            "action": "ESCALATE",
            "message": (
                "Do not rely on the voice alone. "
                "Verify using callback, MFA, or escalate "
                "for further review."
            ),
        },
    }

    if risk_level not in recommendations:
        raise ValueError(
            f"Unknown risk level: {risk_level}"
        )

    return recommendations[risk_level]


def calculate_risk(spoof_probability):
    """
    Convert detector probability into a 3-tier risk assessment.

    spoof_probability:
        Raw classifier probability that the audio is spoofed.

    Returns:
        risk_score
        risk_level
        confidence
        recommendation
    """

    if not 0.0 <= spoof_probability <= 1.0:
        raise ValueError(
            "spoof_probability must be between 0.0 and 1.0"
        )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = spoof_probability * 100.0

    # --------------------------------------------------------
    # Three-tier risk classification
    # --------------------------------------------------------

    if spoof_probability < LOW_THRESHOLD:

        risk_level = "LOW"

    elif spoof_probability < HIGH_THRESHOLD:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------
    #
    # This represents confidence in the binary decision,
    # rather than claiming that the probability itself has
    # been statistically calibrated.
    #
    # 0.50 probability → 0% decision confidence
    # 0.75 probability → 50% decision confidence
    # 0.90 probability → 80% decision confidence
    # 0.99 probability → 98% decision confidence
    #
    # --------------------------------------------------------

    confidence = (
        abs(spoof_probability - 0.5)
        * 2.0
        * 100.0
    )

    confidence = min(
        100.0,
        confidence
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = calculate_recommendation(
        risk_level
    )

    return {
        "risk_score": round(
            risk_score,
            2
        ),

        "risk_level": risk_level,

        "confidence": round(
            confidence,
            2
        ),

        "recommendation": recommendation,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=== RISK ENGINE TEST ==="
    )

    test_probabilities = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    ]

    for probability in test_probabilities:

        result = calculate_risk(
            probability
        )

        print(
            f"\nProbability: "
            f"{probability:.2f}"
        )

        print(
            f"Risk Score: "
            f"{result['risk_score']:.2f}/100"
        )

        print(
            f"Risk Level: "
            f"{result['risk_level']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Action: "
            f"{result['recommendation']['action']}"
        )

        print(
            f"Recommendation: "
            f"{result['recommendation']['message']}"
        )