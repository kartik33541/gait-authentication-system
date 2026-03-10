from google import genai


def diagnose_gait_pattern(api_key, energy, variance, score, threshold, confidence_distribution="N/A"):
    """
    LLM-based gait anomaly analysis and authentication explanation.
    """

    try:

        client = genai.Client(api_key=api_key)

        # Improve formatting for LLM readability
        if isinstance(confidence_distribution, list):
            confidence_distribution = ", ".join([f"{v:.3f}" for v in confidence_distribution])

        prompt = f"""
You are an AI Biometric Gait Authentication Analyzer.

Analyze the authentication attempt using the following metrics.

--------------------------------
Authentication Metrics
--------------------------------

Similarity Score : {score:.4f}
Security Threshold : {threshold}

Walk Energy : {energy:.4f}

Step Variance : {variance:.6f}

Window Confidence Scores :
{confidence_distribution}

--------------------------------
Interpretation Guidelines
--------------------------------

Energy < 1.0
→ Device likely stationary.

Energy 1.5 – 2.8
→ Normal walking intensity.

Energy > 3.0
→ Running or rushed movement.

Variance < 0.002
→ Very stable cadence.

Variance 0.002 – 0.008
→ Normal walking rhythm.

Variance > 0.008
→ Irregular steps or inconsistent gait.

Similarity Score:
Score >= Threshold
→ Gait matches the enrolled profile.

Score slightly below threshold
→ Similar walking pattern but inconsistent.

Score far below threshold
→ Likely a different person.

--------------------------------
Task
--------------------------------

Explain the authentication result in clear plain English (2–3 sentences).

If access is granted, confirm that the gait matches the registered profile.

If access is denied, explain the anomaly in the walking pattern that caused the rejection.

Avoid technical jargon.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        if text:
            return text

        return "Biometric authentication completed."

    except Exception:

        # Safe fallback explanation if LLM fails
        if score >= threshold:
            return "Access granted. Your walking pattern matches the registered biometric profile."

        else:
            return "Access denied. The walking pattern does not match the registered biometric profile."