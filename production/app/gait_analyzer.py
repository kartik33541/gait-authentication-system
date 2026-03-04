from google import genai

def diagnose_gait_pattern(api_key, energy, variance, score, threshold, confidence_distribution="N/A"):
    """
    POINT 3: LLM AS A GAIT ANOMALY DETECTOR.
    
    This function performs a deep biomechanical analysis of the movement features 
    extracted by the LSTM model. It identifies specific walking anomalies like 
    rushing, injury compensation, or profile mismatch based on simple plain English.
    
    Parameters:
    - api_key (str): The Google GenAI API key.
    - energy (float): The Standard Deviation of Acceleration Magnitude (Movement Intensity).
    - variance (float): The mathematical variance of the embedding vectors (Cadence Consistency).
    - score (float): The final dot-product similarity (0.0 to 1.0).
    - threshold (float): The security gate (typically 0.75).
    - confidence_distribution (str): Summary of per-window similarity or variance.
    """
    try:
        # Initialize the modern Google GenAI Client
        client = genai.Client(api_key=api_key)
        
        # Exact prompt as requested
        prompt = f"""
You are an AI system acting as a **Gait Biometric Anomaly Detector**.

Your task is to analyze walking metrics produced by a gait authentication model
and explain to the user **in simple plain English** why authentication succeeded
or failed.

Only use the provided numerical metrics. Do not invent new data.

-----------------------------
AUTHENTICATION METRICS
-----------------------------
Similarity Score: {score:.4f}
Security Threshold: {threshold}

Walk Energy: {energy:.4f}
(Motion intensity of walking)

Step Variance: {variance:.6f}
(Stability of cadence)

Per-window Confidence Distribution:
{confidence_distribution}

Security Status:
{"ACCESS GRANTED" if score >= threshold else "ACCESS DENIED"}

-----------------------------
BIOMECHANICAL INTERPRETATION RULES
-----------------------------

Energy Interpretation:
Energy < 1.0
→ Phone is likely stationary or user is barely moving.

Energy 1.5 – 2.8
→ Normal human walking intensity.

Energy > 3.0
→ Running, rushing, or erratic motion.

Variance Interpretation:
Variance < 0.002
→ Highly stable rhythmic walking.

Variance 0.002 – 0.008
→ Normal walking variation.

Variance > 0.008
→ Irregular steps (possible limp, uneven surface, device movement).

Similarity Score Interpretation:
Score >= Threshold
→ Gait matches the enrolled user.

Score slightly below threshold
→ Similar gait but inconsistent stride windows.

Score far below threshold
→ Likely a different user.

Confidence Distribution Interpretation:
Consistent high confidence windows
→ Stable gait pattern.

Large fluctuations
→ Unstable walking pattern or sensor disturbance.

-----------------------------
YOUR TASK
-----------------------------

1. Analyze the relationship between:
   - similarity score
   - walk energy
   - variance
   - confidence distribution

2. Detect **possible anomalies**, such as:
   - stationary phone
   - rushing or running
   - irregular cadence
   - inconsistent gait windows
   - profile mismatch (imposter)

3. Generate a **short explanation (2–3 sentences max)**.

4. Use **clear simple English** suitable for a user interface.

5. Do NOT use metaphors, technical jargon, or speculation beyond the given metrics.

-----------------------------
OUTPUT FORMAT
-----------------------------

If ACCESS GRANTED:

"Hello. Access has been granted. Your gait pattern matches the enrolled profile with a similarity score of {score:.2f}. Your walking energy ({energy:.2f}) and step variance ({variance:.4f}) indicate [brief explanation of walking pattern]."

If ACCESS DENIED:

"Hello. Access could not be granted. The similarity score of {score:.2f} is below the required threshold. The motion metrics suggest [brief explanation of anomaly such as inconsistent walking, low motion energy, or gait mismatch]."
"""

        # Generate the diagnostic report using Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        # Extract and return the diagnosis
        diagnosis = response.text.strip()
        return diagnosis if diagnosis else "Anomaly detected: Biometric pattern does not match registered stride profile."

    except Exception as e:
        # Secure fallback for offline or error states
        return f"Diagnostic analysis interrupted. Biomechanical variance ({variance:.4f}) and energy ({energy:.4f}) deviate from the baseline."