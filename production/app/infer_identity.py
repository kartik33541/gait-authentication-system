
import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft
from collections import Counter
import os

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ================== UTIL FUNCTIONS ==================
def sliding_windows(signal, window_size=128, step=32):
    return [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]


def gait_features(signal):
    fft_vals = np.abs(fft(signal))[:len(signal) // 2]

    return [
        np.mean(signal),
        np.std(signal),
        np.sqrt(np.mean(signal ** 2)),
        np.ptp(signal),
        np.argmax(fft_vals),
        np.sum(fft_vals ** 2),
        np.std(fft_vals)
    ]


# ================== MAIN INFERENCE ==================
def predict_person(csv_path):

    if not os.path.exists(csv_path):
        return "ACCESS_DENIED (file not found)"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    expected_cols = ["ax", "ay", "az", "wx", "wy", "wz"]

    if not all(col in df.columns for col in expected_cols):
        return "ACCESS_DENIED (bad csv format)"

    df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if len(df) < 200:
        return "ACCESS_DENIED (insufficient data)"

    acc = df[["ax", "ay", "az"]].values
    gyro = df[["wx", "wy", "wz"]].values

    # ================== STATIC DETECTION ==================
    motion_energy = np.mean(np.std(acc, axis=0))

    if motion_energy < 0.15:
        print("⚠ Static data detected")
        return "ACCESS_DENIED (static)"

    # ================== FEATURE EXTRACTION ==================
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    acc_w = sliding_windows(acc)
    gyro_w = sliding_windows(gyro)
    accm_w = sliding_windows(acc_mag)
    gyrom_w = sliding_windows(gyro_mag)

    if len(acc_w) == 0:
        return "ACCESS_DENIED (no windows)"

    X = []

    for i in range(len(acc_w)):
        feats = []

        for axis in range(3):
            feats += gait_features(acc_w[i][:, axis])
        feats += gait_features(accm_w[i])

        for axis in range(3):
            feats += gait_features(gyro_w[i][:, axis])
        feats += gait_features(gyrom_w[i])

        X.append(feats)

    X = np.array(X)
    X_scaled = scaler.transform(X)

    window_preds = rf_model.predict(X_scaled)

    # ================== MAJORITY VOTING ==================
    counter = Counter(window_preds)
    final_pred = counter.most_common(1)[0][0]
    vote_count = counter[final_pred]
    vote_ratio = vote_count / len(window_preds)

    print("------ DEBUG INFO ------")
    print("Total windows:", len(window_preds))
    print("Window predictions:", window_preds)
    print("Vote ratio:", vote_ratio)
    print("Predicted person:", final_pred)
    print("------------------------")

    # ================== THRESHOLD ==================
    THRESHOLD = 0.45

    if vote_ratio >= THRESHOLD:
        return f"ACCESS_GRANTED (Person{final_pred})"
    else:
        return f"ACCESS_DENIED (closest: Person{final_pred})"
