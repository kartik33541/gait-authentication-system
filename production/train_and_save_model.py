import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ================== SETTINGS ==================
BASE_DIR = os.path.join("data", "RealWorldLive")
MODEL_DIR = "model"
WINDOW_SIZE = 128
STEP_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

# ================== UTIL FUNCTIONS ==================
def sliding_windows(signal):
    return [
        signal[i:i + WINDOW_SIZE]
        for i in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE)
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


# ================== DATA LOADING ==================
X = []
y = []

persons = sorted(os.listdir(BASE_DIR))

for person_id, person in enumerate(persons, start=1):

    person_path = os.path.join(BASE_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Processing {person}")

    for file in os.listdir(person_path):

        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(person_path, file)

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ["ax", "ay", "az", "wx", "wy", "wz"]

        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file} (bad format)")
            continue

        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        if len(df) < WINDOW_SIZE:
            continue

        acc = df[["ax", "ay", "az"]].values
        gyro = df[["wx", "wy", "wz"]].values

        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        acc_w = sliding_windows(acc)
        gyro_w = sliding_windows(gyro)
        accm_w = sliding_windows(acc_mag)
        gyrom_w = sliding_windows(gyro_mag)

        for i in range(len(acc_w)):

            feats = []

            for axis in range(3):
                feats += gait_features(acc_w[i][:, axis])

            feats += gait_features(accm_w[i])

            for axis in range(3):
                feats += gait_features(gyro_w[i][:, axis])

            feats += gait_features(gyrom_w[i])

            X.append(feats)
            y.append(person_id)

print("Total samples:", len(X))

X = np.array(X)
y = np.array(y)

# ================== SCALING ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== MODEL ==================
rf_model = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_scaled, y)

# ================== SAVE ==================
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Training complete")
print("✅ Persons detected:", len(persons))
print("✅ Model saved to /model")
