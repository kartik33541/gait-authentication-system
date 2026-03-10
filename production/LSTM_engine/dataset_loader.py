import os
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_DIR = os.path.join(BASE_DIR, "RealWorldData")
SYNTH_DATA_DIR = os.path.join(REAL_DATA_DIR, "SyntheticUsers")

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

WINDOW_SIZE = 256
STEP_SIZE = 128


def apply_bandpass_filter(data, lowcut=0.3, highcut=12.0, fs=38.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


# ---------------------------------------------------------
# Real data augmentation (identity preserving)
# ---------------------------------------------------------
def augment_window(window):

    w = window.copy()

    noise = np.random.normal(0, 0.02, w.shape)
    w = w + noise

    scale = np.random.uniform(0.95, 1.05)
    w = w * scale

    shift = np.random.randint(-5, 5)
    w = np.roll(w, shift, axis=0)

    return w


def collect_all_filtered_data():

    all_data = []

    for root_dir in [REAL_DATA_DIR, SYNTH_DATA_DIR]:

        if not os.path.exists(root_dir):
            continue

        for folder in os.listdir(root_dir):

            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):

                if not file.endswith(".csv"):
                    continue

                try:
                    df = pd.read_csv(os.path.join(folder_path, file))

                    cols = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
                    if all(c in df.columns for c in cols):
                        data = df[cols].values.astype(np.float32)
                    else:
                        data = df.iloc[:, :6].values.astype(np.float32)

                    filtered = apply_bandpass_filter(data)
                    all_data.append(filtered)

                except Exception:
                    pass

    return np.vstack(all_data)


def load_scaler():

    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)

    print("🔧 Fitting global scaler...")

    all_data = collect_all_filtered_data()

    scaler = StandardScaler()
    scaler.fit(all_data)

    joblib.dump(scaler, SCALER_PATH)

    return scaler


def extract_windows(df, scaler):

    cols = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']

    if all(c in df.columns for c in cols):
        data = df[cols].values.astype(np.float32)
    else:
        data = df.iloc[:, :6].values.astype(np.float32)

    filtered = apply_bandpass_filter(data)
    scaled = scaler.transform(filtered)

    windows = []

    for start in range(0, len(scaled) - WINDOW_SIZE + 1, STEP_SIZE):

        w = scaled[start:start + WINDOW_SIZE]
        windows.append(w)

    return windows


def load_all_users():

    scaler = load_scaler()

    X_data = []
    y_labels = []

    label_map = {}
    current_label = 0

    real_windows = []
    synthetic_windows = []

    real_labels = []

    for root_dir, target_list, is_real in [
        (REAL_DATA_DIR, real_windows, True),
        (SYNTH_DATA_DIR, synthetic_windows, False)
    ]:

        if not os.path.exists(root_dir):
            continue

        for folder in sorted(os.listdir(root_dir)):

            folder_path = os.path.join(root_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label_map[folder] = current_label
            user_label = current_label
            current_label += 1

            if is_real:
                real_labels.append(user_label)

            for file in os.listdir(folder_path):

                if not file.endswith(".csv"):
                    continue

                try:

                    df = pd.read_csv(os.path.join(folder_path, file))

                    windows = extract_windows(df, scaler)

                    for w in windows:

                        target_list.append((w, user_label))

                        if is_real and np.random.rand() < 0.5:
                            aug = augment_window(w)
                            target_list.append((aug, user_label))

                except Exception:
                    pass

    for w, l in real_windows + synthetic_windows:
        X_data.append(w)
        y_labels.append(l)

    print(f"Loaded {len(X_data)} windows from {current_label} users")

    return np.array(X_data), np.array(y_labels), label_map, real_windows, synthetic_windows, real_labels


# ---------------------------------------------------------
# Balanced Siamese Pair Generator
# ---------------------------------------------------------
def generate_siamese_pairs(X, y, real_labels):

    num_pairs = len(X) * 2

    pair_images = np.zeros((num_pairs, 2, WINDOW_SIZE, 6), dtype=np.float32)
    pair_labels = np.zeros(num_pairs, dtype=np.float32)

    class_indices = {}

    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    # ensure real_labels only contains labels that exist
    valid_real_labels = [l for l in real_labels if l in class_indices]

    pair_idx = 0

    while pair_idx < num_pairs:

        # ---- anchor from real users ----
        anchor_user = random.choice(valid_real_labels)
        idx1 = random.choice(class_indices[anchor_user])

        # ---- positive sample (same user but different window) ----
        idx2 = idx1
        while idx2 == idx1:
            idx2 = random.choice(class_indices[anchor_user])

        pair_images[pair_idx, 0] = X[idx1]
        pair_images[pair_idx, 1] = X[idx2]
        pair_labels[pair_idx] = 1.0
        pair_idx += 1

        if pair_idx >= num_pairs:
            break

        # ---- negative sample (different real user) ----
        neg_user = random.choice(valid_real_labels)

        while neg_user == anchor_user:
            neg_user = random.choice(valid_real_labels)

        idx3 = random.choice(class_indices[neg_user])

        pair_images[pair_idx, 0] = X[idx1]
        pair_images[pair_idx, 1] = X[idx3]
        pair_labels[pair_idx] = 0.0
        pair_idx += 1

    indices = np.arange(num_pairs)
    np.random.shuffle(indices)

    return pair_images[indices], pair_labels[indices]