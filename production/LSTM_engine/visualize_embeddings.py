# production/LSTM_engine/visualise_embeddings.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.decomposition import PCA

from dataset_loader import extract_windows
from build_encoder import get_encoder

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# production/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "RealWorldData")

SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")

WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")


# =========================
# LOAD TRAINED ENCODER
# =========================

def get_loaded_encoder():

    input_shape = (256, 6)

    encoder = get_encoder(input_shape=input_shape, embedding_dim=256)

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    out_a = encoder(input_a)
    out_b = encoder(input_b)

    dist = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))
    )([out_a, out_b])

    model = tf.keras.models.Model([input_a, input_b], dist)

    model.load_weights(WEIGHTS_PATH)

    return encoder


# =========================
# LOAD DATA + GENERATE EMBEDDINGS
# =========================

def load_user_embeddings(encoder, scaler):

    embeddings = []
    labels = []

    for i in range(1, 11):

        user = f"Person{i}"
        user_folder = os.path.join(REAL_DATA_DIR, user)

        if not os.path.exists(user_folder):
            continue

        for file in os.listdir(user_folder):

            if not file.endswith(".csv"):
                continue

            csv_path = os.path.join(user_folder, file)

            df = pd.read_csv(csv_path)

            windows = extract_windows(df, scaler)

            if len(windows) == 0:
                continue

            windows = np.array(windows, dtype=np.float32)

            emb = encoder.predict(windows, verbose=0)

            for e in emb:
                embeddings.append(e)
                labels.append(user)

    return np.array(embeddings), labels


# =========================
# VISUALIZE USING PCA
# =========================

def visualize_pca(embeddings, labels):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(labels))

    for label in unique_labels:

        idx = [i for i, l in enumerate(labels) if l == label]

        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            label=label,
            alpha=0.7
        )

    plt.title("Gait Embedding Space (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)

    plt.show()


# =========================
# MAIN
# =========================

def main():

    print("\n🧠 Loading trained encoder...")

    encoder = get_loaded_encoder()

    print("✅ Encoder ready")

    print("\n📦 Loading scaler...")

    scaler = joblib.load(SCALER_PATH)

    print("✅ Scaler loaded")

    print("\n📊 Extracting embeddings from dataset...")

    embeddings, labels = load_user_embeddings(encoder, scaler)

    print(f"Total embeddings extracted: {len(embeddings)}")

    print("\n📈 Visualizing PCA...")

    visualize_pca(embeddings, labels)


if __name__ == "__main__":
    main()