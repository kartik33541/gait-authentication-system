
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.cluster import KMeans

from dataset_loader import extract_windows
from build_encoder import get_encoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "RealWorldData")

SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")

WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")

VAULT_PATH = os.path.join(BASE_DIR, "vault.json")


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():

    # rebuild encoder architecture
    encoder = get_encoder((256, 6))

    # recreate siamese wrapper only for loading weights
    input_a = tf.keras.layers.Input((256, 6))
    input_b = tf.keras.layers.Input((256, 6))

    out_a = encoder(input_a)
    out_b = encoder(input_b)

    dist = tf.keras.layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))
    )([out_a, out_b])

    model = tf.keras.models.Model([input_a, input_b], dist)

    model.load_weights(WEIGHTS_PATH)

    scaler = joblib.load(SCALER_PATH)

    vault = {}

    for i in range(1, 11):

        user = f"Person{i}"

        folder = os.path.join(REAL_DATA_DIR, user)

        if not os.path.exists(folder):
            continue

        windows = []

        for file in os.listdir(folder):

            if not file.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(folder, file))

            w = extract_windows(df, scaler)

            windows.extend(w)

        if len(windows) == 0:
            continue

        windows = np.array(windows)

        emb = encoder.predict(windows, verbose=0)

        # remove unstable embeddings (top 20% outliers)
        norms = np.linalg.norm(emb, axis=1)

        threshold = np.percentile(norms, 20)

        emb = emb[norms >= threshold]

        # normalize embeddings
        emb = np.array([l2_normalize(e) for e in emb])

        # -------------------------------
        # KMEANS TEMPLATE CREATION
        # -------------------------------
        num_templates = 3

        templates = []

        if len(emb) >= num_templates:

            kmeans = KMeans(n_clusters=num_templates, random_state=42)
            labels = kmeans.fit_predict(emb)

            for cluster_id in range(num_templates):

                cluster_embeddings = emb[labels == cluster_id]

                if len(cluster_embeddings) == 0:
                    continue

                template = np.mean(cluster_embeddings, axis=0)
                template = l2_normalize(template)

                templates.append(template.tolist())

        else:

            template = np.mean(emb, axis=0)
            template = l2_normalize(template)

            templates.append(template.tolist())

        vault[user] = templates

        print(user, "enrolled with", len(templates), "templates")

    with open(VAULT_PATH, "w") as f:
        json.dump(vault, f, indent=4)

    print("\nVault saved to:", VAULT_PATH)


if __name__ == "__main__":
    main()