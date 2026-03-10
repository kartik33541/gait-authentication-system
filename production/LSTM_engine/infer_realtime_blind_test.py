import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib

from dataset_loader import extract_windows
from build_encoder import get_encoder

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# project root → gait_authentication
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")
VAULT_PATH = os.path.join(BASE_DIR, "vault.json")

SCALER_PATH = os.path.join(PROJECT_ROOT, "production", "scaler.pkl")

BLIND_FOLDER = os.path.join(PROJECT_ROOT, "Blind_Test_Data")

SECURITY_THRESHOLD = 0.70

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# ==========================================
# LOAD TRAINED ENCODER
# ==========================================

def get_loaded_encoder():

    input_shape = (256, 6)

    encoder = get_encoder(input_shape=input_shape, embedding_dim=256)

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    out_a = encoder(input_a)
    out_b = encoder(input_b)

    dist = tf.keras.layers.Lambda(euclidean_distance)([out_a, out_b])

    full_model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=dist)

    full_model.load_weights(WEIGHTS_PATH)

    return encoder

# ==========================================
# IDENTIFICATION ENGINE
# ==========================================

def identify_user(live_csv_path, encoder, scaler):

    if not os.path.exists(VAULT_PATH):
        print("❌ Vault not found!")
        return

    with open(VAULT_PATH, "r") as f:
        vault = json.load(f)

    df = pd.read_csv(live_csv_path)

    # use SAME scaler as training
    windows = extract_windows(df, scaler)

    if len(windows) == 0:
        print("🚨 Recording too short for analysis.")
        return

    windows_arr = np.array(windows, dtype=np.float32)

    embeddings = encoder.predict(windows_arr, verbose=0)

    # normalize each embedding
    embeddings = np.array([l2_normalize(e) for e in embeddings])

    vote_counter = {user: 0 for user in vault.keys()}
    score_accumulator = {user: [] for user in vault.keys()}

    # compare each window embedding against all templates
    for emb in embeddings:

        similarities = {}

        for user, template in vault.items():

            # support both single-template and multi-template vault
            if isinstance(template[0], list):

                scores = []

                for t in template:
                    template_vec = l2_normalize(np.array(t))
                    scores.append(np.dot(emb, template_vec))

                score = max(scores)

            else:

                template_vec = l2_normalize(np.array(template))
                score = np.dot(emb, template_vec)

            similarities[user] = score

        best_user = max(similarities, key=similarities.get)

        vote_counter[best_user] += 1

        for u, s in similarities.items():
            score_accumulator[u].append(s)

    # average similarity scores
    avg_scores = {u: np.mean(score_accumulator[u]) for u in score_accumulator}

    sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    best_user, best_score = sorted_scores[0]

    print("\n📊 AVERAGE SIMILARITY SCORES")

    for user, score in sorted_scores:
        print(f"{user:10s} → {score:.4f}")

    print("\n🗳️ WINDOW VOTES")

    for user, v in sorted(vote_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{user:10s} → {v}")

    print("\n🏆 FINAL DECISION")

    print(f"User: {best_user}")
    print(f"Score: {best_score:.4f}")
    print(f"Threshold: {SECURITY_THRESHOLD:.4f}")

    if best_score >= SECURITY_THRESHOLD:
        print(f"✅ IDENTIFIED AS: {best_user}")
    else:
        print("🚨 UNKNOWN PERSON")

# ==========================================
# MAIN BLIND TEST PIPELINE
# ==========================================

def main():

    print("\n🛡️ STARTING IDENTIFICATION BLIND TEST")

    print(f"📂 Blind Folder: {BLIND_FOLDER}")

    if not os.path.exists(BLIND_FOLDER):
        print("❌ Blind test folder missing")
        return

    print("\n🧠 Loading Trained Encoder...")

    encoder = get_loaded_encoder()

    print("✅ Encoder ready")

    # load global scaler
    if not os.path.exists(SCALER_PATH):
        print("❌ scaler.pkl missing!")
        return

    scaler = joblib.load(SCALER_PATH)

    test_files = [f for f in os.listdir(BLIND_FOLDER) if f.lower().endswith(".csv")]

    if not test_files:
        print("⚠️ No CSV files found")
        return

    print(f"\n📋 Found {len(test_files)} files")

    for file_name in sorted(test_files):

        file_path = os.path.join(BLIND_FOLDER, file_name)

        print("\n========================================")
        print(f"Testing File → {file_name}")
        print("========================================")

        identify_user(file_path, encoder, scaler)

    print("\n✨ Blind test completed")


if __name__ == "__main__":
    main()