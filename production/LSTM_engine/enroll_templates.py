import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler

from dataset_loader import extract_windows
from build_encoder import get_encoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "RealWorldLive")

WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")
VAULT_PATH = os.path.join(BASE_DIR, "vault.json")

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def main():
    print("🔐 Initiating Biometric Enrollment Protocol...")
    
    print("🧠 Rebuilding 256D Architecture...")
    input_shape = (256, 6)
    encoder = get_encoder(input_shape=input_shape, embedding_dim=256)
    
    try:
        # Reconstruct the exact Siamese wrapper to inject weights without errors
        input_a = tf.keras.layers.Input(shape=input_shape)
        input_b = tf.keras.layers.Input(shape=input_shape)
        out_a = encoder(input_a)
        out_b = encoder(input_b)
        
        dist = tf.keras.layers.Lambda(euclidean_distance)([out_a, out_b])
        full_siamese = tf.keras.models.Model(inputs=[input_a, input_b], outputs=dist)
        
        full_siamese.load_weights(WEIGHTS_PATH)
        print("✅ Weights successfully injected into the LSTM engine.")
    except Exception as e:
        print(f"❌ Weight injection failed: {e}")
        return

    vault = {}
    real_users = [f"Person{i}" for i in range(1, 11)]
    
    for user_folder in real_users:
        folder_path = os.path.join(REAL_DATA_DIR, user_folder)
        if not os.path.exists(folder_path):
            continue
            
        print(f"⏳ Processing Biometrics for {user_folder}...")
        user_windows = []
        scaler = StandardScaler()
        
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(file_path)
                    windows = extract_windows(df, scaler)
                    user_windows.extend(windows)
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")
                    
        if len(user_windows) == 0:
            continue
            
        user_windows_arr = np.array(user_windows, dtype=np.float32)
        embeddings = encoder.predict(user_windows_arr, verbose=0)
        
        # Mean pooling + final normalization step
        master_vector = np.mean(embeddings, axis=0)
        master_vector = l2_normalize(master_vector) 
        
        vault[user_folder] = master_vector.tolist()
        print(f"   ✅ {user_folder} enrolled successfully.")
        
    with open(VAULT_PATH, "w") as f:
        json.dump(vault, f, indent=4)
        
    print(f"\n✨ Enrollment Complete! Vault saved to: {VAULT_PATH}")

if __name__ == "__main__":
    main()