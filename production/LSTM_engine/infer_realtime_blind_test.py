import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler

# Import your exact preprocessing and architecture
from dataset_loader import extract_windows
from build_encoder import get_encoder

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_AUTH_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")
VAULT_PATH = os.path.join(BASE_DIR, "vault.json")
BLIND_FOLDER = os.path.join(GAIT_AUTH_ROOT, "Blind_Test_Data")

# Cosine similarity threshold. Your model hit 0.98+ for matches.
SECURITY_THRESHOLD = 0.75 

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def get_loaded_encoder():
    """Rebuilds architecture and injects weights safely"""
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

def authenticate_user(claimed_identity, live_csv_path, encoder):
    if not os.path.exists(VAULT_PATH):
        print("❌ Vault not found!")
        return False
        
    with open(VAULT_PATH, "r") as f:
        vault = json.load(f)
        
    if claimed_identity not in vault:
        print(f"🚨 ACCESS DENIED: User '{claimed_identity}' missing in vault.")
        return False
        
    master_template = np.array(vault[claimed_identity])
    
    try:
        # Load the dataframe and initialize the scaler (matches your dataset_loader.py exactly)
        df = pd.read_csv(live_csv_path)
        scaler = StandardScaler()
        # Passes BOTH arguments so you don't get the positional argument error
        windows = extract_windows(df, scaler)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    if len(windows) == 0:
        print("🚨 ACCESS DENIED: Data too short for analysis.")
        return False
        
    # Convert windows directly to numpy array (data is ALREADY scaled inside extract_windows)
    windows_arr = np.array(windows, dtype=np.float32)
        
    # Extract the live signature
    embeddings = encoder.predict(windows_arr, verbose=0)
    live_signature = l2_normalize(np.mean(embeddings, axis=0))
    
    # Calculate similarity score (Dot product of L2 normalized vectors = Cosine Similarity)
    similarity_score = np.dot(live_signature, master_template)
    
    print(f"\n📊 BIOMETRIC ANALYSIS for {claimed_identity}:")
    print(f"   📐 Match Score: {similarity_score:.4f}")
    print(f"   🛡️ Threshold:   {SECURITY_THRESHOLD:.4f}")
    
    if similarity_score >= SECURITY_THRESHOLD:
        print("✅ AUTHENTICATED. TRUE OWNER VERIFIED.")
        return True
    else:
        print("🚨 IMPOSTER DETECTED. SYSTEM LOCKED.")
        return False

def main():
    print("\n🛡️ STARTING AUTOMATED BLIND TEST...")
    print(f"📂 Target Folder: {BLIND_FOLDER}")

    if not os.path.exists(BLIND_FOLDER):
        print(f"❌ ERROR: Cannot find folder at: {BLIND_FOLDER}")
        return

    # Load Brain
    print("🧠 Rebuilding Architecture and Injecting Weights...")
    encoder = get_loaded_encoder()
    print("✅ Brain is online.")

    # Find ALL .csv files in the blind test folder
    test_files = [f for f in os.listdir(BLIND_FOLDER) if f.endswith(".csv")]
    
    if not test_files:
        print("⚠️ No .csv files found in the Blind_Test_Data folder!")
        return

    print(f"📋 Found {len(test_files)} test files. Commencing analysis...\n")

    # Loop through every single file found in the folder
    for file_name in sorted(test_files):
        # Extract the user ID from the filename (e.g., 'person12_walk3.csv' -> 'Person12')
        parts = file_name.split("_")
        claimed_id = parts[0].capitalize()
        file_path = os.path.join(BLIND_FOLDER, file_name)
        
        print(f"\n--- Testing File: {file_name} ---")
        authenticate_user(claimed_id, file_path, encoder)

    print("\n✨ All tests concluded.")

if __name__ == "__main__":
    main()