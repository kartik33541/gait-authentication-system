# import os
# import json
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler

# # Import our exact preprocessing pipeline
# from dataset_loader import extract_windows

# # ==========================================
# # CONFIGURATION
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ENCODER_PATH = os.path.join(BASE_DIR, "production_encoder.h5")
# VAULT_PATH = os.path.join(BASE_DIR, "vault.json")

# SECURITY_THRESHOLD = 0.80 

# def l2_normalize(vector):
#     norm = np.linalg.norm(vector)
#     if norm == 0: return vector
#     return vector / norm

# def process_live_login(claimed_identity, live_csv_path):
#     """
#     Designed for API usage. Takes the uploaded file and identity,
#     and returns a structured dictionary for the Flask JSON response.
#     """
#     # 1. Open the Vault
#     if not os.path.exists(VAULT_PATH):
#         return {"authenticated": False, "score": 0.0, "message": "Server Error: Vault missing."}
        
#     with open(VAULT_PATH, "r") as f:
#         vault = json.load(f)
        
#     if claimed_identity not in vault:
#         return {"authenticated": False, "score": 0.0, "message": f"User '{claimed_identity}' not registered."}
        
#     master_template = np.array(vault[claimed_identity])
    
#     # 2. Load the AI Brain
#     if not os.path.exists(ENCODER_PATH):
#         return {"authenticated": False, "score": 0.0, "message": "Server Error: Biometric model offline."}
        
#     encoder = load_model(ENCODER_PATH, compile=False)
    
#     # 3. Read and Clean the Live Walk
#     try:
#         df = pd.read_csv(live_csv_path)
#     except Exception as e:
#         return {"authenticated": False, "score": 0.0, "message": "Failed to read uploaded CSV data."}
        
#     scaler = StandardScaler()
#     windows = extract_windows(df, scaler)
    
#     if len(windows) == 0:
#         return {"authenticated": False, "score": 0.0, "message": "Data rejected: Walk recording was too short."}
        
#     # 4. Extract the Live 128D Vector
#     live_windows = np.array(windows)
#     embeddings = encoder.predict(live_windows, verbose=0)
    
#     live_signature = np.mean(embeddings, axis=0)
#     live_signature = l2_normalize(live_signature)
    
#     # 5. THE DECISION
#     similarity_score = float(np.dot(live_signature, master_template)) # Cast to float for JSON serialization
    
#     # Package the final API response
#     if similarity_score >= SECURITY_THRESHOLD:
#         return {
#             "authenticated": True, 
#             "score": round(similarity_score, 4), 
#             "message": "Access Granted. Identity Verified."
#         }
#     else:
#         return {
#             "authenticated": False, 
#             "score": round(similarity_score, 4), 
#             "message": "Access Denied. Biometric mismatch."
#         }

# # No main() block is needed here because this file acts as a module 
# # that your flask_server.py will import and call directly.




import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler

# Import our exact preprocessing pipeline and architecture
from dataset_loader import extract_windows
from build_encoder import get_encoder

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")
VAULT_PATH = os.path.join(BASE_DIR, "vault.json")

# Cosine similarity threshold. Set to 0.75 based on our blind test results.
SECURITY_THRESHOLD = 0.75 

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: return vector
    return vector / norm

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

# ==========================================
# ⚡ FAST API INITIALIZATION
# Load the model ONCE globally so logins are instant
# ==========================================
try:
    GLOBAL_ENCODER = get_loaded_encoder()
    print("✅ Live API Brain is online and ready for incoming connections.")
except Exception as e:
    GLOBAL_ENCODER = None
    print(f"⚠️ Warning: Could not load model weights on startup: {e}")

def process_live_login(claimed_identity, live_csv_path):
    """
    Designed for API usage. Takes the uploaded file and identity,
    and returns a structured dictionary for the Flask JSON response.
    """
    # 1. Open the Vault
    if not os.path.exists(VAULT_PATH):
        return {"authenticated": False, "score": 0.0, "message": "Server Error: Vault missing."}
        
    with open(VAULT_PATH, "r") as f:
        vault = json.load(f)
        
    if claimed_identity not in vault:
        return {"authenticated": False, "score": 0.0, "message": f"User '{claimed_identity}' not registered."}
        
    master_template = np.array(vault[claimed_identity])
    
    # 2. Check the AI Brain
    if GLOBAL_ENCODER is None:
        return {"authenticated": False, "score": 0.0, "message": "Server Error: Biometric model offline."}
        
    # 3. Read and Clean the Live Walk
    try:
        df = pd.read_csv(live_csv_path)
        scaler = StandardScaler()
        # Uses the format exactly matched to your local dataset_loader
        windows = extract_windows(df, scaler)
    except Exception as e:
        return {"authenticated": False, "score": 0.0, "message": f"Failed to read uploaded CSV data: {e}"}
        
    if len(windows) == 0:
        return {"authenticated": False, "score": 0.0, "message": "Data rejected: Walk recording was too short."}
        
    # 4. Extract the Live 256D Vector (Fast API Prediction)
    live_windows = np.array(windows, dtype=np.float32)
    embeddings = GLOBAL_ENCODER.predict(live_windows, verbose=0)
    
    live_signature = l2_normalize(np.mean(embeddings, axis=0))
    
    # 5. THE DECISION
    similarity_score = float(np.dot(live_signature, master_template)) # Cast to float for JSON serialization
    
    # Package the final API response
    if similarity_score >= SECURITY_THRESHOLD:
        return {
            "authenticated": True, 
            "score": round(similarity_score, 4), 
            "message": "Access Granted. Identity Verified."
        }
    else:
        return {
            "authenticated": False, 
            "score": round(similarity_score, 4), 
            "message": "Access Denied. Biometric mismatch."
        }

# No main() block is needed here because this file acts as a module 
# that your flask_server.py will import and call directly.