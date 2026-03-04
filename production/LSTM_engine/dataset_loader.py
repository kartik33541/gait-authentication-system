# import os
# import random
# import numpy as np
# import pandas as pd
# from scipy.signal import butter, filtfilt
# from sklearn.preprocessing import StandardScaler

# # ==========================================
# # CONFIGURATION
# # ==========================================
# # Points to the 'production' folder
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# # Points to 'production/RealWorldLive'
# REAL_DATA_DIR = os.path.join(BASE_DIR, "RealWorldLive")

# # FIXED: Points to synthetic_data INSIDE RealWorldLive based on your actual structure
# SYNTH_DATA_DIR = os.path.join(REAL_DATA_DIR, "synthetic_data")

# WINDOW_SIZE = 256
# STEP_SIZE = 128  # 50% overlap for sliding window

# def butter_bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=38.0, order=4):
#     """
#     Frequency Normalization (DSP):
#     Strips gravity (0Hz) and high-frequency static (>15Hz) from the raw signal.
#     """
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data, axis=0)
#     return y

# def extract_windows(df, scaler):
#     """
#     Applies the normalizations and chops the variable-length CSV into 256-row blocks.
#     """
#     if 'timestamp' in df.columns:
#         df = df.drop(columns=['timestamp'])
        
#     raw_data = df.values
    
#     # Frequency Normalization
#     filtered_data = butter_bandpass_filter(raw_data)
    
#     # Spatial Normalization (Z-Score Scaling)
#     scaled_data = scaler.fit_transform(filtered_data)
    
#     windows = []
#     # Temporal Normalization (Sliding Window)
#     for start in range(0, len(scaled_data) - WINDOW_SIZE + 1, STEP_SIZE):
#         window = scaled_data[start:start + WINDOW_SIZE]
#         windows.append(window)
        
#     return windows

# def load_all_users():
#     """
#     Loads all users from RealWorldLive and synthetic_data folders.
#     """
#     print("⚙️ Loading Data pipeline...")
#     X_data = []
#     y_labels = []
    
#     label_map = {}
#     current_label_id = 0
#     scaler = StandardScaler()
    
#     # Combine both paths into a single search list
#     paths_to_search = [REAL_DATA_DIR, SYNTH_DATA_DIR]
    
#     total_files_processed = 0
    
#     for search_dir in paths_to_search:
#         if not os.path.exists(search_dir):
#             print(f"⚠️ Warning: Directory not found: {search_dir}")
#             continue
            
#         print(f"📁 Searching in: {search_dir}")
#         for folder in os.listdir(search_dir):
#             # Check for "Person" (Real) or "User" (Synthetic) folders
#             if not (folder.lower().startswith("person") or folder.lower().startswith("user")):
#                 continue
                
#             folder_path = os.path.join(search_dir, folder)
#             if not os.path.isdir(folder_path):
#                 continue
                
#             if folder not in label_map:
#                 label_map[folder] = current_label_id
#                 current_label_id += 1
                
#             user_label = label_map[folder]
            
#             for file in os.listdir(folder_path):
#                 if file.endswith(".csv"):
#                     file_path = os.path.join(folder_path, file)
#                     try:
#                         df = pd.read_csv(file_path)
#                         windows = extract_windows(df, scaler)
                        
#                         for w in windows:
#                             X_data.append(w)
#                             y_labels.append(user_label)
                            
#                         total_files_processed += 1
#                     except Exception as e:
#                         print(f"⚠️ Skipping {file}: {e}")
                        
#     print(f"✅ Extracted {len(X_data)} perfect 256-timestep windows from {total_files_processed} files.")
#     print(f"🧬 Total unique human identities loaded: {current_label_id}")
    
#     return np.array(X_data), np.array(y_labels), label_map

# def generate_siamese_pairs(X, y):
#     """
#     Generates balanced Positive (Same User) and Negative (Different User) pairs.
#     """
#     print("⚖️ Generating balanced Siamese training pairs...")
#     pair_images = []
#     pair_labels = []
    
#     num_classes = len(np.unique(y))
#     if num_classes < 2:
#         raise ValueError("❌ Not enough users found to create Siamese pairs! Check your data folders.")

#     class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
#     for idx1 in range(len(X)):
#         # --- Create a POSITIVE pair (Same User) ---
#         current_user = y[idx1]
#         idx2 = random.choice(class_indices[current_user])
#         pair_images.append([X[idx1], X[idx2]])
#         pair_labels.append(1) 
        
#         # --- Create a NEGATIVE pair (Different User) ---
#         different_user = random.choice([i for i in range(num_classes) if i != current_user])
#         idx3 = random.choice(class_indices[different_user])
#         pair_images.append([X[idx1], X[idx3]])
#         pair_labels.append(0) 
        
#     pair_images = np.array(pair_images)
#     pair_labels = np.array(pair_labels).astype('float32')
    
#     indices = np.arange(len(pair_labels))
#     np.random.shuffle(indices)
    
#     pair_images = pair_images[indices]
#     pair_labels = pair_labels[indices]
    
#     print(f"✅ Generated {len(pair_labels)} total pairs ({len(pair_labels)//2} Positive, {len(pair_labels)//2} Negative).")
#     return pair_images, pair_labels

# if __name__ == "__main__":
#     X, y, mapping = load_all_users()
#     if len(X) > 0:
#         pairs, labels = generate_siamese_pairs(X, y)
#         print(f"Pair Array Shape: {pairs.shape}")

import os
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION - BASED EXACTLY ON YOUR IMAGE
# ==========================================
# BASE_DIR is 'production'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
REAL_DATA_DIR = os.path.join(BASE_DIR, "RealWorldLive")
SYNTH_DATA_DIR = os.path.join(REAL_DATA_DIR, "synthetic_data")

WINDOW_SIZE = 256
STEP_SIZE = 128 

def apply_bandpass_filter(data, lowcut=0.3, highcut=12.0, fs=38.0, order=3):
    """Normalization 1: Frequency Normalization (38Hz hardware calibrated)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def extract_windows(df, scaler):
    """Normalizations 2 & 3: Z-Scaling and Zero-Mean Centering"""
    cols_to_use = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    if not all(col in df.columns for col in cols_to_use):
        data = df.iloc[:, :6].values.astype(np.float32)
    else:
        data = df[cols_to_use].values.astype(np.float32)
        
    filtered_data = apply_bandpass_filter(data, fs=38.0)
    scaled_data = scaler.fit_transform(filtered_data)
    
    windows = []
    # Normalization 4: Temporal Windowing
    for start in range(0, len(scaled_data) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(scaled_data[start:start + WINDOW_SIZE])
    return windows

def load_all_users():
    """Bulletproof explicitly targeted loader."""
    print(f"⚙️  Initializing Data Pipeline (Hardware Rate: 38Hz)...")
    X_data = []
    y_labels = []
    label_map = {}
    current_label_id = 0
    scaler = StandardScaler()
    total_files = 0
    
    # --- PHASE 1: Load Real Users (Person1 to Person10) ---
    print(f"📁 Scanning Phase 1: Real Users in {REAL_DATA_DIR}")
    if os.path.exists(REAL_DATA_DIR):
        for folder in sorted(os.listdir(REAL_DATA_DIR)):
            folder_path = os.path.join(REAL_DATA_DIR, folder)
            if not os.path.isdir(folder_path): continue
            
            if folder.lower().startswith("person"):
                label_map[folder] = current_label_id
                user_label = current_label_id
                current_label_id += 1
                
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(folder_path, file))
                            for w in extract_windows(df, scaler):
                                X_data.append(w)
                                y_labels.append(user_label)
                            total_files += 1
                        except Exception: pass
    
    real_count = current_label_id
    print(f"   ✅ Found {real_count} Real Users.")

    # --- PHASE 2: Load Synthetic Users (User1 to User5010) ---
    print(f"📁 Scanning Phase 2: Synthetic Users in {SYNTH_DATA_DIR}")
    if os.path.exists(SYNTH_DATA_DIR):
        for folder in sorted(os.listdir(SYNTH_DATA_DIR)):
            folder_path = os.path.join(SYNTH_DATA_DIR, folder)
            if not os.path.isdir(folder_path): continue
            
            if folder.lower().startswith("user") or folder.lower().startswith("person"):
                label_map[folder] = current_label_id
                user_label = current_label_id
                current_label_id += 1
                
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(folder_path, file))
                            for w in extract_windows(df, scaler):
                                X_data.append(w)
                                y_labels.append(user_label)
                            total_files += 1
                        except Exception: pass
                        
                if current_label_id % 1000 == 0:
                    print(f"   🔄 Processed {current_label_id} synthetic identities...")

    print(f"\n✅ TOTAL SUCCESS: Loaded {len(X_data)} windows from {current_label_id} unique identities.")
    return np.array(X_data, dtype=np.float32), np.array(y_labels, dtype=np.int32), label_map

def generate_siamese_pairs(X, y):
    print("⚖️  Generating balanced Siamese pairs (Pos/Neg)...")
    if len(X) == 0: return np.array([]), np.array([])
    
    num_classes = len(np.unique(y))
    num_pairs = len(X) * 2
    
    pair_images = np.zeros((num_pairs, 2, WINDOW_SIZE, 6), dtype=np.float32)
    pair_labels = np.zeros(num_pairs, dtype=np.float32)

    class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    pair_idx = 0
    for idx1 in range(len(X)):
        curr_user = y[idx1]
        
        # Positive
        idx2 = random.choice(class_indices[curr_user])
        pair_images[pair_idx, 0] = X[idx1]
        pair_images[pair_idx, 1] = X[idx2]
        pair_labels[pair_idx] = 1.0 
        pair_idx += 1
        
        # Negative
        diff_user = random.choice([i for i in range(num_classes) if i != curr_user])
        idx3 = random.choice(class_indices[diff_user])
        pair_images[pair_idx, 0] = X[idx1]
        pair_images[pair_idx, 1] = X[idx3]
        pair_labels[pair_idx] = 0.0 
        pair_idx += 1
        
    indices = np.arange(num_pairs)
    np.random.shuffle(indices)
    print(f"✅ Generated {len(pair_labels)} total pairs.")
    return pair_images[indices], pair_labels[indices]