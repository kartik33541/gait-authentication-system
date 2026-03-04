# import os
# import json
# import time
# import pandas as pd
# from google import genai

# # ==========================================
# # CONFIGURATION
# # ==========================================
# # Add your 3 API keys here
# API_KEYS = [
#     "AIzaSyCjSB3KzMn3QljQMWYc3TZlXmth5NSV-N8", 
#     "AIzaSyBH9Mjh_UjdV8FGlDqHyr2cYTM8b68C_CM", 
#     "AIzaSyA38cxEDiNdwbEiZ_ZvmtETZ3MET9o92pQ"
# ]

# TOTAL_USERS = 5000
# BATCH_SIZE = 50 
# OUTPUT_FILE = "biomechanical_profiles.json"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def analyze_ground_truth():
#     print("📊 Analyzing Ground Truth from 10 Real Users (30 files total)...")
#     all_data = []
#     valid_folders = [f"Person{i}" for i in range(1, 11)]

#     for folder_name in valid_folders:
#         folder_path = os.path.join(BASE_DIR, folder_name)
#         if os.path.exists(folder_path):
#             for file in os.listdir(folder_path):
#                 if file.endswith(".csv"):
#                     try:
#                         df = pd.read_csv(os.path.join(folder_path, file))
#                         df.columns = df.columns.str.lower().str.strip()
#                         cols = ["ax", "ay", "az", "wx", "wy", "wz"]
#                         if all(c in df.columns for c in cols):
#                             all_data.append(df[cols])
#                     except Exception as e:
#                         print(f"⚠️ Skipping {file}: {e}")
                        
#     if not all_data:
#         raise ValueError("❌ Could not find real CSV files! Ensure script is in RealWorldLive.")
        
#     combined_df = pd.concat(all_data, ignore_index=True)
#     return {
#         "acc_z_mean": float(round(combined_df['az'].mean(), 3)),
#         "acc_z_std": float(round(combined_df['az'].std(), 3)),
#         "acc_y_std": float(round(combined_df['ay'].std(), 3)),
#         "acc_x_std": float(round(combined_df['ax'].std(), 3)),
#         "gyro_x_std": float(round(combined_df['wx'].std(), 3)), 
#     }

# def get_llm_prompt(batch_size, stats, batch_idx):
#     archetypes = [
#         "Athletic/Fast (High Cadence, High Pitch)", 
#         "Elderly/Shuffling (Low Cadence, High Asymmetry)", 
#         "Heavy-set/Impactful (High Vertical G, Low Pitch)", 
#         "Tall/Long-stride (Low Cadence, High Swing)", 
#         "Average/Standard (Balanced)"
#     ]
#     current_type = archetypes[batch_idx % len(archetypes)]

#     return f"""
#     Generate a JSON array of {batch_size} unique, physically possible human walking profiles.
#     Target Archetype: {current_type}

#     GROUND TRUTH CONSTRAINTS (30Hz Smartphone):
#     - Base Vertical Mean: {stats['acc_z_mean']}g (Std: {stats['acc_z_std']})
#     - Avg Forward/Lateral Sway: {stats['acc_y_std']}g / {stats['acc_x_std']}g
#     - Avg Thigh Pitch: {stats['gyro_x_std']} rad/s

#     BIOMECHANICAL LOGIC RULES:
#     1. Height/Cadence: taller height (185cm+) MUST have lower cadence (1.4-1.7Hz).
#     2. Weight/Impact: heavier weight (90kg+) MUST have higher 'acc_z_vertical_g' and 'heel_strike_sharpness'.
#     3. Age/Stability: older age (60+) MUST have higher 'step_asymmetry' and lower 'gyro_pitch_thigh'.
#     4. Correlation: 'acc_z_vertical_g' and 'acc_y_forward_g' must be positively correlated (physics of propulsion).

#     Return ONLY a valid JSON array of objects. No prose.
#     Fields: [age_years, height_cm, weight_kg, cadence_hz, acc_z_vertical_g, acc_y_forward_g, acc_x_lateral_g, 
#     gyro_pitch_thigh_rad_s, gyro_roll_pocket_rad_s, gyro_yaw_pelvis_rad_s, step_asymmetry, heel_strike_sharpness]
#     """

# def main():
#     try:
#         stats = analyze_ground_truth()
#     except Exception as e:
#         print(str(e)); return

#     output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    
#     if os.path.exists(output_path):
#         with open(output_path, "r") as f:
#             final_output = json.load(f)
#         print(f"📂 Resuming. Already have {len(final_output)} profiles.")
#     else:
#         final_output = {}

#     total_batches = TOTAL_USERS // BATCH_SIZE
    
#     # Rotation logic variables
#     current_key_idx = 0
#     fail_count = 0
#     client = genai.Client(api_key=API_KEYS[current_key_idx])
    
#     while len(final_output) < TOTAL_USERS:
#         current_batch_idx = (len(final_output) // BATCH_SIZE) + 1
#         print(f"⏳ [Key {current_key_idx+1}] Requesting Batch {current_batch_idx}/{total_batches}...")
        
#         try:
#             response = client.models.generate_content(
#                 model='gemini-2.5-flash',
#                 contents=get_llm_prompt(BATCH_SIZE, stats, current_batch_idx)
#             )
            
#             text = response.text.strip()
#             if "```" in text:
#                 text = text.split("```")[1]
#                 if text.startswith("json"): text = text[4:]
            
#             batch_data = json.loads(text.strip())
            
#             if isinstance(batch_data, list):
#                 for profile in batch_data:
#                     if len(final_output) < TOTAL_USERS:
#                         person_id = f"person{11 + len(final_output)}"
#                         final_output[person_id] = profile
                
#                 with open(output_path, "w") as f:
#                     json.dump(final_output, f, indent=4)
                
#                 print(f"✅ Batch {current_batch_idx} saved. Total users: {len(final_output)}")
                
#                 # Success: reset fail count and brief breather
#                 fail_count = 0
#                 time.sleep(10) 
            
#         except Exception as e:
#             if "429" in str(e):
#                 fail_count += 1
#                 print(f"⚠️ Rate limit on Key {current_key_idx+1}. Fail count: {fail_count}")
                
#                 # Switch key if we fail twice with the current one
#                 if fail_count >= 2:
#                     current_key_idx = (current_key_idx + 1) % len(API_KEYS)
#                     print(f"🔄 ROTATING: Switching to API Key {current_key_idx+1}...")
#                     client = genai.Client(api_key=API_KEYS[current_key_idx])
#                     fail_count = 0
#                     time.sleep(5)
#                 else:
#                     print("😴 Sleeping 30s before retry...")
#                     time.sleep(30)
#             else:
#                 print(f"❌ Error: {e}. Retrying in 15s...")
#                 time.sleep(15)

#     print(f"\n✨ SUCCESS: 5,000 unique profiles saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()

import os
import json
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv

# Load environment variables from production/.env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ==========================================
# CONFIGURATION
# ==========================================
# Add your 3 API keys here
API_KEYS = [
    os.getenv("API_KEY_1"), 
    os.getenv("API_KEY_2"), 
    os.getenv("API_KEY_3")
]

TOTAL_USERS = 5000
BATCH_SIZE = 50 
OUTPUT_FILE = "biomechanical_profiles.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_ground_truth():
    print("📊 Analyzing Ground Truth from 10 Real Users (30 files total)...")
    all_data = []
    valid_folders = [f"Person{i}" for i in range(1, 11)]
    
    for folder_name in valid_folders:
        folder_path = os.path.join(BASE_DIR, folder_name)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(folder_path, file))
                        df.columns = df.columns.str.lower().str.strip()
                        cols = ["ax", "ay", "az", "wx", "wy", "wz"]
                        if all(c in df.columns for c in cols):
                            all_data.append(df[cols])
                    except Exception as e:
                        print(f"⚠️ Skipping {file}: {e}")
                        
    if not all_data:
        raise ValueError("❌ Could not find real CSV files! Ensure script is in RealWorldLive.")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return {
        "acc_z_mean": float(round(combined_df['az'].mean(), 3)),
        "acc_z_std": float(round(combined_df['az'].std(), 3)),
        "acc_y_std": float(round(combined_df['ay'].std(), 3)),
        "acc_x_std": float(round(combined_df['ax'].std(), 3)),
        "gyro_x_std": float(round(combined_df['wx'].std(), 3)), 
    }

def get_llm_prompt(batch_size, stats, batch_idx):
    archetypes = [
        "Athletic/Fast (High Cadence, High Pitch)", 
        "Elderly/Shuffling (Low Cadence, High Asymmetry)", 
        "Heavy-set/Impactful (High Vertical G, Low Pitch)", 
        "Tall/Long-stride (Low Cadence, High Swing)", 
        "Average/Standard (Balanced)"
    ]
    current_type = archetypes[batch_idx % len(archetypes)]

    return f"""
    Generate a JSON array of {batch_size} unique, physically possible human walking profiles.
    Target Archetype: {current_type}

    GROUND TRUTH CONSTRAINTS (30Hz Smartphone):
    - Base Vertical Mean: {stats['acc_z_mean']}g (Std: {stats['acc_z_std']})
    - Avg Forward/Lateral Sway: {stats['acc_y_std']}g / {stats['acc_x_std']}g
    - Avg Thigh Pitch: {stats['gyro_x_std']} rad/s

    BIOMECHANICAL LOGIC RULES:
    1. Height/Cadence: taller height (185cm+) MUST have lower cadence (1.4-1.7Hz).
    2. Weight/Impact: heavier weight (90kg+) MUST have higher 'acc_z_vertical_g' and 'heel_strike_sharpness'.
    3. Age/Stability: older age (60+) MUST have higher 'step_asymmetry' and lower 'gyro_pitch_thigh'.
    4. Correlation: 'acc_z_vertical_g' and 'acc_y_forward_g' must be positively correlated (physics of propulsion).

    Return ONLY a valid JSON array of objects. No prose.
    Fields: [age_years, height_cm, weight_kg, cadence_hz, acc_z_vertical_g, acc_y_forward_g, acc_x_lateral_g, 
    gyro_pitch_thigh_rad_s, gyro_roll_pocket_rad_s, gyro_yaw_pelvis_rad_s, step_asymmetry, heel_strike_sharpness]
    """

def main():
    try:
        stats = analyze_ground_truth()
    except Exception as e:
        print(str(e)); return

    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            final_output = json.load(f)
        print(f"📂 Resuming. Already have {len(final_output)} profiles.")
    else:
        final_output = {}

    total_batches = TOTAL_USERS // BATCH_SIZE
    
    # Rotation logic variables
    current_key_idx = 0
    fail_count = 0
    client = genai.Client(api_key=API_KEYS[current_key_idx])
    
    while len(final_output) < TOTAL_USERS:
        current_batch_idx = (len(final_output) // BATCH_SIZE) + 1
        print(f"⏳ [Key {current_key_idx+1}] Requesting Batch {current_batch_idx}/{total_batches}...")
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=get_llm_prompt(BATCH_SIZE, stats, current_batch_idx)
            )
            
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]
            
            batch_data = json.loads(text.strip())
            
            if isinstance(batch_data, list):
                for profile in batch_data:
                    if len(final_output) < TOTAL_USERS:
                        person_id = f"person{11 + len(final_output)}"
                        final_output[person_id] = profile
                
                with open(output_path, "w") as f:
                    json.dump(final_output, f, indent=4)
                
                print(f"✅ Batch {current_batch_idx} saved. Total users: {len(final_output)}")
                
                # Success: reset fail count and brief breather
                fail_count = 0
                time.sleep(10) 
            
        except Exception as e:
            if "429" in str(e):
                fail_count += 1
                print(f"⚠️ Rate limit on Key {current_key_idx+1}. Fail count: {fail_count}")
                
                # Switch key if we fail twice with the current one
                if fail_count >= 2:
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    print(f"🔄 ROTATING: Switching to API Key {current_key_idx+1}...")
                    client = genai.Client(api_key=API_KEYS[current_key_idx])
                    fail_count = 0
                    time.sleep(5)
                else:
                    print("😴 Sleeping 30s before retry...")
                    time.sleep(30)
            else:
                print(f"❌ Error: {e}. Retrying in 15s...")
                time.sleep(15)

    print(f"\n✨ SUCCESS: 5,000 unique profiles saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()