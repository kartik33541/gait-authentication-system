import os
import json
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(BASE_DIR, "biomechanical_profiles.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "synthetic_data")

def analyze_real_hardware():
    """
    Analyzes Person 1 to 10 CSVs to extract the specific 
    physical baselines (Z-axis gravity roughly ~9.8 m/s²) 
    and hardware row counts (~560 rows).
    """
    print("📊 Extracting ground-truth hardware baseline...")
    stats = {'row_counts': [], 'az_baseline': []}
    
    for i in range(1, 11):
        folder = os.path.join(BASE_DIR, f"Person{i}")
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(folder, f))
                        # Standardize columns
                        df.columns = df.columns.str.lower().str.strip()
                        if 'az' in df.columns:
                            stats['row_counts'].append(len(df))
                            stats['az_baseline'].append(df['az'].mean())
                    except Exception as e:
                        pass
    
    # Defaults just in case real folders are missing
    return {
        'avg_rows': int(np.mean(stats['row_counts'])) if stats['row_counts'] else 560,
        'baseline_az': np.mean(stats['az_baseline']) if stats['az_baseline'] else 9.81
    }

def simulate_realistic_window(profile, hw_dna, walk_idx):
    """
    Fourier Physics Engine precisely constrained to MIT App
    recording window (38Hz).
    """
    # -------------------------------------------------------------------
    # 1. HARDWARE TIMESTAMPS: 15-second MIT fix (~38Hz)
    # -------------------------------------------------------------------
    # App logic randomly delays start between 23-38ms
    start_ts = np.random.randint(23, 39) 
    # Strict stop slightly after 15,000ms limit
    target_end = 15000 + np.random.randint(5, 25)
    total_duration_ms = target_end - start_ts
    
    # Draw an exact sample count based on the MIT hardware variance 
    # (Typically ~540 to 580 samples over 15s)
    num_samples = int(np.random.normal(hw_dna['avg_rows'], 12))
    
    # Generate noisy ~26ms hardware gaps, then mathematically stretch/shrink 
    # them so the final accumulated gap exactly matches total_duration_ms
    avg_gap = total_duration_ms / (num_samples - 1)
    gaps = np.random.normal(avg_gap, 1.8, num_samples - 1)
    gaps *= (total_duration_ms / np.sum(gaps)) 
    
    # Construct exact timestamp array
    timestamp_ms = np.zeros(num_samples)
    timestamp_ms[0] = start_ts
    timestamp_ms[1:] = start_ts + np.cumsum(gaps)
    timestamp_ms = np.round(timestamp_ms).astype(int)
    # Force last tick to prevent floating point drift
    timestamp_ms[-1] = target_end 

    t = (timestamp_ms - start_ts) / 1000.0 # Standardize to seconds

    # -------------------------------------------------------------------
    # 2. EXTRACT LLM PROFILE
    # -------------------------------------------------------------------
    # Adds subtle <1.5% cadence shift for alternate walks to prevent exact cloning
    cad_variance = np.random.uniform(0.985, 1.015) if walk_idx > 1 else 1.0
    cadence = profile["cadence_hz"] * cad_variance
    asymmetry = profile["step_asymmetry"]
    sharpness = profile["heel_strike_sharpness"]

    # -------------------------------------------------------------------
    # 3. VERTICAL ACCELERATION (az) - Fourier Harmonic Synthesis
    # -------------------------------------------------------------------
    az_amp = profile["acc_z_vertical_g"]
    
    # Use real gravity baseline + 3 harmonics for the "double hump" gait wave
    az = hw_dna['baseline_az'] + (
        az_amp * np.sin(2 * np.pi * cadence * t) +
        0.3 * az_amp * np.sin(4 * np.pi * cadence * t) +  # 2nd Harmonic
        0.1 * az_amp * np.sin(6 * np.pi * cadence * t)    # 3rd Harmonic
    )

    # Heel Strike Transients & Step Asymmetry 
    step_times = np.arange(0, t[-1], 1/cadence)
    for i, st in enumerate(step_times):
        idx = np.searchsorted(t, st)
        # Apply profile asymmetry to alternate footfalls
        mod = (1 - asymmetry) if i % 2 == 1 else 1.0
        
        # Exponential shockwave decay matching the sharpness profile
        for d in range(min(6, len(az)-idx)):
            az[idx+d] += (sharpness * 1.5 * mod) * np.exp(-d * 0.7)

    # -------------------------------------------------------------------
    # 4. Y/X ACCELERATION & GYROSCOPE SYNTHESIS
    # -------------------------------------------------------------------
    # Forward sway (phase-shifted by 90deg)
    ay = profile["acc_y_forward_g"] * np.sin(2 * np.pi * cadence * t + np.pi/2)
    # Lateral sway (half frequency - side to side over two full steps)
    ax = profile["acc_x_lateral_g"] * np.sin(np.pi * cadence * t) 

    # Gyroscopic rotation physics
    wx = profile["gyro_pitch_thigh_rad_s"] * np.sin(2 * np.pi * cadence * t)
    wy = profile["gyro_roll_pocket_rad_s"] * np.sin(2 * np.pi * cadence * t + np.pi/4)
    wz = profile["gyro_yaw_pelvis_rad_s"] * np.sin(np.pi * cadence * t)

    # -------------------------------------------------------------------
    # 5. SENSOR NOISE
    # -------------------------------------------------------------------
    noise_acc = 0.05
    noise_gyro = 0.15
    
    return pd.DataFrame({
        'timestamp': timestamp_ms,
        'ax': ax + np.random.normal(0, noise_acc, num_samples),
        'ay': ay + np.random.normal(0, noise_acc, num_samples),
        'az': az + np.random.normal(0, noise_acc, num_samples),
        'wx': wx + np.random.normal(0, noise_gyro, num_samples),
        'wy': wy + np.random.normal(0, noise_gyro, num_samples),
        'wz': wz + np.random.normal(0, noise_gyro, num_samples)
    })

def main():
    print("⚙️ Initializing Fourier Physics Synthesizer...")
    
    # Analyze the real ~38Hz gaps and gravity Z baselines
    hw_dna = analyze_real_hardware()
    print(f"✅ Baseline Calibrated: Z={hw_dna['baseline_az']:.3f}, TargetRows={hw_dna['avg_rows']}")
    
    if not os.path.exists(PROFILES_FILE):
        raise FileNotFoundError(f"❌ Could not find {PROFILES_FILE}.")
        
    with open(PROFILES_FILE, "r") as f:
        profiles = json.load(f)
        
    print(f"🧬 Loaded {len(profiles)} unique identity profiles.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_files = 0
    
    user_keys = sorted(profiles.keys(), key=lambda x: int(x.replace('person', '')))
    
    for user_id in user_keys:
        profile = profiles[user_id]
        folder_name = user_id.capitalize()
        user_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(user_folder, exist_ok=True)
        
        for walk_idx in range(1, 4):
            df = simulate_realistic_window(profile, hw_dna, walk_idx)
            file_name = f"{user_id}_walk{walk_idx}.csv"
            save_path = os.path.join(user_folder, file_name)
            
            # Formats rounding to maintain original data structure match
            df = df.round({'ax': 4, 'ay': 4, 'az': 4, 'wx': 4, 'wy': 4, 'wz': 4})
            df.to_csv(save_path, index=False)
            total_files += 1
            
        if int(user_id.replace('person', '')) % 100 == 0:
            print(f"✅ Synthesized physical data up to {folder_name}...")

    print(f"\n✨ COMPLETE! Successfully synthesized {total_files} strictly compliant CSV files.")
    print(f"📁 Stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()