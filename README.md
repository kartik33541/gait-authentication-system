# AI-Powered Contactless Gait Authentication System

This project implements a **Gait-based Biometric Authentication System** using smartphone accelerometer and gyroscope data. It demonstrates that human gait patterns are unique behavioral signatures that can be used for contactless security.

## Repository Structure
The repository is structured into two distinct environments:

### Research Phase
- Benchmarking and feature engineering using traditional Machine Learning (Random Forest) on the UCI HAR and Physics Toolbox datasets.

### Production Phase
- A high-precision, real-time authentication pipeline using:
  - Siamese 1D-CNN Encoder
  - Triplet Loss embeddings
  - AI-driven synthetic data scaling.

---

# 🛠️ General Environment Setup

## Python Version
All experiments and production code were tested on: **Python 3.12.0**

## Step 1 – Clone and Navigate
```bash
git clone https://github.com/kartik33541/gait-auth-api.git
cd gait_authentication
```

## Step 2 - Virtual Environment (Highly Recommended)
```bash
python -m venv venv
```
- *For Windows:* 
  ```bash
  venv\Scripts\activate
  ```
- *For Mac/Linux:* 
  ```bash
  source venv/bin/activate
  ```

## Step 3 – Install Dependencies
```bash
pip install -r requirements.txt
```
---

# Research and Production System Overview

## 🔬 OPTION A — Validate Research Results
This section is for evaluating the initial feasibility of the project using handcrafted features and traditional classifiers.

### Research Goals:
- Validate UCI HAR results (>80% accuracy)
- Evaluate real-world Physics Toolbox dataset (8 users)
- Analyze window-level vs. file-level majority voting

### Execution Order:
1. Navigate to `research/notebooks/` and run:
   - `01_uci_har_person_identification.ipynb`: Validates the UCI HAR dataset with 80:20, 70:30, and 60:40 splits.
   - `02_real_world_baseline.ipynb`: Benchmarks the 8-person RealWorld1 dataset.
   - `03_real_world_improvements.ipynb`: Explores window overlap and feature scaling.
   - `04_authentication_simulation.ipynb`: Simulates real-world "Access Granted/Denied" scenarios.

---

## 🚀 OPTION B — Real-Time Production System (Siamese 1D-CNN)
The production system is a hardened, scalable API designed for real-world deployment.

### 🧬 Model Architecture & Logic
- **Architecture:** Siamese 1D-Convolutional Neural Network (1D-CNN)
- **Core Engine:** Deep Metric Learning using Triplet Loss to map gait cycles into a 128-D embedding space.
- **Preprocessing:** Butterworth Bandpass Filter (0.5Hz – 3Hz) removes gravity bias and sensor jitter.
- **Security Guard:** Physics-based Walk Energy Score (`std(√(a_x^2 + a_y^2 + a_z^2)) ≥ 1.0`) to defeat "lift-and-drop" spoofing attacks.
- **Authentication:** Cosine Similarity threshold of 0.70 for biometric matching.

# 📝 Step-by-Step Production Setup

## 1. Navigate to Production
```bash
cd production
```

## 2. Scaling the Dataset (AI Generator)
To prevent overfitting, the system expands the 10 real users to 110 unique identities using biological variance injection.

### Bash:
```bash
python generate_synthetic_gait.py
```
Navigate back:
```bash
cd ..
```

### How it works:
This script reads the real-world "seed" data and applies Time Warping, Jittering, and Phase Shifting to create 100 new synthetic walkers. This ensures the model learns gait mechanics rather than memorizing 10 specific people.

## 3. Train the Siamese Brain
```bash
python cnn_engine/train_siamese.py
```
This script trains the 1D-CNN encoder to recognize unique gait signatures across all 110 identities.

## 4. Enroll Authorized Users
```bash
python cnn_engine/enroll_templates.py
```
This generates the mathematical `.pkl` templates for Person 1-10. These templates are what the live walk is compared against.

## 5. Launch the Robust Flask Server
```bash
python app/flask_server.py
```
The server includes an AI Warm-up Routine to eliminate the first-request lag and an API Key security layer.

## 6. Mobile App Integration 
- Install `GaitAuth_Live.apk` on your Android device.
- **Network Setup:** Connect your phone and laptop to the same Mobile Hotspot (to bypass campus firewalls).
- Find your laptop IP using `ipconfig` and enter it into the app (e.g., `http://192.168.1.5:8000`).
- **Test:** Walk naturally for 15 seconds. The app sends the CSV, and the server returns the biometric decision.

---
# 🔑 Authentication Responses
| Response | Description |
| --- | --- |
| **GRANTED** | `[Name]`: Similarity ≥ 0.70 with an enrolled template |
| **ACCESS_DENIED** | Similarity < 0.70 (Unrecognized pattern) |
| **STATIC/FAKE WALK DETECTED** | Energy Score < 1.0 (Fake walk attempt) |

---

# 📲 6. Mobile App Connectivity & Access Methods

The system supports two primary ways to connect the Android application to the inference server. Depending on your setup, ensure you install the correct `.apk` from the `production/mobile_app/` directory.

## 🏠 Method A: Local IP Address (Development)
**Use Case:** Testing the system on your local WiFi or Mobile Hotspot without deploying to the cloud.

**App to Use:** `GaitAuth_Flask2.apk`

### Setup Steps:
1. Connect both your laptop and phone to the same Mobile Hotspot (recommended to bypass firewall restrictions).
2. Find your laptop's Local IP by typing `ipconfig` in the terminal (e.g., `192.168.1.5`).
3. Open the app and enter the URL: `http://YOUR_IP:8000/predict` (or port 7860 if running the production script).

**Advantage:** Minimal latency and works without an active internet connection.

## 🌐 Method B: Live Link (Cloud Verification)
**Use Case:** Accessing the server via the public internet once deployed on Render or Hugging Face.

**App to Use:** `GaitAuth_Live.apk`

### Setup Steps:
1. Ensure your server is deployed and the status is **Live/Running**.
2. Copy your public URL from the Render/Hugging Face dashboard (e.g., `https://gait-secure-api.onrender.com`).
3. Open the app and enter the URL: `https://your-app-name.onrender.com/predict`.

**Advantage:** Allows for remote authentication from any location with internet access.

## ⚠️ Connectivity Troubleshooting
- **API Key:** Both methods require the header `X-API-KEY: GAIT_SECURE_2026` to be configured within the app logic.
- **Cold Start:** If using Method B on Render's free tier, the first request may take ~50 seconds to "wake up" the server.
- **Endpoint:** Always ensure the URL ends with the `/predict` route.

---

### 📂 Further Information
* **Project Structure:** Refer to the directory tree layout provided in the main README.
* **Research vs. Production:** See the comparison table in the sections above for phase differences.
* **Research Documentation:** [research/README.md](research/README.md)
* **Deployment Details:** [production/README.md](production/README.md)
* **LLM Usage:** [llm_usage.md](llm_usage.md)
* **Documentation:** [Documentation.md](Documentation.md)

**Final Note:** This system demonstrates that subtle human gait patterns captured through smartphone sensors can be engineered into a functional biometric system, validated through the research phase and proven in real-world deployment.
