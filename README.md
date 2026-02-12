# AI-Powered Contactless Gait Authentication System

This project implements a **Gait-based Biometric Authentication System** using smartphone accelerometer and gyroscope data.

It consists of two major parts:

1. **Research Phase** – Model development, validation, and experimentation.
2. **Production Phase** – Real-time authentication system using a mobile app and HTTP server.

The system demonstrates that human gait patterns can be used as a biometric identifier.

---

# Python Version

All experiments and production code were tested on:

**Python 3.12.0 (Recommended)**

Virtual environment is optional but strongly recommended.

---

# Quick Setup

## Step 1 – Clone Repository

```bash
cd gait_authentication
```
## Step 2 - Create Virtual Environment (Recommended)
python -m venv venv
# For Windows:
venv\Scripts\activate  
# Note: Results and experiments were originally trained on a Conda environment (Recommended)
```
## Step 3 – Install Dependencies
```bash
pip install -r requirements.txt
```
## Choose What You Want to Run
You have two main options:

### OPTION A — Validate Research Results (Pre-Collected Data)
If you want to:
- Validate UCI HAR results (>80% accuracy)
- Evaluate real-world Physics Toolbox dataset (8 users)
- See experimental analysis and comparisons
- Inspect window-level and file-level accuracy
Navigate to:
- `research/notebooks/`
Run notebooks in this order:
- `01_uci_har_person_identification.ipynb` — To validate UCI HAR dataset 
- `02_real_world_baseline.ipynb` — Notebook 2,3,4 tested on real-world dataset inside `Research/RealWorld1` (dataset of 8 users collected from Physics Toolbox Sensor suite)
- `03_real_world_improvements.ipynb`
- `04_authentication_simulation.ipynb`
These notebooks include:
- 80:20, 70:30, 60:40 validation splits (UCI HAR)
- Window-level & file-level metrics
- Majority voting analysis
- Real-world dataset experiments (Physics Toolbox)
For full research explanation, methodology, and analysis:
documentation/research/README.md`

## OPTION B — Run Real-Time Authentication System (Production Mode)

If you want to experience live gait authentication, this section allows you to collect your own walking data, train a custom model, and perform real-time authentication via the mobile app.

### 🚀 Production System Overview
The data flow for the real-time system is defined as:
**Mobile App** → **CSV Data** → **HTTP POST** → **Server** → **Model** → **Access Decision**


### 🛠️ Model Configuration
The production system uses the following default parameters:
* **Algorithm:** Random Forest (800 trees)
* **Feature Engineering:** 56 total features per window
* **Window Size:** 128 samples (approx. 6.4s duration)
* **Sampling Rate:** ≈20Hz
* **Overlap:** 75% (32-sample step size)
* **Decision Logic:** Majority voting (Threshold = 0.45)
* **Preprocessing:** Static motion detection enabled

---

### 📝 Detailed Setup Steps

1.  **Environment Setup:** Activate your virtual environment:
    ```bash
    venv\Scripts\activate
    ```
2.  **Navigation:** Move to the production directory:
    ```bash
    cd production
    ```
3.  **Data Preparation:** Prepare training data following the directory structure in: 
    `production/data/RealWorldLive/`
4.  **Model Training:** Generate your model and scaler files:
    ```bash
    python train_and_save_model.py
    ```
5.  **Start Server:** Launch the inference server:
    ```bash
    python app/server.py
    ```
6.  **Mobile App Installation:** Install `gait_app.apk` (built with MIT App Inventor) on your Android device.First open your Laptop connect it with phone hotspot (Campus wifi may    block http post) type "ipconfig" on terminal type your ip on app , now start the server.py first then read steps below.
7.  **Connectivity:** Ensure both your laptop and mobile device are on the **same WiFi network or hotspot**.
8.  **Authentication:** Open the app, enter your laptop's IP address, and start walking. Sensor data is sent automatically for real-time inference.

---

### 🔑 Authentication Logic & Responses
Decisions are returned based on sliding window predictions and majority voting logic. Possible responses include:
* `ACCESS_GRANTED`
* `ACCESS_DENIED` (due to unauthorized gait, static motion, or insufficient data)

---

### 📂 Further Information
* **Project Structure:** Refer to the directory tree layout provided in the main README.
* **Research vs. Production:** See the comparison table in the sections above for phase differences.
* **Research Documentation:** [research/README.md](research/README.md)
* **Deployment Details:** [production/README.md](production/README.md)
* **AI Transparency:** [llm_usage.md](llm_usage.md)

**Final Note:** This system demonstrates that subtle human gait patterns captured through smartphone sensors can be engineered into a functional biometric system, validated through the research phase and proven in real-world deployment.
