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
## Step 2 – Create Virtual Environment (Recommended)
```bash
yaml -m venv venv
yaml\Scripts\activate   Windows   results and model was trained on conda  environment
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
### OPTION B — Run Real-Time Authentication System (Production Mode)
If you want to experience live gait authentication, this section allows you to:
- Collect your own walking data (`production/mobile_app/gait_app.apk`) which will automatically send data via HTTP POST after starting `server.py`.
e.g., start server.py first.
'the app made from MIT App Inventor'
to collect data.
to train your model,
to perform real-time authentication via mobile app.
definition of the production system overview:
mobile app → CSV Data → HTTP POST → Server → Model → Access Decision.
default configuration for the model includes:
rForest with 800 trees,
total features per window =56,
wsize=128 samples,
sampling rate ≈20Hz,
duration≈6.4s,
overlap=75%,
voting threshold=0.45,
st static motion detection enabled.
detailed setup steps are as follows:
activate virtual environment (`venv\Scripts\activate`),
navigate to production folder (`cd production`),
pick or prepare training data following structure in `production/data/RealWorldLive/`,
generate model files by running `python train_and_save_model.py`, start server with `python app/server.py`, install mobile app APK (`gait_app.apk`) on Android device, ensure laptop and mobile are connected to same WiFi network or hotspot, open the app, enter IP address, start walking; sensor data is sent automatically for inference; access decision returned based on majority voting with threshold of 0.45.
defining possible access responses such as `ACCESS_GRANTED`, `ACCESS_DENIED`, etc., based on sliding window predictions and majority voting logic.
detailed project structure provided in the directory tree layout above.
differences between research phases vs production deployment summarized in tabular form above.
further documentation links provided for detailed research experiments (`research/README.md`) and deployment details (`production/README.md`).
further information about AI assistance transparency available in `llm_usage.md`. 
the final note emphasizes that subtle human gait patterns captured through sensors can be engineered into a functional biometric system validated through research phase and demonstrated in real-world deployment.