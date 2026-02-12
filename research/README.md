# Research Documentation – Gait-Based Person Identification

This document describes the model development, experimental validation, and analytical observations performed during the research phase of the AI-Powered Contactless Employee Authentication System.

This research phase is completely separate from the production deployment pipeline.

---

## 1. Research Objective

The objective of the research phase was to:

- Build a gait-based person identification model.
- Validate it on a benchmark dataset (UCI HAR).
- Evaluate performance on real-world collected data.
- Analyze scalability and domain shift.
- Understand why performance differs between controlled and real-world environments.

---

## 2. Datasets Used

### 2.1 UCI HAR Dataset (Benchmark Validation)

Location:  research/UCI HAR Dataset/

- 30 subjects
- 50 Hz sampling frequency
- 128-sample sliding windows
- Window duration = 2.56 seconds
- Pre-segmented data
- Walking activities filtered (1, 2, 3)

Purpose:
To validate whether gait patterns contain identifiable personal signatures in a controlled environment.

---

### 2.2 RealWorld1 Dataset (Physics Toolbox Data)

Location:  research/RealWorld1/


- Data collected using Physics Toolbox Sensor Suite.
- Per-person structure:
PersonX/
train/
test/

- Linear acceleration + gyroscope signals.
- Separate train/test recordings.
- Up to 8 users evaluated.

Purpose:
To evaluate performance on unconstrained real-world data.

⚠️ This dataset is different from production data (which is collected using a custom mobile app at 20 Hz).

---

## 3. Feature Engineering

For each sliding window:

Signals used:
- Accelerometer (ax, ay, az)
- Gyroscope (wx, wy, wz)
- Accelerometer magnitude
- Gyroscope magnitude

For each signal (8 total), the following features were extracted:

Time-domain:
- Mean
- Standard deviation
- RMS
- Peak-to-peak

Frequency-domain (FFT):
- Dominant frequency
- Spectral energy
- Frequency spread

Total features per window:

8 signals × 7 features = 56 features


---

## 4. Model Configuration

Classifier:

RandomForestClassifier


Parameters:
- n_estimators = 1000
- max_features = "sqrt"
- random_state = 42
- n_jobs = -1

Preprocessing:
- StandardScaler applied to features

---

## 5. UCI HAR Validation Results

Model was validated using multiple train-validation splits to ensure robustness and avoid overfitting.

| Split Ratio | Validation Accuracy |
|------------|--------------------|
| 80:20 | 0.8995 |
| 70:30 | 0.8833 |
| 60:40 | 0.8812 |

Observation:

- Accuracy remains stable as validation size increases.
- No dramatic collapse in performance.
- Indicates the model is **not memorizing data**.
- Demonstrates good generalization on controlled benchmark data.

Conclusion:
Gait patterns are distinguishable in controlled lab conditions.

---

## 6. RealWorld1 Baseline Performance

Using:
- Window size = 128
- 50% overlap
- Random Forest
- Up to 8 users

Observed:

- Window-level accuracy dropped significantly (~60–70% range).
- File-level (majority voting) accuracy was higher.

---

## 7. Why Accuracy Dropped in Real-World Data

Performance degradation occurred due to multiple factors:

### 7.1 Domain Shift
- UCI HAR collected in controlled lab.
- RealWorld1 collected in unconstrained environment.

### 7.2 Sensor Placement Variability
- Waist-mounted (UCI) vs handheld variations (RealWorld1).

### 7.3 Sampling Frequency Differences
- UCI: 50 Hz
- RealWorld1: device-dependent frequency

### 7.4 Smaller Dataset Size
- 30 subjects (UCI) vs ≤8 subjects (RealWorld1).
- Fewer training windows per person.

### 7.5 Natural Gait Variability
- Real walking sessions contain inconsistent speed and posture.

These collectively reduced separability between users.

---

## 8. Improvement Experiments

Several assumptions were tested to improve real-world performance:

### 8.1 Larger Window Size
Increasing window duration to capture more gait cycles.

Rationale:
Longer windows contain more periodic walking information.

### 8.2 Higher Overlap (75%)
Increasing window overlap to:
- Generate more training samples.
- Improve temporal continuity.

### 8.3 Scalability Analysis (8 vs 5 Users)

Reducing enrolled users improved separability and reduced confusion.

Observation:
Biometric systems scale non-linearly with number of enrolled identities.

---

## 9. Key Insights

1. Gait identification works strongly in controlled conditions.
2. Real-world deployment introduces domain shift challenges.
3. Window-level accuracy alone is insufficient for authentication.
4. Majority voting significantly stabilizes predictions.
5. Reducing enrolled users improves reliability.
6. Longer windows increase stability but reduce response speed.

---

## 10. Research Conclusion

- Random Forest was chosen due to stability and interpretability.
- 56 engineered features were sufficient for identity separation.
- UCI HAR validation confirms model capability.
- Real-world data requires aggregation strategies.
- Scalability and environmental robustness are key challenges.

The research phase validated the feasibility of gait-based authentication while revealing important deployment constraints.
