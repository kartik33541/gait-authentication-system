# Project Documentation  
## AI-Powered Contactless Gait Authentication System

---

# 1. Problem Overview

The objective of this project was to design a **contactless employee authentication system** using smartphone-based gait analysis.

When an employee approaches the system, their smartphone’s inertial sensor data (accelerometer + gyroscope) is analyzed to automatically grant or deny access.

The project includes:

- Research phase (model development & validation)
- Production phase (real-time authentication system)

---

# 2. Methodology Overview

## 2.1 Dataset Foundation

The project used:

- **UCI HAR Dataset (30 subjects)** as the primary benchmark dataset.

- Real-world datasets collected using:
  - Physics Toolbox Sensor Suite (Research Phase – RealWorld1)
  - Custom MIT App Inventor application (Production – RealWorldLive)

---

## 2.2 Signal Processing & Feature Engineering

Each walking session was segmented into sliding windows.

For each window:

- 3 Accelerometer axes (x, y, z)
- Accelerometer magnitude
- 3 Gyroscope axes (x, y, z)
- Gyroscope magnitude

For each signal, the following 7 features were extracted:

1. Mean  
2. Standard Deviation  
3. RMS  
4. Peak-to-Peak  
5. Dominant Frequency (FFT)  
6. Spectral Energy  
7. Frequency Spread  

Total features per window:

8 signals × 7 features = 56 features


### Why 56 Features?

The goal was to capture both:

- Time-domain characteristics (motion strength, variation)
- Frequency-domain characteristics (rhythm of walking)

FFT-based features were included because gait is inherently rhythmic and periodic.

---

## 2.3 Model Selection

Random Forest was finalized as the production model because:

- Dataset size was relatively small.
- Tree-based models handle non-linear patterns effectively.
- It provided stable accuracy without complex tuning.
- Easier to interpret and deploy compared to boosting-based models.

---

## 2.4 Windowing & Decision Logic

## 2.4 Sliding Windows & Temporal Segmentation

### Window Size

We used a fixed window size of **128 samples**.

---

### Sampling Frequency Differences

#### UCI HAR Dataset
- Sampling frequency: **50 Hz**
- 128 samples → 128 / 50 = **2.56 seconds per window**

This dataset was collected under controlled lab conditions with consistent sensor sampling.

---

#### Production System (RealWorldLive)

- Sampling frequency: **~20 Hz**
- 128 samples → 128 / 20 = **~6.4 seconds per window**

Why 20 Hz?

- The MIT App Inventor mobile app uses a clock timer set to 50 ms.
- 1000 ms / 50 ms ≈ 20 samples per second.
- This was a practical design decision because:
  - Lower sampling reduces mobile CPU usage.
  - Reduces battery consumption.
  - Ensures stable real-time transmission over WiFi.
  - Prevents unnecessary high-frequency noise.

Although 20 Hz is lower than UCI’s 50 Hz, it is sufficient to capture walking rhythm because human gait frequency typically lies below 5 Hz.

---

### Overlap Strategy

We experimented with **50% to 75% overlap**.

Overlap is calculated as:



### Majority Voting

Instead of taking a decision on a single window:

- All window predictions are aggregated.
- Final decision is based on majority class.
- Vote ratio threshold applied.

This was preferred because:

> A single window may produce incorrect classification, but majority voting increases stability and real-world reliability.

---

## 2.5 Threshold Selection

Initial threshold = 0.65  
Problem observed:

- Legitimate users were denied occasionally.

Final threshold = 0.45  

This improved usability while maintaining reasonable security.

Trade-off acknowledged:

- Lower threshold slightly increases risk of false acceptance.

---

# 3. Smartphone Sensors – What Is Being Measured?

## 3.1 Accelerometer

Measures **linear acceleration** along x, y, z axes.

It captures:

- Step force
- Directional movement
- Body motion intensity

## 3.2 Gyroscope

Measures **angular velocity** (rotational motion).

It captures:

- Phone orientation changes
- Subtle hand/body rotation patterns

Together, these sensors capture a unique walking signature.

---

# 4. The 30-Person Limitation & Dataset Expansion Strategy

The original dataset contained only 30 subjects.

For a real production system, this is insufficient.

### Assumptions Made

- Phone is carried in similar orientation.
- User must be walking.
- Same device class.
- Indoor environment.
- Static motion (low energy) → authentication denied.

---

## 4.1 Expansion Strategies Applied

1. Real-world data collection (5 → 8 users).
2. Multiple sessions per user.
3. Increased window overlap (up to 75%).
4. Larger window experiments (3.84s).
5. Majority voting at session level.

---

## 4.2 Future Scalability Strategy (For 100+ Users)

To scale:

- Increase number of sessions per user.
- Collect data at multiple walking speeds.
- Include different device placements.
- Use adaptive thresholding.
- Implement continuous model retraining.
- Explore synthetic augmentation (carefully validated).

---

# 5. Validation Strategy

### UCI HAR Results

- 80:20 → ~89.95%
- 70:30 → ~88.33%
- 60:40 → ~88.12%

Consistency across splits indicated:

> Model is generalizing, not memorizing.

---

### RealWorld1 (Physics Toolbox)

Window-level accuracy dropped (~70%) due to:

- Noise
- Frequency mismatch
- Device placement variation

---

### Production (RealWorldLive)

Session-level authentication achieved using:

- Majority voting
- Static filtering
- Threshold tuning

---

# 6. Why UCI ≈ 90% but Real-World ≈ 70%?

UCI HAR:

- Strict 50Hz sampling
- Controlled environment
- Fixed device placement
- Low noise

Real World:

- Inconsistent frequency
- Higher noise
- Mixed walking patterns
- Variable phone position
- Environmental disturbance

This domain shift explains performance drop.

---

# 7. Critical Technical Realization

Earlier belief:

> Normalizing data always improves accuracy.

Observation:

When real-world data was normalized, window-level accuracy dropped significantly.

Reason:

In identity detection, micro-variations act as unique signatures.  
Excess normalization can remove individual-specific characteristics.

This was a key learning point in the project.
---

# 8. 🔬 Methodology

The system is built as a session-level gait biometric authentication model using Linear Accelerometer and Gyroscope data. Sensor readings (`ax, ay, az, wx, wy, wz`) are collected via smartphone and segmented into sliding windows (128 samples). For each window, 56 handcrafted features are extracted using both time-domain (mean, std, RMS, peak-to-peak) and frequency-domain (FFT-based dominant frequency, spectral energy, frequency spread) characteristics.

A **Random Forest classifier** (800 trees) was finalized due to its robustness on small structured datasets and ability to model non-linear gait patterns. Instead of relying on single-window predictions, **majority voting** across overlapping windows (75% overlap in production) is used to improve decision stability.

**Static detection** (motion energy < 0.15) prevents false authentication when no gait movement is present. A **decision threshold of 0.45** on vote ratio balances usability and security. The production system performs real-time inference by receiving CSV data via HTTP POST from a custom-built mobile application.

---

# 9. Screenshots of Working System

### UCI Validation Accuracy
![UCI Accuracy](results/screenshots/uci_accuracy.png)

### RealWorld Window-Level Accuracy
![RealWorld Accuracy](results/screenshots/realworld_accuracy.png)

### Production Server Output
![Server Output](results/screenshots/server_output.png)

### Mobile Application Interface
![Mobile App](production/mobile_app/img/homepage.png)

![Mobile App](production/mobile_app/img/logic.png)

### Result Comparison Graph
![Results Graph](results/images/uci_vs_realworld_8users.png)

---

# 10. LLM Usage

LLMs were used as a technical assistant for:

- Feature engineering refinement.
- Validation reasoning.
- Debugging path issues.
- Brainstorming model improvement ideas.
- MIT App Inventor logic suggestions.

All results were experimentally validated before documentation.

For detailed explanation:

See:
llm_usage.md


---

# 11. Future Work

If extended further:

- Deploy system with full frontend.
- Multi-user registration interface.
- Cloud-based inference server.
- Larger dataset collection.
- Adaptive learning system.

---

# Final Conclusion

This project demonstrates that human gait patterns captured through smartphone sensors can be transformed into a functional biometric authentication system.

The research phase validated the methodology.  
The production phase demonstrated real-time deployment feasibility.

The project highlights both the strengths and challenges of biometric gait authentication in practical environments.