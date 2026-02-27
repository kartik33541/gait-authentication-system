# LLM Usage Log

This document details how Large Language Models (LLMs) were used as a technical co-pilot during the development of this biometric system. The focus was on accelerating development, debugging hardware constraints, and scaling data.

---

## 1. Research Phase (UCI HAR Dataset)

### Model Selection & Evaluation
For the initial research phase using the UCI HAR dataset, I used **Random Forest**. The LLM helped brainstorm statistical features (Mean, Std, Energy) and suggested signal-processing features like **FFT-based frequency features** and **spectral energy**. I selectively implemented these to see which actually improved the Random Forest's ability to classify activities.

### The "Normalization" Rejection (Key Learning)
When moving to real-world data, the LLM suggested heavy normalization to reduce noise. **I rejected this.** Experimentation showed that over-normalizing stripped away the micro-variations that make a person's gait unique. This was a major turning point: learning that for biometrics, some "noise" is actually the "signature."

---

## 2. Production System (RealWorldLive Data)

### From Classification to Siamese Networks
For the actual security system, I shifted from Random Forest to a **Deep Metric Learning architecture (Siamese 1D-CNN)**. The LLM was instrumental in architecting the Triplet Loss logic and the embedding-based comparison system.

### Scaling Data with Synthetic Users
Because deep learning models require massive amounts of data to avoid **overfitting**, I used the LLM to write a **Synthetic Gait Generator**. 
- **Purpose:** It took my 10 real-world users and generated 100 "synthetic" identities (Person 11-110).
- **Benefit:** Without these synthetic users, the model would have simply memorized the 10 real people (overfitting) rather than learning the generalized "concept" of a human walk. This allows the system to remain robust even when a completely new person tries to use it.

### On-the-Fly Data Augmentation
The LLM helped implement biological variance logic (Warping, Jitter, and Time-shifting) for real-time data augmentation.
- **Purpose:** It creates infinite variations of a single walk.
- **Benefit:** This forces the model to ignore minor tremors or different phone positions and focus on the core gait signature, significantly improving real-world reliability.

---

## 3. Engineering & Debugging

### Physical Security Logic (Walk Energy Score)
During testing, I found a "spoofing" bug where lifting and dropping the phone could trick the system. I used the LLM to brainstorm a more robust **3D Magnitude-based Walk Energy Score**. This replaced a simple standard deviation check, ensuring that only sustained, rhythmic walking energy (Threshold > 1.0) can trigger a prediction call.

### Mobile App & HTTP Flow
I used the LLM to scaffold the HTTP POST communication between **MIT App Inventor** and the **Flask API**, specifically handling the API Key verification and CSV formatting to ensure the data arrived at the server intact.

---

## 4. Validation Strategy

Every LLM suggestion went through a strict **"Test-First"** protocol:
1. **Logic Check:** Did the suggested physics/math make sense for gait?
2. **Local Validation:** Did it run on my Windows laptop without errors?
3. **Empirical Tuning:** Suggestions for thresholds (like the initial 0.65 similarity) were adjusted based on real-world walk tests.

---

## 5. Production System Usage

### App Development Assistance

The mobile application was built using **MIT App Inventor**.

While I developed the app independently, I used the LLM to:

- Brainstorm how to implement accelerometer + gyroscope capture logic  
- Design the clock-based polling system  
- Structure CSV formatting logic  
- Plan HTTP POST communication flow  

The implementation itself was manually developed and tested.

---

## 6. What LLM Was NOT Used For

- No results were fabricated.
- All accuracy metrics were manually computed and verified.
- No synthetic datasets were generated.
- The mobile application was not auto-generated.
- Documentation was not blindly copied.
- No unnecessary libraries or external models were added without understanding.
- Model improvements were not accepted without experimental validation.

---

## 5. Summary Reflection

The LLM acted as a high-speed research assistant. It allowed me to solve complex threading issues in the cloud and scale my dataset from 10 people to 110 in a matter of minutes. However, the final security decisions—like the 0.70 authentication threshold and the rejection of over-normalization—were based entirely on my own physical testing and performance logs.