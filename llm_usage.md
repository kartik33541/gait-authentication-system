# LLM Usage Log

This document briefly describes how Large Language Models (LLMs) were used during both the research and production phases of the AI-Powered Contactless Employee Security System.

The LLM was used as a technical assistant and reasoning partner — not as a black-box solution generator.

---

## 1. Research Phase Usage

### Feature Engineering

I initially had the intuition that combining **linear accelerometer and gyroscope data** would better distinguish individuals based on motion patterns.

I already knew fundamental statistical features such as:
- Mean  
- Standard deviation  
- Energy  

I used the LLM to explore additional signal-processing features. It suggested:
- FFT-based frequency features  
- Dominant frequency  
- Spectral energy  
- Frequency spread  

After understanding the reasoning behind these features, I selectively implemented them rather than blindly applying suggestions.

---

### Train–Validation Split Strategy

From the beginning, I was concerned about **overfitting**, since:

- The UCI HAR dataset is relatively small.
- Real-world collected data was also limited.

Initially, I evaluated performance using an 80:20 split.  
Based on my own reasoning, I decided to test:

- 70:30 split  
- 60:40 split  

to ensure the model was not memorizing the data.

The LLM assisted in reasoning about generalization and consistency across splits, but all results were manually re-run and verified before documentation.

The consistent accuracy across splits confirmed that the model was learning patterns rather than memorizing samples.

---

### Where LLM suggestion was Rejected -- Real-World Accuracy Drop Analysis

When shifting from the UCI HAR dataset to real-world data (Physics Toolbox recordings), window-level accuracy dropped.

The LLM suggested normalization to reduce noise. After experimentation, I observed that normalization reduced identity-specific characteristics. In identity detection, certain micro-variations can act as unique behavioral signatures.As a result accuracy dopped so that suggestion was the one of the big learning and rejected suggestion by LLM in this project.

This was experimentally validated before being documented that more normalisation on dataset will reduce the accuracy as it will reduce unique behavioural changes.

Additionally, I used the LLM to brainstorm potential accuracy improvements, including:

- Increasing overlap percentage  
- Increasing window duration  
- Adjusting sampling frequency assumptions  
- Window balancing strategies  

Not all suggestions improved performance, and decisions were based strictly on experimental results.

---

## 2. Real-World Debugging & System Logic

### Static Detection Logic

A static detection rule was implemented to deny access if motion energy was too low.

This logic aligns with the system design:

- Gait authentication requires walking.
- No walking → no gait pattern → access denied.

The LLM assisted in validating this reasoning, but the final logic was implemented and tested manually.

---

### Path and Model Loading Debugging

During restructuring of research and production folders, relative path issues occurred.

The LLM helped identify:
- Working directory confusion
- Use of `os.getcwd()` for debugging

Final resolution and restructuring were handled manually.

---

### Window Balancing and Larger Window Experiments

I alongside with suggestions from LLm's implemented experiments with:
- Larger window sizes
- Higher overlap percentages
- Balanced window strategies

These were validated experimentally by me on notebooks before being retained or discarded.

---

## 3. Production System Usage

### App Development Assistance

The mobile application was built using **MIT App Inventor**.

While I developed the app independently, I used the LLM to:

- Brainstorm how to implement accelerometer + gyroscope capture logic  
- Design the clock-based polling system  
- Structure CSV formatting logic  
- Plan HTTP POST communication flow  

The implementation itself was manually developed and tested.

---

### HTTP Server & Deployment

The HTTP server structure was initially scaffolded with LLM assistance.

However:

- Data flow (sensor → CSV → HTTP POST → server → prediction) was architected manually.
- Network debugging (campus WiFi restriction issue) was resolved independently.
- Production model integration was implemented and validated manually.

---

### Threshold Tuning

The authentication threshold was initially suggested at 0.65 by LLM when i initially suggeted the idea of threshold based access denial.

During real-world testing, legitimate users were denied due to strict voting criteria. After observing vote ratios experimentally, I adjusted the threshold to 0.45 to balance security and usability.

This final value was chosen based on empirical validation.

---

## 4. What LLM Was NOT Used For

- No results were fabricated.
- All accuracy metrics were manually computed and verified.
- No synthetic datasets were generated.
- The mobile application was not auto-generated.
- Documentation was not blindly copied.
- No unnecessary libraries or external models were added without understanding.
- Model improvements were not accepted without experimental validation.

---

## 5. Key Learning Reflection

This project reinforced several important engineering lessons:

- Model accuracy and real-world system reliability are not the same.
- Real-time HTTP data transfer requires careful debugging.
- Normalization does not always improve biometric identification, it may decrease it also.
- Brainstorming ideas with LLM is useful — but validation must be experimental.

Most importantly, this project demonstrated that subtle human gait patterns can be engineered into a functioning biometric authentication system.

---

LLMs were used as a technical co-pilot — not as a primary tool for all setup and reasoning.
