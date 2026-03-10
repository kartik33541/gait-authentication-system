import sys
import os

# add LSTM_engine to python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(PROJECT_ROOT, "production", "LSTM_engine")

sys.path.append(ENGINE_PATH)

from dataset_loader import load_scaler

scaler = load_scaler()

print("✅ scaler.pkl created successfully")