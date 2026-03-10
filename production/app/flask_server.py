# from flask import Flask, request, jsonify
# import os
# import time
# import logging
# import pandas as pd
# import sys
# import uuid
# import json
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras.backend as K
# from sklearn.preprocessing import StandardScaler
# from gait_analyzer import diagnose_gait_pattern
# from dotenv import load_dotenv

# # Load environment variables from production/.env
# load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# # NEW GOOGLE GENAI SDK
# from google import genai

# # Silence background library noise (TensorFlow, HTTPX, Werkzeug)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# logging.getLogger('werkzeug').setLevel(logging.ERROR)
# logging.getLogger("httpx").setLevel(logging.WARNING)

# # Ensure the server can see files in the parent 'production' folder
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Import directly from the LSTM engine folder
# from LSTM_engine.dataset_loader import extract_windows
# from LSTM_engine.build_encoder import get_encoder

# app = Flask(__name__)

# # ================= CONFIG =================
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "received_gait.csv"))

# # Setup Paths for the AI Brain
# LSTM_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "LSTM_engine"))
# WEIGHTS_PATH = os.path.join(LSTM_DIR, "siamese_lstm.weights.h5")
# VAULT_PATH = os.path.join(LSTM_DIR, "vault.json")

# MAX_REQUEST_SIZE = 2 * 1024 * 1024
# app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE

# # --- Security Tolerances ---
# EXPECTED_DURATION_MS = 15000
# DURATION_TOLERANCE = 3000  # Network latency buffer
# MAX_AGE_MS = 300000        # 5 minute replay protection

# API_KEY = "GAIT_SECURE_2026"
# SECURITY_THRESHOLD = 0.75  # Set to 0.75 based on blind testing

# # ================= LLM EXPLAINER (Connects to gait_analyzer.py) =================
# def generate_llm_explanation(scenario, **kwargs):
#     """
#     Point 2: LLM as an Authentication Explainer.
#     Connects to gait_analyzer.py to provide diagnostic-aware explanations.
#     """
#     try:
#         # Initialize the modern Client
#         client = genai.Client(api_key=GEMINI_API_KEY)
#         sys_prompt = "You are a professional Biometric Security Specialist. Provide a concise, plain English explanation (1-2 sentences) of the movement analysis."
        
#         if scenario == "TEST":
#             prompt = f"{sys_prompt} Smartphone ping: '{kwargs.get('text')}'. Acknowledge the biometric server is online and ready for sensor data stream."
            
#         elif scenario == "STATIC":
#             prompt = f"{sys_prompt} Gait energy is {kwargs.get('energy'):.2f} (Threshold 1.0). Explain the device is static and they must walk normally to authenticate."
            
#         elif scenario == "AUTH":
#             status = kwargs.get('status')
#             score = kwargs.get('score')
#             energy = kwargs.get('energy')
#             variance = kwargs.get('variance')
            
#             # --- POINT 3: CALL THE DIAGNOSTIC ENGINE ---
#             # We import this inside or at the top of the file
#             from gait_analyzer import diagnose_gait_pattern
#             diagnosis = diagnose_gait_pattern(GEMINI_API_KEY, energy, variance, score, SECURITY_THRESHOLD)
            
#             if status == "ACCESS GRANTED":
#                 prompt = f"""{sys_prompt} 
#                 Analysis: {status}. Match Score: {score:.4f}, Energy: {energy:.2f}, Variance: {variance:.4f}. 
#                 Diagnostic Insight: '{diagnosis}'.
#                 Briefly confirm the physical gait matches the registered owner perfectly based on this insight."""
#             else:
#                 # Specialized prompt for Denials (Using Diagnostic Insight)
#                 prompt = f"""{sys_prompt} 
#                 Analysis: {status} (IMPOSTER). 
#                 METRICS: Match Score {score:.4f} (Threshold {SECURITY_THRESHOLD}), Walk Energy {energy:.2f}, Step Variance {variance:.4f}.
#                 DIAGNOSTIC INSIGHT: '{diagnosis}'.
                
#                 TASK: Generate a plain English explanation of this failure. Incorporate the Diagnostic Insight to explain exactly what the anomaly in the gait pattern implies (e.g. why they don't match the owner physically)."""
            
#         response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
#         return response.text.replace('\n', ' ').strip()
        
#     except Exception as e:
#         logging.error(f"LLM Explainer Error: {e}")
#         if scenario == "TEST": return "Server connection established successfully."
#         if scenario == "STATIC": return "Static device detected. Please walk to authenticate."
#         return f"Authentication {kwargs.get('status')}. (Detailed AI reasoning offline)."

# # ================= AI BRAIN INITIALIZATION =================
# def l2_normalize(vector):
#     norm = np.linalg.norm(vector)
#     return vector / norm if norm > 0 else vector

# def euclidean_distance(vects):
#     x, y = vects
#     sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))

# def get_loaded_encoder():
#     input_shape = (256, 6)
#     encoder = get_encoder(input_shape=input_shape, embedding_dim=256)
#     input_a = tf.keras.layers.Input(shape=input_shape)
#     input_b = tf.keras.layers.Input(shape=input_shape)
#     out_a = encoder(input_a)
#     out_b = encoder(input_b)
#     dist = tf.keras.layers.Lambda(euclidean_distance)([out_a, out_b])
#     full_model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=dist)
#     full_model.load_weights(WEIGHTS_PATH)
#     return encoder

# try:
#     GLOBAL_ENCODER = get_loaded_encoder()
#     print("✅ 1:N Live API Brain is online and ready.")
# except Exception:
#     GLOBAL_ENCODER = None
#     print("⚠️ Warning: Could not load model weights.")

# print("🚀 Secure Biometric Flask server running on port 8000...\n")

# # ================= ROUTE =================
# @app.route("/predict", methods=["POST"])
# def predict():
#     start_time = time.time()
#     # START OF REQUEST: 4 BLANK LINES AND SEPARATOR
#     print("\n" * 4 + "—" * 60)
#     print("📥 New request received from MIT App")
    
#     temp_csv_path = None 

#     try:
#         # 1. API KEY VERIFICATION
#         client_key = request.headers.get("X-API-KEY")
#         if client_key != API_KEY:
#             print("🚫 Unauthorized Request (Invalid API Key)")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "UNAUTHORIZED"}), 401

#         # 2. READ BODY
#         csv_text = request.get_data(as_text=True)
#         if not csv_text or not csv_text.strip():
#             print("❌ Empty Payload")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "EMPTY"}), 400

#         # 3. OVERWRITE CLASSIC FILE
#         with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
#             f.write(csv_text)

#         # 4. SERVER TEST HANDLER
#         if "," not in csv_text:
#             print(f"🧪 Test Signal: {csv_text}")
#             llm_message = generate_llm_explanation("TEST", text=csv_text)
#             print(f"✨ {llm_message}")
#             return jsonify({"result": "SERVER_TEST_RECEIVED", "message": llm_message}), 200

#          # 5. CREATE THREAD-SAFE TEMPORARY FILE FOR ML
#         unique_id = uuid.uuid4().hex
#         temp_csv_path = os.path.abspath(os.path.join(BASE_DIR, "..", f"received_gait_{unique_id}.csv"))

#         with open(temp_csv_path, "w", encoding="utf-8", newline="") as f:
#             f.write(csv_text)
#             f.flush()
#             os.fsync(f.fileno())
            
#         logging.info(f"📄 Thread-safe CSV saved | ID: {unique_id}")


#         # 6. LOAD AND VALIDATE STRUCTURE
#         df = pd.read_csv(temp_csv_path)
#         if "timestamp" not in df.columns:
#             print("❌ Bad Data Format (Missing Timestamp)")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "BAD_FORMAT"}), 200

#         # 7. MANIPULATION PROTECTION (Duration Check)
#         duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
#         print(f"⏱ Recording duration: {duration} ms")

#         if not (EXPECTED_DURATION_MS - DURATION_TOLERANCE <= duration <= EXPECTED_DURATION_MS + DURATION_TOLERANCE):
#             print("🚫 Stitched/Manipulated Recording Detected")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "INVALID_DURATION"}), 200

#         # 8. TRUE REPLAY PROTECTION (Time Freshness Check)
#         last_timestamp = df["timestamp"].iloc[-1]
#         if last_timestamp > 1500000000000:
#             current_time_ms = int(time.time() * 1000)
#             age_ms = current_time_ms - last_timestamp
#             if age_ms > MAX_AGE_MS:
#                 print(f"🚫 REPLAY ATTACK DETECTED (Data is {age_ms/1000:.1f}s old)")
#                 return jsonify({"result": "ACCESS_DENIED", "reason": "STALE_DATA_REPLAY"}), 200
#         else:
#             print("ℹ️ Using Relative Timestamps")

#         # 9. ROBUST STATIC DETECT
#         df['acc_mag'] = (df['ax']**2 + df['ay']**2 + df['az']**2)**0.5
#         walk_energy = df['acc_mag'].std()
#         print(f"📊 Walk Energy Score: {walk_energy:.4f}")

#         if walk_energy < 1.0: 
#             print("🧍 Static Walk Detected")
#             llm_message = generate_llm_explanation("STATIC", energy=walk_energy)
#             print(f"✨ {llm_message}")
#             return jsonify({"result": "ACCESS_DENIED", "score": 0.0, "message": llm_message}), 200

#         # 10. 1:N SIAMESE IDENTIFICATION
#         if GLOBAL_ENCODER is None:
#             return jsonify({"result": "ERROR", "reason": "MODEL_OFFLINE"}), 500

#         with open(VAULT_PATH, "r") as f: vault = json.load(f)

#         df_ml = pd.read_csv(temp_csv_path, usecols=['ax', 'ay', 'az', 'wx', 'wy', 'wz'], dtype=np.float32)
#         windows = extract_windows(df_ml, StandardScaler())
        
#         if len(windows) == 0:
#             print("🚨 Data Too Short")
#             return jsonify({"result": "ACCESS_DENIED", "reason": "DATA_TOO_SHORT"}), 200

#         live_windows = np.array(windows, dtype=np.float32)
#         embeddings = GLOBAL_ENCODER.predict(live_windows, verbose=0)
#         step_variance = float(np.mean(np.var(embeddings, axis=0)))
#         live_signature = l2_normalize(np.mean(embeddings, axis=0))

#         best_match_id, highest_score = "IMPOSTER", -1.0
#         for person_id, template_list in vault.items():
#             score = float(np.dot(live_signature, np.array(template_list)))
#             if score > highest_score:
#                 highest_score, best_match_id = score, person_id

#         if highest_score >= SECURITY_THRESHOLD:
#             result, status = best_match_id, "ACCESS GRANTED"
#         else:
#             result, status = "IMPOSTER", "ACCESS DENIED"

#         print(f"🧠 {status} {result} (Match Score: {highest_score:.4f})")
        
#         # LLM REASONING
#         llm_message = generate_llm_explanation("AUTH", status=status, user=result, score=highest_score, energy=walk_energy, variance=step_variance)
#         print(f"✨ {llm_message}")

#         return jsonify({
#             "result": result,
#             "score": round(highest_score, 4),
#             "message": llm_message,
#             "response_time_sec": round(time.time() - start_time, 3)
#         }), 200

#     except Exception as e:
#         print(f"🔥 Server Error: {e}")
#         return jsonify({"result": "ERROR"}), 500
        
#     finally:
#         if temp_csv_path and os.path.exists(temp_csv_path):
#             try: os.remove(temp_csv_path)
#             except: pass

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, threaded=True)





from flask import Flask, request, jsonify
import os
import time
import logging
import pandas as pd
import sys
import uuid
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib
from dotenv import load_dotenv


# ================= FIX IMPORT PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Allow Python to see production folder
sys.path.append(PROJECT_ROOT)

from google import genai

from LSTM_engine.dataset_loader import extract_windows
from LSTM_engine.build_encoder import get_encoder
from gait_analyzer import diagnose_gait_pattern


# ================= ENV =================
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

LSTM_DIR = os.path.join(PROJECT_ROOT, "LSTM_engine")

WEIGHTS_PATH = os.path.join(LSTM_DIR, "siamese_lstm.weights.h5")
VAULT_PATH = os.path.join(LSTM_DIR, "vault.json")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")

CSV_PATH = os.path.join(PROJECT_ROOT, "received_gait.csv")


# ================= SECURITY =================
API_KEY = "GAIT_SECURE_2026"
SECURITY_THRESHOLD = 0.75

EXPECTED_DURATION_MS = 15000
DURATION_TOLERANCE = 3000
MAX_AGE_MS = 300000


# ================= APP =================
app = Flask(__name__)


# ================= UTIL =================
def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


# ================= LOAD AI BRAIN =================
def get_loaded_encoder():

    input_shape = (256, 6)

    encoder = get_encoder(input_shape=input_shape, embedding_dim=256)

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    out_a = encoder(input_a)
    out_b = encoder(input_b)

    dist = tf.keras.layers.Lambda(euclidean_distance)([out_a, out_b])

    full_model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=dist)

    full_model.load_weights(WEIGHTS_PATH)

    return encoder


print("\n🧠 Loading Biometric Engine...")

GLOBAL_ENCODER = get_loaded_encoder()
SCALER = joblib.load(SCALER_PATH)

print("✅ Encoder loaded")
print("✅ Scaler loaded")
print("🚀 Secure Biometric Server Running\n")


# ================= ROUTE =================
@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    print("\n" + "="*60)
    print("📥 AUTHENTICATION REQUEST RECEIVED")
    print("="*60)

    temp_csv_path = None

    try:

        # ---------- API KEY ----------
        client_key = request.headers.get("X-API-KEY")

        if client_key != API_KEY:
            print("🚫 Unauthorized request")
            return jsonify({"result":"ACCESS_DENIED","reason":"UNAUTHORIZED"}),401


        # ---------- READ BODY ----------
        csv_text = request.get_data(as_text=True)

        if not csv_text:
            return jsonify({"result":"ACCESS_DENIED","reason":"EMPTY"}),400


        # ---------- SERVER TEST HANDLER ----------
        if "," not in csv_text:

            # Save debug copy so user can see server ping
            with open(CSV_PATH, "w", encoding="utf-8") as f:
                f.write(csv_text)

            print("\n🧪 SERVER CONNECTIVITY TEST")
            print("-"*40)
            print(f"Signal Received : {csv_text}")

            message = "Biometric server online and ready for gait authentication."

            print("Status          : SERVER ONLINE")
            print("Debug CSV Updated : received_gait.csv")
            print("="*60)

            return jsonify({
                "result":"SERVER_TEST_RECEIVED",
                "message":message
            }),200

        # ---------- THREAD SAFE CSV ----------
        unique_id = uuid.uuid4().hex

        temp_csv_path = os.path.join(PROJECT_ROOT, f"received_gait_{unique_id}.csv")

        with open(temp_csv_path,"w",encoding="utf-8") as f:
            f.write(csv_text)

        # Overwrite monitoring CSV for debugging
        with open(CSV_PATH,"w",encoding="utf-8") as f:
            f.write(csv_text)


        df = pd.read_csv(temp_csv_path)


        # ---------- Duration check ----------
        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

        print(f"\n⏱ Recording Duration : {duration} ms")

        if not (EXPECTED_DURATION_MS-DURATION_TOLERANCE <= duration <= EXPECTED_DURATION_MS+DURATION_TOLERANCE):

            print("🚫 Invalid recording duration")

            return jsonify({"result":"ACCESS_DENIED","reason":"INVALID_DURATION"}),200


        # ---------- Replay protection ----------
        last_timestamp = df["timestamp"].iloc[-1]

        if last_timestamp > 1500000000000:

            age = int(time.time()*1000) - last_timestamp

            if age > MAX_AGE_MS:

                print("🚫 Replay attack detected")

                return jsonify({"result":"ACCESS_DENIED","reason":"REPLAY_ATTACK"}),200


        # ---------- Static detection ----------
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

        walk_energy = float(df['acc_mag'].std())

        print(f"📊 Walk Energy        : {walk_energy:.4f}")

        if walk_energy < 1.0:

            print("🧍 Static device detected")

            explanation = diagnose_gait_pattern(
                os.getenv("GEMINI_API_KEY"),
                walk_energy,
                0,
                0,
                SECURITY_THRESHOLD,
                "No movement"
            )

            return jsonify({
                "result":"ACCESS_DENIED",
                "score":0,
                "message":explanation
            }),200


        # ---------- WINDOW EXTRACTION ----------
        df_ml = df[['ax','ay','az','wx','wy','wz']].astype(np.float32)

        windows = extract_windows(df_ml, SCALER)

        if len(windows)==0:

            return jsonify({"result":"ACCESS_DENIED","reason":"DATA_TOO_SHORT"}),200


        windows_arr = np.array(windows,dtype=np.float32)


        # ---------- EMBEDDINGS ----------
        embeddings = GLOBAL_ENCODER.predict(windows_arr,verbose=0)

        embeddings = np.array([l2_normalize(e) for e in embeddings])

        step_variance = float(np.mean(np.var(embeddings,axis=0)))


        # ---------- VAULT ----------
        with open(VAULT_PATH) as f:
            vault=json.load(f)


        vote_counter={u:0 for u in vault}
        score_accumulator={u:[] for u in vault}

        confidence_dist=[]


        for emb in embeddings:

            similarities={}

            for user,templates in vault.items():

                scores=[]

                for t in templates:

                    t_vec=l2_normalize(np.array(t))
                    scores.append(np.dot(emb,t_vec))

                score=max(scores)

                similarities[user]=score


            best=max(similarities,key=similarities.get)

            vote_counter[best]+=1

            confidence_dist.append(similarities[best])

            for u,s in similarities.items():
                score_accumulator[u].append(s)


        avg_scores={u:np.mean(score_accumulator[u]) for u in score_accumulator}

        sorted_scores=sorted(avg_scores.items(),key=lambda x:x[1],reverse=True)

        best_user,best_score=sorted_scores[0]


        # ---------- DECISION ----------
        if best_score>=SECURITY_THRESHOLD:

            status="ACCESS GRANTED"
            result=best_user

        else:

            status="ACCESS DENIED"
            result="IMPOSTER"


        # ---------- LOGGING ----------
        print("\n🧠 GAIT AUTHENTICATION RESULT")
        print("-"*40)
        print(f"Best Match          : {best_user}")
        print(f"Similarity Score    : {best_score:.4f}")
        print(f"Threshold           : {SECURITY_THRESHOLD}")
        print(f"Decision            : {status}")


        print("\n🗳 Window Votes")

        for u,v in sorted(vote_counter.items(),key=lambda x:x[1],reverse=True):
            print(f"{u:10s} → {v}")


        # ---------- LLM ----------
        explanation = diagnose_gait_pattern(
            os.getenv("GEMINI_API_KEY"),
            walk_energy,
            step_variance,
            best_score,
            SECURITY_THRESHOLD,
            confidence_distribution=str(confidence_dist[:10])
        )


        print("\n🤖 AI Explanation")
        print(explanation)

        print("="*60)


        return jsonify({
            "result":result,
            "score":round(best_score,4),
            "message":explanation,
            "response_time_sec":round(time.time()-start_time,3)
        }),200


    except Exception as e:

        print(f"🔥 Server Error: {e}")

        return jsonify({"result":"ERROR"}),500


    finally:

        if temp_csv_path and os.path.exists(temp_csv_path):

            try:
                os.remove(temp_csv_path)
            except:
                pass


if __name__=="__main__":

    app.run(host="0.0.0.0",port=8000,threaded=True)