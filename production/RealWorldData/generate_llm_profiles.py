import os
import json
import time
from google import genai
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR,"biomechanical_profiles.json")

TOTAL_USERS = 5000
BATCH_SIZE = 100

load_dotenv()

API_KEYS=[
os.getenv("GEMINI_API_KEY_1"),
os.getenv("GEMINI_API_KEY_2"),
os.getenv("GEMINI_API_KEY_3"),
os.getenv("GEMINI_API_KEY_4"),
os.getenv("GEMINI_API_KEY_5")
]

key_index=0
client=genai.Client(api_key=API_KEYS[key_index])


def rotate_key():
    global key_index,client
    key_index=(key_index+1)%len(API_KEYS)
    client=genai.Client(api_key=API_KEYS[key_index])
    print("🔄 Switching API key",key_index+1)


def valid(p):

    if not isinstance(p,dict):
        return False

    if "age" not in p: return False
    if "height_cm" not in p: return False
    if "weight_kg" not in p: return False
    if "fitness" not in p: return False

    if not 18<=p["age"]<=85: return False
    if not 150<=p["height_cm"]<=205: return False
    if not 50<=p["weight_kg"]<=120: return False
    if not 1<=p["fitness"]<=10: return False

    return True


PROMPT=f"""
Generate {BATCH_SIZE} realistic human walking biomechanical profiles.

Return JSON array with fields:

age
height_cm
weight_kg
fitness

Constraints:
age: 18-85
height_cm: 150-205
weight_kg: 50-120
fitness: 1-10

Correlations:
taller people → longer stride
heavier people → stronger impacts
older people → slightly lower cadence
higher fitness → smoother gait

Return ONLY JSON array.
"""


# --------------------------------------------------
# Resume logic
# --------------------------------------------------

if os.path.exists(OUTPUT_FILE):

    try:
        with open(OUTPUT_FILE,"r") as f:
            profiles=json.load(f)

        print("📂 Resuming from",len(profiles),"existing profiles")

    except:
        print("⚠️ Could not read existing JSON. Starting fresh.")
        profiles={}

else:
    profiles={}


# --------------------------------------------------
# Main generation loop
# --------------------------------------------------

while len(profiles)<TOTAL_USERS:

    try:

        r=client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT
        )

        text=r.text.strip()

        # Clean markdown formatting if present
        if "```" in text:
            parts=text.split("```")
            text=parts[1] if len(parts)>1 else parts[0]
            if text.startswith("json"):
                text=text[4:]

        batch=json.loads(text)

        if not isinstance(batch,list):
            raise ValueError("LLM output not list")

        added=0

        for p in batch:

            if len(profiles)>=TOTAL_USERS:
                break

            if valid(p):

                pid=f"person{11+len(profiles)}"
                profiles[pid]=p
                added+=1

        print(f"✅ Added {added} profiles | Total: {len(profiles)}")

        # Save progress after each batch
        with open(OUTPUT_FILE,"w") as f:
            json.dump(profiles,f,indent=2)

        time.sleep(4)

    except Exception as e:

        print("⚠️ Error:",e)

        rotate_key()

        time.sleep(10)


print("🎉 Finished generating",len(profiles),"profiles")