import json
import math

INPUT_FILE = "biomechanical_profiles.json"
OUTPUT_FILE = "biomechanical_profiles_clean.json"

allowed_keys = {"age", "height_cm", "weight_kg", "fitness"}

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

clean = {}
seen = set()

stats = {
    "extra_fields_removed":0,
    "duplicates_removed":0,
    "bmi_removed":0,
    "range_removed":0,
    "correlation_removed":0
}

for person, profile in data.items():

    # remove hallucinated fields
    p = {k:profile[k] for k in allowed_keys if k in profile}

    if len(p) != 4:
        continue

    age = p["age"]
    h = p["height_cm"]
    w = p["weight_kg"]
    fitness = p["fitness"]

    # range check
    if not (18 <= age <= 85 and
            150 <= h <= 205 and
            45 <= w <= 120 and
            1 <= fitness <= 10):
        stats["range_removed"] += 1
        continue

    # BMI filter
    bmi = w / ((h/100)**2)
    if not (17 <= bmi <= 35):
        stats["bmi_removed"] += 1
        continue

    # correlation sanity
    if age > 75 and fitness > 8:
        stats["correlation_removed"] += 1
        continue

    if age < 20 and fitness < 3:
        stats["correlation_removed"] += 1
        continue

    key = (age,h,w,fitness)

    if key in seen:
        stats["duplicates_removed"] += 1
        continue

    seen.add(key)

    clean[person] = p

# reindex persons sequentially
final_profiles = {}

for i,(k,v) in enumerate(clean.items()):
    final_profiles[f"person{i+11}"] = v

with open(OUTPUT_FILE,"w") as f:
    json.dump(final_profiles,f,indent=2)

print("Original profiles:",len(data))
print("Final profiles:",len(final_profiles))
print("Removed duplicates:",stats["duplicates_removed"])
print("Removed BMI outliers:",stats["bmi_removed"])
print("Removed range errors:",stats["range_removed"])
print("Removed correlation errors:",stats["correlation_removed"])