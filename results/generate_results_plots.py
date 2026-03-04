import os
import matplotlib.pyplot as plt

# Ensure images directory exists
OUTPUT_DIR = "results/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1️⃣ UCI HAR Split Comparison
# -------------------------------

uci_splits = ["80:20", "70:30", "60:40"]
uci_accuracy = [0.8995, 0.8834, 0.8813]

plt.figure()
plt.plot(uci_splits, uci_accuracy, marker='o')
plt.title("UCI HAR - Validation Split Comparison")
plt.xlabel("Train : Validation Split")
plt.ylabel("Accuracy")
plt.ylim(0.85, 0.92)
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/uci_split_comparison.png")
plt.close()


# ---------------------------------------------
# 2️⃣ UCI vs RealWorld (8 Users Comparison)
# ---------------------------------------------

datasets = ["UCI HAR", "RealWorld (8 Users) - Window", "RealWorld (8 Users) - File"]
accuracy_values = [0.8995, 0.7099, 1.00]

plt.figure()
plt.bar(datasets, accuracy_values)
plt.title("UCI vs RealWorld (8 Users)")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)
plt.savefig(f"{OUTPUT_DIR}/uci_vs_realworld_8users.png")
plt.close()


# -------------------------------------------------
# 3️⃣ 8 Users vs 5 Users (Window-Level Accuracy)
# -------------------------------------------------

user_counts = ["8 Users", "5 Users"]
window_accuracy = [0.7099, 0.74]  # Adjust if needed

plt.figure()
plt.bar(user_counts, window_accuracy)
plt.title("Window-Level Accuracy: 8 vs 5 Users")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig(f"{OUTPUT_DIR}/8_vs_5_users_window.png")
plt.close()


# -------------------------------------------------
# 4️⃣ Window-Level vs File-Level (8 Users)
# -------------------------------------------------

levels = ["Window-Level", "File-Level (Majority Voting)"]
level_accuracy = [0.7099, 1.00]

plt.figure()
plt.bar(levels, level_accuracy)
plt.title("Window vs File-Level Accuracy (8 Users)")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.savefig(f"{OUTPUT_DIR}/window_vs_file.png")
plt.close()


# -------------------------------------------------
# 5️⃣ Window Size Impact
# -------------------------------------------------

window_types = ["2.56s (UCI)", "3.84s (RealWorld)", "6.4s (Production)"]
window_accuracy = [0.8995, 0.7059, 0.73]

plt.figure()
plt.plot(window_types, window_accuracy, marker='o')
plt.title("Impact of Window Size on Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0.65, 0.95)
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/window_size_effect.png")
plt.close()


print("All result plots generated successfully in results/images/")
