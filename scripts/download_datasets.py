import os
import requests

# ----------------------------
# Folder to save datasets
# ----------------------------
SAVE_FOLDER = "data/raw"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# Dataset URLs
# ----------------------------
DATASETS = {
    "matches.csv": "https://footystats.org/c-dl.php?type=matches&comp=16494",
    "teams.csv": "https://footystats.org/c-dl.php?type=teams&comp=16494",
    "teams2.csv": "https://footystats.org/c-dl.php?type=teams2&comp=16494",
    "players.csv": "https://footystats.org/c-dl.php?type=players&comp=16494",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

print("=" * 60)
print("Downloading FootyStats datasets")
print("=" * 60)

success = 0

for filename, url in DATASETS.items():

    print(f"\nDownloading {filename}...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code == 200:

            filepath = os.path.join(SAVE_FOLDER, filename)

            with open(filepath, "wb") as file:
                file.write(response.content)

            print("✓ Downloaded successfully")
            success += 1

        else:
            print(f"✗ HTTP Error {response.status_code}")

    except Exception as e:
        print(f"✗ {e}")

print("\n" + "=" * 60)
print(f"Finished: {success}/{len(DATASETS)} datasets downloaded.")
print(f"Saved in: {SAVE_FOLDER}")
print("=" * 60)