import json
import os

DATASET_PATH = "data/processed/dataset.json"

def save_sample(chunk, summary):
    os.makedirs("data/processed", exist_ok=True)

    sample = {
        "input": chunk,
        "output": summary
    }

    if os.path.exists(DATASET_PATH):
        data = json.load(open(DATASET_PATH))
    else:
        data = []

    data.append(sample)

    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved sample. Total: {len(data)}")