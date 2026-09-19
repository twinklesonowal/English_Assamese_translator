from datasets import load_dataset
import json, os

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading English-Assamese parallel data...")
print("This may take a few minutes (downloads ~100MB).\n")

# Samanantar is AI4Bharat's dataset — 'as' means Assamese
dataset = load_dataset("ai4bharat/samanantar", "as", split="train", trust_remote_code=True)

print(f"Loaded {len(dataset)} sentence pairs.")
print("Sample entry:", dataset[0])

# Split into train / val / test
split = dataset.train_test_split(test_size=0.1, seed=42)
train_val = split["train"].train_test_split(test_size=0.1, seed=42)

train_data = train_val["train"]
val_data   = train_val["test"]
test_data  = split["test"]

print(f"\nSplit sizes:")
print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")
print(f"  Test:  {len(test_data)}")

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} pairs to {path}")

save_jsonl(train_data, os.path.join(SAVE_DIR, "train.jsonl"))
save_jsonl(val_data,   os.path.join(SAVE_DIR, "val.jsonl"))
save_jsonl(test_data,  os.path.join(SAVE_DIR, "test.jsonl"))

print("\nData preparation complete!")
print("Next step: run  python 2finetune.py")