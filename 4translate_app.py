"""
Step 4 (optional): Interactive translation app.
Run this after training: python 4_translate_app.py
"""

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_DIR = "models/en_asm_finetuned"
ASSAMESE_PREFIX = ">>as<< "

print("Loading model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
model     = MarianMTModel.from_pretrained(MODEL_DIR)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
model.eval()

print("\n" + "=" * 50)
print("  English → Assamese Translator")
print("  Type 'quit' to exit")
print("=" * 50 + "\n")

while True:
    english = input("English: ").strip()
    if english.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not english:
        continue

    inputs = tokenizer(
        [ASSAMESE_PREFIX + english],
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, num_beams=4, max_length=128)

    assamese = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Assamese: {assamese}\n")