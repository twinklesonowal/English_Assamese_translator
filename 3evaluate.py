"""
Step 3: Evaluate your fine-tuned model and translate new sentences.

BLEU score interpretation for NMT:
  < 10  = Almost unusable
  10-20 = Gets the gist
  20-30 = Understandable
  30-40 = Good quality
  > 40  = Near human (rare without lots of data)
"""

import json, torch
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

MODEL_DIR  = "models/en_asm_finetuned"   # path to your saved model
DATA_DIR   = "data"
ASSAMESE_PREFIX = ">>as<< "

# ── Load fine-tuned model ─────────────────────────────────────────────────────
print(f"Loading fine-tuned model from: {MODEL_DIR}\n")
tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
model     = MarianMTModel.from_pretrained(MODEL_DIR)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
model.eval()

# ── Translation function ──────────────────────────────────────────────────────
def translate(english_texts: list[str], batch_size: int = 16) -> list[str]:
    """Translate a list of English sentences to Assamese."""
    results = []
    # Add language prefix to each sentence
    prefixed = [ASSAMESE_PREFIX + t for t in english_texts]

    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=4,          # beam search for better quality
                max_length=128,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ── Quick demo translations ───────────────────────────────────────────────────
demo_sentences = [
    "Hello, how are you?",
    "The weather is very nice today.",
    "I want to learn Assamese language.",
    "Where is the nearest hospital?",
    "My name is Rohan and I live in Guwahati.",
]

print("=" * 60)
print("DEMO TRANSLATIONS (English → Assamese)")
print("=" * 60)
translations = translate(demo_sentences)
for en, asm in zip(demo_sentences, translations):
    print(f"\nEN:  {en}")
    print(f"AS:  {asm}")

# ── Full BLEU evaluation on test set ─────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

test_src, test_ref = [], []
with open(f"{DATA_DIR}/test.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if "translation" in r:
            en = r["translation"].get("en", "")
            asm = r["translation"].get("as", "")
        else:
            en  = r.get("src", "")
            asm = r.get("tgt", "")
        if en.strip() and asm.strip():
            test_src.append(en)
            test_ref.append(asm)

print(f"Test sentences: {len(test_src)}")
print("Translating... (this may take a few minutes)")

hypotheses = translate(test_src)
references = [[ref] for ref in test_ref]   # sacrebleu format

bleu = sacrebleu.corpus_bleu(hypotheses, references)
print(f"\nBLEU score: {bleu.score:.2f}")

# ── Save predictions ──────────────────────────────────────────────────────────
with open("outputs/test_predictions.txt", "w", encoding="utf-8") as f:
    for en, hyp, ref in zip(test_src, hypotheses, test_ref):
        f.write(f"EN:  {en}\n")
        f.write(f"HYP: {hyp}\n")
        f.write(f"REF: {ref}\n")
        f.write("-" * 50 + "\n")

print("\nPredictions saved to outputs/test_predictions.txt")
print("\nEvaluation complete!")