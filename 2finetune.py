"""
Step 2: Fine-tune Helsinki-NLP/opus-mt-en-mul on English→Assamese.

Helsinki-NLP's opus-mt-en-mul is pretrained on 100+ languages including
Assamese (code: ">>as<<"). We fine-tune it on our parallel data.

GPU recommended (Google Colab free tier works great).
CPU-only training is possible but will be very slow.
"""

import os, json, torch
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import sacrebleu

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME  = "Helsinki-NLP/opus-mt-en-mul"   # pretrained multilingual model
DATA_DIR    = "data"
OUTPUT_DIR  = "models/en_asm_finetuned"
MAX_LENGTH  = 128    # max tokens per sentence (increase for longer sentences)
BATCH_SIZE  = 16     # reduce to 8 if you run out of GPU memory
NUM_EPOCHS  = 5
LEARNING_RATE = 5e-5
ASSAMESE_PREFIX = ">>as<< "   # tells the model to output Assamese

# ── Load tokenizer & model ────────────────────────────────────────────────────
print(f"Loading pretrained model: {MODEL_NAME}")
print("This downloads ~300MB on first run...\n")

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

# Adjust field names to match your JSONL keys
# Common patterns: {"en": "...", "as": "..."} or {"translation": {"en": ..., "as": ...}}
def extract_pairs(records, src_key="src", tgt_key="tgt"):
    sources, targets = [], []
    for r in records:
        if "translation" in r:          # nested format
            src = r["translation"].get(src_key, "")
            tgt = r["translation"].get(tgt_key, "")
        else:
            src = r.get(src_key, "")
            tgt = r.get(tgt_key, "")
        if src.strip() and tgt.strip():
            sources.append(ASSAMESE_PREFIX + src)  # add language prefix
            targets.append(tgt)
    return sources, targets

print("Loading data...")
train_src, train_tgt = extract_pairs(load_jsonl(f"{DATA_DIR}/train.jsonl"))
val_src,   val_tgt   = extract_pairs(load_jsonl(f"{DATA_DIR}/val.jsonl"))
print(f"Train pairs: {len(train_src)}, Val pairs: {len(val_src)}")

# ── Tokenise ──────────────────────────────────────────────────────────────────
def tokenize(sources, targets):
    model_inputs = tokenizer(
        sources,
        text_target=targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    return Dataset.from_dict(model_inputs)

print("Tokenising data (may take a minute)...")
train_dataset = tokenize(train_src, train_tgt)
val_dataset   = tokenize(val_src,   val_tgt)

# ── BLEU metric ───────────────────────────────────────────────────────────────
def compute_bleu(eval_preds):
    preds, labels = eval_preds
    # Replace -100 (padding) with pad token id
    labels = [
        [(l if l != -100 else tokenizer.pad_token_id) for l in label]
        for label in labels
    ]
    decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)
    decoded_labels = [[ref] for ref in decoded_labels]  # sacrebleu expects list-of-lists

    result = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    return {"bleu": round(result.score, 2)}

# ── Training arguments ────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",             # change to "wandb" if you want experiment tracking
    fp16=torch.cuda.is_available(),  # use mixed precision on GPU
)

# ── Trainer ───────────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_bleu,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train! ────────────────────────────────────────────────────────────────────
print("\nStarting fine-tuning...")
print("Watch the 'bleu' score — higher is better (0–100 scale).\n")
trainer.train()

# Save final model
print(f"\nSaving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete! Next step: run  python 3_evaluate.py")