# English → Assamese NMT — Step-by-Step Guide

## What is this?
A Neural Machine Translation model that translates English to Assamese.
We fine-tune Helsinki-NLP's pretrained multilingual model (opus-mt-en-mul)
which already knows 100+ languages including Assamese.

---

## Project Structure
```
en_asm_nmt/
├── data/              ← parallel sentence pairs (auto-downloaded)
├── models/            ← your fine-tuned model (saved here after training)
├── outputs/           ← evaluation results and translations
├── 1_prepare_data.py  ← Step 1: download and split data
├── 2_finetune.py      ← Step 2: fine-tune the model
├── 3_evaluate.py      ← Step 3: compute BLEU score + sample translations
├── 4_translate_app.py ← Step 4: interactive CLI translator
└── requirements.txt
```

---

## Setup (do once)

### 1. Install Python 3.10+
Download from https://python.org

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Recommended) Use Google Colab for free GPU
Go to https://colab.research.google.com
Upload these scripts and run them there — it's free and much faster.

---

## Running the Pipeline

### Step 1 — Get data
```bash
python 1_prepare_data.py
```
Downloads the AI4Bharat Samanantar English-Assamese corpus.
Creates data/train.jsonl, data/val.jsonl, data/test.jsonl

### Step 2 — Fine-tune
```bash
python 2_finetune.py
```
- Downloads Helsinki-NLP/opus-mt-en-mul (~300 MB, once only)
- Fine-tunes on your Assamese data
- Saves best model to models/en_asm_finetuned/
- On GPU: ~30 min per epoch | On CPU: ~4–8 hours per epoch

### Step 3 — Evaluate
```bash
python 3_evaluate.py
```
Shows BLEU score and sample translations. Saves all to outputs/

### Step 4 — Translate interactively
```bash
python 4_translate_app.py
```
Type English sentences and see Assamese output.

---

## Key Concepts (for beginners)

| Term | What it means |
|------|--------------|
| NMT | Neural Machine Translation — AI that translates text |
| Fine-tuning | Taking a pretrained model and training it more on your specific data |
| BLEU score | 0–100 metric for translation quality (>20 is good) |
| Tokenization | Breaking text into small pieces the model understands |
| Beam search | Smarter decoding that explores multiple translation options |
| Epoch | One full pass through your training data |

---

## Tips for Better Results

1. **More data = better model.** Try to get 500K+ sentence pairs.
   Best source: https://huggingface.co/datasets/ai4bharat/samanantar

2. **Use a GPU.** Google Colab (free) or Kaggle Notebooks (free).

3. **Increase epochs carefully.** Watch val BLEU — stop when it stops improving.

4. **Try mBART for even better results:**
   Replace model name with "facebook/mbart-large-50-many-to-many-mmt"
   (needs more memory but is more powerful)

---

## Data Sources for Assamese
- AI4Bharat Samanantar: https://huggingface.co/datasets/ai4bharat/samanantar
- OPUS corpus: https://opus.nlpl.eu
- PMIndia: https://huggingface.co/datasets/pmindia (Parliament of India speeches)
