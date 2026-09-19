# English → Assamese Neural Machine Translator

A fine-tuned Neural Machine Translation (NMT) model and web app that translates English text to Assamese (অসমীয়া).

---

## Demo

![Translator Web App](https://img.shields.io/badge/BLEU%20Score-35.93-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)

---

## Features

- English to Assamese translation powered by a fine-tuned Helsinki-NLP model
- Clean, minimal web interface built with Flask
- Dark mode and light mode toggle
- Loading spinner animation during translation
- Voice pronunciation of Assamese output (browser TTS)
- Copy translation to clipboard
- Recent translation history (last 5 translations)
- GPU accelerated inference (CUDA)
- Ctrl + Enter shortcut to translate

---

## Model

| Detail | Value |
|---|---|
| Base model | Helsinki-NLP/opus-mt-en-mul |
| Dataset | AI4Bharat Samanantar (Assamese) |
| Training pairs | 114,393 |
| Validation pairs | 12,711 |
| Test pairs | 14,123 |
| BLEU score | 35.93 |
| Epochs | 3 |
| Device | NVIDIA RTX 4060 Laptop GPU |

---

## Project Structure

```
nmt_pretrained/
├── templates/
│   └── index.html          ← Frontend UI
├── app.py                  ← Flask backend
├── 1prepare_data.py        ← Download and split dataset
├── 2finetune.py            ← Fine-tune the model
├── 3evaluate.py            ← Evaluate BLEU score
├── 4translate_app.py       ← CLI translator
├── requirements.txt
└── README.md
```

> Note: `models/` and `data/` folders are excluded from this repo due to file size. See setup instructions below to recreate them.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/panthoiba/English_Assamese_translator.git
cd English_Assamese_translator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install GPU-enabled PyTorch (recommended)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Download and prepare data

```bash
python 1prepare_data.py
```

### 5. Fine-tune the model

```bash
python 2finetune.py
```

This downloads Helsinki-NLP/opus-mt-en-mul (~300MB) and fine-tunes it on the Assamese dataset. Training takes approximately 30–60 minutes on a GPU.

### 6. Evaluate

```bash
python 3evaluate.py
```

### 7. Run the web app

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## Share with friends using ngrok

```bash
pip install pyngrok
```

In a second terminal while `app.py` is running:

```python
from pyngrok import ngrok
ngrok.set_auth_token('your_token_here')
url = ngrok.connect(5000)
print("Share this link:", url)
input("Press Enter to stop...")
```

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 4GB+ VRAM for inference
- 8GB+ VRAM for training

---

## Tech Stack

- **Model**: Helsinki-NLP/opus-mt-en-mul (MarianMT)
- **Training**: Hugging Face Transformers, Seq2SeqTrainer
- **Dataset**: AI4Bharat Samanantar
- **Backend**: Flask + Waitress
- **Frontend**: HTML, CSS, JavaScript
- **Evaluation**: SacreBLEU

---

## Sample Translations

| English | Assamese |
|---|---|
| Hello, how are you? | হে, কেনেকুৱা? |
| The weather is very nice today. | আজি বতৰখন খুব ভাল। |
| I want to learn Assamese language. | মই অসমীয়া ভাষা শিকিবলৈ বিচাৰোঁ। |
| Where is the nearest hospital? | নিকটীয়া চিকিৎসালয় ক'ত? |
| My name is Rohan and I live in Guwahati. | মোৰ নাম ৰোহন আৰু মই গুৱাহাটীত বাস কৰিছো। |

---

## Author

Made by [Twinkle Sonowal](https://github.com/twinklesonowal)