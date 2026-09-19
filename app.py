from flask import Flask, render_template, request, jsonify
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

MODEL_DIR = "models/en_asm_finetuned"
ASSAMESE_PREFIX = ">>as<< "

print("Loading model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
model = MarianMTModel.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}!")

def translate(text):
    inputs = tokenizer(
        [ASSAMESE_PREFIX + text],
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, num_beams=4, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json()
    english_text = data.get("text", "").strip()
    if not english_text:
        return jsonify({"error": "No text provided"}), 400
    assamese = translate(english_text)
    return jsonify({"translation": assamese})

if __name__ == "__main__":
    from waitress import serve
    print("Starting production server on http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000)