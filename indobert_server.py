"""
IndoBERT Local Embedding Server
Model: indobenchmark/indobert-base-p2
Port:  5001

Cara pakai:
    .venv/bin/python indobert_server.py

POST /embed
    Body:    {"texts": ["kalimat 1", "kalimat 2"]}
    Returns: {"embeddings": [[768 floats], [768 floats]], "model": "..."}

GET /health
    Returns: {"status": "ok", "model": "indobenchmark/indobert-base-p2", "dim": 768}
"""

from flask import Flask, request, jsonify
from transformers import BertTokenizer, AutoModel
import torch
import sys

MODEL_NAME = "indobenchmark/indobert-base-p2"

print(f"Loading {MODEL_NAME}...", flush=True)
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"Model loaded. Hidden size: {model.config.hidden_size}", flush=True)
except Exception as e:
    print(f"ERROR loading model: {e}", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'dim': model.config.hidden_size,
    })


@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json(force=True, silent=True)
    if not data or 'texts' not in data:
        return jsonify({'error': 'Body harus berisi {"texts": [...]}'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts harus list non-kosong'}), 400

    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        out = model(**enc)

    # Attention-mask weighted mean pool: [batch, seq, 768] → [batch, 768]
    mask   = enc['attention_mask'].unsqueeze(-1).float()  # [batch, seq, 1]
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    return jsonify({
        'embeddings': pooled.tolist(),
        'model': MODEL_NAME,
        'dim': pooled.shape[1],
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    print(f"Server running on http://127.0.0.1:{args.port}", flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False)
