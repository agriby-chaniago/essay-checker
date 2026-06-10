# NLP Essay Checker

Sistem penilaian esai otomatis berbasis NLP menggunakan **IndoBERT** + TF-IDF + BM25 + Jaccard Similarity.

---

## Stack

| Komponen | Teknologi |
|---|---|
| Frontend | HTML + Vanilla JS |
| Backend API | PHP 8.x |
| Embedding Model | IndoBERT Base P2 (`indobenchmark/indobert-base-p2`) |
| Model Server | Python 3.11 + Flask |
| Fallback | TF-IDF Enhanced + Pre-built IDF |

---

## Requirements

- PHP 8.x (`php -S`)
- Python 3.11
- Virtual environment: `.venv-pytorch/` (sudah ada di project)

---

## Instalasi (pertama kali)

```bash
cd /home/agribychaniago/www/essay-checker

# Buat virtual environment
python3 -m venv .venv-pytorch

# Install dependencies
.venv-pytorch/bin/python -m pip install torch transformers flask
```

> Download torch + transformers ~2GB. Lakukan sekali saja.

---

## Cara Menjalankan

### 1. Start IndoBERT Server (Terminal 1)

```bash
cd /home/agribychaniago/www/essay-checker

.venv-pytorch/bin/python indobert_server.py
```

Tunggu hingga muncul:
```
Model loaded. Hidden size: 768
Server running on http://127.0.0.1:5001
 * Running on http://127.0.0.1:5001
```

> **Catatan:** Load pertama kali butuh ~1-2 menit (download model ~500MB ke cache).
> Load berikutnya instan karena model sudah di-cache di `~/.cache/huggingface/hub/`.

### 2. Start PHP Server (Terminal 2)

```bash
cd /home/agribychaniago/www/essay-checker

php -S localhost:8080
```

### 3. Buka Browser

```
http://localhost:8080
```

---

## Cara Pakai

1. **Kunci Jawaban** — isi poin-poin jawaban yang benar (bisa lebih dari satu)
2. **Jawaban Mahasiswa** — ketik teks jawaban atau upload PDF/gambar
3. **Konfigurasi** — atur threshold, bobot semantic vs kelengkapan
4. Klik **Analisis Sekarang**

---

## Mode Analisis

| Mode | Kondisi | Akurasi |
|---|---|---|
| **IndoBERT** (768 dim) | `indobert_server.py` aktif | Tinggi — mengerti makna semantik |
| **TF-IDF Enhanced** | Server Python mati | Menengah — berbasis kata kunci + pre-built IDF |

Jika IndoBERT server tidak aktif, sistem otomatis fallback ke TF-IDF Enhanced.

---

## Troubleshooting

**Port 5001 sudah dipakai:**
```bash
# Cek proses
lsof -i :5001

# Kill dan restart
kill -9 $(lsof -ti :5001)
.venv-pytorch/bin/python indobert_server.py
```

**Port berbeda:**
```bash
.venv-pytorch/bin/python indobert_server.py --port 5002
# Lalu update baris di api.php:
# curl_init('http://127.0.0.1:5002/embed');
```

**Cek IndoBERT server aktif:**
```bash
curl http://127.0.0.1:5001/health
# {"status":"ok","model":"indobenchmark/indobert-base-p2","dim":768}
```

**Cek PHP server aktif:**
```bash
curl http://localhost:8080/index.html
```

---

## Struktur File

```
essay-checker/
├── index.html          # Frontend UI
├── api.php             # Backend PHP (TF-IDF + BM25 + Jaccard + IndoBERT proxy)
├── indobert_server.py  # Python Flask server untuk IndoBERT
├── .venv-pytorch/      # Python virtual environment (torch + transformers + flask)
└── README.md
```

---

## Metode Penilaian

```
Skor Akhir = (Skor Semantik × w_semantic) + (Skor Kelengkapan × w_kelengkapan)

Skor Semantik    = (cosine_sim × 0.85 + jaccard × 0.15) × 100
Skor Kelengkapan = (soft_completeness per poin + BM25 boost 15%) × 100
```

Default bobot: semantic 60%, kelengkapan 40% (bisa diubah di UI).
