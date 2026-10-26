# Article Named Entity Recognition (NER) Service 

Proyek ini merupakan layanan REST API dan antarmuka web sederhana (*Embedded UI*) untuk **Ekstraksi Entitas Terkategori (*Named Entity Recognition*)** pada artikel berita dan finansial. Sistem dibangun menggunakan **FastAPI** di sisi backend, model bahasa **spaCy** yang dikalibrasi dengan aturan khusus (*EntityRuler*), serta dilengkapi dengan skrip otomatisasi *data profiling*, *dataset splitting*, evaluasi metrik quantitatif, dan pengujian otomatis (*automated testing*).

---


## 📁 Struktur Direktori Proyek

```text
article-ner-service-v2/
├── data/                             # Direktori data & pemrosesan dataset
│   ├── raw/                          # File CSV mentah (Kliping Maybank)
│   ├── merged_articles_clean.csv     # Dataset gabungan bebas duplikat
│   ├── train.csv                     # Split data latih (70%)
│   ├── val.csv                       # Split data validasi (15%)
│   └── test.csv                      # Split data uji (15%)
├── src/                              # Source code utama
│   ├── data_profiling.py             # Skrip analisis & pembersihan data awal
│   ├── prepare_data.py              # Skrip pembagian dataset & pembuat sampel evaluasi
│   ├── train.py                      # Skrip adaptasi EntityRuler & evaluasi metrik
│   └── main.py                       # Aplikasi FastAPI & Web UI Panel
├── tests/                            # Unit testing otomatis
│   └── test_api.py                   # Skenario pengujian endpoint API
├── evaluation_examples.json          # Datasets sampel evaluasi ground truth (10 contoh)
├── pytest.ini                        # Konfigurasi modul pytest
├── requirements.txt                  # Dependensi Python proyek
└── README.md                         # Dokumentasi utama proyek
```

---

## ⚙️ Panduan Instalasi & Penggunaan

### 1. Prasyarat Sistem
* Python 3.10+
* Virtual Environment (disarankan)

### 2. Instalasi Dependensi
 Jalankan perintah berikut pada terminal di root direktori proyek:

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Mengunduh Model Language spaCy
Unduh model dasar Bahasa Inggris yang digunakan oleh sistem:
```bash
python -m spacy download en_core_web_sm
```

---

## 🔄 Pipeline Pemrosesan Data & Pelatihan Model

### Langkah 1: Data Profiling & Cleaning
Menggabungkan seluruh dataset kliping CSV mentah dari direktori `data/raw/`, membersihkan duplikasi judul/bodi, dan menganalisis distribusi bahasa.
```bash
python src/data_profiling.py
```

### Langkah 2: Dataset Splitting
Membagi dataset bersih menjadi proporsi **70% Train**, **15% Validation**, dan **15% Test**, serta secara otomatis menyiapkan file `evaluation_examples.json`.
```bash
python src/prepare_data.py
```

### Langkah 3: Evaluasi Model (Baseline vs Adapted)
Menjalankan pengujian evaluasi perbandingan antara model *baseline* (`en_core_web_sm`) dan model *adapted* yang dikalibrasi aturan `EntityRuler`:
```bash
python src/train.py
```

---

## 🚀 Pengoperasian Service API

### 1. Menjalankan Server FastAPI
Jalankan Uvicorn server untuk mengaktifkan layanan API:
```bash
uvicorn src.main:app --reload --port 8000
```
Server akan berjalan secara lokal di URL: `http://127.0.0.1:8000`

---

## 🌐 Dokumentasi Endpoint REST API

### 1. Web UI Dashboard (Panel Pengujian Interaktif)
* **Endpoint:** `GET /`
* **Format Respon:** `text/html`
* **Deskripsi:** Menampilkan antarmuka halaman web interaktif untuk menguji ekstraksi entitas secara langsung melalui peramban (browser).

---

### 2. Ekstraksi Entitas Artikel
* **Endpoint:** `POST /extract-entities`
* **Headers:** `Content-Type: application/json`

#### Contoh Request Body:
```json
{
  "title": "Maybank Launching Feature in Kuala Lumpur",
  "body": "Maybank announced a new digital banking initiative in Kuala Lumpur today."
}
```

#### Contoh Response Berhasil (`200 OK`):
```json
{
  "entities": [
    {
      "text": "Maybank",
      "label": "ORG",
      "start": 0,
      "end": 7
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 21,
      "end": 33
    },
    {
      "text": "Maybank",
      "label": "ORG",
      "start": 43,
      "end": 50
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 94,
      "end": 106
    },
    {
      "text": "today",
      "label": "DATE",
      "start": 107,
      "end": 112
    }
  ],
  "meta": {
    "entity_count": 5,
    "model": "en_core_web_sm"
  }
}
```

#### Contoh Response Gagal - Input Kosong (`400 Bad Request`):
```json
{
  "detail": "Judul dan bodi artikel tidak boleh kosong keduanya."
}
```

#### Contoh Response Gagal - Panjang Teks Melebihi Batas (`400 Bad Request`):
```json
{
  "detail": "Input melebihi batas maksimum 50000 karakter."
}
```

---

## 🧪 Automated Testing (Pengujian Otomatis)

Proyek ini telah dilengkapi dengan unit test berbasis `pytest` yang mencakup 5 skenario utama:
1. **Normal Input**: Memastikan ekstraksi entitas berjalan normal.
2. **Empty Input**: Memastikan server menolak payload kosong dengan HTTP status 400.
3. **Exceed Max Length**: Memastikan server menolak input yang melebihi 50.000 karakter.
4. **Special Characters**: Memastikan sistem mampu menangani simbol dan karakter khusus.
5. **Response Structure**: Memastikan skema response konsisten (posisi `start` & `end` valid).

Jalankan pengujian dengan perintah:
```bash
pytest
```

---

## ⚠️ Batasan Sistem (Known Limitations)

1. **Panjang Karakter Input:** Total gabungan panjang karakter `title` dan `body` dibatasi maksimal **50.000 karakter**.
2. **Dependensi Bahasa Model:** Model teroptimasi untuk artikel berbahasa Inggris (*English*) dan istilah entitas lokal Malaysia/Regional (seperti *Bursa Malaysia*, *Maybank*, *CIMB*, *Klang*).
3. **Cakupan EntityRuler:** Pengenalan entitas khusus terikat pada pola (*patterns*) yang terdaftar di `src/train.py`.