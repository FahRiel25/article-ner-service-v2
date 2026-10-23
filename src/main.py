import spacy
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Article Named-Entity Recognition Service", version="1.0.0"
)

MODEL_NAME = "en_core_web_sm"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    from spacy.cli import download

    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)


# Skema Input Request JSON
class ArticleRequest(BaseModel):
    title: str = Field(..., description="Judul artikel")
    body: str = Field(..., description="Isi bodi artikel")


MAX_TEXT_LENGTH = 50000  # Batas maksimum karakter


@app.post("/extract-entities", status_code=status.HTTP_200_OK)
def extract_entities(payload: ArticleRequest):
    title = payload.title.strip()
    body = payload.body.strip()

    # Validasi input: Tolak jika judul dan body keduanya kosong
    if not title and not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Judul dan bodi artikel tidak boleh kosong keduanya.",
        )

    # Validasi input: Tolak jika melebihi batas panjang karakter
    if len(title) + len(body) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input melebihi batas maksimum {MAX_TEXT_LENGTH} karakter.",
        )

    # Penggabungan Title dan Body sesuai instruksi tugas
    combined_text = f"{title}. {body}" if title and body else title or body

    # Ekstraksi Entitas dengan spaCy
    doc = nlp(combined_text)
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]

    return {
        "entities": entities,
        "meta": {"entity_count": len(entities), "model": MODEL_NAME},
    }


@app.get("/", response_class=HTMLResponse)
def serve_panel():
    """Tampilan Panel Antarmuka Web (UI) Sederhana."""
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Article NER Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #1a365d; margin-top: 0; }
            label { font-weight: bold; display: block; margin-top: 15px; margin-bottom: 5px; }
            input[type="text"], textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { margin-top: 20px; padding: 12px 20px; background-color: #2b6cb0; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
            button:hover { background-color: #2c5282; }
            #loading { display: none; margin-top: 15px; color: #d69e2e; font-weight: bold; text-align: center; }
            #error { display: none; margin-top: 15px; padding: 10px; background-color: #fed7d7; color: #9b2c2c; border-radius: 4px; }
            #results { margin-top: 25px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { border: 1px solid #e2e8f0; padding: 10px; text-align: left; }
            th { background-color: #edf2f7; }
            .badge { background-color: #ebf8ff; color: #2b6cb0; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Article Named-Entity Recognition Service</h2>
            <form id="nerForm">
                <label for="title">Judul Artikel:</label>
                <input type="text" id="title" placeholder="Masukkan judul artikel...">
                
                <label for="body">Bodi Artikel:</label>
                <textarea id="body" rows="6" placeholder="Masukkan isi bodi artikel..."></textarea>
                
                <button type="submit">Ekstrak Entitas</button>
            </form>

            <div id="loading">Memproses artikel...</div>
            <div id="error"></div>
            <div id="results"></div>
        </div>

        <script>
            document.getElementById('nerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const title = document.getElementById('title').value;
                const body = document.getElementById('body').value;
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const results = document.getElementById('results');

                loading.style.display = 'block';
                error.style.display = 'none';
                results.innerHTML = '';

                try {
                    const response = await fetch('/extract-entities', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title, body })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Terjadi kesalahan pada server.');
                    }

                    if (data.entities.length === 0) {
                        results.innerHTML = '<p><em>Tidak ada entitas yang ditemukan.</em></p>';
                    } else {
                        let html = `<h3>Hasil Ekstraksi (${data.meta.entity_count} entitas)</h3>`;
                        html += `<table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Teks Entitas</th>
                                    <th>Kategori (Label)</th>
                                    <th>Karakter (Start - End)</th>
                                </tr>
                            </thead>
                            <tbody>`;
                        data.entities.forEach((ent, index) => {
                            html += `<tr>
                                <td>${index + 1}</td>
                                <td><strong>${ent.text}</strong></td>
                                <td><span class="badge">${ent.label}</span></td>
                                <td>${ent.start} - ${ent.end}</td>
                            </tr>`;
                        });
                        html += '</tbody></table>';
                        results.innerHTML = html;
                    }
                } catch (err) {
                    error.innerText = err.message;
                    error.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """