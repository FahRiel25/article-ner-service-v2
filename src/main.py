from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from src.ner_model import NERModel
import os

app = FastAPI(title="Article NER Service")

# Setup template engine untuk antarmuka web
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model secara global saat aplikasi berjalan
ner_model = NERModel()

class ArticleInput(BaseModel):
    title: str
    body: str

@app.get("/")
async def serve_panel(request: Request):
    # Menggunakan keyword arguments (request, name, context) agar kompatibel dengan versi terbaru
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

@app.post("/extract-entities")
async def extract_entities_endpoint(article: ArticleInput):
    # Validasi: Menolak input jika title atau body kosong (hanya berisi spasi)
    if not article.title.strip() or not article.body.strip():
        raise HTTPException(status_code=400, detail="Title and body cannot be empty.")
    
    # Menggabungkan title dan body sebelum diproses
    combined_text = f"{article.title}. {article.body}"
    
    try:
        entities = ner_model.extract_entities(combined_text)
        return {
            "entities": entities,
            "meta": {
                "entity_count": len(entities),
                "model": "en_core_web_sm (spaCy baseline)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))