from fastapi.testclient import TestClient
from src.main import app
from src.ner_model import NERModel

client = TestClient(app)

def test_api_valid_input():
    response = client.post("/extract-entities", json={
        "title": "Apple Inc. CEO Tim Cook visits Indonesia",
        "body": "Tim Cook arrived in Jakarta on Tuesday, April 2026."
    })
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "meta" in data
    assert len(data["entities"]) > 0

def test_api_empty_title():
    response = client.post("/extract-entities", json={
        "title": "   ",
        "body": "The body has content."
    })
    assert response.status_code == 400
    assert "Title and body cannot be empty" in response.json()["detail"]

def test_api_empty_body():
    response = client.post("/extract-entities", json={
        "title": "Valid Title",
        "body": ""
    })
    assert response.status_code == 400

def test_api_missing_fields():
    response = client.post("/extract-entities", json={
        "title": "Only title provided"
    })
    # FastAPI otomatis mengembalikan 422 jika field wajib (body) tidak ada
    assert response.status_code == 422

def test_ner_model_determinism():
    model = NERModel()
    text = "Maybank opens a new branch in Kuala Lumpur."
    result1 = model.extract_entities(text)
    result2 = model.extract_entities(text)
    assert result1 == result2  # Memastikan output konsisten untuk input yang sama