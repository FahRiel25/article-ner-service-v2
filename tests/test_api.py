import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


# Test 1: Skenario Input Normal
def test_extract_entities_success():
    payload = {
        "title": "Maybank Malaysia Launches New Feature",
        "body": "Maybank announced a new digital banking feature in Kuala Lumpur today.",
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "meta" in data
    assert isinstance(data["entities"], list)
    assert data["meta"]["entity_count"] == len(data["entities"])


# Test 2: Skenario Input Kosong (Keduanya Kosong -> Status 400)
def test_extract_entities_empty_input():
    payload = {"title": "", "body": ""}
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 400
    assert (
        "Judul dan bodi artikel tidak boleh kosong" in response.json()["detail"]
    )


# Test 3: Skenario Input Melebihi Batas Panjang Karakter (Status 400)
def test_extract_entities_exceed_max_length():
    payload = {
        "title": "Judul Panjang",
        "body": "A" * 50001,  # Melebihi MAX_TEXT_LENGTH (50000)
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 400
    assert "melebihi batas maksimum" in response.json()["detail"]


# Test 4: Skenario Karakter Spesial dan Simbol
def test_extract_entities_special_characters():
    payload = {
        "title": "Maybank & CIMB! @Kuala Lumpur #2026",
        "body": "Bank Negara Malaysia (BNM) released RM1,000,000 funds.",
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    assert "entities" in response.json()


# Test 5: Skenario Struktur Response & Determinisme Offset
def test_extract_entities_structure():
    payload = {
        "title": "Maybank",
        "body": "Maybank operates in Kuala Lumpur.",
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    for entity in data["entities"]:
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity
        assert entity["start"] < entity["end"]