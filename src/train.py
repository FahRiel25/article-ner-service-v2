import json
import os
import pandas as pd
import spacy

EVAL_JSON = "evaluation_examples.json"
VAL_CSV = "data/val.csv"


def create_adapted_nlp():
    """Membuat model spaCy yang di-adaptasi/dikalibrasi dengan EntityRuler."""
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    # Aturan tambahan untuk entitas finansial/berita lokal
    patterns = [
        {"label": "ORG", "pattern": "Maybank"},
        {"label": "ORG", "pattern": "Bursa Malaysia"},
        {"label": "ORG", "pattern": "CIMB"},
        {"label": "ORG", "pattern": "PTT Synergy"},
        {"label": "GPE", "pattern": "Kuala Lumpur"},
        {"label": "GPE", "pattern": "Klang"},
        {"label": "GPE", "pattern": "Singapore"},
    ]
    ruler.add_patterns(patterns)
    return nlp


def evaluate_ground_truth(nlp, model_name="Baseline"):
    """Mengevaluasi performa model menggunakan evaluation_examples.json."""
    if not os.path.exists(EVAL_JSON):
        print(f"Error: File {EVAL_JSON} tidak ditemukan.")
        return

    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        examples = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for example in examples:
        text = example["text"]
        expected_entities = {
            (e["start"], e["end"], e["label"]) for e in example["entities"]
        }

        doc = nlp(text)
        predicted_entities = {
            (ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
        }

        tp = len(expected_entities.intersection(predicted_entities))
        fp = len(predicted_entities - expected_entities)
        fn = len(expected_entities - predicted_entities)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"\n=== HASIL EVALUASI METRIK QUANTITATIVE ({model_name}) ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")


def sample_val_extraction(nlp, n_samples=3):
    """Mengambil sampel ekstraksi entitas dari data/val.csv."""
    if not os.path.exists(VAL_CSV):
        print(f"Warning: File {VAL_CSV} tidak ditemukan.")
        return

    df_val = pd.read_csv(VAL_CSV, low_memory=False, on_bad_lines="skip")
    df_val.columns = df_val.columns.str.strip().str.replace("'", "")

    print("\n=== HASIL EKSTRAKSI SAMPLE VAL SET ===")

    title_col = next((c for c in df_val.columns if "title" in c.lower()), None)
    body_col = next((c for c in df_val.columns if "body" in c.lower()), None)

    for idx, row in df_val.head(n_samples).iterrows():
        title_val = (
            row[title_col]
            if title_col and pd.notna(row[title_col])
            else (row.iloc[1] if len(row) > 1 else "")
        )
        body_val = (
            row[body_col]
            if body_col and pd.notna(row[body_col])
            else (row.iloc[2] if len(row) > 2 else "")
        )

        title = str(title_val).strip()
        body = str(body_val).strip()
        text = f"{title} {body}"

        doc = nlp(text)

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

        print(f"\nJudul Artikel : {title[:80]}...")
        print(f"Jumlah Entitas: {len(entities)}")
        print(
            f"Entitas Sampel (3 Teratas): {json.dumps(entities[:3], indent=2)}"
        )


if __name__ == "__main__":
    print("Memuat Model Baseline vs Adapted...")
    
    nlp_baseline = spacy.load("en_core_web_sm")
    evaluate_ground_truth(nlp_baseline, "Baseline (en_core_web_sm)")

    nlp_adapted = create_adapted_nlp()
    evaluate_ground_truth(nlp_adapted, "Adapted (Calibrated Rules)")

    sample_val_extraction(nlp_adapted)