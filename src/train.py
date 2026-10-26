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


def compute_metrics(tp, fp, fn):
    """Menghitung nilai Precision, Recall, dan F1-Score."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


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
    label_stats = {}

    for example in examples:
        # Pembacaan teks aman (fallback jika kunci 'text' tidak tersedia)
        text = example.get("text") or f"{example.get('title', '')}. {example.get('body', '')}".strip()

        expected_entities = {
            (e["start"], e["end"], e["label"]) for e in example.get("entities", [])
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

        # Perhitungan breakdown metrik per-label
        all_labels = {e[2] for e in expected_entities}.union({p[2] for p in predicted_entities})
        for lbl in all_labels:
            if lbl not in label_stats:
                label_stats[lbl] = {"tp": 0, "fp": 0, "fn": 0}
            exp_lbl = {e for e in expected_entities if e[2] == lbl}
            pred_lbl = {p for p in predicted_entities if p[2] == lbl}

            label_stats[lbl]["tp"] += len(exp_lbl.intersection(pred_lbl))
            label_stats[lbl]["fp"] += len(pred_lbl - exp_lbl)
            label_stats[lbl]["fn"] += len(exp_lbl - pred_lbl)

    precision, recall, f1 = compute_metrics(true_positives, false_positives, false_negatives)

    print(f"\n=== HASIL EVALUASI METRIK QUANTITATIVE ({model_name}) ===")
    print(f"Overall Exact-Span Precision : {precision:.4f}")
    print(f"Overall Exact-Span Recall    : {recall:.4f}")
    print(f"Overall Exact-Span F1-Score  : {f1:.4f}")

    print("\n--- Breakdown Metrik Per-Label ---")
    for lbl, st in sorted(label_stats.items()):
        lp, lr, lf1 = compute_metrics(st["tp"], st["fp"], st["fn"])
        print(f" Label {lbl:<10} | Precision: {lp:.4f} | Recall: {lr:.4f} | F1-Score: {lf1:.4f}")


def sample_val_extraction(nlp, n_samples=3):
    """Mengambil sampel ekstraksi entitas dari data/val.csv."""
    if not os.path.exists(VAL_CSV):
        print(f"Warning: File {VAL_CSV} tidak ditemukan.")
        return

    df_val = pd.read_csv(VAL_CSV, sep=";", low_memory=False, on_bad_lines="skip")
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
        text = f"{title}. {body}".strip() if title and body else title or body

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