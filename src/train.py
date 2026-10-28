import json
import os
import pandas as pd
import spacy

class BaselineNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text):
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]

class AdaptedNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "ORG", "pattern": "Bursa Malaysia"},
            {"label": "ORG", "pattern": "FTSE Bursa Malaysia KLCI"},
            {"label": "ORG", "pattern": "FBM KLCI"},
            {"label": "ORG", "pattern": "Maybank"},
            {"label": "ORG", "pattern": "CIMB"},
            {"label": "ORG", "pattern": "Public Bank"},
            {"label": "GPE", "pattern": "Kuala Lumpur"},
        ]
        ruler.add_patterns(patterns)

    def extract(self, text):
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]

def evaluate_model(model, eval_data, labels):
    tp_total, fp_total, fn_total = 0, 0, 0
    label_stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}

    for item in eval_data:
        text = item["text"]
        expected_set = {(e["text"], e["label"]) for e in item.get("expected_entities", [])}
        preds = model.extract(text)
        pred_set = {(e["text"], e["label"]) for e in preds}

        tp_set = expected_set.intersection(pred_set)
        fp_set = pred_set - expected_set
        fn_set = expected_set - pred_set

        tp_total += len(tp_set)
        fp_total += len(fp_set)
        fn_total += len(fn_set)

        for text_val, lbl in tp_set:
            if lbl in label_stats:
                label_stats[lbl]["tp"] += 1
        for text_val, lbl in fp_set:
            if lbl in label_stats:
                label_stats[lbl]["fp"] += 1
        for text_val, lbl in fn_set:
            if lbl in label_stats:
                label_stats[lbl]["fn"] += 1

    p_over = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r_over = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_over = 2 * p_over * r_over / (p_over + r_over) if (p_over + r_over) > 0 else 0.0

    print(f"Overall Exact-Span Precision : {p_over:.4f}")
    print(f"Overall Exact-Span Recall    : {r_over:.4f}")
    print(f"Overall Exact-Span F1-Score  : {f1_over:.4f}\n")
    print("--- Breakdown Metrik Per-Label ---")

    for lbl in sorted(labels):
        tp = label_stats[lbl]["tp"]
        fp = label_stats[lbl]["fp"]
        fn = label_stats[lbl]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"Label {lbl:<12} | Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f1:.4f}")

def run_training_and_eval():
    print("Memuat Model Baseline vs Adapted...\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_path = os.path.join(base_dir, "evaluation_examples.json")
    val_path = os.path.join(base_dir, "data", "processed", "val.csv")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    labels = ["CARDINAL", "DATE", "EVENT", "GPE", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]

    baseline_model = BaselineNER()
    adapted_model = AdaptedNER()

    print("=== HASIL EVALUASI METRIK QUANTITATIVE (Baseline (en_core_web_sm)) ===")
    evaluate_model(baseline_model, eval_data, labels)
    print("\n")

    print("=== HASIL EVALUASI METRIK QUANTITATIVE (Adapted (Calibrated Rules)) ===")
    evaluate_model(adapted_model, eval_data, labels)
    print("\n")

    # Evaluasi Sampel Val Set
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        print("=== HASIL EKSTRAKSI SAMPLE VAL SET ===\n")
        for i in range(min(2, len(val_df))):
            row = val_df.iloc[i]
            title = str(row["title"])
            full_text = str(row["full_text"])
            ents = adapted_model.extract(full_text)
            
            print(f"Judul Artikel : '{title[:65]}...'")
            print(f"Jumlah Entitas: {len(ents)}")
            print("Entitas Sampel (3 Teratas): [")
            for e in ents[:3]:
                print("  {")
                print(f'    "text": "{e["text"]}",')
                print(f'    "label": "{e["label"]}",')
                print(f'    "start": {e["start"]},')
                print(f'    "end": {e["end"]}')
                print("  }" if e == ents[:3][-1] else "  },")
            print("]\n")

if __name__ == "__main__":
    run_training_and_eval()