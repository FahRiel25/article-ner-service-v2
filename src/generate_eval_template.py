import pandas as pd
import json
import os

def create_eval_template():
    # Menyiapkan path file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'data', 'processed', 'test.csv')
    out_path = os.path.join(base_dir, 'evaluation_examples.json')

    # Memuat data test.csv
    try:
        df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {test_path}")
        return

    # Mengambil 10 baris acak secara deterministik (seed=42)
    sample_df = df.sample(n=10, random_state=42)

    eval_data = []
    for _, row in sample_df.iterrows():
        # Karena artikel bisa sangat panjang, kita batasi panjangnya maksimal 500 karakter 
        # agar Anda tidak terlalu lelah melabeli secara manual
        short_text = str(row["full_text"])[:500] + "..." if len(str(row["full_text"])) > 500 else str(row["full_text"])
        
        eval_data.append({
            "original_id": str(row.get("original_id", "unknown")),
            "text": short_text,
            "expected_entities": [
                # Contoh format pengisian (Anda harus menggantinya dengan entitas yang sebenarnya):
                # {"text": "Maybank", "label": "ORG"},
                # {"text": "Malaysia", "label": "GPE"}
            ]
        })

    # Menyimpan ke dalam file JSON
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)
    
    print(f"Berhasil! Template evaluasi berisi 10 artikel disimpan di:\n{out_path}")
    print("Silakan buka file tersebut dan isi bagian 'expected_entities' secara manual.")

if __name__ == "__main__":
    create_eval_template()