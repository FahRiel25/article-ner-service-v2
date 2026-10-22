import json
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(clean_csv_path: str, seed: int = 42):
    df = pd.read_csv(clean_csv_path, sep=";")

    # Menggabungkan Title dan Body sebelum analisis
    df["full_text"] = (
        df["title"].fillna("") + ". " + df["body"].fillna("")
    ).str.strip()

    # 1. Split 70% Train dan 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)

    # 2. Split Temp menjadi 50% Val (15% total) dan 50% Test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)

    print("=== HASIL PEMBAGIAN DATASET ===")
    print(f"Total Dataset : {len(df)} artikel")
    print(f"Training set  : {len(train_df)} artikel (70%)")
    print(f"Validation set: {len(val_df)} artikel (15%)")
    print(f"Test set      : {len(test_df)} artikel (15%)")

    # Simpan split data
    train_df.to_csv("data/train.csv", index=False, sep=";")
    val_df.to_csv("data/val.csv", index=False, sep=";")
    test_df.to_csv("data/test.csv", index=False, sep=";")
    print("\nFile train.csv, val.csv, dan test.csv berhasil disimpan di folder 'data/'.")

    # 3. Buat template evaluation_examples.json (10 sampel dari test_df)
    sample_eval = test_df.head(10)
    eval_list = []

    for idx, row in sample_eval.iterrows():
        eval_list.append(
            {
                "article_id": row.get("original_id", str(idx)),
                "title": row["title"],
                "body": row["body"][:300]
                + "...",  # Ambil eksplanasi pendek 300 karakter
                "entities": [],  # Siap diisi annotasi span [text, label, start, end]
            }
        )

    with open("evaluation_examples.json", "w", encoding="utf-8") as f:
        json.dump(eval_list, f, indent=2, ensure_ascii=False)

    print(
        "Template 'evaluation_examples.json' (10 sampel) berhasil dibuat di root proyek.\n"
    )


if __name__ == "__main__":
    split_data("data/merged_articles_clean.csv")