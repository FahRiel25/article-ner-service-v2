import glob
import pandas as pd


def profile_dataset(data_dir: str):
    # Memuat seluruh file CSV Maybank
    file_paths = sorted(glob.glob(f"{data_dir}/Maybank_*.csv"))
    print(f"Ditemukan {len(file_paths)} file CSV.")

    dataframes = []
    for path in file_paths:
        df_item = pd.read_csv(path, sep=";", low_memory=False)
        dataframes.append(df_item)

    df = pd.concat(dataframes, ignore_index=True)

    # Profiling data awal
    total_rows = len(df)
    missing_title = df["title"].isna().sum()
    missing_body = df["body"].isna().sum()
    duplicate_rows = df.duplicated(subset=["title", "body"]).sum()
    lang_dist = df["language"].value_counts(dropna=False).to_dict()

    print("\n=== HASIL PROFILING DATASET ===")
    print(f"Total baris dimuat     : {total_rows}")
    print(f"Judul kosong (missing) : {missing_title}")
    print(f"Bodi kosong (missing)  : {missing_body}")
    print(f"Baris duplikat         : {duplicate_rows}")
    print("\nDistribusi Bahasa:")
    for lang, count in lang_dist.items():
        print(f"  - {lang}: {count} artikel")

    # Simpan dataset gabungan tanpa duplikat
    df_clean = df.drop_duplicates(subset=["title", "body"]).reset_index(
        drop=True
    )
    df_clean.to_csv("data/merged_articles_clean.csv", index=False, sep=";")
    print(
        f"\nDataset bersih berhasil disimpan ke 'data/merged_articles_clean.csv' ({len(df_clean)} baris)."
    )


if __name__ == "__main__":
    profile_dataset("data/raw")