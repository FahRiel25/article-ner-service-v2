import pandas as pd
from sklearn.model_selection import train_test_split
import os

def run_data_split():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    data_path = os.path.join(processed_dir, 'cleaned_articles.csv')
    
    if not os.path.exists(data_path):
        print("Error: cleaned_articles.csv belum ada. Jalankan profiling.py terlebih dahulu.")
        return

    df = pd.read_csv(data_path)
    # Memfilter artikel bahasa Inggris sesuai dengan kemampuan model baseline
    df_en = df[df['language'].str.contains('en', na=False)].copy()
    
    # Split 70% Train, 15% Val, 15% Test (seed=42)
    train_df, temp_df = train_test_split(df_en, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # Simpan CSV
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Buat ringkasan teks untuk laporan
    summary_text = f"""==================================================
LAPORAN PEMBAGIAN DATA (DATA SPLIT SUMMARY)
==================================================
Random Seed        : 42
Total Data Bersih  : {len(df_en)} artikel (Bahasa Inggris)

1. Training Set (70%):
   - Jumlah Dokumen: {len(train_df)} baris
   - File Path     : data/processed/train.csv

2. Validation Set (15%):
   - Jumlah Dokumen: {len(val_df)} baris
   - File Path     : data/processed/val.csv

3. Test Set (15%):
   - Jumlah Dokumen: {len(test_df)} baris
   - File Path     : data/processed/test.csv
==================================================
"""
    print("=== TAHAP 2: PEMBAGIAN DATA (DATA SPLIT) ===")
    print(summary_text)
    
    summary_file = os.path.join(processed_dir, 'split_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"File ringkasan pembagian data berhasil dibuat di: {summary_file}\n")

if __name__ == "__main__":
    run_data_split()