import pandas as pd
import glob
import os

def run_profiling():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if not csv_files:
        print("Peringatan: Tidak ada file CSV ditemukan di data/raw/")
        return

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', on_bad_lines='skip')
            df_list.append(df)
        except Exception as e:
            print(f"Gagal memuat {file}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    
    print("=== TAHAP 1: PROFILING DATA ===")
    print(f"Total baris dimuat         : {len(full_df)}")
    print(f"Jumlah baris duplikat      : {full_df.duplicated(subset=['title', 'body']).sum()}")
    print(f"Missing Title              : {full_df['title'].isnull().sum()}")
    print(f"Missing Body               : {full_df['body'].isnull().sum()}")
    if 'language' in full_df.columns:
        print("\nDistribusi Bahasa:")
        print(full_df['language'].value_counts(dropna=False))

    # Pembersihan
    clean_df = full_df.dropna(subset=['title', 'body'], how='all').copy()
    clean_df['title'] = clean_df['title'].fillna('')
    clean_df['body'] = clean_df['body'].fillna('')
    clean_df['full_text'] = clean_df['title'] + ". " + clean_df['body']
    clean_df = clean_df.drop_duplicates(subset=['full_text'])

    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'cleaned_articles.csv')
    clean_df.to_csv(output_path, index=False)
    
    print(f"\nData bersih ({len(clean_df)} baris) disimpan ke: {output_path}\n")

if __name__ == "__main__":
    run_profiling()