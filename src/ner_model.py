import pandas as pd
import spacy
import os
from sklearn.model_selection import train_test_split

class NERModel:
    def __init__(self, model_name="en_core_web_sm"):
        # Memuat model NER bawaan spaCy sebagai baseline
        print(f"Memuat model NER: {model_name}...")
        self.nlp = spacy.load(model_name)
        self.supported_labels = self.nlp.get_pipe("ner").labels

    def extract_entities(self, text):
        """
        Mengekstrak entitas dari teks yang diberikan.
        Mengembalikan daftar dictionary berisi text, label, start, end.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            # Filter hanya label utama sesuai instruksi minimal (PERSON, ORG, GPE/LOC, DATE)
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities

def split_and_prepare_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    data_path = os.path.join(processed_dir, 'cleaned_articles.csv')
    
    # Muat data bersih
    df = pd.read_csv(data_path)
    
    # Filter hanya artikel bahasa Inggris (atau yang terdeteksi 'en) untuk konsistensi model
    df_en = df[df['language'].str.contains('en', na=False)].copy()
    
    # Split Data: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(df_en, test_size=0.30, random_state=42)
    # Split Temp: 50% Val (15% total), 50% Test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # Simpan hasil split
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    print(f"Data Split Selesai (Seed: 42):")
    print(f" - Train: {len(train_df)} artikel")
    print(f" - Validation: {len(val_df)} artikel")
    print(f" - Test: {len(test_df)} artikel")
    
    # Uji Coba Baseline NER pada 1 sampel dari test data
    print("\n--- Uji Coba Baseline Model ---")
    ner = NERModel()
    sample_text = test_df.iloc[0]['full_text']
    print(f"Teks Sampel: {sample_text[:100]}...\n")
    
    extracted = ner.extract_entities(sample_text)
    for ent in extracted:
        print(f"[{ent['label']}] {ent['text']} (Offset: {ent['start']}-{ent['end']})")

if __name__ == "__main__":
    split_and_prepare_data()