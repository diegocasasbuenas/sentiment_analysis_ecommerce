import pandas as pd
import spacy
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm  # Usamos la versión auto-detectable
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de paths usando pathlib
BASE_DIR = Path("/mnt/c/Users/diego/OneDrive/Documentos/Data science projects/nlp 2/petsentiment_analysis")
DATA_PATH = BASE_DIR / "data/raw/balanced_data.csv"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Mantener caracteres especiales en español
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(file_path: Path) -> pd.DataFrame:
    """Carga datos y realiza limpieza inicial"""
    logger.info("Loading data...")
    
    dtypes = {
        "rating": "float32",
        "text": "string",
        "user_id": "string",
        "asin": "string",
        "timestamp": "string"
    }
    
    data = pd.read_csv(file_path, dtype=dtypes, usecols=["rating", "text", "user_id", "parent_asin", "timestamp"])
    
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    except ValueError:
        data["timestamp"] = pd.to_datetime(data["timestamp"], infer_datetime_format=True, errors="coerce")
        n_errors = data["timestamp"].isna().sum()
        logger.warning(f"Couldn't parse {n_errors} timestamps")
    
    return data.dropna(subset=["timestamp"])

def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Filtra y transforma datos para el análisis"""
    data = data[data["rating"] != 3.0]
    data["Sentiment_target"] = (data["rating"] > 3).astype("int8")
    data["word_count"] = data["text"].str.split().str.len()
    q95 = data["word_count"].quantile(0.95)
    return data[data["word_count"].between(4, q95)]

def setup_spacy() -> spacy.language.Language:
    """Configura spaCy solo para tokenización"""
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "entity_linker", "textcat", "lemmatizer"])
    return nlp

def tokenize_text(nlp: spacy.language.Language, texts: pd.Series) -> pd.Series:
    """Tokeniza el texto usando spaCy después de limpiarlo"""
    processed = []
    progress_bar = tqdm(total=len(texts), desc="Tokenizing texts", unit="doc", dynamic_ncols=True, leave=True)
    
    for text in texts:
        cleaned_text = clean_text(text)
        doc = nlp(cleaned_text)
        tokens = [token.text.lower() for token in doc]
        processed.append(" ".join(tokens))
        progress_bar.update(1)
    
    progress_bar.close()
    return pd.Series(processed, index=texts.index)

def main():
    tqdm.pandas()
    data = load_data(DATA_PATH)
    data = preprocess_dataset(data)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Sentiment_target"])
    
    nlp = setup_spacy()
    logger.info("Tokenizing training data...")
    train_data["processed_text"] = tokenize_text(nlp, train_data["text"])
    
    logger.info("Tokenizing test data...")
    test_data["processed_text"] = tokenize_text(nlp, test_data["text"])
    
    train_data.to_parquet(OUTPUT_DIR / "train_data2.parquet", index=False)
    test_data.to_parquet(OUTPUT_DIR / "test_data2.parquet", index=False)
    logger.info("✅ Datos procesados y guardados exitosamente")

if __name__ == "__main__":
    main()
