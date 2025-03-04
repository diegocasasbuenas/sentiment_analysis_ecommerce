import pandas as pd
import spacy
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm  
import logging

# Configure logging for tracking progress and potential issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory and file paths
BASE_DIR = Path("/mnt/c/Users/diego/OneDrive/Documentos/Data science projects/nlp 2/petsentiment_analysis")
DATA_PATH = BASE_DIR / "data/raw/balanced_data.csv"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

def clean_text(text):
    """Clean text by converting to lowercase, removing special characters, and stripping extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)  # Remove non-alphanumeric characters except spaces and apostrophes
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file and perform initial cleaning."""
    logger.info("Loading data...")

    # Define data types for efficiency
    dtypes = {
        "rating": "float32",
        "text": "string",
        "user_id": "string",
        "asin": "string",
        "timestamp": "string"
    }
    
    # Load the dataset with selected columns
    data = pd.read_csv(file_path, dtype=dtypes, usecols=["rating", "text", "user_id", "parent_asin", "timestamp"])
    
    # Convert timestamp column to datetime format
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    except ValueError:
        # If parsing fails, infer datetime format and handle errors
        data["timestamp"] = pd.to_datetime(data["timestamp"], infer_datetime_format=True, errors="coerce")
        n_errors = data["timestamp"].isna().sum()
        logger.warning(f"Couldn't parse {n_errors} timestamps")
    
    # Drop rows where timestamp could not be converted
    return data.dropna(subset=["timestamp"])

def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Filter and transform dataset for sentiment analysis."""
    # Remove neutral reviews (rating = 3)
    data = data[data["rating"] != 3.0]

    # Assign sentiment labels: 1 (positive) for rating > 3, 0 (negative) otherwise
    data["Sentiment_target"] = (data["rating"] > 3).astype("int8")

    # Compute word count for each review
    data["word_count"] = data["text"].str.split().str.len()

    # Set an upper threshold for word count (95th percentile) and filter out very short or long reviews
    q95 = data["word_count"].quantile(0.95)
    return data[data["word_count"].between(4, q95)]

def setup_spacy() -> spacy.language.Language:
    """Load spaCy model with only the tokenizer enabled."""
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "entity_linker", "textcat", "lemmatizer"])
    return nlp

def tokenize_text(nlp: spacy.language.Language, texts: pd.Series) -> pd.Series:
    """Tokenize text using spaCy after cleaning it."""
    processed = []
    progress_bar = tqdm(total=len(texts), desc="Tokenizing texts", unit="doc", dynamic_ncols=True, leave=True)
    
    for text in texts:
        cleaned_text = clean_text(text)  # Clean text before tokenization
        doc = nlp(cleaned_text)  # Tokenize text with spaCy
        tokens = [token.text.lower() for token in doc]  # Convert tokens to lowercase
        processed.append(" ".join(tokens))  # Join tokens into a single string
        progress_bar.update(1)  # Update progress bar
    
    progress_bar.close()
    return pd.Series(processed, index=texts.index)

def main():
    """Main pipeline for loading, processing, tokenizing, and saving data."""
    tqdm.pandas()  # Enable pandas integration for tqdm progress bars

    # Load and preprocess data
    data = load_data(DATA_PATH)
    data = preprocess_dataset(data)
    
    # Split data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Sentiment_target"])
    
    # Load spaCy model for tokenization
    nlp = setup_spacy()

    # Tokenize training data
    logger.info("Tokenizing training data...")
    train_data["processed_text"] = tokenize_text(nlp, train_data["text"])
    
    # Tokenize test data
    logger.info("Tokenizing test data...")
    test_data["processed_text"] = tokenize_text(nlp, test_data["text"])
    
    # Save processed data as Parquet files for efficient storage
    train_data.to_parquet(OUTPUT_DIR / "train_data2.parquet", index=False)
    test_data.to_parquet(OUTPUT_DIR / "test_data2.parquet", index=False)
    logger.info("Processed data successfully saved.")

# Run the script when executed
if __name__ == "__main__":
    main()
