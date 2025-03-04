### FastText model ###
import torch
import umap
import plotly.express as px
import pandas as pd
import re
import joblib
import fasttext
import random
from sklearn.model_selection import train_test_split
import os

os.chdir("/mnt/c/Users/diego/OneDrive/Documentos/Data science projects/nlp 2/petsentiment_analysis")

data = pd.read_parquet("data/processed/train_data1.parquet")

# Load dataset and take a sample
df_sample = data.sample(n=50000, random_state=42).reset_index(drop=True)

# Split into train and test sets
train_data, test_data = train_test_split(df_sample, test_size=0.2, random_state=42)

# Save data in text files for FastText
def save_fasttext_format(df, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for text, label in zip(df["processed_text"], df["Sentiment_target"]):
            f.write(f"__label__{label} {text}\n")

save_fasttext_format(train_data, "train.txt")
save_fasttext_format(test_data, "test.txt")

# Function to evaluate FastText models
def evaluate_fasttext(model, test_file):
    results = model.test(test_file)
    return results[1]  # Accuracy

# Define hyperparameter search space
param_grid = {
    "epoch": [10, 20, 30, 40, 50], 
    "lr": [0.1, 0.3, 0.5, 0.7, 1.0], 
    "wordNgrams": [1, 2, 3],
    "dim": [50, 100, 150, 200, 300],
    "loss": ["softmax", "hs", "ns"]
}

# Random search
num_iterations = 20
best_model = None
best_acc = 0
best_params = {}

for i in range(num_iterations):
    print(f"\n🔹 Iteration {i+1}/{num_iterations}")

    # Select random parameters
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print(f"Testing parameters: {params}")

    # Train model with selected parameters
    model = fasttext.train_supervised(
        input="train.txt",
        epoch=params["epoch"],
        lr=params["lr"],
        wordNgrams=params["wordNgrams"],
        dim=params["dim"],
        loss=params["loss"]
    )

    # Evaluate model
    acc = evaluate_fasttext(model, "test.txt")
    print(f"Accuracy obtained: {acc:.4f}")

    # Save the best model
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_params = params

# Save the best model found
if best_model:
    model_path = os.path.join("models", "fasttext_model.bin")
    best_model.save_model(model_path)
    print("\nBest model saved with parameters:")
    print(best_params)
    print(f"Best accuracy: {best_acc:.4f}")
