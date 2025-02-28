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


# Cargar dataset y tomar una muestra
df_sample = data.sample(n=150000, random_state=42).reset_index(drop=True)

# Función para limpiar texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"http\S+|www\S+|@\w+|\d+", "", text)  # Eliminar URLs, menciones y números
    text = re.sub(r"[^a-zA-Záéíóúñü\s]", "", text)  # Eliminar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # Eliminar espacios extras
    return text

df_sample["cleaned_text"] = df_sample["text"].apply(clean_text)

# Separar en train y test
train_data, test_data = train_test_split(df_sample, test_size=0.2, random_state=42)

# Guardar los datos en archivos de texto para FastText
def save_fasttext_format(df, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for text, label in zip(df["cleaned_text"], df["Sentiment_target"]):
            f.write(f"__label__{label} {text}\n")

save_fasttext_format(train_data, "train.txt")
save_fasttext_format(test_data, "test.txt")

# Función para evaluar modelos FastText
def evaluate_fasttext(model, test_file):
    results = model.test(test_file)
    return results[1]  # Accuracy

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    "epoch": [10, 20, 30, 40, 50], 
    "lr": [0.1, 0.3, 0.5, 0.7, 1.0], 
    "wordNgrams": [1, 2, 3],
    "dim": [50, 100, 150, 200, 300],
    "loss": ["softmax", "hs", "ns"]
}

# Búsqueda aleatoria
num_iterations = 10
best_model = None
best_acc = 0
best_params = {}

for i in range(num_iterations):
    print(f"\n🔹 Iteración {i+1}/{num_iterations}")

    # Seleccionar parámetros aleatorios
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print(f"Probando parámetros: {params}")

    # Entrenar modelo con los parámetros seleccionados
    model = fasttext.train_supervised(
        input="train.txt",
        epoch=params["epoch"],
        lr=params["lr"],
        wordNgrams=params["wordNgrams"],
        dim=params["dim"],
        loss=params["loss"]
    )

    # Evaluar modelo
    acc = evaluate_fasttext(model, "test.txt")
    print(f"Accuracy obtenido: {acc:.4f}")

    # Guardar el mejor modelo
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_params = params

# Guardar el mejor modelo encontrado
if best_model:
    best_model.save_model("best_fasttext_model.bin")
    print("\n✅ Mejor modelo guardado con parámetros:")
    print(best_params)
    print(f"🎯 Mejor accuracy: {best_acc:.4f}")