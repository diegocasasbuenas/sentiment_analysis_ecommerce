import torch
import re
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import Dict, Optional

# Cambiar directorio de trabajo (si es necesario)
os.chdir("/mnt/c/Users/diego/OneDrive/Documentos/Data science projects/nlp 2/petsentiment_analysis/models")

# Configuración de FastAPI
app = FastAPI(title="Pet Sentiment Analysis API", description="API para análisis de sentimientos en reseñas de productos para mascotas.")

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Dispositivo en uso: {device}")

# Cargar tokenizer y modelo entrenado
try:
    tokenizer = AutoTokenizer.from_pretrained("sentiment_transformer_model")
    model = AutoModelForSequenceClassification.from_pretrained("sentiment_transformer_model").to(device)
    model.eval()  # Poner el modelo en modo evaluación
    print("✅ Modelo y tokenizer cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    raise

# Clase para recibir datos en la API
class TextInput(BaseModel):
    text: str

# Función para limpiar texto
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\sáéíóúñü]", "", text)  # Mantener caracteres especiales en español
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]  # Limitar longitud máxima

# Función para predecir sentimiento
def predict_sentiment(text: str) -> Dict[str, str]:
    try:
        text = clean_text(text)
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        prediction = torch.argmax(outputs.logits, axis=1).cpu().item()
        sentiment_map = {0: "Negativo", 1: "Positivo"}
        return {"sentiment": sentiment_map.get(prediction, "Desconocido")}
    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la solicitud.")

# Endpoint de la API
@app.post("/predict", response_model=Dict[str, str])
def get_prediction(input_data: TextInput) -> Dict[str, str]:
    """
    Endpoint para predecir el sentimiento de un texto.
    - **text**: Texto a analizar (máximo 512 caracteres).
    """
    try:
        return predict_sentiment(input_data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar con: uvicorn nombre_archivo:app --reload