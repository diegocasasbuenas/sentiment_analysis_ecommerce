import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
import os

os.chdir("/mnt/c/Users/diego/OneDrive/Documentos/Data science projects/nlp 2/petsentiment_analysis")

data = pd.read_parquet("data/processed/train_data2.parquet")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset and take a sample
df_sample = data.sample(n=100000, random_state=42).reset_index(drop=True)

df_sample["cleaned_text"] = df_sample["text"].apply(clean_text)

# Convert Sentiment_target to integer
df_sample["labels"] = df_sample["Sentiment_target"].astype(int)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

dataset = Dataset.from_pandas(df_sample[["cleaned_text", "labels"]])
dataset = dataset.map(tokenize_function, batched=True)

# Split data into train and test sets
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Define model for multi-class classification
num_labels = len(df_sample["labels"].unique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Model evaluation
y_pred = trainer.predict(test_dataset).predictions.argmax(axis=1)
y_true = test_dataset["labels"]

print(f"\nBest test accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred))

# Save final model
model.save_pretrained("sentiment_transformer_model")
tokenizer.save_pretrained("sentiment_transformer_model")
print("Training successfully completed!")
