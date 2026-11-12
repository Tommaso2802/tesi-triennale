import pandas as pd
import torch
from transformers import pipeline

FILE_INPUT = "Dataset_finale.csv"   
MODEL_NAME = "tabularisai/multilingual-emotion-classification"

df = pd.read_csv(FILE_INPUT, sep=";", encoding="utf-8-sig")

# Pulizia: il modello accetta solo stringhe non vuote
df["text"] = df["text"].fillna("").astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

# Classificazione emotiva di ogni battuta
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1,
)
risultati = classifier(df["text"].tolist(), batch_size=32, truncation=True)

scores = pd.DataFrame([{r["label"]: r["score"] for r in ris} for ris in risultati])
df = pd.concat([df, scores], axis=1)

df.to_csv("emotions_battute.csv", sep=";", index=False, encoding="utf-8")
