import pandas as pd

FILE_INPUT = "emotions_battute.csv"   

df = pd.read_csv(FILE_INPUT, sep=";", encoding="utf-8-sig")

df["turno"] = df.groupby("conversation_id").cumcount() + 1

# Le colonne delle emozioni sono tutte quelle che non descrivono la battuta
emozioni = df.columns.drop(["id", "conversation_id", "speaker", "text", "turno"]).tolist()

df[["conversation_id", "turno"] + emozioni].to_csv("serie_temporale_valori_emotivi.csv", sep=";", index=False, encoding="utf-8")
