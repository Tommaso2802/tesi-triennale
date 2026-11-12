import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ALGORITMI = {"kmeans": "punteggi_kmeans.csv", "isolation_forest": "punteggi_isolation_forest.csv"}

# Etichette: 1 = tono variabile, 0 = tono stabile
etichette = pd.read_csv("etichette_conversazioni.csv", sep=";", encoding="utf-8-sig")
y = etichette.set_index("conversation_id")["etichetta"].eq("variabile").astype(int)
print(f"Conversazioni a tono variabile")


def metriche(d):
    return {
        "accuracy": accuracy_score(d["y"], d["anomala"]),
        "precision": precision_score(d["y"], d["anomala"]),
        "recall": recall_score(d["y"], d["anomala"]),
        "F1": f1_score(d["y"], d["anomala"]),
        "ROC-AUC": roc_auc_score(d["y"], d["punteggio"]),                     # area sotto la curva ROC
        "average precision": average_precision_score(d["y"], d["punteggio"]),   # riassume la curva precision-recall
    }


# Metriche di ogni algoritmo in ciascuna delle 50 divisioni
risultati, coppie = {}, {}
for nome, file in ALGORITMI.items():
    p = pd.read_csv(file, sep=";", encoding="utf-8-sig")
    p["y"] = p["conversation_id"].map(y)
    if p["y"].isna().any():
        raise ValueError(f"{file}: ci sono conversazioni senza etichetta")
    risultati[nome] = pd.DataFrame([metriche(d) for _, d in p.groupby("divisione")])
    coppie[nome] = set(zip(p["divisione"], p["conversation_id"]))

# Media ± deviazione standard sulle 50 divisioni
tabella = pd.DataFrame({nome: m.mean().map("{:.3f}".format) + " ± " + m.std().map("{:.3f}".format)
                        for nome, m in risultati.items()})
print(tabella)
tabella.to_csv("metriche.csv", sep=";", encoding="utf-8-sig")

# Test di Wilcoxon: per ogni metrica, i 50 valori dei due algoritmi confrontati divisione per divisione
a, b = risultati["kmeans"], risultati["isolation_forest"]
test = pd.DataFrame({m: wilcoxon(a[m], b[m]) for m in a}, index=["W", "p-value"]).T
test["(p < 0.05)"] = test["p-value"] < 0.05
print("\nTest di Wilcoxon: kmeans contro isolation_forest")
print(test.to_string(formatters={"W": "{:.0f}".format, "p-value": "{:.2e}".format}))
test.to_csv("wilcoxon.csv", sep=";", encoding="utf-8-sig")
