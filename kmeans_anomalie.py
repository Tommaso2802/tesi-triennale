import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ShuffleSplit

PERCENTILE_SOGLIA = 90   # anomala se la distanza supera il 90° percentile
RIPETIZIONI = 50
N_ULTIME = 12

df = pd.read_csv("serie_temporale_valori_emotivi.csv", sep=";", encoding="utf-8-sig")
emozioni = df.columns.drop(["conversation_id", "turno"]).tolist()

# Feature di ogni conversazione
g = df.groupby("conversation_id", sort=False)[emozioni]
diff = g.diff()                 # x_t - x_(t-1), vuoto alla prima battuta
dalla_fine = df.groupby("conversation_id").cumcount(ascending=False).to_numpy()  
diff_ultime = diff.fillna(0).set_index(df["conversation_id"])
medie = g.mean()

X = pd.concat({
    "media": medie,
    "std": g.std(),
    "delta": g.last() - g.first(),
    "salto_max": diff.abs().groupby(df["conversation_id"], sort=False).max(),
    # differenza con la battuta precedente nelle ultime 12 battute (diff_12 = ultima)
    **{f"diff_{p:02d}": diff_ultime[dalla_fine == N_ULTIME - p].reindex(medie.index, fill_value=0)
       for p in range(1, N_ULTIME + 1)},
}, axis=1)
X.columns = [f"{e}_{s}" for s, e in X.columns]

# Scelta di K con la silhouette
silhouette = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X))
              for k in range(2, 11)}
K = max(silhouette, key=silhouette.get)
print(f"K = {K} (silhouette {silhouette[K]:.3f})")


def addestra(X_train, seme):
    km = KMeans(n_clusters=K, n_init=10, random_state=seme).fit(X_train)
    distanza = lambda D: km.transform(D).min(axis=1)   # distanza dal centroide più vicino
    soglia = np.percentile(distanza(X_train), PERCENTILE_SOGLIA)
    return distanza, soglia


# Monte Carlo: 50 divisioni casuali 80/20
divisioni = ShuffleSplit(n_splits=RIPETIZIONI, test_size=0.2, random_state=42)
righe = []
for r, (train, val) in enumerate(divisioni.split(X)):
    distanza, soglia = addestra(X.iloc[train], r)
    punteggio = distanza(X.iloc[val])
    righe.append(pd.DataFrame({"divisione": r, "conversation_id": X.index[val],
                               "punteggio": punteggio, "anomala": (punteggio > soglia).astype(int)}))

# Punteggi di validazione, da passare a valutazione.py
pd.concat(righe).to_csv("punteggi_kmeans.csv", sep=";", index=False, encoding="utf-8")

# Modello finale su tutte le conversazioni
distanza, soglia = addestra(X, 42)
anomala = pd.Series(distanza(X) > soglia, index=X.index, name="anomala")
anomala.map({True: "yes", False: "no"}).to_csv("anomalie_kmeans.csv", sep=";", encoding="utf-8")
print(f"{anomala.sum()} conversazioni anomale su {len(X)}")
