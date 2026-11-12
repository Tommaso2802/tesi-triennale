import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ShuffleSplit

QUOTA_ANOMALIE = 0.10   # 10% di anomale, cioè soglia al 90° percentile come nel k-means
RIPETIZIONI = 50
MAX_SAMPLES = 128       # conversazioni per albero
N_ULTIME = 12
FILE_INPUT = "serie_temporale_valori_emotivi.csv"   # per usare un altro file basta cambiare qui

df = pd.read_csv(FILE_INPUT, sep=None, engine="python", encoding="utf-8-sig")  # separatore riconosciuto da solo
emozioni = df.columns.drop(["conversation_id", "turno"]).tolist()

# Feature di ogni conversazione 
g = df.groupby("conversation_id", sort=False)[emozioni]
diff = g.diff()                                                  # x_t - x_(t-1), vuoto alla prima battuta
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


def foresta(seme):
    # contamination fissa la soglia al 90° percentile dei punteggi di addestramento
    return IsolationForest(n_estimators=300, max_samples=MAX_SAMPLES,
                           contamination=QUOTA_ANOMALIE, random_state=seme)


# Monte Carlo: 50 divisioni casuali 80/20 (le stesse del k-means)
divisioni = ShuffleSplit(n_splits=RIPETIZIONI, test_size=0.2, random_state=42)
righe = []
for r, (train, val) in enumerate(divisioni.split(X)):
    modello = foresta(r).fit(X.iloc[train])
    righe.append(pd.DataFrame({"divisione": r, "conversation_id": X.index[val],
                               "punteggio": -modello.score_samples(X.iloc[val]),   # più alto = più anomala
                               "anomala": (modello.predict(X.iloc[val]) == -1).astype(int)}))

# Punteggi di validazione, da passare a valutazione.py
pd.concat(righe).to_csv("punteggi_isolation_forest.csv", sep=";", index=False, encoding="utf-8")

# Modello finale su tutte le conversazioni
iso = foresta(42).fit(X)
anomala = pd.Series(iso.predict(X) == -1, index=X.index, name="anomala")
anomala.map({True: "yes", False: "no"}).to_csv("anomalie_isolation_forest.csv", sep=";", encoding="utf-8")
print(f"{anomala.sum()} conversazioni anomale su {len(X)}")
