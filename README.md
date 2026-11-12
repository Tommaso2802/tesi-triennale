# Rilevamento di anomalie nel tono emotivo di conversazioni

Codice della tesi di laurea triennale in Informatica di **Tommaso Persello**, Università degli Studi di Udine.

## Panoramica

Il progetto propone un framework che va dalla generazione di conversazioni in stile chat, con tono emotivo stabile o variabile, fino all'individuazione automatica delle conversazioni in cui il tono cambia.

1. Ogni battuta viene classificata in **11 emozioni** con un modello linguistico multilingue.
2. Ogni conversazione diventa una **serie temporale multivariata**, una probabilità per emozione a ogni turno.
3. Due algoritmi **non supervisionati**, Isolation Forest e k-means, assegnano a ogni conversazione un punteggio di anomalia.
4. Le etichette assegnate in fase di generazione (*tono variabile* / *tono stabile*) sono usate **solo per la valutazione**: precision, recall, F1, accuracy, ROC-AUC, average precision e test di Wilcoxon.
