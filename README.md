# 🧠 Qtsumego — Quantum Go Life Analyzer
### Born Machine Variational Quantum Algorithm for Life-and-Death in Go

![Logo](docs/logo.svg)

Qtsumego è un framework ibrido **quantum–classical** per analizzare problemi di **vita/morte nel Go** utilizzando:

- Born Machines (circuiti quantistici parametrizzati)
- Board classica con regole di cattura complete
- Valutazione quantistica del valore atteso di vita
- Ottimizzazione variational (gradient-free)

Il progetto esplora un nuovo paradigma:  
**modellare la lettura locale nel Go come inferenza quantistica**, non come ricerca.

---

## 🚀 Funzionalità principali

- ✔ Board 9×9 con catture, gruppi, libertà
- ✔ Generazione automatica delle mosse candidate
- ✔ Born Machine parametrica (RealAmplitudes)
- ✔ Valore atteso quantistico della vita del gruppo
- ✔ Ottimizzazione dei parametri θ
- ✔ Architettura modulare e estendibile
- ✔ Paper scientifico incluso (LaTeX)

---

## 📦 Installazione

```bash
git clone https://github.com/AlbertoVari/Qtsumego.git
cd Qtsumego
pip install -r requirements.txt
