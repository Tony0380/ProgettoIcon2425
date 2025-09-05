# Mapping Argomenti Progetto ↔ Teoria Ingegneria della Conoscenza

## 📋 Panoramica

Questo documento fornisce una mappatura precisa tra gli argomenti implementati nel progetto **Assistente Pianificazione Viaggi Intelligente** e il materiale teorico del corso di Ingegneria della Conoscenza.

---

## 🔍 **ARGOMENTO 1: Algoritmi di Ricerca**

### **Implementazione nel Progetto**
- **File principale**: `search_algorithms/pathfinder.py`
- **Algoritmi implementati**:
  - A* con euristica haversine per ottimizzazione multi-criterio
  - Floyd-Warshall per calcolo di tutte le distanze minime
  - Multi-objective A* con weighted sum approach
  - Beam Search con pruning euristico

### **📚 PDF di Riferimento**
**`2. Ricerca di Soluzioni in Spazi di Stati.pdf`**

### **📖 Argomenti Specifici da Studiare**
- **Capitolo 3**: Spazi di stati e formulazione problemi
  - Rappresentazione nodi e archi
  - Funzioni di costo e euristica
- **Sezione 3.5**: Ricerca informata - Algoritmo A*
  - Funzioni f(n) = g(n) + h(n)
  - Proprietà di ammissibilità e ottimalità
  - Euristiche consistenti
- **Sezione 3.6**: Algoritmi per grafi
  - Dijkstra per shortest path
  - Gestione grafi pesati
- **Appendice**: Algoritmi di programmazione dinamica
  - Floyd-Warshall per all-pairs shortest path

---

## 🤖 **ARGOMENTO 2: Machine Learning**

### **Implementazione nel Progetto**
- **File principali**: 
  - `ml_models/predictor_models.py` (predizione prezzi/tempi)
  - `ml_models/preference_classifier.py` (classificazione profili utente)
  - `ml_models/dataset_generator.py` (generazione dataset sintetici)
- **Modelli implementati**:
  - Gradient Boosting Regressor per predizione prezzi
  - Logistic Regression per classificazione profili
  - Random Forest come modello ensemble
  - K-fold Cross-Validation per valutazione rigorosa

### **📚 PDF di Riferimento**
**`7. Apprendimento Supervisionato.pdf`**

### **📖 Argomenti Specifici da Studiare**
- **Sezione 7.1**: Fondamenti apprendimento supervisionato
  - Bias induttivo e trade-off bias-varianza
  - Overfitting e underfitting
- **Sezione 7.3**: Modelli lineari
  - Regressione lineare e Ridge regression
  - Regressione logistica e regolarizzazione
- **Sezione 7.4**: Ensemble methods
  - Bagging e Random Forest
  - Boosting e Gradient Boosting
- **Sezione 7.6**: Valutazione modelli
  - Cross-validation e metriche di performance
  - Curve ROC e precision-recall

---

## 🎲 **ARGOMENTO 3: Reti Bayesiane**

### **Implementazione nel Progetto**
- **File principale**: `bayesian_network/uncertainty_models.py`
- **Implementazione**:
  - Rete Bayesiana a 6 nodi per gestione incertezza
  - Conditional Probability Tables (CPTs) strutturate
  - Variable Elimination per inferenza esatta
  - Likelihood Weighting per inferenza approssimata

### **📚 PDF di Riferimento**
**`9. Ragionamento su Modelli di Conoscenza Incerta.pdf`**

### **📖 Argomenti Specifici da Studiare**
- **Sezione 9.1**: Teoria della probabilità
  - Probabilità condizionate e regola di Bayes
  - Indipendenza e indipendenza condizionale
- **Sezione 9.3**: Reti Bayesiane
  - Struttura DAG e conditional independence
  - Costruzione reti e definizione CPTs
- **Sezione 9.4**: Algoritmi di inferenza
  - Variable Elimination Algorithm
  - Complessità computazionale dell'inferenza
- **Sezione 9.5**: Metodi di campionamento
  - Forward Sampling e Likelihood Weighting
  - Gibbs Sampling e MCMC

---

## 🧠 **ARGOMENTO 4: Programmazione Logica**

### **Implementazione nel Progetto**
- **File principali**:
  - `prolog_kb/prolog_interface.py` (interfaccia Python-Prolog)
  - `prolog_kb/travel_rules.pl` (knowledge base con 200+ regole)
- **Implementazione**:
  - Predicati first-order per vincoli di viaggio
  - Regole per constraint satisfaction
  - SLD Resolution con backtracking
  - Unificazione per pattern matching

### **📚 PDF di Riferimento**
**`5. Rappresentazione e Ragionamento Relazionale.pdf`**

### **📖 Argomenti Specifici da Studiare**
- **Sezione 5.1**: Logica del primo ordine
  - Sintassi: termini, formule, quantificatori
  - Semantica: interpretazioni e modelli
- **Sezione 5.3**: Programmazione logica e Prolog
  - Clausole di Horn e regole definite
  - SLD Resolution e procedura di dimostrazione
- **Sezione 5.4**: Unificazione
  - Algoritmo di unificazione
  - Most General Unifier (MGU)
- **Sezione 5.5**: Constraint Logic Programming
  - Constraint Satisfaction Problems
  - Propagazione vincoli e backtracking

---

## 🔗 **INTEGRAZIONE MULTI-PARADIGMA**

### **Implementazione nel Progetto**
- **File principale**: `intelligent_travel_planner.py`
- **Architettura ibrida**:
  - Orchestratore centrale per coordinamento paradigmi
  - Fusion decisionale con pesi adattivi
  - Cross-validation tra diversi approcci

### **📚 PDF di Riferimento Aggiuntivi**

#### **Per Constraint Satisfaction:**
**`3. Ragionamento con Vincoli.pdf`**
- **Sezione 3.2**: CSP e tecniche di risoluzione
- **Sezione 3.4**: Consistency checking e propagazione

#### **Per Knowledge Graphs:**
**`6. Knowledge Graph e Ontologie.pdf`**
- **Sezione 6.1**: Triple RDF e rappresentazione conoscenza
- **Sezione 6.3**: Ontologie e logiche descrittive

#### **Per Sistemi Ibridi:**
**`1. Ingegneria della Conoscenza.pdf`**
- **Sezione 1.3**: Architetture sistemi intelligenti
- **Sezione 1.4**: Dimensioni spazio progettazione

---

## 📊 **VALUTAZIONE E METODOLOGIA**

### **Implementazione nel Progetto**
- **File principale**: `evaluation/comprehensive_evaluation.py`
- **Framework di valutazione**:
  - K-fold Cross-Validation (k=5)
  - Metriche quantitative con media ± deviazione standard
  - Confronto sistematico baseline vs enhanced
  - Statistical significance testing

### **📚 PDF di Riferimento**
**`7. Apprendimento Supervisionato.pdf`** - **Sezione 7.6**: Metodologie di valutazione

### **📖 Concetti Chiave**
- Cross-validation e generalization error
- Metriche di classificazione (accuracy, precision, recall, F1)
- Metriche di regressione (MAE, RMSE, R²)
- Statistical testing e confidence intervals

---

## 🎯 **TABELLA RIASSUNTIVA**

| **Argomento Progetto** | **PDF Teoria** | **Sezioni Principali** | **File Implementazione** |
|------------------------|----------------|------------------------|---------------------------|
| **Algoritmi di Ricerca** | `2. Ricerca di Soluzioni in Spazi di Stati.pdf` | 3.5 (A*), 3.6 (Dijkstra), Appendice (Floyd-Warshall) | `search_algorithms/pathfinder.py` |
| **Machine Learning** | `7. Apprendimento Supervisionato.pdf` | 7.1 (Fondamenti), 7.3 (Modelli lineari), 7.4 (Ensemble) | `ml_models/*.py` |
| **Reti Bayesiane** | `9. Ragionamento su Modelli di Conoscenza Incerta.pdf` | 9.1 (Probabilità), 9.3 (Reti), 9.4 (Inferenza) | `bayesian_network/uncertainty_models.py` |
| **Programmazione Logica** | `5. Rappresentazione e Ragionamento Relazionale.pdf` | 5.1 (FOL), 5.3 (Prolog), 5.4 (Unificazione) | `prolog_kb/*.py`, `prolog_kb/*.pl` |
| **Constraint Satisfaction** | `3. Ragionamento con Vincoli.pdf` | 3.2 (CSP), 3.4 (Consistency) | Integrato in tutti i moduli |
| **Sistemi Intelligenti** | `1. Ingegneria della Conoscenza.pdf` | 1.3 (Architetture), 1.4 (Design) | `intelligent_travel_planner.py` |

---

## 💡 **Suggerimenti per lo Studio**

### **Sequenza di Studio Consigliata**
1. **Prima** → `1. Ingegneria della Conoscenza.pdf` (panoramica generale)
2. **Poi** → PDF specifici per ogni argomento implementato
3. **Infine** → `6. Knowledge Graph e Ontologie.pdf` (per integrazione)

### **Focus per l'Esame**
- **Comprendere le motivazioni** dietro la scelta di ogni paradigma
- **Saper spiegare l'integrazione** tra i diversi approcci
- **Conoscere vantaggi e limitazioni** di ogni metodologia
- **Padroneggiare le metriche** di valutazione utilizzate

### **Domande Tipiche**
- "Perché hai scelto A* invece di Dijkstra per questo problema?"
- "Come gestisci l'incertezza nella pianificazione viaggi?"
- "Qual è il vantaggio dell'approccio ibrido rispetto ai singoli paradigmi?"
- "Come valuti la qualità del tuo sistema?"

---

**📝 Nota**: Questo mapping è specifico per il progetto di pianificazione viaggi. Per altri progetti, gli argomenti teorici rimangono gli stessi ma le implementazioni possono variare.