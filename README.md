# Assistente Pianificazione Viaggi - Progetto

## Panoramica del Progetto

Sistema intelligente per la pianificazione di viaggi che integra quattro paradigmi dell'Intelligenza Artificiale:

- **Algoritmi di Ricerca**: A*, Floyd-Warshall, Dijkstra
- **Machine Learning**: Predizione e classificazione
- **Ragionamento Probabilistico**: Reti Bayesiane
- **Programmazione Logica**: Knowledge Base Prolog

## Struttura Repository

```
ProgettoIcon2425/
├── data_collection/
│ ├── __init__.py
│ ├── transport_data.py # Grafo 20 città italiane
│ └── weather_integration.py # OpenWeatherMap API
├── search_algorithms/
│ ├── __init__.py
│ └── pathfinder.py # A*, Floyd-Warshall, Dijkstra
├── ml_models/
│ ├── __init__.py
│ ├── dataset_generator.py # Generazione dati sintetici
│ ├── predictor_models.py # Predizione prezzi/tempi
│ ├── preference_classifier.py # Classificazione profili utente
│ └── ml_pathfinder_integration.py # Integrazione ML
├── bayesian_network/
│ ├── __init__.py
│ └── uncertainty_models.py # Rete Bayesiana 6 nodi
├── prolog_kb/
│ ├── __init__.py
│ ├── travel_rules.pl # 200+ regole Prolog
│ └── prolog_interface.py # Python-Prolog bridge
├── evaluation/
│ ├── __init__.py
│ └── comprehensive_evaluation.py # Framework valutazione ICon
├── results/ # Risultati sperimentali
├── esempi_doc/ # Template documentazione
├── teoria_ICON/ # Materiale teorico corso
├── intelligent_travel_planner.py # Sistema principale
├── test_complete_system.py # Test integrazione
├── generate_final_results.py # Generatore risultati
├── context.md # Contesto e progressi
└── Documentazione_Tecnica.md # Documentazione tecnica completa
```

## Componenti Principali

### Data Collection

- **transport_data.py**: Grafo di 20 citta italiane con NetworkX
- **weather_integration.py**: Integrazione API meteo OpenWeatherMap

### Search Algorithms

- **pathfinder.py**: Implementazione A*, Floyd-Warshall, Dijkstra, Multi-objective A*

### Machine Learning

- **dataset_generator.py**: Generazione dataset sintetici con 29 features
- **predictor_models.py**: Modelli predizione prezzi/tempi (Gradient Boosting, Ridge)
- **preference_classifier.py**: Classificazione profili utente (Logistic Regression, Random Forest)
- **ml_pathfinder_integration.py**: Integrazione ML con algoritmi ricerca

### Bayesian Network

- **uncertainty_models.py**: Rete Bayesiana 6 nodi per gestione incertezza

### Prolog Knowledge Base

- **travel_rules.pl**: 200+ regole logiche per vincoli di viaggio
- **prolog_interface.py**: Bridge Python-Prolog per integrazione

### Evaluation

- **comprehensive_evaluation.py**: Framework valutazione compliant con K-fold CV

## Installazione e Utilizzo

### Prerequisiti

```bash
pip install networkx scikit-learn pgmpy numpy pandas
```

### Esecuzione

```bash
# Demo completa del sistema
python intelligent_travel_planner.py --demo

# Test sistema completo
python test_complete_system.py

# Generazione risultati ICon
python generate_final_results.py
```

## Documentazione Tecnica

Per l'analisi dettagliata dell'implementazione, risultati sperimentali e valutazione compliant:

📋 **[Documentazione_Tecnica.md](Documentazione_Tecnica.md)**

## Conformità

✓ Integrazione multi-paradigma (Ricerca + ML + Bayesiana + Logica)
✓ Valutazione quantitativa con cross-validation K-fold
✓ Risultati con media ± deviazione standard
✓ Confronto sistematico algoritmi baseline vs potenziati
✓ Sistema ibrido originale per pianificazione viaggi
