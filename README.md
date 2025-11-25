# 🌌 Total Perspective Vortex — EEG Brain-Computer Interface (BCI)

<div align="center">

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/github/license/raveriss/Total_Perspective_Vortex)
[![CI](https://github.com/raveriss/Total_Perspective_Vortex/actions/workflows/ci.yml/badge.svg?branch=main)]()
![lint](https://img.shields.io/badge/lint-ruff%20✔-yellow)
![mypy](https://img.shields.io/badge/mypy-checked-purple)
[![Mutation](https://img.shields.io/badge/mutmut-≥90%25-orange.svg)]()
[![codecov](https://codecov.io/github/raveriss/Total_Perspective_Vortex/graph/badge.svg?token=LSR1U908CU)](https://codecov.io/github/raveriss/Total_Perspective_Vortex)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?label=pre--commit)]()
![sklearn](https://img.shields.io/badge/scikit--learn-pipeline-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![Security](https://img.shields.io/badge/security-bandit-green.svg)]()
![mne](https://img.shields.io/badge/MNE-EEG%20Analysis-orange)
![numpy](https://img.shields.io/badge/numpy-math%20core-blue)
![pandas](https://img.shields.io/badge/pandas-data%20analysis-green)

</div>

---
## 📑 Table des matières

- [🌌 Total Perspective Vortex — EEG Brain-Computer Interface (BCI)](#total-perspective-vortex--eeg-brain-computer-interface-bci)
- [📌 Overview](#overview)
- [🧠 Objectifs pédagogiques (42 / IA / ML)](#objectifs-pédagogiques-42--ia--ml)
- [🧩 Architecture du projet](#architecture-du-projet)
- [🔬 1. Préprocessing & parsing EEG (MNE)](#1-préprocessing--parsing-eeg-mne)
- [🎛️ 2. Extraction de features](#2-extraction-de-features)
- [🧮 3. Réduction de dimension (PCA, CSP, ICA…)](#3-réduction-de-dimension-pca-csp-ica)
- [🧠 4. Pipeline scikit-learn](#4-pipeline-scikit-learn)
- [🔍 5. Entraînement](#5-entraînement)
- [⚡ 6. Prédiction en pseudo temps réel](#6-prédiction-en-pseudo-temps-réel)
- [🧪 Tests & qualité logicielle](#tests--qualité-logicielle)
- [📚 Stack technique](#stack-technique)
  - [Traitement du signal / maths](#traitement-du-signal--maths)
  - [Machine Learning](#machine-learning)
  - [Qualité & Murphy Map](#qualité--murphy-map)
- [🔎 Pourquoi cette stack ?](#pourquoi-cette-stack-)
 [© Licence](#licence)
- [📖 Ressources utilisées](#ressources-utilisées)
- [👤 Auteur](#auteur)

---

# 📌 Overview

**Total Perspective Vortex** est un projet de **Brain-Computer Interface (BCI)** utilisant des données **EEG** pour déterminer, en quasi temps réel, l’intention motrice d’un individu (mouvement A ou B).

Il implémente un pipeline complet :

* 🧠 **Parsing & preprocessing EEG** (MNE, filtres 8–40 Hz)
* 🎚️ **Extraction de features** (spectre, puissance, canaux × temps)
* 🔻 **Réduction de dimension** implémentée manuellement (CSP, PCA, ICA…)
* 🔗 **Pipeline scikit-learn** (baseEstimator + transformerMixin)
* 🤖 **lassification supervisée**
* ⏱️ **Prediction < 2 secondes** (lecture pseudo temps réel)
* 📈 **Validation croisée (cross_val_score)**
* 🧪 **Accuracy ≥ 60 % sur sujets non vus – métrique obligatoire**

Le travail final ne contient **que le code Python** ; le dataset EEG Physionet n’est pas versionné.

---

# 🧠 Objectifs pédagogiques (42 / IA / ML)

* Concevoir un **pipeline ML complet** sur données EEG
* Implémenter un **algorithme mathématique de réduction de dimension**
* Intégrer ce module dans un **pipeline scikit-learn**
* Traiter un flux **temps réel**
* Travailler sur un dataset bruité (EEG réel)
* Manipuler **MNE**, **NumPy**, **Pandas**, **SciPy**, **scikit-learn**
* Construire des métriques reproductibles et un score fiable
* Préparer une défense solide (norme 42 + compréhension algorithmique)

---

# 🧩 Architecture du projet

```
Total_Perspective_Vortex/
├── docs
│   ├── assets
│   │   ├── image01.png
│   │   └── image02.png
│   ├── project
│   │   ├── gantt_tpv.png
│   │   ├── roadmap.md
│   │   └── wbs_tpv_v1.md
│   ├── risk
│   │   └── tpv_murphy_map_v8.csv
│   ├── total_perspective_vortex.en.checklist.pdf
│   └── Total_Perspective_Vortex.en.subject.pdf
├── LICENSE
├── Makefile
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── README.md
├── scripts
│   ├── predict.py
│   ├── train.py
│   └── visualize_raw_filtered.py
├── src
│   └── tpv
│       ├── classifier.py
│       ├── dimensionality.py
│       ├── features.py
│       ├── __init__.py
│       ├── pipeline.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── realtime.py
│       ├── train.py
│       └── utils.py
└── tests
    ├── test_classifier.py
    ├── test_dimensionality.py
    ├── test_pipeline.py
    ├── test_preprocessing.py
    └── test_realtime.py
```

---

# 🔬 1. Préprocessing & parsing EEG (MNE)

* Lecture des fichiers Physionet
* Visualisation du signal brut
* Filtrage bande-passante 8–40 Hz
* Découpage des epochs (t0–tn)
* Extraction des événements motrices (Left Hand / Right Hand / Feet)

Exemple :

```bash
poetry run python scripts/visualize_raw_filtered.py ./path/to/eeg/
```

---

# 🎛️ 2. Extraction de features

* Puissances par fréquence
* Spectrogrammes ou FFT
* Projection channel × time
* Agrégation temporelle

Tu décides des features que tu veux envoyer à ta matrice X ∈ R^(d × N).

---

# 🧮 3. Réduction de dimension (PCA, CSP, ICA…)

🔐 **Partie obligatoire du sujet : implémenter l’algorithme soi-même**

* calcul des matrices de covariance
* décomposition SVD / eigendecomposition
* normalisation
* projection WᵀX → X'
* tests de cohérence dimensionnelle

Exemple :

```python
from tpv.dimensionality import CSP
transformer = CSP(n_components=4)
X_reduced = transformer.fit_transform(X, y)
```

---

# 🧠 4. Pipeline scikit-learn

Le sujet exige :

* héritage de `baseEstimator` et `TransformerMixin`
* pipeline → `[Preprocessing → Dimensionality → Classifier]`
* utilisation de `cross_val_score`

Exemple :

```python
pipeline = Pipeline([
    ("reduce", CSP(n_components=4)),
    ("clf", LinearDiscriminantAnalysis())
])
```

---

# 🔍 5. Entraînement

L’interface CLI unifiée `mybci.py` lance les modules `tpv.train` et `tpv.predict` avec
des identifiants explicites :

```bash
python mybci.py S01 R01 train
```

Raccourci Makefile avec des valeurs par défaut modifiables :

```bash
make train TRAIN_SUBJECT=S01 TRAIN_RUN=R01
```

Affiche :

* scores cross_val_score
* statistiques par run
* moyenne ≥ 60 % requise sur sujets jamais vus

---

# ⚡ 6. Prédiction en pseudo temps réel

Réutilise la même CLI pour la phase inference :

```bash
python mybci.py S01 R01 predict
```

Ou via le Makefile :

```bash
make predict PREDICT_SUBJECT=S01 PREDICT_RUN=R01
```

Contraintes :

* lecture par chunks simulant un flux
* prédiction < **2 secondes** après réception
* sortie de classe {1, 2}

---

# 🧪 Tests & qualité logicielle

* pytest
* ruff
* black
* mypy
* coverage
* mutation testing (mutmut)
* CI GitHub Actions + Codecov

---

# 📚 Stack technique

### Traitement du signal / maths

* **NumPy** (matrices, opérations vectorisées)
* **SciPy** (eigenvalues, SVD)
* **MNE** (EEG parsing)

### Machine Learning

* **scikit-learn** (pipeline, classif, cross-validation)

### Qualité & Murphy Map

* ruff, black, mypy
* pytest, coverage, mutmut
* GitHub Actions, Codecov

Les fichiers de cartographie des risques (Loi de Murphy) se trouvent dans :

- `docs/qa/murphy_map_tpv_v1.csv`
- `docs/qa/murphy_map_tpv_v2.csv`

---

## 📖 Ressources utilisées

Les contenus suivants ont été essentiels pour comprendre l’EEG, les
filtres spatiaux (CSP) et la mise en place d’un pipeline d’analyse
monotrial robuste :

- 🎥 [Playlist YouTube — Machine Learning from Scratch](https://www.youtube.com/playlist?list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)  
  Série pédagogique pour consolider les bases de l’apprentissage supervisé
  (modèles linéaires, descente de gradient, régularisation) utilisées pour
  entraîner le classifieur sur les features EEG.

- 📄 [Wikipédia — Électroencéphalographie](https://fr.wikipedia.org/wiki/%C3%89lectroenc%C3%A9phalographie)  
  Notions fondamentales sur l’EEG, l’acquisition du signal et le rôle des
  électrodes, indispensables pour interpréter les données brutes.

- 📄 [Wikipédia — Common spatial pattern](https://en.wikipedia.org/wiki/Common_spatial_pattern)  
  Présentation du principe des filtres spatiaux CSP, de la maximisation de
  la variance entre classes et de leur utilisation en BCI.

- 📄 [Blankertz et al., *Optimizing Spatial Filters for Robust EEG Single-Trial Analysis*](https://doc.ml.tu-berlin.de/bbci/publications/BlaTomLemKawMue08.pdf)  
  Article de référence décrivant les stratégies d’optimisation de filtres
  spatiaux pour améliorer la robustesse de l’analyse EEG monotrial.

---

# © Licence

MIT License.

---
# 👤 Auteur

**Rafael Verissimo**  
Étudiant IA/Data — École 42 Paris  
GitHub : https://github.com/raveriss  
LinkedIn : https://www.linkedin.com/in/verissimo-rafael/  
