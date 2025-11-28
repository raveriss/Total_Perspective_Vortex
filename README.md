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
- [✅ Contraintes officielles du sujet](#-contraintes-officielles-du-sujet)
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

## 🚀 Installation et gestion des dépendances (Poetry uniquement)

L’environnement est géré exclusivement avec **Poetry** (aucun fichier
`requirements.txt` n’est utilisé).

```bash
poetry install --with dev
```

Les commandes CLI existantes restent accessibles via Poetry, par exemple :

```bash
poetry run python mybci.py S01 R01 train
poetry run python mybci.py S01 R01 predict
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


# ✅ Contraintes officielles du sujet

Ces exigences doivent être **présentes et respectées** dans toute la documentation et le code :

1. **Finalité** : classer en temps « réel » un signal EEG (imagination de mouvement A ou B).
2. **Source des données** : utiliser **exclusivement Physionet (EEG motor imagery)** ; les signaux sont des matrices **channels × time** avec runs découpés et labellisés proprement.
3. **Prétraitement obligatoire** :
   - visualiser le signal brut dans un script dédié ;
   - filtrer les bandes utiles (theta, alpha, beta… au choix) ;
   - visualiser après prétraitement ;
   - extraire les features (spectre, PSD, etc.) ;
   - 🚫 interdiction implicite : ne pas utiliser `mne-realtime`.
4. **Pipeline ML** :
   - utilisation obligatoire de `sklearn.pipeline.Pipeline` ;
   - transformer maison héritant de `BaseEstimator` et `TransformerMixin` ;
   - implémenter soi-même la réduction **PCA, ICA, CSP ou CSSP** (NumPy/SciPy autorisés, pas de version prête de sklearn ou MNE).
5. **Entraînement/validation/test** :
   - `cross_val_score` sur l’ensemble du pipeline ;
   - splits **Train / Validation / Test** distincts pour éviter l’overfit ;
   - moyenne d’**accuracy ≥ 60 %** sur **tous les sujets du jeu de test** et les **6 runs** d’expériences, sur des données **jamais apprises**.
6. **Temps réel** : le script `predict` lit un flux simulé (lecture progressive d’un fichier) et produit une prédiction en **< 2 secondes** après chaque chunk.
7. **Architecture** : fournir un script **train** et un script **predict** ; le dépôt final contient **uniquement le code Python** (pas le dataset).
8. **Bonus facultatifs** : wavelets pour le spectre, classifieur maison ou autres datasets EEG.
9. **Formalisme mathématique** : pour le transformer, avec X ∈ R^{d × N}, produire une matrice W telle que W^T X = X_{CSP}/X_{PCA}/X_{ICA}.

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

- `docs/qa/murphy_map_tpv.csv`

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
