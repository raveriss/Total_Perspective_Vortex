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

# 📌 Overview

**Total Perspective Vortex** est un projet de **Brain-Computer Interface (BCI)** utilisant des données **EEG** pour déterminer, en quasi temps réel, l’intention motrice d’un individu (mouvement A ou B).

Il implémente un pipeline complet :

* 🧠 **Parsing & preprocessing EEG** (MNE, filtres 8–40 Hz)
* 🎚️ **Extraction de features** (spectre, puissance, canaux × temps)
* 🔻 **Réduction de dimension** implémentée manuellement (CSP, PCA, ICA…)
* 🔗 **Pipeline scikit-learn** (baseEstimator + transformerMixin)
* 🤖 **Classification supervisée**
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
├── src/
│   └── tpv/
│       ├── __init__.py
│       ├── preprocessing.py        # Parsing + filtrage EEG (MNE)
│       ├── features.py             # Extraction spectre, puissance, channels
│       ├── dimensionality.py       # PCA/CSP implémenté manuellement
│       ├── pipeline.py             # Pipeline sklearn complet
│       ├── classifier.py           # Classif sklearn (LDA/Ridge/SVM…)
│       ├── realtime.py             # Lecture playback + prédiction <2s
│       ├── train.py                # Script d'entraînement
│       ├── predict.py              # Script de prédiction
│       └── utils.py
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── visualize_raw_filtered.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_dimensionality.py
│   ├── test_pipeline.py
│   ├── test_classifier.py
│   └── test_realtime.py
│
├── docs/
|   ├── project/
|   |    ├── wbs_tpv_v1.md        # ton WBS
|   |    ├── gantt_tpv.png        # (optionnel)
│   |    └── roadmap.md
│   |
|   ├── risk/
│   │   └── tpv_murphy_map_v8.csv
│   |
│   ├── Total_Perspective_Vortex.en.subject.pdf
|   └── total_perspective_vortex.en.checklist.pdf
│
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
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

```bash
poetry run python scripts/train.py subject_id run_id
```

Affiche :

* scores cross_val_score
* statistiques par run
* moyenne ≥ 60 % requise sur sujets jamais vus

---

# ⚡ 6. Prédiction en pseudo temps réel

```bash
poetry run python scripts/predict.py subject_id run_id
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

# 🔎 Pourquoi cette stack ?

* **MNE** est la librairie standard pour EEG scientifique ; indispensable.
* **scikit-learn** permet un pipeline structuré, stable et lisible par les recruteurs.
* **CSP/PCA implémentés à la main** → démontre tes compétences mathématiques et ML.
* **NumPy/SciPy** → calculs lourds, eigenvectors, projections.
* **CI/CD complète** → montre une maturité “niveau entreprise”.

---

# © Licence

MIT License.

---

# 👤 Auteur

**Rafael Verissimo**  
Étudiant IA/Data — École 42 Paris  
GitHub : https://github.com/raveriss  
LinkedIn : https://www.linkedin.com/in/verissimo-rafael/
