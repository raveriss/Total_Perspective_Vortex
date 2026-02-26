# EEG Brain-Computer Interface (BCI) - Total Perspective Vortex

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
- [📥 Télécharger le dépôt](#-télécharger-le-dépôt)
- [🧠 Objectifs pédagogiques (42 / IA / ML)](#objectifs-pédagogiques-42--ia--ml)
- [🧩 Architecture du projet](#architecture-du-projet)
- [🔬 1. Préprocessing & parsing EEG (MNE)](#1-préprocessing--parsing-eeg-mne)
- [📊 Visualiser raw vs filtré](#-visualiser-raw-vs-filtré)
- [🧭 Justification scientifique : canaux & fenêtres temporelles](#-justification-scientifique--canaux--fenêtres-temporelles)
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
- [🧭 Vue d’ensemble documentation](#-vue-densemble-documentation)
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
* 🧪 **Accuracy ≥ 75 % sur sujets non vus – métrique obligatoire**

Le travail final ne contient **que le code Python** ; le dataset EEG Physionet n’est pas versionné.

---

## 📥 Télécharger le dépôt

Cloner le projet depuis GitHub :

```bash
git clone https://github.com/raveriss/Total_Perspective_Vortex.git
cd Total_Perspective_Vortex
```
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
.
├── AGENTS.md
├── author
├── codecov.yml
├── create_tpv_fields.sh
├── data
├── docs
│   ├── assets
│   │   ├── image01.png
│   │   └── image02.png
│   ├── index.md
│   ├── metrics
│   │   ├── eval_payload.json
│   │   └── eval_payload.npz
│   ├── project
│   │   ├── benchmark_results.json
│   │   ├── benchmark_results.md
│   │   ├── checklist_wbs_matrix.md
│   │   ├── gantt_tpv.png
│   │   ├── physionet_dataset.md
│   │   ├── roadmap.md
│   │   ├── splits_metrics.md
│   │   ├── split_strategy.md
│   │   └── wbs_tpv.md
│   ├── risk
│   │   └── tpv_murphy_map.csv
│   ├── total_perspective_vortex.en.checklist.pdf
│   └── Total_Perspective_Vortex.en.subject.pdf
├── LICENSE
├── Makefile
├── mybci.py
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── README.md
├── scripts
│   ├── aggregate_accuracy.py
│   ├── aggregate_experience_scores.py
│   ├── aggregate_scores.py
│   ├── benchmark.py
│   ├── fetch_physionet.py
│   ├── __init__.py
│   ├── predict.py
│   ├── prepare_physionet.py
│   ├── sync_dataset.py
│   ├── train.py
│   └── visualize_raw_filtered.py
├── src
│   └── tpv
│       ├── classifier.py
│       ├── dimensionality.py
│       ├── features.py
│       ├── __init__.py
│       ├── pipeline.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── __pycache__
│       │   ├── classifier.cpython-310.pyc
│       │   ├── dimensionality.cpython-310.pyc
│       │   ├── features.cpython-310.pyc
│       │   ├── __init__.cpython-310.pyc
│       │   ├── pipeline.cpython-310.pyc
│       │   ├── preprocessing.cpython-310.pyc
│       │   └── utils.cpython-310.pyc
│       ├── realtime.py
│       ├── train.py
│       └── utils.py
└── tests
    ├── test_aggregate_scores_cli.py
    ├── test_benchmark.py
    ├── test_classifier.py
    ├── test_dimensionality.py
    ├── test_docs.py
    ├── test_experience_scores.py
    ├── test_features.py
    ├── test_fetch_physionet.py
    ├── test_mybci.py
    ├── test_pipeline.py
    ├── test_predict_cli.py
    ├── test_predict_evaluate_run.py
    ├── test_predict_load_data.py
    ├── test_predict_reports.py
    ├── test_prepare_physionet.py
    ├── test_preprocessing.py
    ├── test_realtime.py
    ├── test_scripts_roundtrip.py
    ├── test_sync_dataset.py
    ├── test_tpv_entrypoints.py
    ├── test_train_cli.py
    ├── test_train.py
    ├── test_utils.py
    └── test_visualize_raw_filtered.py
```

---

## 🚀 Mise en route : données, installation, entraînement, prédiction (Poetry + Makefile)

Le projet utilise **Poetry exclusivement** (aucun `requirements.txt`).
Le **Makefile** expose des raccourcis vers les commandes `poetry run ...`.

---

| Objectif | Commande recommandée | Commande équivalente |
|---|---|---|
| Installer | `make install` | `poetry install --with dev` |
| Linter | `make lint` | `poetry run ruff check .` |
| Formatter | `make format` | `poetry run ruff format . && poetry run ruff check --fix .` |
| Type-check | `make type` | `poetry run mypy src scripts tests` |
| Tests | `make test` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest -vv` |
| Coverage | `make cov` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run coverage run -m pytest ...` |
| Mutation | `make mut` | `MUTMUT_USE_COVERAGE=1 ... poetry run mutmut run` |
| Entraîner | `make train` | `poetry run python mybci.py 109 3 train` *(par défaut)* |
| Prédire | `make predict` | `poetry run python mybci.py 109 3 predict` *(par défaut)* |
| Temps réel | `make realtime <subject> <run>` | `poetry run python src/tpv/realtime.py <subject> <run>` |
| Visualiser brut/filtré | `make visualizer <subject> <run>` | `poetry run python scripts/visualize_raw_filtered.py <subject> <run>` |
| Moyenne des moyennes | `make compute-mean-of-means` | `poetry run python scripts/aggregate_experience_scores.py` |
| Benchmark global | `make mybci` | `poetry run python mybci.py` |
| Nettoyer | `make clean` | supprime `./artifacts` + les `*.npy` (hors `.venv`, `.git`, `artifacts`) |

---

### 📦 Générer les artefacts manquants avant l'évaluation globale

L'exécution de `make mybci` sans arguments déclenche
l'évaluation des 6 expériences (3 → 14) sur 109 sujets. Pour éviter
les avertissements "aucun modèle disponible", assurez-vous que
`artifacts/<subject>/<run>/model.joblib` existe pour chaque run visé.

---

### 🖥️ Exemples d'exécution (sorties tronquées)

#### `make install`

```bash
raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make install
poetry install --with dev
Creating virtualenv total-perspective-vortex in /home/raveriss/Desktop/Total_Perspective_Vortex/.venv
Installing dependencies from lock file

Package operations: 87 installs, 1 update, 0 removals

  - Updating pip (25.2 -> 25.3)
  - Installing mdurl (0.1.2)
  - Installing uc-micro-py (1.0.3)
  - Installing boolean-py (5.0)
  - Installing certifi (2025.11.12)
  - Installing charset-normalizer (3.4.4)
  - Installing defusedxml (0.7.1)
  - Installing idna (3.11)
  - Installing linkify-it-py (2.0.3)
  - Installing markdown-it-py (4.0.0)
  - Installing numpy (1.26.4)
  - Installing pygments (2.19.2)
  - Installing six (1.17.0)
  - Installing typing-extensions (4.15.0)
  - Installing urllib3 (2.6.3)
  - Installing colorama (0.4.6): Installing...
  - Installing colorama (0.4.6)
  - Installing contourpy (1.3.2): Installing...
  - Installing cycler (0.12.1)
  - Installing distlib (0.4.0): Installing...
  - Installing exceptiongroup (1.3.1)
  - Installing filelock (3.20.3)
  - Installing cycler (0.12.1)
  - Installing distlib (0.4.0): Installing...
  - Installing exceptiongroup (1.3.1)
  - Installing filelock (3.20.3)
  - Installing contourpy (1.3.2)
  - Installing cycler (0.12.1)
  - Installing distlib (0.4.0): Installing...
  - Installing exceptiongroup (1.3.1)
  - Installing filelock (3.20.3)
  - Installing exceptiongroup (1.3.1)
  - Installing filelock (3.20.3)
  - Installing distlib (0.4.0)
  - Installing exceptiongroup (1.3.1)
  - Installing filelock (3.20.3)
  - Installing fonttools (4.61.0): Pending...
  - Installing iniconfig (2.3.0)
  - Installing kiwisolver (1.4.9): Installing...
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing iniconfig (2.3.0)
  - Installing kiwisolver (1.4.9): Installing...
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing fonttools (4.61.0): Installing...
  - Installing iniconfig (2.3.0)
  - Installing kiwisolver (1.4.9): Installing...
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing kiwisolver (1.4.9)
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing iniconfig (2.3.0)
  - Installing kiwisolver (1.4.9)
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing fonttools (4.61.0)
  - Installing iniconfig (2.3.0)
  - Installing kiwisolver (1.4.9)
  - Installing license-expression (30.4.4)
  - Installing mando (0.7.1)
  - Installing markupsafe (3.0.3)
  - Installing mdit-py-plugins (0.5.0)
  - Installing msgpack (1.1.2)
  - Installing packageurl-python (0.17.6)
  - Installing pillow (12.0.0)
  - Installing platformdirs (4.5.0)
  - Installing packaging (25.0)
  - Installing pluggy (1.6.0)
  - Installing py-serializable (2.1.0)
  - Installing pyparsing (3.2.5)
  - Installing python-dateutil (2.9.0.post0)
  - Installing pyyaml (6.0.3)
  - Installing requests (2.32.5)
  - Installing rich (14.2.0)
  - Installing sortedcontainers (2.4.0)
  - Installing tomli (2.3.0)
  - Installing cachecontrol (0.14.4): Installing...
  - Installing cachecontrol (0.14.4)
  - Installing cfgv (3.5.0)
  - Installing click (8.3.1): Installing...
  - Installing coverage (7.12.0): Installing...
  - Installing coverage (7.12.0): Installing...
  - Installing click (8.3.1)
  - Installing coverage (7.12.0): Installing...
  - Installing cyclonedx-python-lib (9.1.0): Installing...
  - Installing decorator (5.2.1)
  - Installing identify (2.6.15)
  - Installing jinja2 (3.1.6)
  - Installing joblib (1.5.2): Installing...
  - Installing lazy-loader (0.4)
  - Installing cyclonedx-python-lib (9.1.0): Installing...
  - Installing decorator (5.2.1)
  - Installing identify (2.6.15)
  - Installing jinja2 (3.1.6)
  - Installing joblib (1.5.2): Installing...
  - Installing lazy-loader (0.4)
  - Installing coverage (7.12.0)
  - Installing cyclonedx-python-lib (9.1.0): Installing...
  - Installing decorator (5.2.1)
  - Installing identify (2.6.15)
  - Installing jinja2 (3.1.6)
  - Installing joblib (1.5.2): Installing...
  - Installing lazy-loader (0.4)
  - Installing libcst (1.8.6): Pending...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing libcst (1.8.6): Installing...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing decorator (5.2.1)
  - Installing identify (2.6.15)
  - Installing jinja2 (3.1.6)
  - Installing joblib (1.5.2): Installing...
  - Installing lazy-loader (0.4)
  - Installing libcst (1.8.6): Installing...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing cyclonedx-python-lib (9.1.0)
  - Installing decorator (5.2.1)
  - Installing identify (2.6.15)
  - Installing jinja2 (3.1.6)
  - Installing joblib (1.5.2): Installing...
  - Installing lazy-loader (0.4)
  - Installing libcst (1.8.6): Installing...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing lazy-loader (0.4)
  - Installing libcst (1.8.6): Installing...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing joblib (1.5.2)
  - Installing lazy-loader (0.4)
  - Installing libcst (1.8.6): Installing...
  - Installing matplotlib (3.10.7): Pending...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing matplotlib (3.10.7): Installing...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing matplotlib (3.10.7): Installing...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing libcst (1.8.6)
  - Installing matplotlib (3.10.7): Installing...
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing matplotlib (3.10.7)
  - Installing mypy-extensions (1.1.0)
  - Installing nodeenv (1.9.1)
  - Installing pathspec (0.12.1)
  - Installing pip-api (0.0.34)
  - Installing pip-requirements-parser (32.0.1)
  - Installing pooch (1.8.2)
  - Installing pytest (8.4.2)
  - Installing pytz (2025.2)
  - Installing radon (6.0.1)
  - Installing scipy (1.15.3)
  - Installing setproctitle (1.3.7)
  - Installing stevedore (5.6.0)
  - Installing textual (6.6.0)
  - Installing threadpoolctl (3.6.0)
  - Installing toml (0.10.2)
  - Installing tqdm (4.67.1)
  - Installing tzdata (2025.2)
  - Installing virtualenv (20.36.1)
  - Installing bandit (1.9.2): Installing...
  - Installing black (24.10.0): Pending...
  - Installing black (24.10.0): Installing...
  - Installing black (24.10.0): Installing...
  - Installing bandit (1.9.2)
  - Installing black (24.10.0): Installing...
  - Installing black (24.10.0)
  - Installing hypothesis (6.148.1)
  - Installing isort (5.13.2)
  - Installing mne (1.11.0)
  - Installing mutmut (3.4.0)
  - Installing mypy (1.18.2)
  - Installing pandas (2.3.3)
  - Installing pip-audit (2.9.0)
  - Installing pre-commit (4.5.0)
  - Installing pytest-cov (5.0.0)
  - Installing pytest-randomly (3.16.0)
  - Installing pytest-timeout (2.4.0)
  - Installing ruff (0.6.9)
  - Installing scikit-learn (1.7.2)
  - Installing xenon (0.9.3)

Installing the current project: total-perspective-vortex (0.1.0)
```

#### `make show-activate`

```bash
raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make show-activate
Commande d'activation (a executer dans le shell courant) :
source /home/raveriss/Desktop/Total_Perspective_Vortex/.venv/bin/activate
raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ 
```

#### `Activer l'environnement`

```bash
raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ source /home/raveriss/Desktop/Total_Perspective_Vortex/.venv/bin/activate
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ 
```

#### `make train <subject> <run>`

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make train 1 4
CV_SPLITS: 10 (scores: 10)
[0.6667 0.6667 0.6667 0.6667 0.3333 0.3333 0.6667 1.0000 0.6667 0.3333]
cross_val_score: 0.6000
```

*Entraîner via Makefile avec une stratégie de features* :

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make train 1 3 wavelet
INFO: dim_method='csp/cssp' appliqué avant l'extraction des features.
[⚡ TPV] Extracting wavelet features...
CV_SPLITS: 10 (scores: 10)
[0.6667 0.3333 0.3333 0.6667 1.0000 0.6667 0.6667 0.6667 0.3333 1.0000]
cross_val_score: 0.6333
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ 
```

#### `make predict <subject> <run>`

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make predict 1 3
epoch nb: [prediction] [truth] equal?
epoch 00: [1] [1] True
epoch 01: [0] [0] True
epoch 02: [0] [0] True
...
epoch 14: [0] [0] True
Accuracy: 1.0000
```

#### `make realtime <subject> <run>`

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make realtime 1 3
realtime prediction window=0 offset=0.000s raw=1 (T2) smoothed=1 (T2) latency=0.000s
realtime prediction window=1 offset=0.500s raw=1 (T2) smoothed=1 (T2) latency=0.000s
realtime prediction window=2 offset=1.000s raw=1 (T2) smoothed=1 (T2) latency=0.000s
...
realtime prediction window=188 offset=94.000s raw=0 (T1) smoothed=0 (T1) latency=0.000s
realtime prediction window=189 offset=94.500s raw=0 (T1) smoothed=0 (T1) latency=0.000s
realtime prediction window=190 offset=95.000s raw=0 (T1) smoothed=0 (T1) latency=0.000s
```

#### `make mybci`

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make mybci
experiment 0: subject 001: accuracy = 1.0000
INFO: modèle absent pour S002 R03, entraînement automatique en cours...
experiment 0: subject 002: accuracy = 0.6923
...
experiment 5: subject 109: accuracy = 1.0000

Mean accuracy of the six different experiments for all 109 subjects:
experiment 0:		accuracy = 0.8894
experiment 1:		accuracy = 0.8799
experiment 2:		accuracy = 0.9056
experiment 3:		accuracy = 0.9013
experiment 4:		accuracy = 0.8935
experiment 5:		accuracy = 0.8972

Mean accuracy of 6 experiments: 0.8945
```

#### `make compute-mean-of-means`

```bash
(total-perspective-vortex-py3.10) raveriss@raveriss-NLx0MU:~/Desktop/Total_Perspective_Vortex$ make compute-mean-of-means
Subject	T1	T2	T3	T4	Mean	Eligible(4/4)	MeetsThreshold_0p75
S001	0.962	0.895	1.000	0.933	0.947	yes	yes
S002	0.846	0.900	1.000	0.857	0.901	yes	yes
...
S108	0.833	1.000	0.900	0.917	0.912	yes	yes
S109	0.900	0.933	1.000	0.867	0.925	yes	yes
Global	0.891	0.889	0.906	0.901	0.897	109 subjects, bonus 5	yes
Worst subjects by Mean (mean_of_means):
- S092: 0.765
- S066: 0.780
- S059: 0.795
- S089: 0.799
- S045: 0.815
```

---

# 🔬 1. Préprocessing & parsing EEG (MNE)

* Lecture des fichiers Physionet
* Visualisation du signal brut
* Filtrage bande-passante 8–40 Hz
* Découpage des epochs (t0–tn)
* Extraction des événements motrices (Left Hand / Right Hand / Feet)

**Structure locale attendue** (non versionnée) : `data/<subject>/<run>.edf`.
Vérifiez l’intégrité et le nombre de runs avant tout parsing :

---

## 📊 Visualiser raw vs filtré

Ce script génère une figure comparative **signal brut vs signal filtré**
(bande-passante 8–40 Hz), afin de valider visuellement le préprocessing
sur un couple **(subject, run)** avant d’enchaîner sur l’extraction de features.

### Commande

> Recommandé : exécuter via Poetry pour garantir l’environnement.

```bash
make visualizer 1 9
```
<div align="center">
  <img src="https://github.com/raveriss/Total_Perspective_Vortex/blob/main/docs/assets/image01.png" alt="scripts visualize">
</div>

---

## 🧭 Justification scientifique : canaux & fenêtres temporelles

Les canaux EEG retenus se concentrent sur la **région sensorimotrice**
(10-20) car l’imagerie motrice induit des variations d’ERD/ERS surtout
sur les sites **C3/Cz/C4** et leurs voisins fronto-centraux et centro-pariétaux.
Cela aligne la sélection `DEFAULT_MOTOR_ROI` (ex: FC3/FC1/FCz/FC2/FC4,
C3/C1/Cz/C2/C4, CP3/CP1/CPz/CP2/CP4) du script
`scripts/visualize_raw_filtered.py` et les recommandations classiques en BCI
(Pfurtscheller & Neuper, 2001; Wolpaw et al., 2002).
Références :
* Pfurtscheller, G. & Neuper, C. (2001). Motor imagery and direct brain-computer communication.
  https://doi.org/10.1109/5.939829
* Wolpaw, J. R. et al. (2002). Brain-computer interfaces for communication.
  https://doi.org/10.1016/S1388-2457(02)00057-3

Les fenêtres temporelles d’epoching **0.5–2.5 s**, **1.0–3.0 s** et **0.0–2.0 s**
correspondent aux valeurs par défaut de `tpv.utils.DEFAULT_EPOCH_WINDOWS`. Elles
évacuent la phase de réaction immédiate à l’indice visuel tout en couvrant la
réponse motrice soutenue, ce qui limite les risques de fenêtres trop courtes
ou mal alignées avec l’ERD/ERS (cf. Murphy TPV-032, TPV-052).

Pour **visualiser l’impact** de ces choix (canaux + filtrage), utiliser le script
de comparaison brut/filtré :

```bash
make visualizer 1 3 C3 Cz C4
```

Ce graphique permet de vérifier visuellement que les canaux sensorimoteurs
présentent bien une dynamique exploitable dans la bande 8–40 Hz avant
d’alimenter la pipeline de features.

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

---

# 🧠 4. Pipeline scikit-learn

Le sujet exige :

* héritage de `baseEstimator` et `TransformerMixin`
* pipeline → `[Preprocessing → Dimensionality → Classifier]`
* utilisation de `cross_val_score`

---

# 🧪 Tests & qualité logicielle

* pytest
* ruff
* black
* mypy
* coverage
* mutation testing (mutmut)
  * la configuration Mutmut couvre `mybci.py`, `src/tpv` et `scripts` pour
    évaluer la qualité sur l'ensemble du pipeline
* CI GitHub Actions + Codecov

---

### Matrice checklist → WBS → tests

| Item checklist TPV | WBS / livrable | Test ou commande reproductible |
| --- | --- | --- |
| Visualisation raw vs filtré | 3.3.1–3.3.4 | `poetry run python scripts/visualize_raw_filtered.py data/S001` ; `poetry run pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Filtre 8–30 Hz + notch 50 Hz | 3.1.1–3.1.3 | `poetry run pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Réduction dimension (PCA/CSP) | 5.2.1–5.2.4 | `poetry run pytest tests/test_dimensionality.py::test_csp_returns_log_variances_and_orthogonality` |
| Pipeline sklearn (BaseEstimator/TransformerMixin) | 5.3.1–5.3.4 | `poetry run pytest tests/test_pipeline.py::test_pipeline_pickling_roundtrip` |
| Train + score via CLI | 6.3.x & 7.1.x | `poetry run pytest tests/test_classifier.py::test_training_cli_main_covers_parser_and_paths` |
| Predict renvoie l’ID de classe | 1.2.x & 6.2.x | `poetry run pytest tests/test_classifier.py::test_predict_cli_main_covers_parser_and_report` |
| Temps réel < 2 s | 8.2.x–8.3.x | `poetry run pytest tests/test_realtime.py::test_realtime_latency_threshold_enforced` |
| Score ≥ 75 % (agrégation) | 7.2.x | `poetry run pytest tests/test_classifier.py::test_aggregate_scores_exports_files_and_thresholds` |

---

### Définition du score global (scripts/aggregate_experience_scores.py)

* **Eligible** = sujet disposant des 4 types d’expérience (T1..T4).
* **Mean** (par sujet) = moyenne des 4 moyennes d’expérience.
* **MeetsThreshold_0p75** = `Mean >= 0.75`.
* **GlobalMean** = `(mean(T1) + mean(T2) + mean(T3) + mean(T4)) / 4`.
* Le script affiche les **10 pires sujets** par `Mean` et retourne **exit code 1**
  si `GlobalMean < 0.75`.

La version complète et maintenable de cette matrice, incluant les références aux risques Murphy, est disponible dans [`docs/project/checklist_wbs_matrix.md`](docs/project/checklist_wbs_matrix.md).

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

- `docs/qa/murphy_map_tpv.csv`

---

## 🧭 Vue d’ensemble documentation

Tous les jalons projet sont récapitulés dans [`docs/index.md`](docs/index.md), avec des liens directs vers le WBS, le diagramme de Gantt, la roadmap et la Murphy map.

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

- 🗄️ [PhysioNet — EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)
- 🏷️ [ICLabel — Tutoriel “EEG Independent Component Labeling”](https://labeling.ucsd.edu/tutorial/labels)
- 📚 [MNE-Python — Tutoriels officiels](https://mne.tools/dev/auto_tutorials/index.html)
- 📝 [Importing EEG data — blog / guide pratique](https://cbrnr.github.io/blog/importing-eeg-data/)

---

# © Licence

MIT License.

---
# 👤 Auteur

**Rafael Verissimo**
Étudiant IA/Data — École 42 Paris  
GitHub : https://github.com/raveriss  
LinkedIn : https://www.linkedin.com/in/verissimo-rafael/
