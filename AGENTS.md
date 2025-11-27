# AGENTS.md — Blueprint de Développement, Qualité, Checklist & Loi de Murphy (ft_linear_regression)

**Contexte cible** : Ubuntu 22.04.5 (Jammy), Python 3.10.18, **pas de sudo**, **Poetry**, exécution **uniquement sur Ubuntu**.

Ce document sert de **plan d’action exécutable**

---

## 🎯 Contraintes BCI obligatoires

Les contraintes suivantes doivent figurer simultanément dans README, AGENTS et Murphy Map et être respectées dans le code :

1. **Finalité** : classer en temps « réel » un signal EEG (imagination de mouvement A ou B).
2. **Source des données** : jeu **Physionet EEG motor imagery** obligatoire ; signaux structurés en matrice **channels × time** avec runs découpés et labellisés proprement.
3. **Prétraitement obligatoire** : visualisation du brut (script dédié), filtrage des bandes utiles (theta/alpha/beta…), visualisation après prétraitement, extraction des features (spectre/PSD…), et interdiction implicite d’utiliser `mne-realtime`.
4. **Pipeline ML** : utilisation de `sklearn.pipeline.Pipeline`, transformer maison héritant de `BaseEstimator` et `TransformerMixin`, réduction de dimension **PCA/ICA/CSP/CSSP implémentée à la main** (NumPy/SciPy autorisés, pas de version prête de sklearn/MNE).
5. **Entraînement/validation/test** : `cross_val_score` sur le pipeline complet, splits **Train/Validation/Test** distincts (pas d’overfit), accuracy moyenne **≥ 60 %** sur **tous les sujets de test** et les **6 runs** sur données **jamais apprises**.
6. **Temps réel** : le script `predict` lit un flux simulé (lecture progressive) et fournit chaque prédiction en **moins de 2 secondes** après réception d’un chunk.
7. **Architecture** : présence d’un script **train** et d’un script **predict** ; le dépôt final versionné contient **uniquement le code Python** (dataset exclu).
8. **Bonus facultatifs** : wavelets pour le spectre, classifieur maison, autres datasets EEG.
9. **Formalisme mathématique** : pour le transformer, avec X ∈ R^{d × N}, produire une matrice W telle que W^T X = X_{CSP}/X_{PCA}/X_{ICA}.
 pour implémenter `ft_linear_regression` à la 42, avec une posture **défense‑proof** : TDD systématique, couverture **100 %** (statements **et** branches), **diff=100 %**, contrôle **par fichier**, CI Ubuntu‑only. Les bonus CI perso sont isolés.

---

## 0) 🏗️ Fondations techniques & outillage

### 0.1 Git & hygiène de repo
- [ ] Init repo + `README.md` (usage, séquence de soutenance, badges CI si voulu)
- [ ] `LICENSE` (MIT) + `author`
- [ ] `.gitignore` : `theta.json`, `htmlcov/`, `.coverage*`, `.pytest_cache/`, `__pycache__/`, `*.pyc`
- [ ] Convention commits : `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

### 0.2 Environnement & dépendances (Poetry, no‑sudo)
- [ ] Installer Poetry (utilisateur) :
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
  poetry config virtualenvs.in-project true
  poetry env use 3.10
  ```
- [ ] `pyproject.toml` — **versions Python verrouillées** :
  ```toml
  [tool.poetry]
  name = "ft-linear-regression"
  version = "0.1.0"
  description = "42 Total_Perspective_Vortex (Ubuntu-only, Poetry)"
  authors = ["raveriss <you@example.com>"]

  [tool.poetry.dependencies]
  python = ">=3.10,<3.11"

  [tool.poetry.group.dev.dependencies]
  pytest = "^8.3"
  pytest-cov = "^5.0"
  pytest-timeout = "^2.3"
  pytest-randomly = "^3.15"
  mypy = "^1.10"
  ruff = "^0.5"
  mutmut = "^3.0"

  [tool.poetry.group.viz]
  optional = true
  [tool.poetry.group.viz.dependencies]
  matplotlib = "^3.9"

  [tool.ruff]
  line-length = 88
  [tool.ruff.lint]
  select = ["E","F","W","I"]
  [tool.ruff.format]
  quote-style = "double"
  ```

### 0.3 Makefile (raccourcis non intrusifs)
```Makefile
.PHONY: install lint format type test cov mut run-train run-predict reqs install-venv run-train-nopoetry run-predict-nopoetry mut
install:
	poetry install --with dev
reqs:
	poetry export -f requirements.txt -o requirements.txt --without-hashes
lint:
	poetry run ruff check .
format:
	poetry run ruff format . && poetry run ruff check --fix .
type:
	poetry run mypy src
test:
	poetry run pytest -q
cov:
	poetry run coverage run -m pytest && \
	poetry run coverage json -o coverage.json && \
	poetry run coverage html --skip-empty --show-contexts && \
	poetry run coverage report --fail-under=100
mut:
	poetry run mutmut run --simple-output
run-train:
	poetry run python3 -m src.train
run-predict:
        poetry run python3 -m src.predict

install-venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run-train-nopoetry:
	. .venv/bin/activate && python3 -m src.train --data data.csv --alpha 1e-7 --iters 100000 --theta theta.json

run-predict-nopoetry:
        . .venv/bin/activate && python3 -m src.predict 85000 --theta theta.json

mut:
	poetry run mutmut run --paths-to-mutate src --tests-dir tests --runner "pytest -q" --use-coverage --simple-output

```

### 0.4 CI/CD (GitHub Actions) — **Ubuntu‑only**
`.github/workflows/ci.yml`
```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  tests:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      - name: Configure Poetry
        run: |
          echo "$HOME/.local/bin" >> $GITHUB_PATH
          poetry config virtualenvs.in-project true
          poetry install --no-root --with dev
      - name: Lint & type
        run: |
          poetry run ruff check .
          poetry run mypy src
      - name: Tests & coverage (100 % global, diff 100 %)
        run: |
          poetry run coverage run -m pytest -q
          poetry run coverage json -o coverage.json
          poetry run coverage xml -o coverage.xml
          poetry run coverage report --fail-under=100
      - name: Enforce per-file 100 %
        run: |
          python - << 'PY'
import json,sys
j=json.load(open('coverage.json'))
miss=[f for f in j['files'].values() if f['summary']['percent_covered']<100]
if miss:
    print('Files below 100%:', [k for k,v in j['files'].items() if v in miss])
    sys.exit(1)
PY
      - name: Upload coverage HTML (artifact)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: htmlcov
          path: htmlcov/

  smoke-no-poetry:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - name: Install with pip (no Poetry)
        run: |
          python -m venv .venv
          . .venv/bin/activate
          pip install -r requirements.txt
      - name: Smoke run train & predict (no Poetry)
        run: |
          . .venv/bin/activate
          python -m src.train --data data.csv --alpha 1e-7 --iters 10 --theta theta.json
        python -m src.predict 85000 --theta theta.json

```

### 0.5 TDD — Red → Green → Refactor (règle d’or)
- **Definition of Ready** : pas de code sans **au moins un test qui échoue**.
- **Definition of Done** : tests verts, **100 %** couverture (branches), CLI/doc à jour.
- **Hooks (local)** :
  - `pre-commit` : `ruff format --check`, `ruff check`, `mypy` (rapide)

---

## 1) 🧩 Architecture minimale (agents)
- **`src/classifier.py`** :
- **`src/dimensionality.py`** :
- **`src/features.py`** :
- **`src/__init__.py`** :
- **`src/pipeline.py`** :
- **`src/predict.py`** :
- **`src/preprocessing.py`** :
- **`src/realtime.py`** :
- **`src/train.py`** :
- **`src/utils.py`** :

- **`tests/`** : unitaires + E2E + erreurs I/O + contrats.
- **Bonus isolé** :

> **Main guard requis** partout : `if __name__ == "__main__": main()` et exécution via `python3 -m src.train` / `python3 -m src.predict`.

---

## 2) 📜 Exigences 42 — conformité stricte
- [ ] **Deux programmes distincts** : `train.py`, `predict.py`.
- [ ] Hypothèse **exacte** : `estimate_price(x) = θ0 + θ1 * x`.
- [ ] **Initialisation** : `θ0 = 0`, `θ1 = 0`.
- [ ] **Mise à jour simultanée** : calculer `tmpθ0`, `tmpθ1` à partir des `θ` **courants**, puis assigner `θ ← θ − tmpθ` en **fin** d’itération.
- [ ] **Avant entraînement** : prédire **0** pour tout `km`.
- [ ] **Pas de lib magique** : **interdit** `numpy.polyfit`, `sklearn.LinearRegression`.
- [ ] **Persistance** : `theta.json` UTF‑8 (`{"theta0":..., "theta1":...}`) ; messages + codes retour ≠0 si manquant/corrompu.
- [ ] **CLI** : options `--alpha`, `--iters`, `--theta` ; **pas de magic numbers** en dur.
- [ ] **Predict interactif par défaut** : prompt si kilométrage non fourni.
- [ ] **Prédiction avant entraînement = 0** : tant que theta.json n’a pas été entraîné/écrit, predict doit renvoyer 0 pour tout kilométrage (hypothèse avec θ0=0, θ1=0). Testable en défense.

**Scénario E2E “défense” (à garder en sous‑puces) :**
- [ ] Étape
...

---

## 3) 🧪 Plan de tests (défense‑proof)
**Objectifs** : 100 % couverture (branches + diff), **contrôle par fichier**, tests rapides.

### 3.1 Unitaires
-
...

### 3.2 E2E
-
- CLI `--help` (exit 0), erreurs d’options (exit ≠ 0, message)
- **Entrée interactive** : prompt

### 3.3 Couverture (outil `coverage`)
- `.coveragerc` implicite via commandes : `branch=True`, `--skip-empty`, `--show-contexts`
- Générer `coverage.json` → script CI vérifie **100 % par fichier**
- **Diff=100 %** (chaque patch couvert)
- CI verrouillée sur **Ubuntu 22.04 uniquement** (pas de Windows/macOS)
- Upload vers **Codecov** (`coverage.xml`) → badge obligatoire pour mandatory

### 3.4 Mutation (CI perso)
- Outil : `mutmut` avec **scope global** sur tout le code **mandatory** (`src/`), pas seulement l’algorithme.
- Commande de référence :
  `mutmut run --paths-to-mutate src --tests-dir tests --runner "pytest -q" --use-coverage --simple-output`
- Objectif : **≥ 90 % de mutants tués** sur l’ensemble du code mandatory.
- Exclusions permises (documentées) : bonus (`src/viz.py`) et tout point d’entrée `__main__` pure glue non testable.
- Tout mutant survivant sur les zones **critiques** (formules, MAJ simultanée, I/O de `theta.json`, gestion d’erreurs CLI) = **échec** jusqu’à ajout de tests.
- CI : publier le rapport des survivants en artefact et lister les justifications résiduelles.

### 3.5 Tolérances numériques (si tests internes)
-
...

## 4) ⚙️ Spécifications d’implémentation

### 4.1 Formules
-
...

### 4.2 CLI (exemples)
```bash

```

### 4.3 Persistance
-
  ```
- **Ne jamais** committer

### 4.4 Structure projet
```
.
├── AGENTS.md
├── author
├── codecov.yml
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

## 5) 🛡️ Loi de Murphy — risques & contre‑mesures (condensé)


---

## 6) ✅ Procédure de validation finale (soutenance)
1. `pytest -q` → **tout vert**
2. `coverage run -m pytest && coverage json && coverage report --fail-under=100` (branches)
3. **Contrôle par fichier** : script CI sur `coverage.json` → **100 % partout**
3bis. **Upload vers Codecov** (`coverage.xml`) + vérif diff=100 %
4. **Mutation testing (scope global mandatory) ≥ 90 %** + aucun survivant sur les zones critiques.
5. Démo E2E : `predict(0)=0` → `train` → `predict≈csv` (MAJ simultanée validée)
6. Vérif visuelle `htmlcov/` (tout vert)
7. README : commande `predict→train→predict`, aucune mention de lib “magique”
8. Vérif environnement : exécution validée uniquement sous **Ubuntu 22.04** (soutenance école 42)


---

## 7) 📎 Annexes — extraits utiles

### 7.1 Bloc d’aide minimal (à snapshot en test)
```
usage: train.py
usage: predict.py
```

### 7.2 Modèle de messages d’erreurs (tests de régression)
- `ERROR:
- `ERROR:
- `ERROR:

---

## 8) 🔭 Bonus CI perso (hors soutenance 42)
- `vulture`, `bandit`, `radon/xenon` (analyse dead‑code/sécurité/complexité)
- Job Python 3.11 Ubuntu (smoke) en plus du 3.10

---

## 9) Pourquoi cette version ?
- **Alignée 42** : Ubuntu‑only, Python 3.10, no‑sudo, 2 programmes, MAJ simultanée, prédiction=0 avant train
- **Efficace** : CI courte, messages d’erreurs testés, contrôle par fichier
- **Évolutive** : bonus CI perso **isolés** ; viz en groupe Poetry optionnel
- **Lisible** : checklists concises, extraits directement copiables
