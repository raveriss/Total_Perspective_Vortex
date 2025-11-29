# AGENTS.md — Blueprint Dev / Qualité / WBS / Loi de Murphy (Total_Perspective_Vortex)

**Contexte cible** : Ubuntu 22.04.5 (Jammy), Python 3.10.18, **pas de sudo**,
**Poetry**, exécution **uniquement sur Ubuntu**.

Ce document sert de **plan d’action exécutable** pour les agents (LLM/Codex)
chargés de modifier le dépôt **Total_Perspective_Vortex**.

Tous les agents doivent considérer comme **sources de vérité** :

- le **WBS** : `docs/project/wbs_tpv.md`
- le **Gantt / roadmap** : `docs/project/gantt_tpv.png`, `docs/project/roadmap.md`
- la **Murphy Map** : `docs/risk/tpv_murphy_map.csv`
- le **GitHub Project** :
  `Total_Perspective_Vortex – WBS & Murphy Map – v1.0 - 2025/11/28`
- les **issues GitHub** du repo : `raveriss/Total_Perspective_Vortex`

Aucune implémentation, refactor ou ajout de fichier ne doit être réalisé
hors de ce cadrage (WBS + risques + issues).


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
10. **Posture défense-proof globale** : pour implémenter `Total_Perspective_Vortex` avec une posture TDD systématique, couverture 100 %, diff=100 %, contrôle par fichier, CI Ubuntu-only.

---

## 🔁 Règles pour les agents (LLM / Codex)

Avant de générer du code, **tout agent** doit :

1. **Identifier le WBS ID concerné**
   - Chercher dans `docs/project/wbs_tpv.md` la tâche correspondante.
   - Si aucune tâche ne correspond, **ne pas inventer de feature** :
     proposer d’abord une mise à jour du WBS.

2. **Consulter la Murphy Map**
   - Filtrer `docs/risk/tpv_murphy_map.csv` sur ce WBS ID.
   - Lister les `Murphy ID` associés et leurs risques (cause, effet).
   - Adapter le design / les tests pour couvrir ces risques.

3. **Travailler via une issue GitHub**
   - Vérifier qu’une issue existe pour ce WBS ID.
   - Si ce n’est pas le cas, proposer une **issue à créer** avec :
     - titre = WBS ID + résumé court,
     - lien vers les sections WBS + Murphy Map concernées.

4. **Mettre à jour l’item dans le GitHub Project**
   - Associer l’issue à l’item du Project.
   - Mettre à jour les champs : `Status`, `Phase`, `Type`, `Priority`,
     `Risk score` si pertinent.

5. **Ne jamais livrer de code sans trace WBS**
   - Tout nouveau module / script / test doit pouvoir être relié à un
     `WBS ID` et, si applicable, à un ou plusieurs `Murphy ID`.
   - En cas de doute, l’agent doit **refuser l’implémentation** et
     demander une clarification WBS / risques.
6. **Respect strict de la structure TPV**
   - Aucun fichier ne doit être créé en dehors de :
     - `src/tpv/` (code ML / EEG)
     - `scripts/` (scripts CLI ou visualisation)
     - `tests/` (tests)
     - `docs/` (documentation)
   - Aucun fichier Python ne doit être ajouté à la racine, sauf `mybci.py`.
   - Toute proposition de nouveau fichier doit pointer vers :
     - un **WBS ID**,
     - une **issue GitHub** existante ou à créer,
     - un ou plusieurs **Murphy ID** associés.


## 0) 🏗️ Fondations techniques & outillage

### 0.1 Git & hygiène de repo
- [ ] Init repo + `README.md` (usage, séquence de soutenance, badges CI si voulu)
- [ ] `LICENSE` (MIT) + `author`
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
  name = "total-perspective-vortex"
  version = "0.1.0"
  description = "EEG Brain-Computer Interface pipeline for the Total Perspective Vortex project."
  authors = ["raveriss <you@example.com>"]
  license = "MIT"
  readme = "README.md"
  packages = [{ include = "tpv", from = "src" }]

  [tool.poetry.dependencies]
  python = ">=3.10,<3.11"
  numpy = "^1.26"
  pandas = "^2.2"
  scipy = "^1.11"
  scikit-learn = "^1.3"
  mne = "^1.6"
  matplotlib = "^3.8"
  joblib = "^1.4"

  [tool.poetry.group.dev.dependencies]
  pytest = "^8.3"
  pytest-cov = "^5.0"
  pytest-timeout = "^2.3"
  pytest-randomly = "^3.15"
  hypothesis = "^6.112"
  mypy = "^1.11"
  ruff = "^0.6"
  black = "^24.10"
  isort = "^5.13"
  bandit = "^1.7"
  mutmut = "^3.0"
  radon = "^6.0"
  xenon = "^0.9"
  pre-commit = "^4.0"
  pip-audit = "^2.7"
  coverage = "^7.6"

  [tool.black]
  line-length = 88
  target-version = ["py310"]

  [tool.isort]
  profile = "black"
  line_length = 88
  known_first_party = ["mybci", "tpv"]
  src_paths = ["src", "scripts", "tests"]

  [tool.ruff]
  line-length = 88
  target-version = "py310"

  [tool.ruff.lint]
  select = ["E", "F", "W", "I", "B", "PL", "C4"]
  ignore = []

  [tool.ruff.lint.isort]
  known-first-party = ["mybci", "tpv"]

  [tool.mutmut]
  paths_to_mutate = ["mybci.py", "src/tpv"]
  tests_dir = "tests"
  pytest_add_cli_args = ["-q"]
  mutate_only_covered_lines = true

  [tool.pytest.ini_options]
  pythonpath = ["src", ".", ".."]

  [tool.mypy]
  python_version = "3.10"
  check_untyped_defs = true
  warn_unused_ignores = true
  warn_return_any = true
  warn_redundant_casts = true
  strict_optional = true
  no_implicit_optional = true
  show_error_codes = true
  pretty = true
  ignore_missing_imports = true
  files = "src scripts tests"

  [build-system]
  requires = ["poetry-core"]
  build-backend = "poetry.core.masonry.api"


  ```

### 0.3 Makefile (raccourcis non intrusifs)
```Makefile
# ========================================================================================
# Makefile - Automatisation pour le projet Total_Perspective_Vortex
# Objectifs :
#   - Simplifier l’installation et la gestion de l’environnement (Poetry / venv)
#   - Automatiser les vérifications (lint, format, type-check, tests, coverage, mutation)
#   - Fournir des commandes pratiques pour l’entraînement et la prédiction du modèle
# ========================================================================================

.PHONY: install lint format type test cov mut train predict  activate deactivate

VENV = .venv
VENV_BIN = $(VENV)/bin/activate

# --- Benchmarks ---------------------------------------------------------------
BENCH_DIR   := data/benchmarks
BENCH_CSVS  := $(wildcard $(BENCH_DIR)/*.csv)

# Utilisation raccourcie de Poetry
POETRY = poetry run

# ----------------------------------------------------------------------------------------
# Installation des dépendances (dev inclus)
# ----------------------------------------------------------------------------------------
install:
	poetry install --with dev

# ----------------------------------------------------------------------------------------
# Vérifications de qualité du code
# ----------------------------------------------------------------------------------------

# Linting avec Ruff (analyse statique rapide)
lint:
	$(POETRY) ruff check .

# Formatage + correction auto avec Ruff
format:
	$(POETRY) ruff format . && $(POETRY) ruff check --fix .

# Vérification des types avec Mypy
type:
	$(POETRY) mypy src scripts tests


# ----------------------------------------------------------------------------------------
# Tests et couverture
# ----------------------------------------------------------------------------------------

# Exécution des tests unitaires
test:
	$(POETRY) pytest -vv

# Analyse de la couverture avec rapport JSON, HTML et console (100% requis)
cov:
	$(POETRY) coverage run -m pytest && \
	$(POETRY) coverage json -o coverage.json && \
	$(POETRY) coverage xml -o coverage.xml && \
	$(POETRY) coverage html --skip-empty --show-contexts && \
	$(POETRY) coverage report --fail-under=100


# Mutation testing avec Mutmut (guidé par la couverture)
mut: cov
	$(POETRY) mutmut run --use-coverage --simple-output




# ----------------------------------------------------------------------------------------
# Commandes liées au modèle (Poetry)
# ----------------------------------------------------------------------------------------

TRAIN_SUBJECT ?= S01
TRAIN_RUN ?= R01
PREDICT_SUBJECT ?= $(TRAIN_SUBJECT)
PREDICT_RUN ?= $(TRAIN_RUN)

# Entraînement du modèle : exemple minimal avec sujet et run de démonstration
train:
	$(POETRY) python mybci.py $(TRAIN_SUBJECT) $(TRAIN_RUN) train

# Prédiction : exemple minimal réutilisant les identifiants ci-dessus
predict:
	$(POETRY) python mybci.py $(PREDICT_SUBJECT) $(PREDICT_RUN) predict



# Affiche la commande pour activer le venv
activate:
	@echo "Chemin de l'environnement Poetry :"
	@poetry env info -p
	@echo
	@echo "Pour activer manuellement cet environnement :"
	@echo "  source $$(poetry env info -p)/bin/activate"

# Affiche la commande pour désactiver le venv
deactivate:
	@echo "Pour quitter l'environnement :"
	@echo "  deactivate"

# ----------------------------------------------------------------------------------------
# Règle générique pour ignorer les cibles numériques (ex. make predict-nocheck 23000)
# ----------------------------------------------------------------------------------------
%:
	@:
```

### 0.4 CI/CD (GitHub Actions) — **Ubuntu‑only**
`.github/workflows/ci.yml`
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
    paths-ignore:
      - '**/*.md'
      - '**/*.txt'
      - '**/*.png'
  pull_request:
    branches: [ main, develop ]
    paths-ignore:
      - '**/*.md'
      - '**/*.txt'
      - '**/*.png'

env:
  # Version Python canonique utilisée par la CI (alignée avec pyproject.toml)
  PYTHON_VERSION: "3.10"

jobs:
  pre-commit:
    name: Pre-commit checks
    runs-on: ubuntu-22.04
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y libasound2-dev

      - name: Install Poetry (retry)
        run: |
          python -m pip install --user --upgrade pip
          for attempt in 1 2 3; do
            python -m pip install --user --retries 3 --timeout 60 poetry==1.8.4 && break
            if [ "$attempt" -eq 3 ]; then
              echo "Poetry installation failed after ${attempt} attempts." >&2
              exit 1
            fi
            echo "Retrying Poetry installation (attempt ${attempt}/3)..." >&2
            sleep 5
          done
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
          poetry config virtualenvs.create true
          poetry config virtualenvs.in-project true

      - name: Cache virtualenv
        id: cache-poetry
        uses: actions/cache@v3
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ env.PYTHON_VERSION }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies (cache miss)
        if: steps.cache-poetry.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --with dev --no-root

      - name: Install project in editable mode
        run: poetry install --no-interaction --with dev

      - name: Run pre-commit
        run: poetry run pre-commit run --all-files

  static-analysis:
    name: Static Analysis
    runs-on: ubuntu-22.04
    needs: pre-commit
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: sudo apt-get update && sudo apt-get install -y libasound2-dev
      - name: Install Poetry (retry)
        run: |
          python -m pip install --user --upgrade pip
          for attempt in 1 2 3; do
            python -m pip install --user --retries 3 --timeout 60 poetry==1.8.4 && break
            if [ "$attempt" -eq 3 ]; then
              echo "Poetry installation failed after ${attempt} attempts." >&2
              exit 1
            fi
            echo "Retrying Poetry installation (attempt ${attempt}/3)..." >&2
            sleep 5
          done
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
          poetry config virtualenvs.create true
          poetry config virtualenvs.in-project true
      - uses: actions/cache@v3
        id: cache-poetry
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ env.PYTHON_VERSION }}-${{ hashFiles('**/poetry.lock') }}
      - name: Install dependencies (cache miss)
        if: steps.cache-poetry.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --with dev --no-root
      - name: Install project
        run: poetry install --no-interaction --with dev

      - name: Run Black check
        run: poetry run black --check .

      - name: Run isort check
        run: poetry run isort --check-only .

      - name: Run Ruff
        run: poetry run ruff check .

      - name: Run MyPy
        run: poetry run mypy src scripts tests

      - name: Audit dependencies with pip-audit
        run: poetry run pip-audit --progress-spinner=off


  tests:
    runs-on: ubuntu-22.04
    needs: static-analysis
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Poetry (retry)
        run: |
          python -m pip install --user --upgrade pip
          for attempt in 1 2 3; do
            python -m pip install --user --retries 3 --timeout 60 poetry==1.8.4 && break
            if [ "$attempt" -eq 3 ]; then
              echo "Poetry installation failed after ${attempt} attempts." >&2
              exit 1
            fi
            echo "Retrying Poetry installation (attempt ${attempt}/3)..." >&2
            sleep 5
          done
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
          poetry config virtualenvs.create true
          poetry config virtualenvs.in-project true

      - name: Cache virtualenv
        id: cache-poetry
        uses: actions/cache@v3
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ env.PYTHON_VERSION }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies (with dev)
        if: steps.cache-poetry.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --with dev

      - name: Ensure project installed
        if: steps.cache-poetry.outputs.cache-hit == 'true'
        run: poetry install --no-interaction --with dev

      - name: Run tests with coverage (Makefile)
        run: make cov

      - name: Generate coverage.xml for Codecov
        run: poetry run coverage xml -o coverage.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          files: ./coverage.xml
          disable_search: true
          flags: unittests
          name: ci-ubuntu-py${{ env.PYTHON_VERSION }}
          slug: raveriss/Total_Perspective_Vortex
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false

      - name: Run mutation tests (coverage-guided)
        run: |
          poetry run mutmut run --use-coverage --simple-output

      - name: Show mutation results
        run: |
          poetry run mutmut results > mutmut-results.txt
          cat mutmut-results.txt

      - name: Upload mutation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: mutmut-results
          path: mutmut-results.txt

      - name: Fail if surviving mutants
        run: |
          if grep -Fq "survived" mutmut-results.txt; then
            echo "Surviving mutants detected" && exit 1
          fi

  build:
    name: Build Package
    runs-on: ubuntu-22.04
    needs: [static-analysis, tests]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: sudo apt-get update && sudo apt-get install -y libasound2-dev
      - name: Install Poetry (retry)
        run: |
          python -m pip install --user --upgrade pip
          for attempt in 1 2 3; do
            python -m pip install --user --retries 3 --timeout 60 poetry==1.8.4 && break
            if [ "$attempt" -eq 3 ]; then
              echo "Poetry installation failed after ${attempt} attempts." >&2
              exit 1
            fi
            echo "Retrying Poetry installation (attempt ${attempt}/3)..." >&2
            sleep 5
          done
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
          poetry config virtualenvs.create true
          poetry config virtualenvs.in-project true
      - run: poetry install --no-interaction --no-root
      - name: Build package
        run: poetry build
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

> Exemple minimal de CI ci-dessous. La configuration réelle utilisée est
> définie dans `.github/workflows/ci.yml` (jobs pré-commit, static-analysis,
> tests, build).


### 0.5 TDD — Red → Green → Refactor (règle d’or)
- **Definition of Ready** : pas de code sans **au moins un test qui échoue**.
- **Definition of Done** : tests verts, **100 %** couverture (branches), CLI/doc à jour.
- **Hooks (local)** :
  - `pre-commit` : `ruff format --check`, `ruff check`, `mypy` (rapide)

---

## 1) 🧩 Architecture minimale (agents)
- **`src/tpv/classifier.py`** :
- **`src/tpv/dimensionality.py`** :
- **`src/tpv/features.py`** :
- **`src/tpv/__init__.py`** :
- **`src/tpv/pipeline.py`** :
- **`src/tpv/predict.py`** :
- **`src/tpv/preprocessing.py`** :
- **`src/tpv/realtime.py`** :
- **`src/tpv/train.py`** :
- **`src/tpv/utils.py`** :

- **`tests/`** : unitaires + E2E + erreurs I/O + contrats.
- **Bonus isolé** :

 **Main guard requis** partout : `if __name__ == "__main__": main()`
 et exécution modulaire via `python -m tpv.train` / `python -m tpv.predict`
 ou via le point d'entrée `python mybci.py <subject> <run> {train,predict}`.

---

## 2) 🔁 Pipeline local avant commit (miroir du CI)

Avant **tout `git commit`**, l’agent doit exécuter **exactement** cette séquence,
dans cet ordre, et **abandonner le commit** dès qu’une étape échoue.

### 2.1 Préparation (si nouveau clone ou `poetry.lock` modifié)

Si l’agent ne sait pas si l’environnement est à jour (nouveau clone,
changement de branche, doute sur `poetry.lock`), il doit exécuter
systématiquement 2.1.1 et 2.1.2.

1. `poetry install --no-interaction --with dev`
2. Vérifier que la commande `poetry run pytest -q` fonctionne au moins une fois.

### 2.2 Pre-commit + Static analysis (miroir CI)

Ces commandes reflètent les jobs `pre-commit` et `static-analysis` de la CI
GitHub. Elles doivent être exécutées **dans cet ordre**.

> La redondance avec `pre-commit` (qui lance déjà une partie de ces outils)
> est **volontaire** pour reproduire fidèlement les jobs de la CI et détecter
> toute divergence locale/CI.

1. `poetry run pre-commit run --all-files`
2. `poetry run black --check .`
3. `poetry run isort --check-only .`
4. `poetry run ruff check .`
5. `poetry run mypy src scripts tests`
6. `poetry run pip-audit --progress-spinner=off`

Si une de ces commandes échoue → **corriger le code** puis **rejouer depuis 2.2.1**.

### 2.3 Tests unitaires + couverture (miroir du job `tests`)
1. `make cov`
   - doit produire `coverage.json`, `coverage.xml`, `htmlcov/`
   - doit finir avec `coverage report --fail-under=100` **OK**
2. Vérifier qu’aucun fichier n’est sous 100 % via le script de la CI
   (contrôle par fichier avec `coverage.json` si script local existant).

### 2.4 Mutation testing (miroir du job `tests` – step Mutmut)
1. `poetry run mutmut run --use-coverage --simple-output`
2. `poetry run mutmut results > mutmut-results.txt`
3. Vérifier **manuellement** ou via script qu’il n’y a **aucune ligne** contenant
   `survived` dans `mutmut-results.txt`.

Si au moins un mutant **survit** → **interdiction de committer** tant que :
- les tests n’ont pas été renforcés, puis
- `poetry run mutmut run` et `poetry run mutmut results` repassent sans `survived`.

### 2.5 Règle d’or pour les agents

- Un commit n’est **valide** que si toutes les étapes 2.2, 2.3 et 2.4 sont **vertes**.
- Tout message “Modifie ton pre-commit car il y a des fails dans le CI” doit déclencher :
  1. Relecture de cette section 2),
  2. Exécution complète de la séquence,
  3. Correction des échecs **avant** de proposer un nouveau commit.

---
cd
## 3) 🧪 Plan de tests (défense‑proof)
**Objectifs** : 100 % couverture (branches + diff), **contrôle par fichier**, tests rapides.

### 3.1 Unitaires
-
...


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
- **Ne jamais** committer les datasets bruts ou fichiers issus de Physionet.

### 4.4 Structure projet
```
.
├── AGENTS.md
├── author
├── codecov.yml
├── create_tpv_fields.sh
├── docs
│   ├── assets
│   │   ├── image01.png
│   │   └── image02.png
│   ├── project
│   │   ├── gantt_tpv.png
│   │   ├── roadmap.md
│   │   └── wbs_tpv.md
│   ├── risk
│   │   └── tpv_murphy_map.csv
│   ├── total_perspective_vortex.en.checklist.pdf
│   └── Total_Perspective_Vortex.en.subject.pdf
├── LICENSE
├── Makefile
├── mybci.py
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── README.md
├── scripts
│   ├── import_murphy_issues.py
│   ├── import_murphy_to_project.py
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
    ├── test_mybci.py
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

## 📚 Documentation du code

Lorsque tu génères du code pour moi, applique **strictement** les règles
de documentation suivantes.

### Règles de commentaires

* **Un commentaire par ligne de code**, placé **juste au-dessus** de la ligne.
* Le commentaire doit expliquer **le “pourquoi”** de la ligne
  (intention, rôle, effet métier, contrainte, robustesse),
  **jamais le “comment”** ni une paraphrase du code.
* Longueur maximale : **80 caractères par commentaire**.
* Les commentaires doivent **respecter l’indentation du code**
  (un commentaire est dans le même bloc que la ligne qu’il décrit).
* **Interdit** :

  * Commentaire en fin de ligne (`…  # commentaire`)
  * Commentaire sous la ligne de code

### Docstrings

* Utiliser des **docstrings uniquement** pour les **fonctions/classes/modules** :

  * But global, paramètres, valeurs de retour, erreurs levées.
  * Ne pas répéter ce qui est déjà expliqué commentaire par commentaire.

---

### Exemple **à ne pas produire** (paraphrase du code, “comment” et non “pourquoi”)

```py
# Calcule la différence entre max_km et min_km,
# ou 1.0 si la différence vaut 0
km_range = max_km - min_km or 1.0  # pragma: no mutate

# Calcule la différence entre max_price et min_price,
# ou 1.0 si la différence vaut 0
price_range = max_price - min_price or 1.0  # pragma: no mutate
```

### Exemple **attendu** (explication du “pourquoi”, pas du “comment”)

```py
# Garantit un intervalle de distance non nul pour éviter une division par zéro
km_range = max_km - min_km or 1.0  # pragma: no mutate

# Garantit un intervalle de prix non nul pour stabiliser la normalisation
price_range = max_price - min_price or 1.0  # pragma: no mutate
```
