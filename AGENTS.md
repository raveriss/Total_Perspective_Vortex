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

## ⚠️ Règle absolue pour les agents (LLM / Codex)

Un agent n’a **jamais** le droit de proposer un `git commit` ou un `git push`
tant qu’il n’a pas, dans **cette réponse précise** :

1. Rappelé la **checklist pré-commit locale** de la section **2)**.
2. Affiché la section :

   > `CHECKLIST 2) – état AVANT COMMIT`

   avec les trois lignes suivantes **exactement** :

   - `2.2 Pre-commit + static analysis : ✅/❌`
   - `→ Commit autorisé : ✅/❌`

3. Listé les **commandes locales** à exécuter, **dans l’ordre 2.1 → 2.3**.

4. Indiqué pour chaque étape si l’état est **réellement connu** :

   - ✅ : l’agent a vu un log de succès ou a lui-même exécuté la commande.
   - ❌ : la commande est en échec.
   - ❌ (inconnu) : l’agent n’a **aucune preuve** que la commande est OK.

> **Règle dure** :
> Si une seule des lignes 2.2, 2.3 est ❌ ou inconnue,
> l’agent doit **écrire explicitement** `→ Commit autorisé : ❌`.

Dans ce cas, l’agent doit :

- **ne pas** proposer de commit/push ni de message de commit,
- **entrer en boucle de remédiation** (fail → fix → re-run) jusqu’à ce que
  2.2 + 2.3 soient ✅, *ou* jusqu’à atteindre une limite d’itérations.

### 🔁 Boucle de remédiation obligatoire (fail → fix → re-run)

But : un échec de check **déclenche une itération de correction**.
Un agent autonome (ex: Codex) n’a pas le droit de “s’arrêter au fail”
si un correctif est possible dans le dépôt.

Règles :
1) L’agent exécute la séquence “MODE DEV” (diagnostic rapide) :
   - `poetry run pre-commit run --all-files --show-diff-on-failure`
2) Si KO :
   - il **isole** le/les hooks en échec,
   - il applique un **patch minimal**,
   - il re-run **uniquement** le/les hooks KO,
   - il répète jusqu’au vert.
3) Ensuite, avant tout commit, l’agent exécute le “MODE AVANT COMMIT (gate)” :
   - 2.2 complet + 2.3 complet (mirroir strict CI)
4) Si le gate est vert :
   - il peut committer.
5) Limite anti-boucle :
   - maximum **5 itérations**.
   - au-delà : l’agent explique la cause racine probable, propose 2 options
     (ex: ajuster un test flaky, corriger typing, corriger un contrat I/O),
     et demande les logs manquants si nécessaire.

Interdictions (anti-contournement) :
- pas de `--no-verify`
- pas de désactivation/suppression de hooks
- pas de “skip” d’un job CI
- pas de modification de seuils (coverage, etc.) pour “faire passer”

Si une étape est **KO** ou non mentionnée, la réponse doit se terminer par :

> « ❌ CI potentiellement en échec : interdiction de committer tant que la
>  checklist 2) n’est pas entièrement verte. »

Toute divergence entre la pipeline locale (section 2) et les jobs CI
(`.github/workflows/ci.yml`) est considérée comme un **échec de l’agent**.

Toute réponse qui ne contient **pas** la section structurée
« CHECKLIST 2) – état AVANT COMMIT » est considérée comme **invalide**.

### 🧭 Mode DEV vs Mode AVANT COMMIT (règle anti-ambiguïté)

**MODE DEV (boucle itérative)** — objectif : garder le dépôt propre pendant
l’itération (formatage/lint/types), même si des tests de couverture ou des
mutants survivent ailleurs.

* Autorisé à tout moment :
  * `poetry run pre-commit run --all-files`
  * et/ou les commandes rapides `make lint`, `make format`, `make type`
* Obligation de formulation :
  * l’agent doit qualifier cela comme un **RUN DEV / diagnostic**,
    et **ne pas** conclure que “tout est bon”.
* Effet sur le gate :
  * un RUN DEV **ne change jamais** la règle “commit interdit”.
  * si 2.3 ne sont pas verts (ou inconnus) :
    `→ Commit autorisé : ❌` reste obligatoire.

* Comportement attendu :
  * si un hook échoue en DEV, l’agent **corrige et re-run** (boucle) au lieu de s’arrêter sur l’échec.

**MODE AVANT COMMIT (gate)** — objectif : miroir strict du CI.

* Obligatoire avant toute suggestion de commit/push :
  * exécuter et valider **2.2 + 2.3 (section 2),
  * afficher le bloc **CHECKLIST 2) – état AVANT COMMIT**,
  * et n’autoriser le commit que si tout est ✅.

* Comportement attendu :
  * si 2.2 ou 2.3 échoue, l’agent **revient en boucle de remédiation**
    (patch → re-run) jusqu’au vert, puis rejoue le gate complet.

### ✅ `pre-commit` autorisé en MODE DEV (diagnostic)

- L’agent **peut exécuter** `poetry run pre-commit run --all-files` en MODE DEV,
  **même si** 2.3 (`make cov`) ne sont pas encore verts.
- Après un RUN DEV, l’agent doit :
  - indiquer explicitement “MODE DEV / diagnostic”,
  - rapporter le résultat de `pre-commit` (OK/KO),
  - proposer un **patch minimal** si un hook échoue,
  - et rappeler que le **gate AVANT COMMIT** reste inchangé.
- Interdictions strictes :
  - présenter un RUN DEV comme une validation “AVANT COMMIT”,
  - suggérer un commit/push tant que la checklist 2) n’est pas entièrement ✅.

---

## 🎯 Contraintes BCI obligatoires

Les contraintes suivantes doivent figurer simultanément dans README, AGENTS et Murphy Map et être respectées dans le code :

1. **Finalité** : classer en temps « réel » un signal EEG (imagination de mouvement A ou B).
2. **Source des données** : jeu **Physionet EEG motor imagery** obligatoire ; signaux structurés en matrice **channels × time** avec runs découpés et labellisés proprement.
3. **Prétraitement obligatoire** : visualisation du brut (script dédié), filtrage des bandes utiles (theta/alpha/beta…), visualisation après prétraitement, extraction des features (spectre/PSD…), et interdiction implicite d’utiliser `mne-realtime`.
4. **Pipeline ML** : utilisation de `sklearn.pipeline.Pipeline`, transformer maison héritant de `BaseEstimator` et `TransformerMixin`, réduction de dimension **PCA/ICA/CSP/CSSP implémentée à la main** (NumPy/SciPy autorisés, pas de version prête de sklearn/MNE).
5. **Entraînement/validation/test** : `cross_val_score` sur le pipeline complet, splits **Train/Validation/Test** distincts (pas d’overfit), accuracy moyenne **≥ 75 %** sur **tous les sujets de test** et les **6 runs** sur données **jamais apprises**.
6. **Temps réel** : le script `predict` lit un flux simulé (lecture progressive) et fournit chaque prédiction en **moins de 2 secondes** après réception d’un chunk.
7. **Architecture** : présence d’un script **train** et d’un script **predict** ; le dépôt final versionné contient **uniquement le code Python** (dataset exclu).
8. **Bonus facultatifs** : wavelets pour le spectre, classifieur maison, autres datasets EEG.
9. **Formalisme mathématique** : pour le transformer, avec X ∈ R^{d × N}, produire une matrice W telle que W^T X = X_{CSP}/X_{PCA}/X_{ICA}.
10. **Posture défense-proof globale** : pour implémenter `Total_Perspective_Vortex` avec une posture TDD systématique, couverture 90 %, diff=190 %, contrôle par fichier, CI Ubuntu-only.

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

# Analyse de la couverture avec rapport JSON, HTML et console (90% requis)
cov:
	$(POETRY) coverage run -m pytest && \
	$(POETRY) coverage json -o coverage.json && \
	$(POETRY) coverage xml -o coverage.xml && \
	$(POETRY) coverage html --skip-empty --show-contexts && \
	$(POETRY) coverage report --fail-under=90





# ----------------------------------------------------------------------------------------
# Commandes liées au modèle (Poetry)
# ----------------------------------------------------------------------------------------

TRAIN_SUBJECT ?= S001
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
- **Definition of Done** : tests verts, CLI/doc à jour.
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

Avant **tout `git commit` ou proposition de commit message**, l’agent doit :

1. **Énoncer cette checklist** dans sa réponse.
2. **Proposer les commandes** à exécuter dans cet ordre exact (2.1).
3. **Demander explicitement** les résultats (logs) si l’agent ne peut pas
   exécuter lui-même les commandes.
4. **Refuser le commit** si une étape n’est pas verte ou inconnue.

Ajout : si une étape est KO, l’agent ne se contente pas de refuser le commit :
il **propose/applique un patch minimal** puis **demande ou exécute** le re-run
jusqu’au vert (boucle de remédiation).
Cette section est le miroir local des jobs CI :

* `pre-commit` (incluant `check yaml/toml`, `fix end of files`,
  `trim trailing whitespace`, `mixed line ending`, `black`, `isort`,
  `ruff`, `mypy`, `bandit`, `radon`, `xenon`, etc.),
* `static-analysis`,
* `tests` (couverture via `make cov`).

L’agent doit toujours rappeler en **texte clair** que :

> « Tant que 2.2, 2.3 ne sont pas toutes ✅, le commit est interdit. »

### 2.1 Préparation (si nouveau clone ou `poetry.lock` modifié)

Si l’agent ne sait pas si l’environnement est à jour (nouveau clone, changement de branche, doute sur `poetry.lock`), il doit exécuter systématiquement 2.1.1 et 2.1.2.

1. `poetry install --no-interaction --with dev`
2. Vérifier que la commande `poetry run pytest -q` fonctionne au moins une fois.

### 2.2 Pre-commit + Static analysis (miroir CI)

L’agent doit proposer cette séquence **exacte** et **dans cet ordre** :

```bash
poetry run pre-commit run --all-files
poetry run black --check .
poetry run isort --check-only .
poetry run ruff check .
poetry run mypy src scripts tests
poetry run pip-audit --progress-spinner=off
```

Règles :

* Si `pre-commit` échoue (`check yaml`, `check toml`, `fix end of files`,
  `trim trailing whitespace`, `mixed line ending`, `black`, `isort`,
  `ruff`, `mypy`, etc.) :

  * l’agent doit :
    * expliquer brièvement l’erreur,
    * appliquer/proposer un **patch minimal**,
    * re-run **d’abord** le(s) hook(s) KO,
    * puis re-run **toute** la séquence 2.2 pour valider le retour au vert.

  * si l’agent a la capacité d’exécuter (ex: Codex), il **doit** re-run lui-même
    et inclure les logs. Sinon, il **demande** les logs au user.
* Tant qu’une commande de 2.2 est **KO** ou non exécutée :

  * l’agent ne doit pas proposer de message de commit,
  * l’agent ne doit pas indiquer que « tout est bon »,
  * l’agent ne doit pas parler de `git push`.

### 2.3 Tests unitaires + couverture (miroir job `tests`)

L’agent doit proposer :

```bash
make cov
```

`make cov` doit :

* générer `coverage.json`, `coverage.xml`, `htmlcov/`,
* se terminer avec `coverage report --fail-under=90` **OK**.

Si `make cov` est **KO**, la réponse doit :

* pointer les fichiers sous-couverts,
* proposer **au moins un test** supplémentaire pour remonter la couverture,
* rappeler de relancer `make cov` jusqu’à obtention de 90 %.

Ajout (boucle) :
- l’agent doit itérer : corriger → re-run `make cov` jusqu’au vert,
  ou s’arrêter après 5 itérations avec diagnostic + options.

### 2.5 Contrat de validité d’un commit

Un commit n’est **valide** que si **toutes** les conditions suivantes sont
satisfaites :

* 2.2 **complète** et **verte**.

La réponse de l’agent doit **toujours** contenir, avant toute suggestion de
message de commit, une synthèse explicite :

> « Checklist 2) :
>
> * 2.2 Pre-commit + static analysis : ✅/❌
>   → Commit autorisé : ✅/❌. »

Si l’agent ne peut pas remplir cette synthèse de façon honnête, il doit
conclure par :

> « ❌ CI potentiellement en échec : commit interdit tant que la checklist 2)
> n’est pas entièrement verte. »

---



## 3) 🧪 Plan de tests (défense‑proof)
**Objectifs** : >= 90 % couverture (branches + diff), **contrôle par fichier**, tests rapides.

### 3.1 Unitaires
-
...


### 3.5 Tolérances numériques (si tests internes)
-
...

## 4) ⚙️ Spécifications d’implémentation

### 4.1 Formules

* Les formules utilisées pour CSP/PCA/ICA doivent être documentées
  (docstring + référence mathématique).

### 4.2 CLI (exemples)

```bash
python mybci.py S001 R01 train
python mybci.py S001 R01 predict
```

### 4.3 Persistance

* Sauvegardes de modèles et paramètres dans un répertoire dédié (`models/`
  ou équivalent), jamais dans `src/`.
* **Ne jamais** committer les datasets bruts ou fichiers issus de Physionet.

---

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

> À maintenir synchronisé avec `docs/risk/tpv_murphy_map.csv`.
> Chaque WBS ID modifié doit rappeler au moins un Murphy ID couvert
> par les tests.

---

## 6) ✅ Procédure de validation finale (soutenance)
1. `pytest -q` → **tout vert**
2. `coverage run -m pytest && coverage json && coverage report --fail-under=90` (branches)
3. **Contrôle par fichier** : script CI sur `coverage.json`
3bis. **Upload vers Codecov** (`coverage.xml`)
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

### 7.2 Modèle de messages d’erreurs
- `ERROR:
- `ERROR:
- `ERROR:

---

## 8) 🔭 Bonus CI perso (hors soutenance 42)
- `vulture`, `bandit`, `radon/xenon` (analyse dead‑code/sécurité/complexité)
- Job Python 3.11 Ubuntu (smoke) en plus du 3.10

---

## 9) 🧾 Format obligatoire des réponses des agents

Toute réponse qui propose **du code ou un changement de fichier** doit
**impérativement** respecter ce format, dans cet **ordre** :

1. **Contexte / WBS / Murphy**

   * WBS ID concerné.
   * Murphy ID concernés.
   * Issue GitHub associée (lien ou titre).

2. **Plan**

   * Liste des étapes de modification du code.
   * Impact attendu sur les risques (Murphy) et sur la BCI (section 🎯).

3. **Checklist pré-commit (section 2)**

   * Rappel des étapes 2.1.
   * Commandes à exécuter **dans l’ordre**.
   * Bloc **obligatoire** :

     ```text
     CHECKLIST 2) – état AVANT COMMIT

     2.2 Pre-commit + static analysis : ✅/❌
     → Commit autorisé                 : ✅/❌
     ```

   * L’agent doit **expliquer** pourquoi il met ✅ ou ❌
     (logs vus, échec, inconnu).

   * Si ❌ :
     - l’agent doit inclure un sous-bloc “BOUCLE DE REMÉDIATION” :
       1) LOG du fail
       2) patch minimal
       3) commandes de re-run (hook KO puis gate complet)
       4) itération N (max 5)

4. **Patch proposé**

   * Diff formaté (un seul patch cohérent).
   * Références claires aux tests impactés ou ajoutés.

5. **Validation**

   * Synthèse explicite (en cohérence avec le bloc CHECKLIST 2) ) :

     * 2.2 : ✅/❌
     * 2.3 : ✅/❌
   * Interdiction d’écrire ou suggérer une commande `git commit` ou
     `git push` si `→ Commit autorisé` n’est pas à ✅.

   * Si une case est ❌ ou inconnue, la réponse doit se terminer par :

     > « ❌ CI potentiellement en échec : commit interdit tant que la
     > checklist 2) n’est pas entièrement verte. »

### 9.3 Exemple attendu (agent autonome : fail → patch → re-run → commit)

Le pattern doit ressembler à :
1) RUN DEV : `pre-commit` KO → log inclus
2) Patch minimal
3) Re-run hook KO → OK
4) RUN AVANT COMMIT : 2.2 + 2.3 → OK
5) Bloc CHECKLIST 2) : ✅ → Commit autorisé ✅ → commit

Toute réponse qui ne suit pas ce format doit être considérée comme **non
conforme** au blueprint AGENTS.

### 9.1 Exemple de réponse idéale (petite modif, tout est vert)

> Cas : correction mineure dans `src/tpv/utils.py` pour clarifier un message
> d’erreur, WBS ID = TPV-UTIL-01.

1. **Contexte / WBS / Murphy**

- WBS ID : TPV-UTIL-01 (utilitaires communs BCI)
- Murphy ID : M-UTIL-01 (messages d’erreurs ambigus)
- Issue GitHub : `TPV-UTIL-01 – Clarifier message d'erreur invalid channel`

2. **Plan**

- Mettre à jour le message d’erreur dans `utils.py` pour mentionner
  explicitement l’ID de canal.
- Ajouter/adapter un test dans `tests/test_utils.py` qui vérifie le message.
- Rejouer la pipeline locale 2.2.

3. **Checklist pré-commit (section 2)**

Commandes à exécuter :

```bash
# 2.1 Préparation (si besoin)
poetry install --no-interaction --with dev
poetry run pytest -q

# 2.2 Pre-commit + static analysis
poetry run pre-commit run --all-files
poetry run black --check .
poetry run isort --check-only .
poetry run ruff check .
poetry run mypy src scripts tests
poetry run pip-audit --progress-spinner=off

# 2.3 Tests + couverture 90 %
make cov

Supposons que toutes ces commandes ont été exécutées avec succès (logs OK).

```text
CHECKLIST 2) – état AVANT COMMIT

2.2 Pre-commit + static analysis : ✅
→ Commit autorisé                 : ✅
```

4. **Patch proposé**

```diff
diff --git a/src/tpv/utils.py b/src/tpv/utils.py
index 1234567..89abcde 100644
--- a/src/tpv/utils.py
+++ b/src/tpv/utils.py
@@ -10,7 +10,8 @@ def validate_channel_index(index: int, max_index: int) -> None:
-    # Vérifie que l'index de canal est dans les bornes autorisées
+    # Refuse explicitement les canaux hors bornes pour faciliter le debug
     if index < 0 or index > max_index:
-        raise ValueError("Invalid channel index")
+        raise ValueError(
+            f"Index de canal invalide: {index} (max autorisé: {max_index})"
+        )
```

Tests associés : ajout/ajustement de `test_validate_channel_index_raises` dans
`tests/test_utils.py`.

5. **Validation**

* 2.2 : ✅ (pre-commit + black + isort + ruff + mypy + pip-audit OK)

→ Commit autorisé : ✅

Message de commit possible :

> `fix: clarifier message d'erreur sur index de canal`


---

### 9.2 Exemple de réponse idéale (KO pre-commit, commit interdit)

> Cas : ajout d’un paramètre dans `mybci.py`, oubli du formatage → ruff/black KO.

1. **Contexte / WBS / Murphy**

- WBS ID : TPV-CLI-02 (CLI mybci – ergonomie)
- Murphy ID : M-CLI-02 (messages usage confus)
- Issue GitHub : `TPV-CLI-02 – Améliorer message d'aide de mybci.py`

2. **Plan**

- Mettre à jour le message d’aide de `mybci.py`.
- Corriger le formatage pour respecter black/ruff.
- Rejouer 2.2 jusqu’à tout vert.

3. **Checklist pré-commit (section 2)**

Commandes à exécuter :

```bash
poetry run pre-commit run --all-files
poetry run black --check .
poetry run isort --check-only .
poetry run ruff check .
poetry run mypy src scripts tests
poetry run pip-audit --progress-spinner=off
make cov
````

Supposons que :

* `poetry run pre-commit run --all-files` échoue sur `ruff` (ligne trop longue),
* les autres commandes **n’ont pas encore été rejouées** après correction.

```text
CHECKLIST 2) – état AVANT COMMIT

2.2 Pre-commit + static analysis : ❌ (ruff KO sur mybci.py)
→ Commit autorisé                 : ❌
```

4. **Patch proposé**

L’agent propose un patch minimal pour corriger la ligne incriminée
(formatage / longueur).
*Aucun message de commit n’est proposé.*

5. **Validation**

* 2.2 : ❌ (doit être rejoué jusqu’à succès)
* 2.3 : ❌ (doit être rejoué après correction)

> « ❌ CI potentiellement en échec : commit interdit tant que la checklist 2)
> n’est pas entièrement verte. »

---

## 📚 Charte de nommage des fonctions, parametres, argumements et variables pédagogique

### 1. Objectif

Cette charte a pour but d’aider un étudiant à écrire du code Python :

* lisible ;
* cohérent ;
* pédagogique ;
* compréhensible par un lecteur non spécialiste.

Le nommage ne sert pas seulement à faire fonctionner le code.
Il sert à **rendre la logique visible**.

Un bon nom doit permettre de comprendre :

* ce que contient la donnée ;
* à quoi elle sert ;
* dans quel contexte elle existe ;
* sans devoir relire toute la fonction.

---

### 2. Principe fondamental

**Un bon nom évite au lecteur de devoir deviner.**

Si le lecteur doit se demander :

* “qu’est-ce que cette donnée contient ?”
* “à quoi sert-elle ?”
* “de quel domaine parle-t-on ?”
* “s’agit-il d’une valeur, d’une collection, d’un compteur, d’un index, d’un chemin, d’un score, d’un coefficient ?”

alors le nom est mauvais ou insuffisant.

### 2.1. Niveau de force des règles

Pour éviter les interprétations trop absolues, les règles de cette charte
doivent être lues selon trois niveaux :

* **à éviter fortement** : mauvais choix dans la majorité des cas ;
* **acceptable selon le contexte** : peut convenir dans un code générique,
  scientifique ou très local ;
* **recommandé** : choix à privilégier dans un code pédagogique clair.

La qualité d’un nom dépend toujours :
* du domaine ;
* du public visé ;
* du niveau d’abstraction ;
* du type de code.

---

### 3. Règle d’or

**Nommer selon le sens réel, en fonction du contexte.**

Par défaut, un bon nom décrit :

1. la réalité du domaine ;
2. le rôle de la donnée ;
3. la nature exacte de la valeur.

Il ne doit pas, sauf besoin réel :

* décrire seulement une formule ;
* décrire seulement une structure interne ;
* utiliser un mot vague qui semble “assez correct” ;
* recopier une notation de cours sans se demander si elle reste claire ici.

#### À éviter fortement

```python
x
y
z
k
m
n
theta
alpha
res
tmp
df
data
input_data
label
labels
result
configuration
row_count
column_count
```

Les noms comme `feature_matrix`, `target_vector`, `gradient`,
`prediction` ou `weights` ne sont pas faux en soi,
mais deviennent insuffisants lorsqu’un nom plus métier
ou plus précis est possible.

#### Meilleur

```python
student_age
student_house_index
current_house_index
student_count
subject_count
house_coefficients
learning_rate
predicted_house_indices
temporary_file_path
student_records
current_house_coefficient_gradient
training_settings
student_subject_scores
```

---

### 4. Priorité de décision

Quand tu choisis un nom, applique cet ordre.

#### 4.1. Le métier réel d’abord

Nommer la réalité du domaine.

#### Exemples

```python
student_house_index
exam_score
fuel_price
invoice_total
customer_email
```

Les exemples pédagogiques peuvent varier selon le domaine :
éducation, finance, santé, logistique, industrie, web, data science, etc.
La règle reste la même : nommer la réalité du domaine,
pas un mot générique interchangeable.

---

#### 4.2. Le rôle réel ensuite

Préciser la fonction de la donnée dans le programme.

#### Exemples

```python
predicted_house_index
validated_email
selected_customer_email
current_iteration
training_configuration
```

---

#### 4.3. La nature exacte de la valeur ensuite

Préciser si c’est un compteur, un chemin, une probabilité, un coefficient, etc.

#### Exemples

```python
student_count
model_file_path
predicted_probability
house_index
accuracy_score
```

---

#### 4.4. La structure seulement si elle est utile

**Règle par défaut :**
la forme interne ne doit pas être le premier niveau de nommage
quand le métier est connu.

#### À éviter en premier choix

```python
matrix
vector
row
column
array
list
dict
```

#### À préférer si le métier est connu

```python
student_subject_scores
house_names
student_house_indices
scores_by_house
student_count
```

---

#### 4.5. La notation mathématique en dernier recours

Des mots comme `theta`, `alpha`, `beta`, `x`, `y`, `z`
ne doivent être gardés que si :

* le contexte est purement mathématique ;
* le bloc est très local ;
* le lecteur visé comprend déjà cette notation.

Dans un projet pédagogique général, ils doivent être renommés.

---

### 5. Contextes de code

La qualité d’un nom dépend du type de code.

#### 5.1. Code pédagogique débutant

Priorité maximale à la clarté immédiate.
On évite fortement :

* notations mathématiques ;
* jargon trop compact ;
* noms orientés structure ;
* mots vagues.

#### 5.2. Code métier applicatif

On privilégie les noms du domaine métier.

#### Exemples

```python
invoice_total
customer_email
shipping_address
payment_status
```

#### 5.3. Code scientifique ou mathématique

Certaines notations peuvent être acceptées **localement**
si elles sont standards pour le public visé.
Mais dès que le code sort d’une démonstration courte,
il faut revenir à des noms explicites.

#### 5.4. Code générique de librairie

Quand le métier n’existe pas, n’est pas connu,
ou ne doit volontairement pas apparaître,
des noms plus structurels peuvent être légitimes.

#### Exemples acceptables

```python
row_index
column_index
value_count
input_values
```

#### Règle pratique

**Plus le contexte métier est connu, plus le nom doit être métier.**
**Plus le code est générique, plus un nom technique peut être `acceptable`.**

---

### 6. Langue à utiliser

#### Règle

Les identifiants Python doivent être écrits en **anglais**.

#### Pourquoi

* c’est la convention standard en Python ;
* cela évite le mélange de langues ;
* cela améliore la réutilisabilité du code ;
* cela facilite la lecture par d’autres développeurs.

#### Tolérance pédagogique

Les commentaires et docstrings peuvent être écrits en français si le public est francophone.

#### Interdit

Mélanger les langues dans les identifiants.

#### Mauvais

```python
maison_labels
nombre_iterations
current_theta
```

#### Correct

```python
house_indices
iteration_count
current_house_coefficients
```

---

### 7. Qualités attendues d’un bon nom

Un bon nom doit être :

* **clair** : le sens apparaît vite ;
* **simple** : il reste lisible ;
* **précis** : il décrit bien ce qu’il représente ;
* **cohérent** : il suit les mêmes règles que le reste du projet ;
* **stable** : il ne devient pas faux au moindre changement interne ;
* **orienté sens** : il parle du contenu avant de parler du contenant.

---

### 8. Ce qu’il faut éviter

### 8.1. Les noms d’une lettre

Sauf cas extrêmement local.

#### Mauvais

```python
i = 0
x = data
y = labels
m, n = scores.shape
```

#### Meilleur

```python
student_index = 0
student_subject_scores = subject_scores
student_house_indices = house_indices
student_count, subject_count = student_subject_scores.shape
```

---

### 8.2. Les abréviations opaques

#### Mauvais

```python
num_iters
grad
preds
cfg
lbls
res
tmp
```

#### Meilleur

```python
iteration_count
coefficient_gradient
predicted_house_indices
training_configuration
house_indices
prediction_result
temporary_file_path
```

---

### 8.3. Les noms trop génériques

#### Mauvais

```python
data
value
item
object
thing
result
test
var
input
label
prediction
configuration
gradient
```

#### Pourquoi

Ils ne disent pas ce que contient réellement la donnée.

#### Meilleur

```python
student_records
predicted_house_index
input_file_path
training_loss
validated_email
current_house_coefficient_gradient
house_classifier_training_settings
student_house_indices
```

---

### 8.4. Les noms purement mathématiques

#### Mauvais

```python
x
y
z
theta
alpha
beta
k
```

#### Meilleur

```python
student_subject_scores
student_house_indices
linear_scores
house_coefficients
learning_rate
regularization_strength
current_house_index
```

---

### 8.5. Les noms trop longs

Un nom ne doit pas devenir une phrase.

#### Trop long

```python
normalized_student_subject_scores_with_bias_term_already_added
```

#### Mieux

```python
student_subject_scores_with_bias
```

Le détail complémentaire doit aller dans :

* la docstring ;
* le commentaire ;
* le nom de la fonction ;
* la documentation.

---

### 8.6. Les noms orientés structure au lieu du contenu

#### Mauvais

```python
row_count
column_count
matrix
vector
table
row_index
column_index
```

#### Pourquoi

Ils décrivent surtout la disposition interne.
Ils deviennent insuffisants dès que le domaine métier est connu
et qu’un nom plus concret est possible.

#### Meilleur quand le métier est connu

```python
student_count
subject_count
student_subject_scores
student_index
subject_index
house_names
```

#### Exception acceptable

Quand le métier est inconnu, temporaire ou volontairement générique,
un nom structurel peut être accepté.

#### Exemple acceptable en code générique

```python
value_count
row_index
column_index
```

Mais dès que le domaine est connu, il faut renommer plus précisément.

---

### 8.7. Les mots corrects mais souvent pédagogiquement insuffisants

Un mot peut être techniquement correct, mais rester insuffisant
dans un code pédagogique si le contexte exige plus de précision.

#### Souvent insuffisants seuls, selon le contexte

```python
data
input
label
labels
prediction
predictions
gradient
result
configuration
score
weights
bias
feature
sample
record
output
model_output
```

#### Pourquoi

Ils ne disent pas assez :

* de quoi on parle ;
* quel est le domaine ;
* quel est le rôle exact ;
* quelle valeur ils portent.

#### Meilleur selon contexte

```python
student_subject_scores
student_house_indices
predicted_house_indices
current_house_coefficient_gradient
house_classifier_training_settings
exam_score
house_coefficients
bias_term
selected_subject_names
training_example_count
student_record
predicted_house_probability
```

---

### 9. Faux amis, termes ambigus et mots trompeurs

Choisir un mot anglais ne suffit pas.
Il faut choisir un mot :

* correct ;
* compréhensible ;
* non ambigu pour le public visé.

#### Règle générale

Avant de conserver un mot anglais dans un identifiant, vérifier :

1. s’il ressemble fortement à un mot français ;
2. si son sens technique anglais est bien le bon ;
3. si un débutant francophone risque de mal l’interpréter ;
4. s’il existe un terme plus explicite dans le contexte réel.

Si un doute subsiste, il faut renommer.

#### Problème pédagogique principal

Un faux ami ne crée pas toujours une erreur d’exécution.
Il crée souvent :

* une mauvaise compréhension ;
* un raisonnement flou ;
* un apprentissage fragile ;
* un code qui semble clair seulement pour son auteur.

---

### 10. Cas fréquents en informatique, data et IA

Dans certains projets de machine learning, des termes comme
`feature`, `label`, `weights` ou `bias` sont standards et acceptables.
La règle pédagogique n’est pas de les interdire,
mais de vérifier s’ils sont suffisamment précis pour le lecteur visé.

#### `sample`

Souvent trop vague.

#### Mauvais

```python
sample_count
samples
sample_data
```

#### Meilleur selon le contexte

```python
observation_count
training_example_count
student_count
record_count
signal_window_count
```

---

#### `feature`

Très ambigu.

#### Mauvais

```python
feature_matrix
features
selected_features
```

#### Meilleur selon le contexte

```python
student_subject_scores
input_variables
selected_subject_names
predictor_values
```

---

#### `label`

Correct en ML, mais trop faible seul dans beaucoup de codes pédagogiques.

#### Mauvais

```python
labels
target_labels
class_labels
```

#### Meilleur selon le contexte

```python
house_indices
target_house_indices
house_names
expected_house_indices
```

---

#### `score`

Acceptable, mais souvent trop vague seul.

#### Mauvais

```python
score
scores
final_score
```

#### Meilleur selon le contexte

```python
exam_score
decision_score
predicted_probability
accuracy_score
house_score
```

---

#### `grade`

Piégeux pour un francophone.

#### Préférer selon contexte

```python
exam_score
discipline_score
student_score
score_level
```

---

#### `record`

Utile, mais parfois ambigu pour un francophone.

#### Acceptable selon contexte

```python
student_record
student_records
record_count
```

---

#### `observation`

Très bon terme en data science.

#### Bon usage

```python
observation_count
observations
customer_observations
```

---

#### `instance`

Correct, mais ambigu en Python orienté objet.

#### Préférer souvent

```python
observation
record
training_example
```

---

#### `bias`

Mot très ambigu.

#### Mauvais

```python
bias
bias_value
```

#### Meilleur selon le contexte

```python
bias_term
intercept
has_bias_term
dataset_bias
```

---

#### `weights`

Mot standard en ML, parfois moins pédagogique qu’un autre.

#### Acceptable

```python
model_weights
```

#### Parfois plus clair

```python
model_coefficients
house_coefficients
```

---

#### `actual`

Très gros faux ami.

#### Pour “réel”

```python
actual_cost
actual_result
```

#### Pour “actuel”

```python
current_cost
current_result
```

---

#### `eventual` / `eventually`

Très dangereux.

Sens correct :

* final ;
* à terme ;
* finalement.

---

#### `issue`

Sur GitHub, signifie souvent :

* problème ;
* ticket ;
* sujet de suivi.

---

#### `consistency`

Signifie :

* cohérence

et non :

* consistance

---

#### `comprehensive`

Signifie :

* complet ;
* exhaustif

et non :

* compréhensif

---

### 11. Règle pratique de remplacement

Quand un mot semble ambigu, remplace-le par un mot
qui répond immédiatement à au moins une de ces questions :

* qu’est-ce que cette donnée contient ?
* de quel domaine parle-t-on ?
* s’agit-il d’une valeur ou d’une collection ?
* s’agit-il d’un compteur, d’un index, d’un score, d’une probabilité, d’un chemin, d’un coefficient ou d’un résultat ?

---

### 12. Règles par type d’élément

### 12.1. Fichiers Python

#### Convention

`snake_case.py`

#### Règle

Le nom du fichier doit refléter ce qu’il contient réellement.

#### Correct

```python
student_data_loader.py
house_classifier.py
model_training.py
prediction_service.py
csv_validator.py
```

#### Mauvais

```python
utils2.py
test3.py
script_final_v2.py
mon_fichier.py
stuff.py
```

#### À éviter

* `final`
* `new`
* `temp`
* `test`
* `misc`
* `utils` sans précision

#### Préférer

```python
date_parser.py
string_formatter.py
path_helpers.py
```

---

### 12.2. Classes

#### Convention

`PascalCase`

#### Règle

Le nom de classe doit désigner une entité, un rôle ou un concept.

#### Correct

```python
StudentRecord
HouseClassifier
CsvReader
PredictionResult
TrainingConfiguration
```

#### Mauvais

```python
data
manager2
testClass
myClass
stuff
```

---

### 12.3. Fonctions

#### Convention

`snake_case`

#### Règle

Le nom d’une fonction doit commencer par un verbe clair.

#### Correct

```python
load_training_data
validate_csv_columns
train_house_classifier
compute_accuracy_score
save_model_coefficients
predict_student_house
```

#### Mauvais

```python
data_training
house_prediction
csv
model
do_stuff
process_data
handle_event
run
```

---

### 12.4. Variables

#### Convention

`snake_case`

#### Règle

Une variable doit décrire ce qu’elle contient,
pas seulement son type,
pas seulement sa structure,
pas seulement un rôle vague.

#### Mauvais

```python
data_frame
list_values
string_value
dict_result
matrix
labels
result
configuration
```

#### Meilleur

```python
student_records
selected_house_names
error_message
house_score_by_student
training_configuration
predicted_house_indices
```

#### Rappel

Le type n’est pas le sens.
La structure n’est pas le sens.
Un mot générique n’est pas une explication.

---

### 12.5. Constantes

#### Convention

`UPPER_CASE`

#### Correct

```python
DEFAULT_LEARNING_RATE = 0.01
MAX_ITERATION_COUNT = 1000
REQUIRED_SUBJECT_NAMES = ["Arithmancy", "Astronomy", "Herbology"]
MODEL_FILE_NAME = "weights.json"
```

---

### 12.6. Booléens

Un booléen doit pouvoir se lire comme une affirmation.

#### Préfixes recommandés

* `is_`
* `has_`
* `can_`
* `should_`
* `was_`

#### Correct

```python
is_valid
has_bias_term
can_train_model
should_save_output
was_loaded_successfully
```

---

### 12.7. Collections

Le nom d’une collection doit indiquer ce qu’elle regroupe.

#### Correct

```python
student_names
house_indices
valid_file_paths
scores_by_house
student_count_by_house
```

#### Bonne pratique pour les dictionnaires

```python
scores_by_house
student_count_by_house
coefficients_by_house
```

---

### 12.8. Compteurs

Le nom doit dire clairement ce qui est compté.

#### Correct

```python
student_count
subject_count
iteration_count
house_count
error_count
record_count
```

---

### 12.9. Index

Le nom doit préciser ce qu’il indexe.

#### Correct

```python
student_index
subject_index
house_index
iteration_index
record_index
```

#### Exception tolérée

Dans une boucle très courte, un nom simple mais explicite reste préférable à une lettre seule.

```python
for student_index, student_name in enumerate(student_names):
    ...
```

---

### 12.10. Chemins, fichiers, URLs

#### Correct

```python
input_file_path
output_directory_path
model_file_path
report_url
csv_file_name
```

---

### 12.11. Exceptions

Une exception doit expliquer ce qui ne va pas.

#### Correct

```python
InvalidCsvFormatError
MissingSubjectColumnError
ModelNotTrainedError
EmptyDatasetError
```

---

### 13. Métier avant technique

Cette règle résume la logique générale de la charte :

1. nommer d’abord le domaine ;
2. préciser ensuite le rôle ;
3. préciser enfin la nature exacte de la valeur ;
4. ne faire apparaître la structure interne que si elle aide réellement.

#### Trop technique

```python
feature_matrix
target_vector
parameter_array
row_count
column_count
```

#### Plus pédagogique

```python
student_subject_scores
student_house_indices
house_coefficients
student_count
subject_count
```

#### Encore mieux si le métier est connu

```python
student_exam_scores
student_house_indices
house_prediction_scores
```

---

### 14. Singulier et pluriel

Le nom doit refléter si la donnée contient une seule valeur ou plusieurs.

#### Correct

```python
house_index
house_indices
student_name
student_names
prediction_score
prediction_scores
```

#### Mauvais

```python
houses_label
student
scores_list
```

---

### 15. Révision avant validation

Avant de valider un nom, poser ces questions :

1. Un débutant comprend-il ce que contient cette donnée ?
2. Son rôle est-il clair sans relire toute la fonction ?
3. Le nom exprime-t-il le sens plutôt que la formule ?
4. Le nom exprime-t-il le contenu plutôt que la disposition ?
5. Est-il assez précis pour éviter le doute ?
6. Est-il assez court pour rester lisible ?
7. Est-il cohérent avec le reste du projet ?
8. Le mot choisi risque-t-il d’être ambigu pour le public visé ?
9. Est-ce un nom adapté au type de code
   (pédagogique, métier, scientifique, générique) ?

Si une réponse pose problème, le nom doit être revu.

---

### 16. Labels d’erreur de nommage

Cette liste peut servir de grille de relecture, de correction ou de feedback.
Elle complète la charte, mais ne remplace pas le jugement de contexte.
Un même nom peut être acceptable ou non selon le type de code,
le public visé et le niveau de précision réellement nécessaire.

#### `NOM-001` — Nom trop court

#### `NOM-002` — Abréviation opaque

#### `NOM-003` — Nom mathématique

#### `NOM-004` — Mélange de langues

#### `NOM-005` — Nom trop générique

#### `NOM-006` — Jargon technique inutile

#### `NOM-007` — Nom trompeur

#### `NOM-008` — Nom trop long

#### `NOM-009` — Singulier/pluriel incohérent

#### `NOM-010` — Détail d’implémentation inutile

#### `NOM-011` — Nom orienté structure

#### `NOM-012` — Mot correct mais pédagogiquement flou

#### `NOM-013` — Nom mal adapté au contexte de code

---

### 17. Méthode pratique de renommage

Quand tu corriges un nom, fais-le dans cet ordre :

1. identifier ce que la donnée représente réellement ;
2. identifier son rôle dans le programme ;
3. identifier le domaine précis ;
4. choisir le mot le plus simple qui reste exact ;
5. ajouter seulement la précision nécessaire ;
6. vérifier que le nom reste lisible ;
7. vérifier qu’il n’exprime pas seulement la structure ;
8. vérifier qu’il est adapté au type de code.

---

### 18. Modèle de décision

#### Mauvais raisonnement

> “C’est ce qu’on voit dans le cours, donc je garde `X`, `y`, `theta`.”

#### Mauvais raisonnement

> “C’est mieux que `x`, donc `input_data`, `labels`, `result` suffisent.”

#### Bon raisonnement

> “Le lecteur doit comprendre ce que cette donnée représente, sans connaître la formule, sans deviner le domaine, et sans devoir interpréter un mot vague.”

---

### 19. Checklist opérationnelle

#### À faire

* utiliser l’anglais pour les identifiants ;
* nommer selon le sens réel ;
* préférer le métier à la structure ;
* choisir des mots simples ;
* préciser ce qui est compté ;
* utiliser un verbe clair pour les fonctions ;
* nommer les booléens comme des affirmations ;
* rester cohérent dans tout le projet ;
* adapter le nom au type de code.

#### À éviter

* lettres seules ;
* abréviations ;
* jargon inutile ;
* mélange de langues ;
* noms trop vagues ;
* noms mathématiques ;
* noms orientés structure ;
* mots corrects mais flous ;
* détails techniques inutiles dans le nom.

---

### 20. Version courte à mémoriser

Par défaut, un bon nom doit dire :

* ce que c’est ;
* ce que ça contient ;
* à quoi ça sert ;
* de quel domaine on parle.

Et si nécessaire, il doit aussi préciser :

* s’il s’agit d’une valeur ou d’une collection ;
* s’il s’agit d’un compteur, d’un index, d’un score,
  d’un chemin, d’une probabilité ou d’un coefficient.

S’il n’aide pas réellement le lecteur à comprendre,
il faut le renommer.

---

### 21. Formule finale

**Le nommage n’est pas une décoration.
C’est une partie de l’explication du programme.**

Le bon nom n’ajoute pas du style :
il retire du doute.

---

### 22. Mini fiche mémo

#### À éviter fortement dans un projet pédagogique

```python
x, y, z, i, j, k, df, tmp, res, obj, val, data
label, labels, result, prediction, predictions
gradient, configuration, matrix, vector, row, column
```

#### Recommandé

```python
student_subject_scores
student_house_indices
current_house_index
student_count
subject_count
house_coefficients
predicted_house_indices
is_valid
output_file_path
training_configuration
current_house_coefficient_gradient
```

#### Fonctions

```python
load_student_records
validate_input_file
train_house_classifier
predict_house_indices
save_prediction_results
```

#### Classes

```python
StudentRecord
HouseClassifier
TrainingConfiguration
PredictionResult
```

#### Constantes

```python
DEFAULT_LEARNING_RATE
MAX_ITERATION_COUNT
REQUIRED_SUBJECT_NAMES
MODEL_FILE_NAME
```


---

## 📚 Documentation du code

Lorsque tu génères, modifies ou réécris du code, applique strictement les
règles suivantes.

### Objectif

Les commentaires doivent documenter **chaque ligne de code**.

Le but est d’avoir **un commentaire juste au-dessus de chaque ligne**,
y compris pour les imports, constantes, affectations, conditions,
retours, appels de fonctions, transformations intermédiaires,
boucles, compréhensions, structures de contrôle et expressions composées.

Cette documentation doit aider :

* un développeur qui reprend le code plus tard ;
* un mainteneur occasionnel qui revient dessus dans 3 ou 5 ans ;
* un lecteur non développeur qui doit malgré tout comprendre
  le rôle local de chaque ligne ;
* un futur modificateur qui doit identifier rapidement
  **où intervenir** pour changer un comportement précis.

Le code reste la source principale d’exécution,
mais le commentaire devient la source principale d’**explication locale**.

---

### Principe fondamental

**Le code doit rester lisible par lui-même.  
Le commentaire doit rendre la lecture, la maintenance et la modification
plus sûres.**

Chaque ligne doit être commentée.

La priorité reste, dans cet ordre :

1. un **meilleur nommage** ;
2. une **variable intermédiaire explicite** ;
3. une **extraction de fonction** ;
4. un **commentaire au-dessus de la ligne**.

Autrement dit :

* on ne garde jamais un code opaque sous prétexte qu’il sera commenté ;
* on améliore d’abord le code ;
* puis on documente chaque ligne pour rendre son rôle clair,
  même à un lecteur non expert.

---

### Hiérarchie de qualité d’un commentaire

Pour chaque ligne, le commentaire doit chercher à fournir,
dans cet ordre de priorité, le niveau le plus utile :

1. **le pourquoi** de la ligne ;
2. **le rôle utile** de la ligne dans le bloc ou l’algorithme ;
3. **l’effet concret utile** de la ligne sur les données,
   le contrat, la structure ou le flux ;
4. **le point de repérage de maintenance** :
   ce que cette ligne pilote, verrouille, influence ou contraint.

Autrement dit :

* si un vrai **pourquoi** existe, il faut l’écrire ;
* sinon, on documente **à quoi sert la ligne ici** ;
* sinon, on documente **ce qu’elle change concrètement** ;
* sinon, on documente **où revenir pour modifier ce comportement**.

---

### Règle absolue

Chaque commentaire doit apporter au moins une des informations suivantes :

* pourquoi cette ligne existe sous cette forme ;
* pourquoi cette implémentation a été retenue ici ;
* quel rôle exact elle joue dans le flux ;
* quel effet concret elle produit sur les données ou le comportement ;
* quel risque elle évite ;
* quelle garantie elle protège ;
* quelle contrainte elle respecte ;
* quel contrat elle préserve ;
* quelle robustesse, stabilité ou compatibilité elle apporte ;
* quel point de modification futur elle constitue.

Un commentaire ne doit jamais :

* se contenter de recopier mot à mot la ligne ;
* paraphraser trivialement la syntaxe ;
* être faux ou approximatif ;
* nommer seulement l’API sans expliquer son intérêt ;
* être décoratif ;
* masquer un code inutilement opaque qu’on aurait pu mieux nommer.

---

### Règle de couverture exhaustive

Puisque l’objectif est d’avoir **un commentaire sur chaque ligne**,
on n’utilise plus la règle :

> « ne commente pas si la ligne est évidente »

À la place, on applique la règle suivante :

> **Même si la ligne est simple, elle doit être documentée,
> avec le niveau d’explication le plus utile disponible.**

Cela signifie :

* une ligne importante reçoit un commentaire riche ;
* une ligne simple peut recevoir un commentaire court ;
* une ligne évidente ne doit pas rester sans commentaire ;
* mais son commentaire doit quand même aider la lecture future.

---

### Test obligatoire avant chaque commentaire

Avant d’écrire un commentaire, vérifie que :

1. il aide à comprendre **pourquoi**, **à quoi sert**,
   **ce que change** ou **où modifier** cette ligne ;
2. il apporte au moins une information utile absente
   de la simple lecture brute du code ;
3. il reste intelligible pour un lecteur non expert ;
4. il améliore réellement la maintenance, le diagnostic,
   la robustesse ou la localisation d’un futur changement.

Si un vrai “pourquoi” n’existe pas,
n’abandonne pas le commentaire :
descends au niveau suivant de la hiérarchie
(rôle utile, effet concret, repérage de maintenance).

---

### Ce qu’un bon commentaire peut expliquer

Un bon commentaire peut documenter :

* une intention de conception ;
* une contrainte technique ou métier ;
* un invariant à préserver ;
* un risque évité ;
* un compromis assumé ;
* une robustesse recherchée ;
* une stabilité de test ;
* une compatibilité inter-OS ou inter-environnements ;
* une exigence de maintenabilité ;
* une contrainte de performance ;
* un diagnostic exploitable ;
* un contrat d’interface ;
* un comportement attendu en cas d’erreur ;
* une normalisation volontaire ;
* une convention retenue pour fiabiliser le système ;
* une hypothèse d’entrée ou de format ;
* une limite connue ;
* une dette technique explicitement assumée ;
* une contrainte imposée par une API, un protocole, un format ou un outil ;
* la structure des données manipulées ;
* le point exact à modifier pour changer un comportement.

---

### Formulation attendue

Le commentaire doit être formulé, selon le cas, comme :

* une **justification** ;
* une **explication de rôle** ;
* une **explication d’effet concret** ;
* une **indication de repérage pour maintenance** ;
* une **contrainte**, une **garantie**, une **hypothèse**
  ou une **limite utile**.

Formulations adaptées :

* Pour garantir…
* Pour éviter…
* Pour préserver…
* Pour stabiliser…
* Pour fiabiliser…
* Pour conserver…
* Pour limiter…
* Pour protéger…
* Pour maintenir…
* Pour distinguer clairement…
* Pour garder un contrat cohérent…
* Pour rendre le diagnostic exploitable…
* Pour éviter qu’un cas limite casse…
* Pour imposer une représentation canonique…
* Pour réduire une ambiguïté de comportement…
* Cette ligne prépare…
* Cette ligne aligne…
* Cette ligne réutilise…
* Cette ligne verrouille…
* Cette ligne pilote…
* Cette ligne sert de point d’entrée pour…
* C’est ici qu’il faut intervenir pour modifier…
* Convention imposée par…
* Compatibilité requise avec…
* Limite volontaire : …
* Hypothèse : …
* Précondition : …
* Postcondition attendue : …

Formulations à éviter en général :

* Importe…
* Initialise…
* Calcule…
* Retourne…
* Vérifie…
* Exécute…
* Normalise…
* Capture…
* Construit…
* Ajoute…
* Supprime…
* Transforme…
* Affiche…

Ces verbes ne sont pas interdits absolument,
mais ils sont insuffisants s’ils décrivent seulement l’action visible
sans expliquer son intérêt local.

---

### Règle spéciale pour attributs, méthodes, fonctions ou API au nom opaque

Quand une ligne contient un attribut, une méthode, une fonction, un module,
une librairie ou une notation dont le nom n’est **pas transparent par lui-même**
(ex. symbole court, abréviation, convention mathématique, API cryptique,
notation implicite comme `.T`, `.dot`, `.iloc`, `np`, `pd`, etc.),
le commentaire peut et doit expliciter :

1. **ce que cet élément fait concrètement** ;
2. **à quoi il sert ici** ;
3. **pourquoi cette opération est utilisée dans ce contexte** ;
4. **où il faut intervenir si l’on veut changer ce comportement**.

Cette exception est importante,
car certaines notations sont compactes pour l’expert,
mais opaques pour un futur lecteur ou modificateur.

#### Ce qu’il faut faire dans ce cas

Quand un nom est opaque, le commentaire peut préciser :

* ce que l’élément fait en termes simples et concrets ;
* quel changement de représentation il produit ;
* quel rôle mathématique, algorithmique ou structurel il joue ;
* pourquoi cette transformation est nécessaire ici ;
* quelle compatibilité de dimensions, de contrat ou de représentation
  elle garantit ;
* quel risque d’ambiguïté elle lève pour le lecteur ;
* à quel endroit il faudra revenir pour modifier cette logique.

#### Exception autorisée pour les notations très opaques

Pour des notations très compactes comme `.T` ou `.dot`,
il est autorisé de mentionner explicitement leur effet concret.

Exemples de reformulations plus parlantes :

* `.T` :
  échange lignes et colonnes pour réorienter la lecture des données ;
* `.dot` :
  combine des valeurs alignées entre deux structures numériques
  pour produire un score, une somme pondérée ou une agrégation vectorisée.

On ne se limite donc pas au terme académique ;
on cherche une formulation compréhensible et exploitable.

#### Ce qu’il ne faut pas faire

Ne pas écrire un commentaire qui :

* se contente de nommer l’API ;
* répète un terme technique sans le rendre plus clair ;
* décrit mécaniquement la syntaxe sans expliquer son intérêt ici ;
* oublie le rôle de la ligne dans le calcul ou dans la maintenance.

#### Formulation attendue en cas de nom opaque

Quand c’est utile, on peut combiner :

1. l’effet concret ;
2. la signification conceptuelle ;
3. la raison locale ;
4. l’impact maintenance.

Exemples de formulations adaptées :

* Pour échanger lignes et colonnes avant d’agréger l’erreur par variable
* Pour combiner chaque erreur avec la variable correspondante
* Pour produire la somme pondérée utilisée par la mise à jour des poids
* C’est ici que se joue l’alignement entre variables et erreurs
* Modifier cette ligne change la manière dont les contributions sont agrégées

---

### Règle spéciale de repérage pour futur modificateur

Comme cette documentation doit aussi servir à un futur lecteur
qui voudra corriger ou faire évoluer le code,
chaque commentaire peut indiquer, quand c’est pertinent :

* ce que la ligne pilote ;
* ce que sa modification changera ;
* quel comportement dépend d’elle ;
* quelle donnée, quelle règle ou quel format elle verrouille ;
* dans quel bloc revenir pour changer une logique précise.

Exemples :

* C’est ici que l’ordre des colonnes est figé
* Modifier cette ligne changera la normalisation appliquée au test
* Cette ligne impose le nom final de la colonne exportée
* C’est ce bloc qu’il faut ajuster pour changer l’imputation des valeurs manquantes

---

### Types de commentaires autorisés

#### 1. Commentaire de ligne

Obligatoire au-dessus de chaque ligne.

#### 2. Commentaire de bloc

Autorisé en plus lorsqu’un ensemble de lignes participe
à une même intention forte.

Le commentaire de bloc ne remplace pas les commentaires de ligne ;
il les complète.

#### 3. Docstring

Réservée aux modules, classes et fonctions.
Elle documente le **contrat global**.

#### 4. Commentaire de convention ou de limite

Autorisé lorsqu’il faut signaler :

* une convention imposée ;
* une hypothèse non évidente ;
* une limite volontaire ;
* une compatibilité requise ;
* une dette technique connue.

---

### Règles de commentaires

* Ajoute un commentaire **au-dessus de chaque ligne de code**.
* Le commentaire doit respecter **l’indentation** du bloc.
* **Langue : français**.
* Les termes techniques anglais sont autorisés uniquement pour :
  * les noms d’API ;
  * les types ;
  * les constantes ;
  * les mots-clés du langage ;
  * les noms de fonctions, classes, modules ou outils.
* **80 caractères maximum par ligne de commentaire**.
* Si une explication complète dépasse 80 caractères,
  la répartir sur plusieurs lignes de commentaire.
* **Interdit** :
  * commentaire en fin de ligne ;
  * commentaire sous la ligne ;
  * paraphrase brute du code ;
  * commentaire décoratif ;
  * commentaire faux ;
  * commentaire générique sans utilité de lecture ou de maintenance ;
  * commentaire inventant une justification absente du contexte.

---

### Règle de réécriture

Si un commentaire existant est trop faible,
réécris-le pour exprimer à la place :

* pourquoi la ligne existe ;
* à quoi elle sert dans le bloc ;
* ce qu’elle change concrètement ;
* ce qu’elle protège ;
* ce qu’elle impose ;
* ce qu’un futur modificateur doit savoir avant d’y toucher ;
* où intervenir pour modifier le comportement concerné.

Si un vrai “pourquoi” ne peut pas être formulé,
le commentaire doit au minimum documenter
le **rôle utile** ou l’**effet concret** de la ligne.

---

### Docstrings

Utilise des **docstrings uniquement** pour :

* les modules ;
* les classes ;
* les fonctions.

Les docstrings doivent couvrir, selon le contexte :

* le but global ;
* les paramètres ;
* la valeur de retour ;
* les erreurs levées ;
* le contrat global d’utilisation ;
* les préconditions utiles ;
* les postconditions utiles ;
* les effets de bord notables ;
* les conventions de format, d’unité ou de représentation si nécessaires.

Les docstrings ne remplacent pas les commentaires ligne par ligne.

---

### Règle d’action

Quand tu traites du code :

1. améliore d’abord le nommage si le code est ambigu ;
2. introduis une variable intermédiaire si elle clarifie l’intention ;
3. extrais une fonction si cela rend le bloc plus lisible ;
4. ajoute ensuite un commentaire au-dessus de chaque ligne ;
5. formule en priorité le pourquoi ;
6. à défaut, formule le rôle utile de la ligne ;
7. à défaut, formule son effet concret ;
8. à défaut, formule son intérêt pour un futur modificateur ;
9. n’invente jamais une justification absente du contexte ;
10. garde des commentaires exacts, utiles et lisibles.

---

### Exemples interdits

#### Exemple interdit 1 : paraphrase pauvre

```py
# On importe Path
from pathlib import Path

# On supprime les espaces
normalized_url = url.strip()

# On retourne 0
return 0
````

Pourquoi c’est interdit :

* le commentaire répète l’action visible ;
* il n’aide ni la maintenance, ni la compréhension ;
* il ne dit pas pourquoi la ligne existe ici.

#### Exemple interdit 2 : API opaque mal documentée

```py
# .T transpose la matrice
transposed_scores = student_scores.T

# .dot fait un produit matriciel
error_sum = transposed_scores.dot(prediction_error_by_student)
```

Pourquoi c’est interdit :

* le terme technique est répété sans être rendu clair ;
* le rôle local dans l’algorithme n’est pas expliqué ;
* un futur modificateur ne sait pas ce que changerait cette ligne.

#### Exemple interdit 3 : commentaire exhaustif mais inutile

```py
# On crée un parser
argument_parser = argparse.ArgumentParser()

# On ajoute un argument
argument_parser.add_argument("--out")

# On parse les arguments
return argument_parser.parse_args()
```

Pourquoi c’est interdit :

* chaque ligne est commentée ;
* mais la documentation reste descriptive et superficielle ;
* elle ne répond ni au pourquoi, ni au rôle, ni au repérage maintenance.

---

### Exemples attendus

#### Exemple attendu 1 : lignes simples mais utiles

```py
# On évite une dépendance au shell et aux séparateurs propres a l'OS
from pathlib import Path

# On assainit l'entree pour eviter qu'un espace parasite fausse la validation
normalized_url = url.strip()

# On garde un code retour neutre car l'erreur a deja ete explicitee avant
return 0
```

#### Exemple attendu 2 : API opaque rendue claire

```py
# On reechange la lecture des donnees pour raisonner par variable
# plutot que par observation avant l'agregation
transposed_scores = student_scores.T

# On combine chaque variable avec les erreurs correspondantes
# pour produire la somme utilisee par la mise a jour
error_sum = transposed_scores.dot(prediction_error_by_student)
```

#### Exemple attendu 3 : repere explicite pour futur modificateur

```py
# Cette ligne fige le nom de la colonne exportee ;
# c'est ici qu'il faut intervenir pour changer le schema de sortie
prediction_output = pd.DataFrame(
    {"Index": index_list_of_students, "Hogwarts House": predicted_house_names}
)

# Cette ligne persiste le resultat final sans index pandas,
# ce qui maintient le format attendu par l'evaluateur
prediction_output.to_csv(output_csv_path, index=False)
```

---

### Exemples attendus avec exigence “un commentaire sur chaque ligne”

#### Exemple attendu 4 : bloc complet documente ligne par ligne

```py
# On recupere le nombre d'observations pour dimensionner la colonne de biais
student_count = standardized_students_discipline_scores.shape[0]

# On prepare une colonne de 1 pour reproduire la convention du modele appris
bias_column = np.ones((student_count, 1))

# On assemble biais et variables normalisees dans l'ordre attendu par les poids
students_discipline_scores_with_bias = np.hstack(
    # Cette sous-structure preserve la colonne de biais en premiere position
    [bias_column, standardized_students_discipline_scores]
)
```

#### Exemple attendu 5 : boucle documentee ligne par ligne

```py
# On parcourt chaque discipline pour reappliquer la logique du train colonne par colonne
for discipline_index, discipline_name in enumerate(students_discipline_scores.columns):
    # Cet indice sert a retrouver la bonne statistique de reference
    reference_average_score = average_discipline_scores[discipline_index]

    # On cible explicitement la colonne a corriger pour garder le schema intact
    students_discipline_scores[discipline_name] = (
        # On remplace les valeurs manquantes par la moyenne du train
        # pour eviter une fuite de donnees provenant du jeu de test
        students_discipline_scores[discipline_name].fillna(
            # C'est ici qu'il faut intervenir si la strategie d'imputation change
            reference_average_score
        )
    )
```

#### Exemple attendu 6 : condition documentee ligne par ligne

```py
# On verrouille la presence d'un identifiant stable avant toute prediction
if "Index" not in raw_students_dataset.columns:
    # Ce message cible directement la cause pour accelerer le diagnostic
    raise ValueError("La colonne 'Index' est manquante dans le fichier CSV.")
```

---

### Résumé opérationnel

Chaque ligne doit être commentée.

Le meilleur commentaire explique **pourquoi** la ligne existe.

Si ce niveau n’est pas accessible,
le commentaire doit au moins expliquer :

* **à quoi sert la ligne ici** ;
* **ce qu’elle change concrètement** ;
* ou **pourquoi un futur mainteneur devra revenir à cet endroit**.

Une ligne simple peut recevoir un commentaire court.
Une ligne sensible doit recevoir un commentaire plus riche.
Mais aucune ligne de code ne doit rester sans commentaire.
