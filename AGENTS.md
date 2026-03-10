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

## 📚 Documentation du code

Lorsque tu génères, modifies ou réécris du code, applique strictement les
règles suivantes.

### Objectif

Les commentaires doivent exprimer le **pourquoi** de **chaque ligne de code**.

Le but est d’avoir **un commentaire au-dessus de chaque ligne**,
y compris pour les imports, constantes, affectations, conditions,
retours, appels de fonctions et transformations intermédiaires.

Quand une justification forte n’existe pas, formule le commentaire avec
le niveau de justification le plus utile disponible, sans paraphrase
grossière du code.

---

### Règle absolue

Un commentaire doit expliquer au moins un des points suivants :

* pourquoi cette ligne existe sous cette forme,
* pourquoi cette implémentation a été retenue ici,
* quel risque elle évite,
* quelle garantie elle protège,
* quelle contrainte elle respecte,
* quel contrat elle préserve,
* quelle robustesse, stabilité ou compatibilité elle apporte.

Un commentaire ne doit jamais :

* décrire ce que fait la ligne,
* reformuler le code,
* expliquer la syntaxe,
* décrire le mécanisme immédiat,
* nommer simplement l’API appelée,
* commenter une évidence lisible directement dans le code.

---

### Priorité absolue

**Aucun commentaire vaut mieux qu’un commentaire faux, évident ou redondant.**

Donc, n’ajoute aucun commentaire si :

* la ligne est évidente,
* sa justification est triviale,
* le lecteur peut déduire seul le commentaire en lisant la ligne,
* le commentaire répond à “quoi” ou “comment” au lieu de “pourquoi”.

---

### Test obligatoire avant chaque commentaire

Avant d’écrire un commentaire, vérifie que :

1. il répond à la question :
   **« Pourquoi cette ligne est-elle nécessaire ici ? »**
   ou
   **« Qu’est-ce qu’on cherche à garantir, éviter ou préserver grâce à elle ? »**
2. il apporte une information absente du code lui-même ;
3. il ne peut pas être déduit simplement en lisant la ligne sans contexte ;
4. il aide réellement la maintenance, le diagnostic, la robustesse,
   la compréhension d’un choix ou d’un compromis.

Si un seul de ces critères échoue, **n’ajoute pas de commentaire**.

---

### Ce qu’un bon commentaire peut expliquer

Un bon commentaire peut justifier :

* une intention de conception,
* une contrainte technique ou métier,
* un invariant à préserver,
* un risque évité,
* un compromis assumé,
* une robustesse recherchée,
* une stabilité de test,
* une compatibilité inter-OS ou inter-environnements,
* une exigence de maintenabilité,
* une contrainte de performance,
* un diagnostic exploitable,
* un contrat d’interface,
* un comportement attendu en cas d’erreur,
* une normalisation volontaire,
* une convention retenue pour fiabiliser le système.

---

### Formulation attendue

Le commentaire doit être formulé comme une **justification**,
pas comme une **description**.

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

Ces verbes décrivent souvent l’action visible, donc le “comment”.

---

### Règles de commentaires

* N’ajoute un commentaire **que si la ligne a un “pourquoi” utile**.
* Lorsqu’un commentaire est justifié, place-le **juste au-dessus** de la ligne.
* Le commentaire doit respecter **l’indentation** du bloc.
* **Langue : français**.
* Les termes techniques anglais sont autorisés uniquement pour :
  * les noms d’API,
  * les types,
  * les constantes,
  * les mots-clés du langage,
  * les noms de fonctions, classes, modules ou outils.
* **80 caractères maximum par ligne de commentaire**.
* **Interdit** :
  * commentaire en fin de ligne,
  * commentaire sous la ligne,
  * paraphrase du code,
  * description syntaxique,
  * commentaire décoratif,
  * commentaire générique sans valeur de maintenance,
  * commentaire redondant avec le nom de la fonction, de l’API ou de la ligne.

---

### Règle de réécriture

Si un commentaire existant décrit l’action de la ligne, réécris-le pour
exprimer à la place :

* quelle garantie est recherchée,
* quel risque est évité,
* quelle contrainte est respectée,
* quel contrat est préservé,
* pourquoi cette implémentation est préférable ici,
* quelle conséquence négative est empêchée.

Si cette réécriture n’est pas possible sans inventer une justification,
**supprime le commentaire**.

---

### Docstrings

Utilise des **docstrings uniquement** pour :

* les modules,
* les classes,
* les fonctions.

Les docstrings doivent couvrir :

* le but global,
* les paramètres,
* la valeur de retour,
* les erreurs levées,
* le contrat global d’utilisation.

Les docstrings ne doivent pas répéter les commentaires ligne par ligne.

---

### Règle d’action

Quand tu traites du code :

1. supprime les commentaires qui décrivent l’action ;
2. conserve uniquement les commentaires réellement utiles ;
3. réécris les commentaires pour exprimer la justification du choix ;
4. n’ajoute de nouveaux commentaires que lorsqu’un “pourquoi” non trivial
   existe clairement ;
5. n’invente jamais une justification absente du contexte.

---

### Exemples interdits

```py
# On importe Path pour gérer les chemins
from pathlib import Path

# On exécute ping pour tester le réseau
ping_result = run_command_diagnostic(("ping", "-c", "1", NETWORK_TEST_IP), runner)

# On retourne 0
return 0

# On supprime les espaces
normalized = url.strip()

# On boucle sur la sortie standard
for line in process.stdout:
    print(line, end="")
```

### Exemples attendus

```py
# On évite une dépendance au shell et aux séparateurs selon l'OS
from pathlib import Path

# On distingue une panne réseau globale d'un simple problème DNS
ping_result = run_command_diagnostic(("ping", "-c", "1", NETWORK_TEST_IP), runner)

# On garde un code retour neutre car l'erreur a déjà été explicitée
return 0

# On assainit l'entrée pour éviter qu'une configuration invalide fausse les probes
normalized = url.strip()

# On expose une progression visible pour éviter un CLI opaque sur un transfert long
for line in process.stdout:
    print(line, end="")
```
