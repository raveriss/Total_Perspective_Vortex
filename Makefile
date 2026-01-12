# ========================================================================================
# Makefile - Automatisation pour le projet Total_Perspective_Vortex
# Objectifs :
#   - Simplifier l’installation et la gestion de l’environnement (Poetry / venv)
#   - Automatiser les vérifications (lint, format, type-check, tests, coverage, mutation)
#   - Fournir des commandes pratiques pour l’entraînement et la prédiction du modèle
# ========================================================================================

.PHONY: \
	install \
	lint \
	format \
	type \
	test \
	cov \
	mut \
	train \
	predict \
	bench \
	activate \
	deactivate \
	clean-mutants \
	clean \
	clean-artifacts \
	ensure-venv \
	clean-npy

# Utilise bash (requis) pour des tests/conditions fiables
SHELL := /bin/bash

VENV = .venv
VENV_BIN = $(VENV)/bin/activate
VENV_PY = $(VENV)/bin/python
PYPROJECT = pyproject.toml
LOCKFILE = poetry.lock
STAMP = $(VENV)/.poetry-installed

# Force Poetry à créer/utiliser le venv dans le repo (./.venv)
# => aucune activation manuelle nécessaire
export POETRY_VIRTUALENVS_IN_PROJECT := true

# --- Benchmarks ---------------------------------------------------------------
BENCH_DIR   := data/benchmarks
BENCH_CSVS  := $(wildcard $(BENCH_DIR)/*.csv)

# Utilisation raccourcie de Poetry
POETRY = poetry run

# Désactive le chargement automatique des plugins pytest globaux (ROS, etc.)
PYTEST_ENV = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Active la collecte de couverture dans les sous-processus (CLI appelés depuis les tests)
COVERAGE_ENV = COVERAGE_PROCESS_START=$(PYPROJECT)

# ----------------------------------------------------------------------------------------
# Installation des dépendances (dev inclus)
# ----------------------------------------------------------------------------------------
install:
	poetry install --with dev
	@mkdir -p $(VENV)
	@touch $(STAMP)

# ----------------------------------------------------------------------------------------
# Bootstrap automatique du venv + deps (zéro activation manuelle)
# ----------------------------------------------------------------------------------------
ensure-venv:
	@set -euo pipefail; \
	if ! command -v poetry >/dev/null 2>&1; then \
		echo "❌ poetry introuvable. Installe Poetry puis relance." >&2; \
		exit 127; \
	fi; \
	needs_install=0; \
	if [[ ! -x "$(VENV_PY)" ]]; then \
		needs_install=1; \
	fi; \
	if [[ ! -f "$(STAMP)" ]]; then \
		needs_install=1; \
	fi; \
	if [[ -f "$(PYPROJECT)" && -f "$(STAMP)" && "$(PYPROJECT)" -nt "$(STAMP)" ]]; then \
		needs_install=1; \
	fi; \
	if [[ -f "$(LOCKFILE)" && -f "$(STAMP)" && "$(LOCKFILE)" -nt "$(STAMP)" ]]; then \
		needs_install=1; \
	fi; \
	if [[ -x "$(VENV_PY)" ]]; then \
		if ! "$(VENV_PY)" -c "import numpy" >/dev/null 2>&1; then \
			needs_install=1; \
		fi; \
	fi; \
	if [[ "$$needs_install" -eq 1 ]]; then \
		echo "🔧 Dépendances absentes/obsolètes → auto-install (poetry install --with dev)"; \
		$(MAKE) --no-print-directory install; \
	fi

# ----------------------------------------------------------------------------------------
# Vérifications de qualité du code
# ----------------------------------------------------------------------------------------

# Linting avec Ruff (analyse statique rapide)
lint: ensure-venv
	$(POETRY) ruff check .

# Formatage + correction auto avec Ruff
format: ensure-venv
	$(POETRY) ruff format . && $(POETRY) ruff check --fix .

# Vérification des types avec Mypy
type: ensure-venv
	$(POETRY) mypy src scripts tests

# ----------------------------------------------------------------------------------------
# Tests et couverture
# ----------------------------------------------------------------------------------------

# Nettoyage du dossier de mutants pour éviter les conflits de tests
clean-mutants:
	rm -rf mutants

# Exécution des tests unitaires (sans plugins pytest externes)
test: ensure-venv clean-mutants
	$(PYTEST_ENV) $(POETRY) pytest -vv

# Analyse de la couverture avec rapport JSON, XML, HTML et console (90% requis)
cov: ensure-venv clean-mutants
	$(PYTEST_ENV) $(COVERAGE_ENV) $(POETRY) coverage erase && \
	$(PYTEST_ENV) $(COVERAGE_ENV) $(POETRY) coverage run --parallel-mode -m pytest && \
	$(POETRY) coverage combine && \
	$(POETRY) coverage json -o coverage.json && \
	$(POETRY) coverage xml -o coverage.xml && \
	$(POETRY) coverage html --skip-empty --show-contexts && \
	$(POETRY) coverage report --fail-under=90

# Mutation testing avec Mutmut (guidé par la couverture)
mut: ensure-venv clean-mutants cov
	MUTMUT_USE_COVERAGE=1 $(PYTEST_ENV) $(POETRY) mutmut run
	$(POETRY) mutmut results > mutmut-results.txt
	@if grep -E "(survived|timeout)" mutmut-results.txt; then \
		echo "Surviving or timed-out mutants detected" >&2; \
		exit 1; \
	fi

# ----------------------------------------------------------------------------------------
# Commandes liées au modèle (Poetry)
# ----------------------------------------------------------------------------------------

TRAIN_SUBJECT ?= S001
TRAIN_RUN ?= R05
TRAIN_ALL ?= true
PREDICT_SUBJECT ?= $(TRAIN_SUBJECT)
PREDICT_RUN ?= $(TRAIN_RUN)

# Entraînement du modèle : exemple minimal avec sujet et run de démonstration
train: ensure-venv
	$(POETRY) python mybci.py $(TRAIN_SUBJECT) $(TRAIN_RUN) train

# Prédiction : exemple minimal réutilisant les identifiants ci-dessus
predict: ensure-venv
	$(POETRY) python mybci.py $(PREDICT_SUBJECT) $(PREDICT_RUN) predict

# Évaluation globale : équivalent à `python mybci.py` du sujet
bench: ensure-venv
	@mkdir -p $(BENCH_DIR)
	$(POETRY) python mybci.py | tee $(BENCH_DIR)/bench_$$(date +%Y%m%d_%H%M%S).log

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

clean: clean-artifacts clean-npy

clean-artifacts:
	@rm -rf ./artifacts ./data/benchmarks

clean-npy:
	@find . -type f -name '*.npy' \
		-not -path './.venv/*' \
		-not -path './.git/*' \
		-not -path './artifacts/*' \
		-delete

# ----------------------------------------------------------------------------------------
# Règle générique pour ignorer les cibles numériques (ex. make predict-nocheck 23000)
# ----------------------------------------------------------------------------------------
%:
	@:
