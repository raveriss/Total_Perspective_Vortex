# ========================================================================================
# Makefile - Automatisation pour le projet Total_Perspective_Vortex
# Objectifs :
#   - Simplifier l’installation et la gestion de l’environnement (Poetry / venv)
#   - Automatiser les vérifications (lint, format, type-check, tests, coverage, mutation)
#   - Fournir des commandes pratiques pour l’entraînement et la prédiction du modèle
# ========================================================================================

.PHONY: install lint format type test cov mut train predict activate deactivate clean-mutants

VENV = .venv
VENV_BIN = $(VENV)/bin/activate

# --- Benchmarks ---------------------------------------------------------------
BENCH_DIR   := data/benchmarks
BENCH_CSVS  := $(wildcard $(BENCH_DIR)/*.csv)

# Utilisation raccourcie de Poetry
POETRY = poetry run

# Désactive le chargement automatique des plugins pytest globaux (ROS, etc.)
PYTEST_ENV = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

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

# Nettoyage du dossier de mutants pour éviter les conflits de tests
clean-mutants:
	rm -rf mutants

# Exécution des tests unitaires (sans plugins pytest externes)
test: clean-mutants
	$(PYTEST_ENV) $(POETRY) pytest -vv

# Analyse de la couverture avec rapport JSON, XML, HTML et console (90% requis)
cov: clean-mutants
	$(PYTEST_ENV) $(POETRY) coverage run -m pytest && \
	$(POETRY) coverage json -o coverage.json && \
	$(POETRY) coverage xml -o coverage.xml && \
	$(POETRY) coverage html --skip-empty --show-contexts && \
	$(POETRY) coverage report --fail-under=90

# Mutation testing avec Mutmut (guidé par la couverture)
mut: clean-mutants cov
	MUTMUT_USE_COVERAGE=1 $(PYTEST_ENV) $(POETRY) mutmut run
	$(POETRY) mutmut results > mutmut-results.txt
	@if grep -E "(survived|timeout)" mutmut-results.txt; then \
		echo "Surviving or timed-out mutants detected" >&2; \
		exit 1; \
	fi

# ----------------------------------------------------------------------------------------
# Commandes liées au modèle (Poetry)
# ----------------------------------------------------------------------------------------

TRAIN_SUBJECT ?= S003
TRAIN_RUN ?= R03
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
