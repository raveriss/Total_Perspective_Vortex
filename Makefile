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
	mybci \
	activate \
	deactivate \
	clean-mutants \
	clean \
	clean-artifacts \
	ensure-venv \
	realtime \
	compute-mean-of-means \
	visualizer \
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

FEATURE_STRATEGY ?=
TRAIN_ARGS ?=
PREDICT_ARGS ?=
BENCH_ARGS ?=

# Entraînement : `make train <subject> <run>`
train: ensure-venv
	@set -euo pipefail; \
	subject="$(word 2,$(MAKECMDGOALS))"; \
	run="$(word 3,$(MAKECMDGOALS))"; \
	positional_strategy="$(word 4,$(MAKECMDGOALS))"; \
	if [[ -z "$$subject" || -z "$$run" ]]; then \
		echo "Usage: make train <subject> <run> [wavelet|welch]" >&2; \
		echo "       make train <subject> <run> FEATURE_STRATEGY=wavelet" >&2; \
		echo "       make train <subject> <run> TRAIN_ARGS='--feature-strategy wavelet'" >&2; \
		echo "Note : GNU make interprète les flags '--*' comme options." >&2; \
		echo "       Utiliser FEATURE_STRATEGY=... ou TRAIN_ARGS=... pour passer des flags." >&2; \
		exit 0; \
	fi; \
	if [[ ! "$$subject" =~ ^[0-9]+$$ || ! "$$run" =~ ^[0-9]+$$ ]]; then \
		echo "❌ <subject> et <run> doivent être des entiers (ex: make train 3 8)" >&2; \
		exit 0; \
	fi; \
	extra_args="$(TRAIN_ARGS)"; \
	feature_strategy="$(FEATURE_STRATEGY)"; \
	if [[ -z "$$feature_strategy" && -n "$$positional_strategy" ]]; then \
		feature_strategy="$$positional_strategy"; \
	fi; \
	if [[ -n "$$feature_strategy" ]]; then \
		extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
	fi; \
	$(POETRY) python scripts/train.py "$$subject" "$$run" $$extra_args

# Prédiction : `make predict <subject> <run>`
predict: ensure-venv
	@set -euo pipefail; \
	subject="$(word 2,$(MAKECMDGOALS))"; \
	run="$(word 3,$(MAKECMDGOALS))"; \
	positional_strategy="$(word 4,$(MAKECMDGOALS))"; \
	if [[ -z "$$subject" || -z "$$run" ]]; then \
		echo "Usage: make predict <subject> <run> [wavelet|welch]" >&2; \
		echo "       make predict <subject> <run> FEATURE_STRATEGY=wavelet" >&2; \
		echo "       make predict <subject> <run> PREDICT_ARGS='--feature-strategy wavelet'" >&2; \
		echo "Note : GNU make interprète les flags '--*' comme options." >&2; \
		echo "       Utiliser FEATURE_STRATEGY=... ou PREDICT_ARGS=... pour passer des flags." >&2; \
		exit 0; \
	fi; \
	if [[ ! "$$subject" =~ ^[0-9]+$$ || ! "$$run" =~ ^[0-9]+$$ ]]; then \
		echo "❌ <subject> et <run> doivent être des entiers (ex: make predict 3 8)" >&2; \
		exit 0; \
	fi; \
	extra_args="$(PREDICT_ARGS)"; \
	feature_strategy="$(FEATURE_STRATEGY)"; \
	if [[ -z "$$feature_strategy" && -n "$$positional_strategy" ]]; then \
		feature_strategy="$$positional_strategy"; \
	fi; \
	if [[ -n "$$feature_strategy" ]]; then \
		extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
	fi; \
	$(POETRY) python scripts/predict.py "$$subject" "$$run" $$extra_args

# realtime : `make realtime <subject> <run>`
realtime: ensure-venv
	@set -euo pipefail; \
	subject="$(word 2,$(MAKECMDGOALS))"; \
	run="$(word 3,$(MAKECMDGOALS))"; \
	if [[ -z "$$subject" || -z "$$run" ]]; then \
		echo "Usage: make realtime <subject> <run>" >&2; \
		exit 0; \
	fi; \
	if [[ ! "$$subject" =~ ^[0-9]+$$ || ! "$$run" =~ ^[0-9]+$$ ]]; then \
		echo "❌ <subject> et <run> doivent être des entiers (ex: make predict 3 8)" >&2; \
		exit 0; \
	fi; \
	$(POETRY) python src/tpv/realtime.py "$$subject" "$$run"

# score : `make score`
compute-mean-of-means: ensure-venv
	@set -euo pipefail; \
	$(POETRY) python scripts/aggregate_experience_scores.py


# realtime : `make visualizer <subject> <run>`
visualizer: ensure-venv
	@set -euo pipefail; \
	subject="$(word 2,$(MAKECMDGOALS))"; \
	run="$(word 3,$(MAKECMDGOALS))"; \
	extra_goals="$(wordlist 4,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))"; \
	channels_args=""; \
	if [[ -z "$$subject" || -z "$$run" ]]; then \
		echo "Usage: make visualizer <subject> <run>" >&2; \
		exit 0; \
	fi; \
	if [[ ! "$$subject" =~ ^[0-9]+$$ || ! "$$run" =~ ^[0-9]+$$ ]]; then \
		echo "❌ <subject> et <run> doivent être des entiers (ex: make visualizer 3 8)" >&2; \
		exit 0; \
	fi; \
	if [[ -n "$$extra_goals" ]]; then \
		read -r -a channel_parts <<< "$$extra_goals"; \
		if [[ "$${channel_parts[0]}" == "--channels" ]]; then \
			channel_parts=("$${channel_parts[@]:1}"); \
		fi; \
		if [[ "$${#channel_parts[@]}" -gt 0 ]]; then \
			channels_args="--channels $${channel_parts[*]}"; \
		fi; \
	fi; \
	$(POETRY) python scripts/visualize_raw_filtered.py "$$subject" "$$run" $$channels_args

# Évaluation globale : équivalent à `python mybci.py` du sujet
mybci: ensure-venv
	@set -euo pipefail; \
	positional_strategy="$(word 2,$(MAKECMDGOALS))"; \
	extra_args="$(BENCH_ARGS)"; \
	feature_strategy="$(FEATURE_STRATEGY)"; \
	if [[ -z "$$feature_strategy" && -n "$$positional_strategy" ]]; then \
		feature_strategy="$$positional_strategy"; \
	fi; \
	if [[ -n "$$feature_strategy" ]]; then \
		extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
	fi; \
	mkdir -p $(BENCH_DIR); \
	$(POETRY) python mybci.py $$extra_args \
		| tee $(BENCH_DIR)/bench_$$(date +%Y%m%d_%H%M%S).log

# Affiche la commande pour activer le venv
activate:
	@echo "Pour activer manuellement cet environnement :"
	@echo "source $$(poetry env info -p)/bin/activate"

# Affiche la commande pour désactiver le venv
deactivate:
	@echo "Pour quitter l'environnement :"
	@echo "deactivate"

clean: clean-artifacts clean-npy clean-epoch-json
	clean

clean-artifacts:
	@rm -rf ./artifacts ./data/benchmarks

clean-npy:
	@find . -type f -name '*.npy' \
		-not -path './.venv/*' \
		-not -path './.git/*' \
		-not -path './artifacts/*' \
		-delete

clean-epoch-json:
	@find ./data -type f \( \
		-name '*_epoch_window.json' -o \
		-name '*_epoch_windows.json' \
	\) -delete

# ----------------------------------------------------------------------------------------
# Règle générique pour ignorer les cibles numériques (ex. make predict-nocheck 23000)
# ----------------------------------------------------------------------------------------
%:
	@:
