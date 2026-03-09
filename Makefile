# ========================================================================================
# Makefile - Automatisation pour le projet Total_Perspective_Vortex
# Objectifs :
#   - Simplifier l’installation et la gestion de l’environnement (Poetry / venv)
#   - Automatiser les vérifications (lint, format, type-check, tests, coverage, mutation)
#   - Fournir des commandes pratiques pour l’entraînement et la prédiction du modèle
# ========================================================================================

.PHONY: \
	install \
	install-deps \
	download_dataset \
	lint \
	format \
	type \
	test \
	cov \
	mut \
	train \
	predict \
	mybci \
	show-activate \
	show-deactivate \
	clean-mutants \
	clean \
	clean-artifacts \
	ensure-venv \
	realtime \
	sanitizer \
	sanitizer-privileged \
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
# Journalise les runs globaux hors dataset pour éviter les collisions de droits
ARTIFACTS_DIR ?= artifacts
BENCH_DIR   ?= $(ARTIFACTS_DIR)/benchmarks
BENCH_CSVS  := $(wildcard $(BENCH_DIR)/*.csv)

# --- Dataset ------------------------------------------------------------------
EEGMMIDB_DATA_DIR ?= data
EEGMMIDB_SENTINEL := $(EEGMMIDB_DATA_DIR)/.eegmmidb.ok
EEGMMIDB_SUBJECT_COUNT ?= 109
EEGMMIDB_RUN_COUNT ?= 14
HANDLED_CLI_ERROR_EXIT_CODE ?= 2

# Utilisation raccourcie de Poetry
POETRY = poetry run
SRC_DIR ?= src
MYBCI_SCRIPT ?= mybci.py
REALTIME_SCRIPT ?= src/tpv/realtime.py
VISUALIZER_SCRIPT ?= scripts/visualize_raw_filtered.py
SANITIZER_SCRIPT ?= scripts/sanitizer.py
AGGREGATE_EXPERIENCE_SCORES_SCRIPT ?= scripts/aggregate_experience_scores.py
TPV_SRC_DIR ?= src/tpv

define ENSURE_SCRIPT_READABLE
script_path="$(1)"; \
if [[ ! -r "$$script_path" ]]; then \
	printf 'INFO: lecture du script %s impossible\n' "$$script_path"; \
	printf '%s\n' "Action: redonnez les droits d'accès au script $$script_path : \`chmod a+rwx $$script_path\`"; \
	exit 0; \
fi; \
true
endef

define ENSURE_TPV_SOURCES_READABLE
src_dir="$(SRC_DIR)"; \
tpv_dir="$(TPV_SRC_DIR)"; \
if [[ -d "$$src_dir" && ( ! -r "$$src_dir" || ! -x "$$src_dir" ) ]]; then \
	printf 'INFO: lecture du dossier %s impossible\n' "$$src_dir"; \
	printf '%s\n' "Action: redonnez les droits d'accès au dossier $$src_dir : \`chmod a+rx $$src_dir\`"; \
	exit 0; \
fi; \
if [[ -d "$$tpv_dir" ]]; then \
	if [[ ! -r "$$tpv_dir" || ! -x "$$tpv_dir" ]]; then \
		printf 'INFO: lecture du dossier %s impossible\n' "$$tpv_dir"; \
		printf '%s\n' "Action: redonnez les droits d'accès au dossier $$tpv_dir et à son contenu : \`chmod -R a+rX $$tpv_dir\`"; \
		exit 0; \
	fi; \
	shopt -s nullglob; \
	blocked_tpv_path=""; \
	for candidate in "$$tpv_dir"/*.py; do \
		if [[ ! -r "$$candidate" ]]; then \
			blocked_tpv_path="$$candidate"; \
			break; \
		fi; \
	done; \
	if [[ -n "$$blocked_tpv_path" ]]; then \
		printf 'INFO: lecture du script %s impossible\n' "$$blocked_tpv_path"; \
		printf '%s\n' "Action: redonnez les droits d'accès au dossier $$tpv_dir et à son contenu : \`chmod -R a+rX $$tpv_dir\`"; \
		exit 0; \
	fi; \
fi; \
true
endef

define ENSURE_ARTIFACTS_TREE_READABLE
artifacts_dir="$(ARTIFACTS_DIR)"; \
if [[ -d "$$artifacts_dir" ]]; then \
	if [[ ! -r "$$artifacts_dir" || ! -x "$$artifacts_dir" ]]; then \
		printf 'INFO: lecture du dossier %s impossible\n' "$$artifacts_dir"; \
		printf '%s\n' "Action: donnez les droits d'accès au dossier $$artifacts_dir : \`chmod a+rx $$artifacts_dir\`"; \
		exit 0; \
	fi; \
	shopt -s nullglob; \
	blocked_artifacts_dir=""; \
	for candidate in "$$artifacts_dir"/S*; do \
		if [[ -d "$$candidate" && ( ! -r "$$candidate" || ! -x "$$candidate" ) ]]; then \
			blocked_artifacts_dir="$$candidate"; \
			break; \
		fi; \
	done; \
	if [[ -n "$$blocked_artifacts_dir" ]]; then \
		printf 'INFO: lecture du dossier %s impossible\n' "$$blocked_artifacts_dir"; \
		printf '%s\n' "Action: donnez les droits d'accès au dossier $$blocked_artifacts_dir : \`chmod a+rx $$blocked_artifacts_dir\`"; \
		exit 0; \
	fi; \
fi; \
true
endef

# Désactive le chargement automatique des plugins pytest globaux (ROS, etc.)
PYTEST_ENV = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Active la collecte de couverture dans les sous-processus (CLI appelés depuis les tests)
COVERAGE_ENV = COVERAGE_PROCESS_START=$(PYPROJECT)

# ----------------------------------------------------------------------------------------
# Installation des dépendances (dev inclus)
# ----------------------------------------------------------------------------------------
install: install-deps download_dataset

install-deps:
	poetry install --with dev
	@mkdir -p $(VENV)
	@touch $(STAMP)

download_dataset:
	@python3 scripts/download_dataset.py \
		--destination "$(EEGMMIDB_DATA_DIR)" \
		--subject-count "$(EEGMMIDB_SUBJECT_COUNT)" \
		--run-count "$(EEGMMIDB_RUN_COUNT)"

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
		$(MAKE) --no-print-directory install-deps; \
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
SANITIZER_ARGS ?=
SANITIZER_COMMAND ?= make -j1 mybci wavelet
SANITIZER_ALLOW_PRIVILEGED_TOOLS ?= 0
TPV_SANITIZER ?= 0

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
	$(call ENSURE_TPV_SOURCES_READABLE); \
		if [[ -n "$$feature_strategy" ]]; then \
			extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
		fi; \
		status=0; \
		$(POETRY) python scripts/train.py "$$subject" "$$run" $$extra_args || status=$$?; \
		if [[ "$$status" -eq "$(HANDLED_CLI_ERROR_EXIT_CODE)" ]]; then \
			exit 0; \
		fi; \
		exit "$$status"

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
		$(call ENSURE_TPV_SOURCES_READABLE); \
		if [[ -n "$$feature_strategy" ]]; then \
			extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
		fi; \
		status=0; \
		$(POETRY) python scripts/predict.py "$$subject" "$$run" $$extra_args || status=$$?; \
		if [[ "$$status" -eq "$(HANDLED_CLI_ERROR_EXIT_CODE)" ]]; then \
			exit 0; \
		fi; \
		exit "$$status"

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
	$(call ENSURE_TPV_SOURCES_READABLE); \
	$(call ENSURE_SCRIPT_READABLE,$(REALTIME_SCRIPT)); \
	$(POETRY) python $(REALTIME_SCRIPT) "$$subject" "$$run"

# score : `make score`
compute-mean-of-means: ensure-venv
	@set -euo pipefail; \
	$(call ENSURE_TPV_SOURCES_READABLE); \
	$(call ENSURE_SCRIPT_READABLE,$(AGGREGATE_EXPERIENCE_SCORES_SCRIPT)); \
	$(POETRY) python $(AGGREGATE_EXPERIENCE_SCORES_SCRIPT)


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
	$(call ENSURE_TPV_SOURCES_READABLE); \
	$(call ENSURE_SCRIPT_READABLE,$(VISUALIZER_SCRIPT)); \
	$(POETRY) python $(VISUALIZER_SCRIPT) "$$subject" "$$run" $$channels_args

# Évaluation globale : équivalent à `python mybci.py` du sujet
mybci: ensure-venv
	@set -euo pipefail; \
	positional_strategy="$(word 2,$(MAKECMDGOALS))"; \
	extra_args="$(BENCH_ARGS)"; \
	sanitizer_mode="$(TPV_SANITIZER)"; \
	feature_strategy="$(FEATURE_STRATEGY)"; \
	if [[ -z "$$feature_strategy" && -n "$$positional_strategy" ]]; then \
		feature_strategy="$$positional_strategy"; \
	fi; \
	if [[ -n "$$feature_strategy" ]]; then \
		extra_args="$$extra_args --feature-strategy $$feature_strategy"; \
	fi; \
	$(call ENSURE_TPV_SOURCES_READABLE); \
	$(call ENSURE_SCRIPT_READABLE,$(MYBCI_SCRIPT)); \
	$(call ENSURE_ARTIFACTS_TREE_READABLE); \
	status=0; \
	if [[ "$$sanitizer_mode" == "1" ]]; then \
		$(POETRY) python $(MYBCI_SCRIPT) $$extra_args || status=$$?; \
	else \
		mkdir -p $(BENCH_DIR); \
		$(POETRY) python $(MYBCI_SCRIPT) $$extra_args \
			| tee $(BENCH_DIR)/bench_$$(date +%Y%m%d_%H%M%S).log || status=$$?; \
	fi; \
	if [[ "$$status" -eq "$(HANDLED_CLI_ERROR_EXIT_CODE)" ]]; then \
		exit 0; \
	fi; \
	exit "$$status"

# Diagnostic / benchmark / profiling autour d'une commande cible
sanitizer: ensure-venv
	@set -euo pipefail; \
	$(call ENSURE_SCRIPT_READABLE,$(SANITIZER_SCRIPT)); \
	privileged_flag=""; \
	if [[ "$(SANITIZER_ALLOW_PRIVILEGED_TOOLS)" == "1" ]]; then \
		sudo -v; \
		privileged_flag="--allow-privileged-tools"; \
	fi; \
	$(POETRY) -- python $(SANITIZER_SCRIPT) $$privileged_flag $(SANITIZER_ARGS) -- $(SANITIZER_COMMAND)

sanitizer-privileged: ensure-venv
	@$(MAKE) sanitizer SANITIZER_ALLOW_PRIVILEGED_TOOLS=1 SANITIZER_ARGS="$(SANITIZER_ARGS)" SANITIZER_COMMAND="$(SANITIZER_COMMAND)"

# Affiche la commande d'activation (make ne peut pas modifier le shell parent)
show-activate:
	@echo "Commande d'activation (a executer dans le shell courant) :"
	@echo "source $$(poetry env info -p)/bin/activate"

# Affiche la commande de desactivation
show-deactivate:
	@echo "Commande de desactivation (a executer dans le shell courant) :"
	@echo "deactivate"

clean: clean-artifacts clean-npy clean-epoch-json
	clear

clean-artifacts:
	@rm -rf ./artifacts ./data/benchmarks

clean-npy:
	@find . -type f -name '*.npy' \
		-not -path './.venv/*' \
		-not -path './.git/*' \
		-not -path './artifacts/*' \
		-delete

clean-epoch-json:
	@if [ -d ./data ]; then \
		find ./data -type f \( \
			-name '*_epoch_window.json' -o \
			-name '*_epoch_windows.json' \
		\) -delete; \
	fi

# ----------------------------------------------------------------------------------------
# Règle générique pour ignorer les cibles numériques (ex. make predict-nocheck 23000)
# ----------------------------------------------------------------------------------------
%:
	@:
