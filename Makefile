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
	cov \il manque dans la section
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

# --- Dataset ------------------------------------------------------------------
EEGMMIDB_URL ?= https://physionet.org/files/eegmmidb/1.0.0/
EEGMMIDB_DATA_DIR ?= data
EEGMMIDB_SENTINEL := $(EEGMMIDB_DATA_DIR)/.eegmmidb.ok
EEGMMIDB_SUBJECT_COUNT ?= 109
EEGMMIDB_RUN_COUNT ?= 14
HANDLED_CLI_ERROR_EXIT_CODE ?= 2

# Utilisation raccourcie de Poetry
POETRY = poetry run

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
	@set -euo pipefail; \
	data_dir="$(EEGMMIDB_DATA_DIR)"; \
	sentinel="$(EEGMMIDB_SENTINEL)"; \
	check_dataset_complete() { \
		local root_dir="$$1"; \
		local subject_index=0; \
		local run_index=0; \
		local subject=""; \
		local run=""; \
		local edf_path=""; \
		local event_path=""; \
		[[ -d "$$root_dir" ]] || { \
			echo "Dataset incomplet: dossier racine manquant ($$root_dir)."; \
			return 1; \
		}; \
		for subject_index in $$(seq 1 $(EEGMMIDB_SUBJECT_COUNT)); do \
			subject="$$(printf "S%03d" "$$subject_index")"; \
			[[ -d "$$root_dir/$$subject" ]] || { \
				echo "Dataset incomplet: dossier sujet manquant ($$root_dir/$$subject)."; \
				return 1; \
			}; \
			for run_index in $$(seq 1 $(EEGMMIDB_RUN_COUNT)); do \
				run="$$(printf "R%02d" "$$run_index")"; \
				edf_path="$$root_dir/$$subject/$${subject}$${run}.edf"; \
				event_path="$$root_dir/$$subject/$${subject}$${run}.edf.event"; \
				[[ -s "$$edf_path" ]] || { \
					echo "Dataset incomplet: fichier manquant ou vide ($$edf_path)."; \
					return 1; \
				}; \
				[[ -s "$$event_path" ]] || { \
					echo "Dataset incomplet: fichier manquant ou vide ($$event_path)."; \
					return 1; \
				}; \
			done; \
		done; \
		return 0; \
	}; \
	if check_dataset_complete "$$data_dir"; then \
		mkdir -p "$$data_dir"; \
		touch "$$sentinel"; \
		echo "Dataset EEGMMIDB déjà complet dans $$data_dir (aucun téléchargement)."; \
		exit 0; \
	fi; \
	rm -f "$$sentinel"; \
	if ! command -v wget >/dev/null 2>&1; then \
		echo "❌ wget introuvable. Installez wget puis relancez make download_dataset." >&2; \
		exit 127; \
	fi; \
	mkdir -p "$$data_dir"; \
	echo "Téléchargement EEGMMIDB PhysioNet (~3.4GB), cela peut prendre du temps..."; \
	echo "Source: $(EEGMMIDB_URL)"; \
	wget_status=0; \
	wget -r -N -c -np -nH --cut-dirs=3 \
		--accept '*.edf,*.edf.event' \
		-R 'index.html*' \
		-P "$$data_dir" \
		"$(EEGMMIDB_URL)" || wget_status=$$?; \
	if check_dataset_complete "$$data_dir"; then \
		touch "$$sentinel"; \
		if [[ "$$wget_status" -ne 0 ]]; then \
			echo "⚠️ wget a retourné $$wget_status, mais le dataset local est complet." >&2; \
		fi; \
		echo "Dataset EEGMMIDB complet et validé dans $$data_dir."; \
	else \
		echo "❌ Dataset EEGMMIDB toujours incomplet après téléchargement." >&2; \
		echo "Relancez make download_dataset pour reprendre (wget -c)." >&2; \
		if [[ "$$wget_status" -ne 0 ]]; then \
			exit "$$wget_status"; \
		fi; \
		exit 1; \
	fi

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
