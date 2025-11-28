#!/usr/bin/env bash
set -euo pipefail

# === CONFIG À ADAPTER =========================================================
# Propriétaire du projet : "@me" ou "raveriss" ou une org
OWNER="raveriss"

# Numéro du projet (visible dans l'URL GitHub Projects)
PROJECT_NUMBER="2"
# ============================================================================
# Vérification rapide
if ! command -v gh >/dev/null 2>&1; then
  echo "Erreur: gh (GitHub CLI) est introuvable." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Erreur: jq est introuvable." >&2
  exit 1
fi

echo "==> Création des champs pour le Project #${PROJECT_NUMBER} (owner=${OWNER})"

# Récupère les champs déjà existants
EXISTING_FIELDS_JSON="$(
  gh project field-list "${PROJECT_NUMBER}" --owner "${OWNER}" --format json
)"

# Définition de tous les champs (JSON)
FIELDS_JSON='{
  "project_fields": [
    { "name": "WBS ID", "type": "CUSTOM", "dataType": "TEXT" },
    { "name": "Description", "type": "CUSTOM", "dataType": "TEXT" },

    {
      "name": "Phase",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "0 – Cadrage & Risques" },
        { "name": "1 – Acquisition données" },
        { "name": "2 – Prétraitement / Filtrage" },
        { "name": "3 – Features / ICA" },
        { "name": "4 – Modélisation / ML" },
        { "name": "5 – Évaluation / Validation" },
        { "name": "6 – Infra / CI/CD" },
        { "name": "7 – Documentation / Deliverables" }
      ]
    },

    {
      "name": "Type",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Task" },
        { "name": "Deliverable" },
        { "name": "Risk" },
        { "name": "Bug" },
        { "name": "Spike (exploration)" },
        { "name": "Chore (infra / organisation)" }
      ]
    },

    {
      "name": "Start date",
      "type": "CUSTOM",
      "dataType": "DATE"
    },
    {
      "name": "Due date",
      "type": "CUSTOM",
      "dataType": "DATE"
    },

    {
      "name": "Effort",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "XS – ≤ 1h" },
        { "name": "S – 2–4h" },
        { "name": "M – ½ journée" },
        { "name": "L – 1 journée" },
        { "name": "XL – 2–3 jours" }
      ]
    },

    {
      "name": "Priority",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Must" },
        { "name": "Should" },
        { "name": "Could" },
        { "name": "Won’t" }
      ]
    },

    {
      "name": "Blocked by",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Deliverable",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Deliverable type",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Code" },
        { "name": "Notebook" },
        { "name": "Dataset" },
        { "name": "Config / Infra" },
        { "name": "Documentation" },
        { "name": "Figure / Graphique" },
        { "name": "Rapport / PDF" }
      ]
    },

    {
      "name": "Repo / Path",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Technical Difficulty",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Low" },
        { "name": "Medium" },
        { "name": "High" },
        { "name": "Extreme" }
      ]
    },

    {
      "name": "Is risk ?",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Yes" },
        { "name": "No" }
      ]
    },

    {
      "name": "Murphy ID",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Risk – Cause",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Risk – Effect",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Risk – Category",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Data" },
        { "name": "Model" },
        { "name": "EEG / Signal" },
        { "name": "Infra / CI/CD" },
        { "name": "Documentation" },
        { "name": "Planning" }
      ]
    },

    {
      "name": "Probability",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "1 – Rare" },
        { "name": "2 – Peu probable" },
        { "name": "3 – Possible" },
        { "name": "4 – Probable" },
        { "name": "5 – Presque certain" }
      ]
    },

    {
      "name": "Impact / Gravité",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "1 – Mineur" },
        { "name": "2 – Modéré" },
        { "name": "3 – Important" },
        { "name": "4 – Critique" },
        { "name": "5 – Catastrophique" }
      ]
    },

    {
      "name": "Risk score",
      "type": "CUSTOM",
      "dataType": "NUMBER"
    },

    {
      "name": "Response strategy",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Avoid" },
        { "name": "Reduce" },
        { "name": "Accept" },
        { "name": "Transfer" }
      ]
    },

    {
      "name": "Mitigation plan",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Contingency plan",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Risk owner",
      "type": "CUSTOM",
      "dataType": "TEXT"
    },

    {
      "name": "Workstream",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "R&D EEG" },
        { "name": "Pipeline ML" },
        { "name": "Dev / Infra" },
        { "name": "Doc / Support" }
      ]
    },

    {
      "name": "Review needed ?",
      "type": "CUSTOM",
      "dataType": "SINGLE_SELECT",
      "options": [
        { "name": "Tech review" },
        { "name": "Scientifique (EEG/ML)" },
        { "name": "Code review" },
        { "name": "Pas de review" }
      ]
    },

    {
      "name": "Links",
      "type": "CUSTOM",
      "dataType": "TEXT"
    }
  ]
}'

# Boucle sur tous les champs CUSTOM
echo "${FIELDS_JSON}" | jq -c '.project_fields[]' | while read -r FIELD; do
  NAME=$(echo "${FIELD}" | jq -r '.name')
  TYPE=$(echo "${FIELD}" | jq -r '.type')
  DATATYPE=$(echo "${FIELD}" | jq -r '.dataType')

  # On ne traite que les champs CUSTOM
  if [[ "${TYPE}" != "CUSTOM" ]]; then
    continue
  fi

  # Vérifie si le champ existe déjà dans le projet
  if echo "${EXISTING_FIELDS_JSON}" | \
     jq -e --arg name "${NAME}" '.fields[] | select(.name == $name)' \
     >/dev/null 2>&1; then
    echo "⏭  Champ déjà présent, on saute : ${NAME}"
    continue
  fi

  echo "➕ Création du champ : ${NAME} (${DATATYPE})"

  # Gestion des différents types
  case "${DATATYPE}" in
    "TEXT"|"DATE"|"NUMBER")
      gh project field-create "${PROJECT_NUMBER}" \
        --owner "${OWNER}" \
        --name "${NAME}" \
        --data-type "${DATATYPE}"
      ;;
    "SINGLE_SELECT")
      OPTIONS_CSV=$(echo "${FIELD}" | jq -r '[.options[].name] | join(",")')
      gh project field-create "${PROJECT_NUMBER}" \
        --owner "${OWNER}" \
        --name "${NAME}" \
        --data-type "SINGLE_SELECT" \
        --single-select-options "${OPTIONS_CSV}"
      ;;
    *)
      echo "⚠️  Type de champ non géré : ${DATATYPE} (champ ${NAME})" >&2
      ;;
  esac
done

echo "✅ Champs custom créés / synchronisés."
