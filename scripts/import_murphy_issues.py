#!/usr/bin/env python3
"""
Importe tpv_murphy_map.csv en créant :

1) Une issue GitHub par ligne dans le repo.
2) Un item dans le Project v2 TPV, relié à l'issue.
3) Les champs du Project (Phase, Probability, etc.) renseignés.

Prérequis :
  - gh (GitHub CLI) >= 2.3 avec scope `project`, `repo`.
  - CSV: docs/risk/tpv_murphy_map.csv avec des en-têtes compatibles.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


# ========================== CONFIG À ADAPTER ========================== #

OWNER = "raveriss"
PROJECT_NUMBER = 2
REPO = "raveriss/Total_Perspective_Vortex"

CSV_PATH = Path("docs/risk/tpv_murphy_map.csv")

# Nom de la colonne utilisée comme titre d'issue
TITLE_COLUMN = "Title"  # adapte si besoin (ex: "Résumé" / "Risk title")

# Mapping nom colonne CSV -> nom du champ dans le Project
FIELD_MAPPING = {
    "WBS ID": "WBS ID",
    "Description": "Description",
    "Phase": "Phase",
    "Type": "Type",
    "Effort": "Effort",
    "Priority": "Priority",
    "Blocked by": "Blocked by",
    "Deliverable": "Deliverable",
    "Deliverable type": "Deliverable type",
    "Repo / Path": "Repo / Path",
    "Technical Difficulty": "Technical Difficulty",
    "Is risk ?": "Is risk ?",
    "Murphy ID": "Murphy ID",
    "Risk – Cause": "Risk – Cause",
    "Risk – Effect": "Risk – Effect",
    "Risk – Category": "Risk – Category",
    "Probability": "Probability",
    "Impact / Gravité": "Impact / Gravité",
    "Risk score": "Risk score",
    "Response strategy": "Response strategy",
    "Mitigation plan": "Mitigation plan",
    "Contingency plan": "Contingency plan",
    "Risk owner": "Risk owner",
    "Workstream": "Workstream",
    "Review needed ?": "Review needed ?",
    "Links": "Links",
}

# ======================== FONCTIONS UTILITAIRES ======================= #


def run_gh(args):
    """Exécute `gh ...` et retourne stdout (str), ou lève une erreur."""
    cmd = ["gh"] + args
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def create_issue(title: str, body: str) -> str:
    """Crée une issue et renvoie son URL."""
    out = run_gh(
        [
            "issue",
            "create",
            "-R",
            REPO,
            "-t",
            title,
            "-b",
            body,
            "--json",
            "url",
        ]
    )
    data = json.loads(out)
    url = data.get("url")
    if not url:
        raise RuntimeError(f"Impossible de récupérer l'URL de l'issue pour '{title}'")
    return url


def add_issue_to_project(issue_url: str) -> str:
    """Ajoute une issue au Project et renvoie l'ID de l'item dans le Project."""
    out = run_gh(
        [
            "project",
            "item-add",
            str(PROJECT_NUMBER),
            "--owner",
            OWNER,
            "--url",
            issue_url,
            "--format",
            "json",
        ]
    )
    data = json.loads(out)
    item_id = data.get("id")
    if not item_id:
        raise RuntimeError(
            f"Impossible de récupérer l'id de l'item pour issue {issue_url}"
        )
    return item_id


def edit_project_item(item_id: str, fields: dict) -> None:
    """Met à jour un item du Project avec des paires champ=valeur."""
    if not fields:
        return

    args = [
        "project",
        "item-edit",
        str(PROJECT_NUMBER),
        "--owner",
        OWNER,
        "--id",
        item_id,
    ]
    for project_field_name, value in fields.items():
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        args.extend(["--field", f"{project_field_name}={value}"])

    run_gh(args)


def build_issue_body(row: dict) -> str:
    """Construit un body d'issue lisible à partir de la ligne CSV."""
    # Récupère quelques infos clés
    wbs = row.get("WBS ID", "").strip()
    mid = row.get("Murphy ID", "").strip()
    phase = row.get("Phase", "").strip()
    category = row.get("Risk – Category", "").strip()
    prob = row.get("Probability", "").strip()
    impact = row.get("Impact / Gravité", "").strip()
    score = row.get("Risk score", "").strip()
    cause = row.get("Risk – Cause", "").strip()
    effect = row.get("Risk – Effect", "").strip()
    mitigation = row.get("Mitigation plan", "").strip()
    contingency = row.get("Contingency plan", "").strip()
    desc = row.get("Description", "").strip()

    body = dedent(
        f"""
        **WBS ID**: {wbs}
        **Murphy ID**: {mid}

        **Phase**: {phase}
        **Catégorie de risque**: {category}

        **Probabilité**: {prob}
        **Impact**: {impact}
        **Score**: {score}

        **Description**
        {desc}

        **Cause**
        {cause}

        **Effet**
        {effect}

        **Plan de mitigation**
        {mitigation}

        **Plan de contingence**
        {contingency}
        """
    ).strip()

    return body


# ============================== MAIN ================================= #


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Erreur : CSV introuvable : {CSV_PATH}", file=sys.stderr)
        return 1

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if TITLE_COLUMN not in reader.fieldnames:
            print(
                f"Erreur : la colonne '{TITLE_COLUMN}' n'existe pas dans le CSV.",
                file=sys.stderr,
            )
            print("Colonnes disponibles :", ", ".join(reader.fieldnames or []))
            return 1

        print(
            f"Import d'issues depuis {CSV_PATH} "
            f"vers repo {REPO} et Project #{PROJECT_NUMBER} ({OWNER})"
        )

        for idx, row in enumerate(reader, start=1):
            title = (row.get(TITLE_COLUMN) or "").strip()
            if not title:
                # fallback : WBS + Murphy ID
                wbs = (row.get("WBS ID") or "").strip()
                mid = (row.get("Murphy ID") or "").strip()
                title = (wbs or mid or f"Item {idx}").strip()

            print(f"\n→ Ligne {idx} : création issue pour '{title}'")

            try:
                body = build_issue_body(row)
                issue_url = create_issue(title, body)
                print(f"  Issue créée : {issue_url}")
            except Exception as exc:
                print(f"  !! Échec création issue : {exc}", file=sys.stderr)
                continue

            try:
                item_id = add_issue_to_project(issue_url)
                print(f"  Item Project créé : {item_id}")
            except Exception as exc:
                print(f"  !! Échec ajout Project : {exc}", file=sys.stderr)
                continue

            # Construire la map {champ_project: valeur}
            fields_to_set: dict[str, str] = {}
            for csv_col, project_field in FIELD_MAPPING.items():
                if csv_col not in row:
                    continue
                fields_to_set[project_field] = row.get(csv_col, "")

            try:
                edit_project_item(item_id, fields_to_set)
                print("  Champs mis à jour.")
            except Exception as exc:
                print(f"  !! Échec mise à jour des champs : {exc}", file=sys.stderr)
                continue

    print("\n✅ Import terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
