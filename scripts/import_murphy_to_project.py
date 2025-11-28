#!/usr/bin/env python3
"""
Importe tpv_murphy_map.csv dans un GitHub Project v2.

Prérequis :
  - gh (GitHub CLI) avec scope `project`
  - fichier CSV compatible avec les champs du Project
"""

import csv
import subprocess
import sys
from pathlib import Path


# ---------------- CONFIG À ADAPTER SI BESOIN ---------------- #

OWNER = "raveriss"
PROJECT_NUMBER = 2

# Chemin vers le CSV (relatif à la racine du repo)
CSV_PATH = Path("docs/risk/tpv_murphy_map.csv")

# Nom de la colonne utilisée comme titre de l'item dans le Project
TITLE_COLUMN = "Title"  # adapte si besoin (ex: "Résumé", "Task Title", etc.)

# Mapping CSV -> champs du Project (noms EXACTS côté Project)
FIELD_MAPPING = {
    # CSV header         # Nom du champ dans le Project
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

# ------------------------------------------------------------ #


def run_gh(args):
    """Exécute une commande gh et renvoie stdout (str) ou lève une erreur."""
    cmd = ["gh"] + args
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def create_project_item(title: str) -> str:
    """Crée un draft item dans le Project et renvoie son ID."""
    stdout = run_gh(
        [
            "project",
            "item-create",
            str(PROJECT_NUMBER),
            "--owner",
            OWNER,
            "--title",
            title,
            "--format",
            "json",
        ]
    )
    # on parse à la main pour éviter d'importer json juste pour une clé
    # stdout ressemble à :
    # {"id":"PVTIF_xxx","...":...}
    # on récupère la valeur après "id":
    import json

    data = json.loads(stdout)
    item_id = data.get("id")
    if not item_id:
        raise RuntimeError(f"Impossible de récupérer l'id de l'item pour '{title}'")
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

    # Pour chaque champ, on ajoute --field "Nom=Valeur"
    for project_field_name, value in fields.items():
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        args.extend(["--field", f"{project_field_name}={value}"])

    run_gh(args)


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Erreur : CSV introuvable : {CSV_PATH}", file=sys.stderr)
        return 1

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_title = TITLE_COLUMN not in reader.fieldnames
        if missing_title:
            print(
                f"Erreur : la colonne '{TITLE_COLUMN}' n'existe pas dans le CSV.",
                file=sys.stderr,
            )
            print("Colonnes trouvées :", ", ".join(reader.fieldnames or []))
            return 1

        print(f"Import depuis {CSV_PATH} dans Project #{PROJECT_NUMBER} ({OWNER})")

        for idx, row in enumerate(reader, start=1):
            title = (row.get(TITLE_COLUMN) or "").strip()
            if not title:
                # fallback : WBS ID + Murphy ID
                wbs = (row.get("WBS ID") or "").strip()
                mid = (row.get("Murphy ID") or "").strip()
                title = f"{wbs} {mid}".strip() or f"Item {idx}"

            print(f"→ Création de l'item {idx}: {title}")

            try:
                item_id = create_project_item(title)
            except Exception as exc:
                print(f"  !! Échec création item '{title}': {exc}", file=sys.stderr)
                continue

            # Construire le dict {nom_champ_project: valeur}
            fields_to_set: dict[str, str] = {}
            for csv_col, project_field in FIELD_MAPPING.items():
                if csv_col not in row:
                    continue
                fields_to_set[project_field] = row.get(csv_col, "")

            try:
                edit_project_item(item_id, fields_to_set)
            except Exception as exc:
                print(
                    f"  !! Échec mise à jour des champs pour '{title}': {exc}",
                    file=sys.stderr,
                )
                continue

    print("✅ Import terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
