# Offre un parsing robuste des arguments CLI
# Garantit le parsing des options CLI de manière déclarative
import argparse

# Fournit l'écriture CSV structurée pour le reporting
import csv

# Fournit une sérialisation JSON native pour la CI
import json

# Garantit des chemins indépendants du système
from pathlib import Path

# Assure des statistiques stables pour les moyennes
import numpy as np

# Réutilise l'évaluation existante pour relire les artefacts sauvegardés
from scripts import predict as predict_cli

# Fige le seuil d'acceptation minimal des runs
MINIMUM_ACCURACY = 0.75
# Fige la cible ambitieuse pour les livrables WBS
TARGET_ACCURACY = 0.81
# Définit le répertoire par défaut où chercher les jeux de données
DEFAULT_DATA_DIR = Path("data")
# Définit le répertoire par défaut où lire les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# Construit un argument parser pour exposer l'agrégation en CLI
def build_parser() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des "
            "artefacts et écrit CSV/JSON"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées pour le scoring",
    )
    # Ajoute une option pour sélectionner un répertoire d'artefacts spécifique
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où sont stockés les modèles et matrices W",
    )
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)",
    )
    # Ajoute une option pour sérialiser les résultats en JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON aligné CI",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Inventorie les runs disponibles en inspectant les artefacts présents
def _discover_runs(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not artifacts_dir.exists():
        # Retourne une liste vide pour signaler l'absence d'artefacts
        return runs
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(artifacts_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Parcourt les dossiers de runs pour le sujet courant
        for run_dir in sorted(subject_dir.iterdir()):
            # Construit le chemin du modèle entraîné pour vérifier l'existence
            model_path = run_dir / "model.joblib"
            # Ajoute le run uniquement si le modèle persiste
            if model_path.exists():
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject_dir.name, run_dir.name))
    # Retourne l'ensemble des couples détectés
    return runs


# Calcule les indicateurs d'un run et ses drapeaux de seuil
def _score_run(subject: str, run: str, data_dir: Path, artifacts_dir: Path) -> dict:
    """Évalue un run et annote les seuils minimum/cible (WBS 7.1)."""

    # Calcule l'accuracy en rechargeant le modèle et les données
    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
    # Construit une structure complète avec les drapeaux de conformité
    return {
        "subject": subject,
        "run": run,
        "accuracy": result["accuracy"],
        "meets_minimum": result["accuracy"] >= MINIMUM_ACCURACY,
        "meets_target": result["accuracy"] >= TARGET_ACCURACY,
    }


# Calcule les accuracies agrégées par run, sujet et global
def aggregate_scores(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé aligné WBS 7.1/7.4."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise la liste des scores individuels pour chaque run
    run_entries: list[dict] = []
    # Initialise le conteneur des scores par sujet pour la moyenne
    subject_scores: dict[str, list[float]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Évalue le run et récupère les drapeaux associés
        entry = _score_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'entrée à la collection globale pour sérialisation
        run_entries.append(entry)
        # Initialise la liste pour le sujet si nécessaire
        if subject not in subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            subject_scores[subject] = []
        # Empile l'accuracy pour la moyenne par sujet
        subject_scores[subject].append(entry["accuracy"])
    # Calcule la moyenne par sujet en conservant les drapeaux globaux
    subject_entries = [
        {
            "subject": subject,
            "accuracy": float(np.mean(scores)),
            "meets_minimum": float(np.mean(scores)) >= MINIMUM_ACCURACY,
            "meets_target": float(np.mean(scores)) >= TARGET_ACCURACY,
        }
        for subject, scores in subject_scores.items()
    ]
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    all_scores = [entry["accuracy"] for entry in run_entries]
    # Calcule la moyenne globale pour le suivi WBS 7.4
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Construit l'entrée synthétique pour l'accuracy globale
    global_entry = {
        "accuracy": global_accuracy,
        "meets_minimum": global_accuracy >= MINIMUM_ACCURACY,
        "meets_target": global_accuracy >= TARGET_ACCURACY,
    }
    # Retourne la structure complète prête pour la sérialisation
    return {
        "runs": run_entries,
        "subjects": subject_entries,
        "global": global_entry,
    }


# Sérialise le rapport agrégé au format CSV
def write_csv(report: dict, csv_path: Path) -> None:
    """Écrit un tableau plat des métriques pour diffusion dataops."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes communes à toutes les lignes
        fieldnames = [
            "type",
            "subject",
            "run",
            "accuracy",
            "meets_minimum",
            "meets_target",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les runs pour écrire les mesures individuelles
        for entry in report["runs"]:
            # Écrit la ligne typée run avec toutes les colonnes
            writer.writerow(
                {
                    "type": "run",
                    "subject": entry["subject"],
                    "run": entry["run"],
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Parcourt les sujets pour écrire les moyennes
        for entry in report["subjects"]:
            # Écrit la ligne typée subject sans identifiant de run
            writer.writerow(
                {
                    "type": "subject",
                    "subject": entry["subject"],
                    "run": "",
                    "accuracy": f"{entry['accuracy']:.6f}",
                    "meets_minimum": entry["meets_minimum"],
                    "meets_target": entry["meets_target"],
                }
            )
        # Écrit la ligne globale pour synthèse WBS 7.4
        writer.writerow(
            {
                "type": "global",
                "subject": "",
                "run": "",
                "accuracy": f"{report['global']['accuracy']:.6f}",
                "meets_minimum": report["global"]["meets_minimum"],
                "meets_target": report["global"]["meets_target"],
            }
        )


# Sérialise le rapport agrégé au format JSON
def write_json(report: dict, json_path: Path) -> None:
    """Écrit une version JSON directement réutilisable par la CI."""

    # Assure la création du répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le JSON
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport complet avec une indentation lisible
        json.dump(report, handle, indent=2)


# Point d'entrée principal pour l'usage en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments puis écrit les rapports demandés."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_scores(args.data_dir, args.artifacts_dir)
    # Sérialise en CSV si un chemin est fourni
    if args.csv_output is not None:
        # Écrit le CSV pour faciliter le partage hors Python
        write_csv(report, args.csv_output)
    # Sérialise en JSON si un chemin est fourni
    if args.json_output is not None:
        # Écrit le JSON pour l'exploitation par la CI
        write_json(report, args.json_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
