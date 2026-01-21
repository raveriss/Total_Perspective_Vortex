# Importe argparse pour fournir une CLI dédiée au reporting
import argparse

# Garantit l'accès aux chemins indépendants de la plateforme
from pathlib import Path

# Offre des statistiques numériques stables pour les agrégations
import numpy as np

# Réutilise l'évaluation existante pour relire les artefacts sauvegardés
from scripts import predict as predict_cli

# Définit le répertoire par défaut où chercher les jeux de données
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut où lire les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# Construit un argument parser pour exposer le reporting en CLI
def build_parser() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation (WBS 7.1, 7.4)."""

    # Crée le parser avec une description orientée traçabilité
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par run, sujet et global à partir des " "artefacts"
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


# Calcule les accuracies en réutilisant les artefacts existants
def aggregate_accuracies(data_dir: Path, artifacts_dir: Path) -> dict:
    """Produit un rapport agrégé par run, sujet et global (WBS 7.1, 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(artifacts_dir)
    # Initialise le dictionnaire des accuracies par run avec des clés uniques
    by_run: dict[str, float] = {}
    # Initialise la collecte des accuracies par sujet pour la moyenne ultérieure
    by_subject_scores: dict[str, list[float]] = {}
    # Initialise la liste globale pour dériver l'accuracy moyenne
    all_scores: list[float] = []
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Construit une clé explicite pour différencier les runs similaires
        run_key = f"{subject}/{run}"
        # Stocke l'accuracy avec une clé unique pour le reporting
        by_run[run_key] = result["accuracy"]
        # Initialise la liste pour le sujet si nécessaire
        if subject not in by_subject_scores:
            # Crée un conteneur dédié pour les scores du sujet
            by_subject_scores[subject] = []
        # Enregistre l'accuracy dans la liste du sujet courant
        by_subject_scores[subject].append(result["accuracy"])
        # Ajoute l'accuracy aux scores globaux pour la moyenne finale
        all_scores.append(result["accuracy"])
    # Calcule la moyenne par sujet en conservant un float natif
    by_subject = {
        key: float(np.mean(values)) for key, values in by_subject_scores.items()
    }
    # Calcule l'accuracy globale en renvoyant 0.0 s'il n'y a pas de run
    global_accuracy = float(np.mean(all_scores)) if all_scores else 0.0
    # Retourne la structure complète prête pour l'affichage ou les tests
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Formate un tableau texte des accuracies calculées
def format_accuracy_table(report: dict) -> str:
    """Construit un tableau lisible pour les accuracies agrégées."""

    # Initialise les lignes avec un en-tête explicite
    lines = ["Run\tSubject\tAccuracy"]
    # Trie les runs pour offrir une lecture stable
    for run_key in sorted(report["by_run"].keys()):
        # Sépare le sujet du run pour l'affichage tabulaire
        subject, run = run_key.split("/")
        # Formate la ligne avec trois colonnes tabulaires
        lines.append(f"{run}\t{subject}\t{report['by_run'][run_key]:.3f}")
    # Ajoute un séparateur visuel entre détails et agrégations
    lines.append("Subject\tMean Accuracy")
    # Trie les sujets pour conserver un ordre reproductible
    for subject in sorted(report["by_subject"].keys()):
        # Ajoute la ligne moyenne par sujet avec trois décimales
        lines.append(f"{subject}\t{report['by_subject'][subject]:.3f}")
    # Ajoute l'accuracy globale en dernière ligne pour synthèse
    lines.append(f"Global\t{report['global']:.3f}")
    # Assemble les lignes avec des retours à la ligne explicites
    return "\n".join(lines)


# Point d'entrée principal pour l'usage en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau d'accuracies."""
    print("\n\naggregate_accuracy.py\n\n")
    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy en lisant les artefacts existants
    report = aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_accuracy_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Retourne 0 pour signaler un succès standard
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
