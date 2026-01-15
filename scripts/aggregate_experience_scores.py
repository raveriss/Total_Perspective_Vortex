# Fournit un parsing CLI structuré pour l'agrégation par expérience
import argparse

# Garantit l'écriture CSV pour le reporting par expérience
import csv

# Garantit un import dynamique sans dépendance de l'ordre sys.path
import importlib

# Garantit la lecture JSON des rapports persistés pour accélérer le scoring
import json

# Garantit l'accès aux modules locaux même via un appel direct
import sys

# Garantit la manipulation de chemins indépendants du système
from pathlib import Path

# Assure des moyennes numériques stables pour les agrégations
import numpy as np

# Localise la racine du dépôt pour construire un sys.path stable
REPO_ROOT = Path(__file__).resolve().parents[1]
# Évite les doublons pour stabiliser l'ordre de résolution
if str(REPO_ROOT) not in sys.path:
    # Priorise la résolution locale des modules du repo
    sys.path.insert(0, str(REPO_ROOT))

# Charge le module predict après l'insertion du repo dans sys.path
predict_cli = importlib.import_module("scripts.predict")

# Définit le répertoire par défaut où chercher les jeux de données
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut où lire les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Déclare le mapping officiel runs -> types d'expériences (R03-R14)
EXPERIENCE_RUNS: dict[str, tuple[str, ...]] = {
    # Regroupe les runs de la tâche motrice réelle main gauche/droite
    "T1": ("R03", "R07", "R11"),
    # Regroupe les runs d'imagerie motrice main gauche/droite
    "T2": ("R04", "R08", "R12"),
    # Regroupe les runs de tâche motrice poings/pieds
    "T3": ("R05", "R09", "R13"),
    # Regroupe les runs d'imagerie motrice poings/pieds
    "T4": ("R06", "R10", "R14"),
}

# Définit l'ordre d'affichage stable des expériences
EXPERIENCE_ORDER = ("T1", "T2", "T3", "T4")

# Définit le seuil de score pour déclencher des points bonus
BONUS_THRESHOLD = 0.75
# Définit le pas de progression des points bonus au-delà du seuil
BONUS_STEP = 0.03


# Construit un argument parser pour exposer l'agrégation en CLI
def build_parser() -> argparse.ArgumentParser:
    """Prépare le parser de la commande d'agrégation par expérience."""

    # Crée le parser avec une description orientée checklist
    parser = argparse.ArgumentParser(
        description=(
            "Agrège les accuracies par sujet et type d'expérience "
            "pour les runs R03-R14"
        ),
    )
    # Ajoute une option pour pointer vers un répertoire de données alternatif
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy utilisées",
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
        help="Chemin de sortie du rapport CSV par sujet",
    )
    # Ajoute un flag pour autoriser l'auto-train en l'absence d'artefacts
    parser.add_argument(
        "--auto-train-missing",
        action="store_true",
        help="Autorise le scan data/ et l'auto-train des runs manquants",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Inventorie les runs disponibles en inspectant les artefacts présents
def _discover_runs_from_artifacts(artifacts_dir: Path) -> list[tuple[str, str]]:
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


# Détecte si des données existent pour un run dans le dataset local
def _has_run_data(data_dir: Path, subject: str, run: str) -> bool:
    """Retourne True si un EDF ou un .npy est disponible pour ce run."""

    # Construit le chemin attendu du fichier EDF pour le run
    edf_path = data_dir / subject / f"{subject}{run}.edf"
    # Construit le chemin attendu des features numpy pour le run
    features_path = data_dir / subject / f"{run}_X.npy"
    # Construit le chemin attendu des labels numpy pour le run
    labels_path = data_dir / subject / f"{run}_y.npy"
    # Confirme la présence d'un support de données exploitable
    return edf_path.exists() or (features_path.exists() and labels_path.exists())


# Inventorie les runs disponibles en inspectant les données brutes
def _discover_runs_from_data(data_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) présents dans le dataset Physionet."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Ignore silencieusement l'exploration si le dossier n'existe pas
    if not data_dir.exists():
        # Retourne une liste vide pour signaler l'absence de données
        return runs
    # Aplatis la liste des runs attendus pour les expériences T1-T4
    expected_runs = [run for runs in EXPERIENCE_RUNS.values() for run in runs]
    # Parcourt les dossiers de sujets pour détecter les runs
    for subject_dir in sorted(data_dir.iterdir()):
        # Ignore les éléments qui ne représentent pas un sujet
        if not subject_dir.is_dir():
            # Passe au chemin suivant pour éviter les collisions
            continue
        # Récupère le nom du sujet pour composer les chemins de runs
        subject = subject_dir.name
        # Parcourt chaque run attendu pour la moyenne par expérience
        for run in expected_runs:
            # Ajoute uniquement les runs disposant de données accessibles
            if _has_run_data(data_dir, subject, run):
                # Enregistre le couple (sujet, run) pour l'agrégation
                runs.append((subject, run))
    # Retourne l'ensemble des couples détectés
    return runs


# Sélectionne l'origine des runs en priorisant les artefacts existants
def _discover_runs(
    data_dir: Path,
    artifacts_dir: Path,
    allow_data_scan: bool,
) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) via artefacts ou dataset."""

    # Tente d'abord de réutiliser les artefacts persistés
    runs = _discover_runs_from_artifacts(artifacts_dir)
    # Retourne immédiatement si des artefacts sont disponibles
    if runs:
        # Préserve le chemin rapide pour éviter un scan lourd des données
        return runs
    # Refuse le scan des données si l'auto-train est désactivé
    if not allow_data_scan:
        # Informe l'utilisateur que seuls les artefacts existants sont évalués
        print(
            "INFO: aucun artefact trouvé, "
            "scan data désactivé (utilisez --auto-train-missing pour activer)."
        )
        # Retourne une liste vide pour éviter un entraînement coûteux
        return []
    # Bascule vers un scan des données pour déclencher l'auto-train
    return _discover_runs_from_data(data_dir)


# Associe un run à un type d'expérience connu
def _map_run_to_experience(run: str) -> str | None:
    """Retourne le type d'expérience pour un run (ou None si baseline)."""

    # Parcourt chaque type d'expérience défini dans le mapping officiel
    for experience, runs in EXPERIENCE_RUNS.items():
        # Retourne le type d'expérience si le run est listé
        if run in runs:
            # Retourne immédiatement le type associé
            return experience
    # Retourne None pour ignorer R01/R02 et les runs inconnus
    return None


# Calcule le nombre de points bonus selon la moyenne globale
def compute_bonus_points(mean_score: float | None) -> int:
    """Retourne le nombre de points bonus au-delà de 75 %."""

    # Retourne zéro si aucun score global n'est disponible
    if mean_score is None:
        # Signale l'absence de bonus si les données sont incomplètes
        return 0
    # Ignore le bonus si la moyenne n'atteint pas le seuil requis
    if mean_score <= BONUS_THRESHOLD:
        # Retourne zéro pour refléter l'absence de bonus
        return 0
    # Calcule l'écart au seuil pour dimensionner les points
    delta = mean_score - BONUS_THRESHOLD
    # Calcule le nombre de points en paliers de 3 %
    return int(delta // BONUS_STEP) + 1


# Construit les chemins d'artefacts attendus pour un run
def _artifact_paths(artifacts_dir: Path, subject: str, run: str) -> tuple[Path, Path]:
    """Retourne les chemins du modèle et de la matrice W attendus."""

    # Calcule le dossier d'artefacts pour accéder aux fichiers
    target_dir = artifacts_dir / subject / run
    # Construit le chemin du modèle sérialisé attendu
    model_path = target_dir / "model.joblib"
    # Construit le chemin de la matrice W attendue
    w_matrix_path = target_dir / "w_matrix.joblib"
    # Retourne les deux chemins pour vérification
    return model_path, w_matrix_path


# Charge une accuracy persistée si un rapport JSON existe
def _load_cached_accuracy(artifacts_dir: Path, subject: str, run: str) -> float | None:
    """Retourne l'accuracy du rapport JSON si disponible."""

    # Calcule le dossier d'artefacts pour accéder au rapport éventuel
    target_dir = artifacts_dir / subject / run
    # Détermine le chemin d'un rapport déjà généré
    report_path = target_dir / "report.json"
    # Retourne None si aucun rapport n'est présent
    if not report_path.exists():
        # Signale l'absence de cache pour déclencher un calcul
        return None
    # Charge le contenu JSON du rapport persisté
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Récupère l'accuracy du rapport si elle existe
    accuracy = report.get("accuracy")
    # Retourne une accuracy typée si elle est valide
    if isinstance(accuracy, (float, int)):
        # Convertit explicitement en float pour homogénéiser le type
        return float(accuracy)
    # Retourne None si la clé est absente ou invalide
    return None


# Ajoute une accuracy au conteneur par sujet et par expérience
def _append_subject_score(
    subject_scores: dict[str, dict[str, list[float]]],
    subject: str,
    experience: str,
    accuracy: float,
) -> None:
    """Enregistre une accuracy dans le dictionnaire d'agrégation."""

    # Initialise le dictionnaire du sujet si nécessaire
    if subject not in subject_scores:
        # Crée un conteneur dédié pour les scores du sujet
        subject_scores[subject] = {}
    # Initialise la liste de scores pour l'expérience si nécessaire
    if experience not in subject_scores[subject]:
        # Crée la liste des scores pour le type courant
        subject_scores[subject][experience] = []
    # Empile l'accuracy pour la moyenne par expérience
    subject_scores[subject][experience].append(accuracy)


# Agrège les scores par sujet en parcourant les runs disponibles
def _collect_subject_scores(
    runs: list[tuple[str, str]],
    data_dir: Path,
    artifacts_dir: Path,
    allow_auto_train: bool,
) -> dict[str, dict[str, list[float]]]:
    """Regroupe les accuracies par sujet et type d'expérience."""

    # Initialise le conteneur des scores par sujet et par type
    subject_scores: dict[str, dict[str, list[float]]] = {}
    # Parcourt chaque run détecté pour calculer l'accuracy associée
    for subject, run in runs:
        # Associe le run au type d'expérience attendu
        experience = _map_run_to_experience(run)
        # Ignore les runs baseline ou hors protocole
        if experience is None:
            # Passe au run suivant pour éviter les fausses moyennes
            continue
        # Charge une accuracy persistée si disponible
        cached_accuracy = _load_cached_accuracy(artifacts_dir, subject, run)
        # Ajoute la valeur persistée si elle existe
        if cached_accuracy is not None:
            # Enregistre l'accuracy dans l'agrégation par sujet
            _append_subject_score(subject_scores, subject, experience, cached_accuracy)
            # Passe au run suivant pour éviter un recalcul
            continue
        # Résout les chemins d'artefacts nécessaires à l'évaluation
        model_path, w_matrix_path = _artifact_paths(artifacts_dir, subject, run)
        # Saute l'auto-train si les artefacts sont incomplets et interdits
        if not allow_auto_train and (
            not model_path.exists() or not w_matrix_path.exists()
        ):
            # Informe l'utilisateur que le run est ignoré par manque d'artefact
            print("INFO: modèle absent pour " f"{subject} {run}, auto-train désactivé.")
            # Passe au run suivant sans entraîner
            continue
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)
        # Ajoute l'accuracy résultante pour l'expérience
        _append_subject_score(subject_scores, subject, experience, result["accuracy"])
    # Retourne le regroupement par sujet
    return subject_scores


# Construit les entrées par sujet avec moyenne par expérience
def _build_subject_entries(
    subject_scores: dict[str, dict[str, list[float]]],
) -> list[dict]:
    """Construit les entrées par sujet pour le reporting."""

    # Initialise la liste des lignes agrégées par sujet
    subject_entries: list[dict] = []
    # Parcourt les sujets dans un ordre déterministe
    for subject in sorted(subject_scores.keys()):
        # Récupère les scores par type d'expérience pour ce sujet
        experience_scores = subject_scores[subject]
        # Calcule les moyennes par type dans l'ordre défini
        means_by_experience = {
            # Calcule la moyenne uniquement si des scores sont présents
            experience: (
                float(np.mean(experience_scores.get(experience, [])))
                if experience_scores.get(experience)
                else None
            )
            # Itère sur l'ordre standard pour un affichage stable
            for experience in EXPERIENCE_ORDER
        }
        # Vérifie si toutes les expériences sont disponibles
        eligible = all(value is not None for value in means_by_experience.values())
        # Filtre les moyennes présentes pour satisfaire mypy
        present_means = [
            value for value in means_by_experience.values() if value is not None
        ]
        # Calcule la moyenne des quatre moyennes si toutes sont présentes
        mean_of_means = (
            # Calcule la moyenne sur les quatre types attendus
            float(np.mean(present_means))
            if eligible
            else None
        )
        # Ajoute l'entrée structurée pour ce sujet
        subject_entries.append(
            {
                "subject": subject,
                "means": means_by_experience,
                "eligible": eligible,
                "mean_of_means": mean_of_means,
            }
        )
    # Retourne les entrées prêtes pour l'affichage
    return subject_entries


# Extrait la liste des moyennes globales éligibles
def _extract_eligible_means(subject_entries: list[dict]) -> list[float]:
    """Retourne les moyennes des sujets complets."""

    # Retourne uniquement les moyennes non nulles
    return [
        # Conserve uniquement les sujets avec quatre expériences valides
        entry["mean_of_means"]
        for entry in subject_entries
        if entry["mean_of_means"] is not None
    ]


# Calcule les moyennes par type d'expérience pour chaque sujet
def aggregate_experience_scores(
    data_dir: Path,
    artifacts_dir: Path,
    allow_auto_train: bool,
) -> dict:
    """Produit un rapport agrégé par sujet et type d'expérience (WBS 7.4)."""

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(data_dir, artifacts_dir, allow_auto_train)
    # Regroupe les accuracies par sujet et type d'expérience
    subject_scores = _collect_subject_scores(
        runs, data_dir, artifacts_dir, allow_auto_train
    )
    # Construit les entrées prêtes pour affichage ou export
    subject_entries = _build_subject_entries(subject_scores)
    # Extrait les moyennes valides pour le score global
    eligible_means = _extract_eligible_means(subject_entries)
    # Calcule la moyenne globale sur les sujets complets
    global_mean = float(np.mean(eligible_means)) if eligible_means else None
    # Calcule les points bonus associés à la moyenne globale
    bonus_points = compute_bonus_points(global_mean)
    # Retourne la structure complète prête pour l'affichage ou export
    return {
        "subjects": subject_entries,
        "global_mean": global_mean,
        "eligible_subjects": len(eligible_means),
        "bonus_points": bonus_points,
    }


# Formate un tableau texte des moyennes par expérience
def format_experience_table(report: dict) -> str:
    """Construit un tableau lisible des moyennes par type d'expérience."""

    # Définit l'en-tête du tableau pour l'affichage CLI
    header = ["Subject", *EXPERIENCE_ORDER, "Mean", "Eligible"]
    # Initialise les lignes avec l'en-tête tabulé
    lines = ["\t".join(header)]
    # Parcourt les entrées pour chaque sujet
    for entry in report["subjects"]:
        # Prépare les valeurs formatées par type d'expérience
        values = []
        # Parcourt les types pour afficher les moyennes dans l'ordre
        for experience in EXPERIENCE_ORDER:
            # Récupère la moyenne ou None pour ce type
            mean_value = entry["means"][experience]
            # Ajoute une valeur formatée ou un placeholder
            values.append(
                "{:.3f}".format(mean_value) if mean_value is not None else "n/a"
            )
        # Formate la moyenne des quatre moyennes si disponible
        mean_of_means = (
            "{:.3f}".format(entry["mean_of_means"])
            if entry["mean_of_means"] is not None
            else "n/a"
        )
        # Ajoute la ligne complète pour le sujet courant
        lines.append(
            "\t".join(
                [
                    entry["subject"],
                    *values,
                    mean_of_means,
                    "yes" if entry["eligible"] else "no",
                ]
            )
        )
    # Ajoute une ligne de synthèse globale si disponible
    if report["global_mean"] is not None:
        # Ajoute la moyenne globale alignée sur les sujets complets
        lines.append(
            "\t".join(
                [
                    "Global",
                    *["-" for _ in EXPERIENCE_ORDER],
                    "{:.3f}".format(report["global_mean"]),
                    (
                        f"{report['eligible_subjects']} subjects,"
                        f" bonus {report['bonus_points']}"
                    ),
                ]
            )
        )
    # Retourne la table formatée prête pour l'impression
    return "\n".join(lines)


# Sérialise le rapport agrégé au format CSV
def write_csv(report: dict, csv_path: Path) -> None:
    """Écrit un tableau CSV des moyennes par sujet et par expérience."""

    # Assure la création du répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier en écriture texte pour créer le CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes du CSV par sujet
        fieldnames = [
            "subject",
            *[f"{experience}_mean" for experience in EXPERIENCE_ORDER],
            "mean_of_means",
            "eligible",
        ]
        # Construit le writer CSV avec l'en-tête défini
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour faciliter l'import tableur
        writer.writeheader()
        # Parcourt les entrées pour chaque sujet
        for entry in report["subjects"]:
            # Prépare une ligne structurée pour le CSV
            row = {
                "subject": entry["subject"],
                "mean_of_means": (
                    f"{entry['mean_of_means']:.6f}"
                    if entry["mean_of_means"] is not None
                    else ""
                ),
                "eligible": entry["eligible"],
            }
            # Remplit les colonnes des types d'expérience
            for experience in EXPERIENCE_ORDER:
                # Récupère la moyenne ou None pour ce type
                mean_value = entry["means"][experience]
                # Stocke une valeur formatée si disponible
                row[f"{experience}_mean"] = (
                    f"{mean_value:.6f}" if mean_value is not None else ""
                )
            # Écrit la ligne CSV pour le sujet courant
            writer.writerow(row)


# Point d'entrée principal pour l'usage en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau par expérience."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy par expérience
    report = aggregate_experience_scores(
        args.data_dir,
        args.artifacts_dir,
        args.auto_train_missing,
    )
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_experience_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Sérialise le CSV si demandé
    if args.csv_output:
        # Écrit le rapport CSV dans le chemin fourni
        write_csv(report, args.csv_output)
    # Retourne 0 pour signaler un succès standard
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
