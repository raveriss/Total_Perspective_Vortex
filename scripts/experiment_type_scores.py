"""Calcule les scores moyens par type d'expérience pour chaque sujet."""

# Préserve argparse pour exposer une CLI dédiée aux scores par expérience
import argparse

# Fournit l'écriture CSV pour les rapports d'accuracy
import csv

# Fournit la sérialisation JSON pour les rapports structurés
import json

# Fournit la dataclass pour regrouper les paramètres d'exécution
from dataclasses import dataclass

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Centralise la moyenne arithmétique pour les agrégations
from statistics import mean

# Garantit l'accès aux types pour clarifier les signatures
from typing import Callable, Iterable, TypedDict

# Réutilise l'évaluation d'un run pour obtenir l'accuracy
from scripts.predict import evaluate_run

# Réutilise la CLI d'entraînement pour générer les artefacts manquants
from scripts.train import TrainingRequest, run_training

# Réutilise la configuration de pipeline partagée avec train.py
from tpv.pipeline import PipelineConfig

# Définit le seuil cible imposé pour les moyennes par expérience
TARGET_ACCURACY = 0.75

# Définit le seuil de bonus en points au-delà de 75 %
BONUS_THRESHOLD = 0.75

# Définit l'incrément de points par tranche de bonus
BONUS_STEP = 0.03

# Définit le répertoire par défaut des données prétraitées
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut pour les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Définit le répertoire par défaut des EDF bruts
DEFAULT_RAW_DIR = Path("data")


# Décrit la structure du rapport agrégé pour les types d'expérience
class ExperimentTypeReport(TypedDict):
    """Structure typée du rapport des moyennes par type."""

    # Associe chaque sujet aux moyennes par type d'expérience
    subjects: dict[str, dict[str, float]]
    # Fournit la moyenne globale par type d'expérience
    by_type: dict[str, float]
    # Stocke la moyenne globale des quatre types
    overall_mean: float
    # Indique si le seuil cible est atteint
    meets_target: bool
    # Stocke le nombre de points bonus calculés
    bonus_points: int


# Regroupe les paramètres d'entraînement pour tous les sujets
@dataclass
class ExperimentTrainingConfig:
    """Stocke les paramètres communs aux entraînements par sujet."""

    # Stocke la configuration de pipeline à appliquer
    pipeline_config: PipelineConfig
    # Fixe le répertoire des données numpy
    data_dir: Path
    # Fixe le répertoire des artefacts à produire
    artifacts_dir: Path
    # Fixe le répertoire des EDF bruts
    raw_dir: Path
    # Active la recherche d'hyperparamètres si demandé
    enable_grid_search: bool = False
    # Définit un nombre de splits spécifique si fourni
    grid_search_splits: int | None = None


# Expose la cartographie des quatre types d'expériences Physionet
def build_experiment_types() -> dict[str, tuple[str, ...]]:
    """Retourne les types d'expériences attendus (4 types)."""

    # Mappe les runs MI gauche/droite pour l'imagerie
    imagery_left_right = ("R03", "R07", "R11")
    # Mappe les runs MI poings/pieds pour l'imagerie
    imagery_fists_feet = ("R04", "R08", "R12")
    # Mappe les runs mouvement gauche/droite pour l'exécution réelle
    movement_left_right = ("R05", "R09", "R13")
    # Mappe les runs mouvement poings/pieds pour l'exécution réelle
    movement_fists_feet = ("R06", "R10", "R14")
    # Retourne la structure complète pour l'agrégation
    return {
        "imagery_left_right": imagery_left_right,
        "imagery_fists_feet": imagery_fists_feet,
        "movement_left_right": movement_left_right,
        "movement_fists_feet": movement_fists_feet,
    }


# Convertit un indice de sujet en identifiant Physionet
def subject_identifier(subject_index: int) -> str:
    """Formate l'identifiant Sxxx attendu par les scripts."""

    # Formate l'index sur trois chiffres avec préfixe S
    return f"S{subject_index:03d}"


# Calcule une moyenne en renvoyant 0.0 si la séquence est vide
def safe_mean(values: Iterable[float]) -> float:
    """Retourne la moyenne ou 0.0 pour éviter ZeroDivisionError."""

    # Convertit l'itérable en liste pour permettre la mesure
    measurements = list(values)
    # Retourne 0.0 si aucune valeur n'est disponible
    if not measurements:
        # Évite une division par zéro en fixant une moyenne neutre
        return 0.0
    # Calcule la moyenne arithmétique des mesures fournies
    return mean(measurements)


# Calcule les points bonus à partir de la moyenne globale
def compute_bonus_points(overall_mean: float) -> int:
    """Retourne le nombre de points bonus au-delà de 75 %."""

    # Retourne zéro si le seuil de bonus n'est pas atteint
    if overall_mean < BONUS_THRESHOLD:
        # Renvoie zéro pour signaler l'absence de bonus
        return 0
    # Calcule l'écart au-dessus du seuil de bonus
    bonus_delta = overall_mean - BONUS_THRESHOLD
    # Calcule le nombre de tranches complètes de 3 %
    bonus_steps = int(bonus_delta // BONUS_STEP)
    # Retourne le nombre de points bonus
    return bonus_steps


# Construit le parser CLI pour la commande
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'évaluation par type."""

    # Crée le parser avec une description explicite
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne chaque sujet et calcule les moyennes par type "
            "d'expérience (4 types Physionet)."
        ),
    )
    # Définit la plage de sujets par défaut
    parser.add_argument(
        "--subject-start",
        type=int,
        default=1,
        help="Indice du premier sujet (ex: 1 pour S001)",
    )
    # Définit la fin de la plage de sujets par défaut
    parser.add_argument(
        "--subject-end",
        type=int,
        default=109,
        help="Indice du dernier sujet inclus (ex: 109 pour S109)",
    )
    # Autorise une liste explicite de sujets pour réduire le périmètre
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Liste CSV de sujets (ex: S001,S002,3) pour écraser la plage",
    )
    # Ajoute la configuration classifier alignée sur scripts/train.py
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm", "centroid"),
        default="lda",
        help="Classifieur final utilisé pour l'entraînement",
    )
    # Ajoute la configuration scaler alignée sur scripts/train.py
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "none"),
        default="none",
        help="Scaler optionnel appliqué après les features",
    )
    # Ajoute la stratégie de features alignée sur scripts/train.py
    parser.add_argument(
        "--feature-strategy",
        choices=("fft", "wavelet"),
        default="fft",
        help="Méthode d'extraction de features spectrales",
    )
    # Ajoute la réduction de dimension alignée sur scripts/train.py
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp", "svd"),
        default="pca",
        help="Méthode de réduction de dimension pour la pipeline",
    )
    # Ajoute le nombre de composantes optionnel
    parser.add_argument(
        "--n-components",
        type=int,
        default=argparse.SUPPRESS,
        help="Nombre de composantes conservées par le réducteur",
    )
    # Ajoute un flag pour désactiver la normalisation des features
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Désactive la normalisation des features extraites",
    )
    # Ajoute la fréquence d'échantillonnage alignée sur scripts/train.py
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage des features",
    )
    # Ajoute un flag pour activer la recherche d'hyperparamètres
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Active la recherche d'hyperparamètres pour l'entraînement",
    )
    # Ajoute un nombre de splits optionnel pour la recherche
    parser.add_argument(
        "--grid-search-splits",
        type=int,
        default=None,
        help="Nombre de splits pour la GridSearchCV (optionnel)",
    )
    # Ajoute le chemin des données numpy
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les matrices numpy",
    )
    # Ajoute le chemin des artefacts d'entraînement
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine pour les modèles entraînés",
    )
    # Ajoute le chemin des EDF bruts
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Répertoire racine contenant les fichiers EDF bruts",
    )
    # Ajoute un chemin optionnel de sortie CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin du rapport CSV des moyennes",
    )
    # Ajoute un chemin optionnel de sortie JSON
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin du rapport JSON des moyennes",
    )
    # Retourne le parser prêt pour l'exécution
    return parser


# Interprète la liste des sujets fournie en CLI
def parse_subjects(subjects_csv: str, start: int, end: int) -> list[str]:
    """Retourne la liste des identifiants Sxxx à traiter."""

    # Retourne la plage par défaut si aucune liste explicite
    if not subjects_csv:
        # Construit les identifiants Sxxx depuis la plage fournie
        return [subject_identifier(idx) for idx in range(start, end + 1)]
    # Initialise la liste de sortie pour l'ordre demandé
    subjects: list[str] = []
    # Découpe la chaîne CSV pour extraire chaque token
    for raw_value in subjects_csv.split(","):
        # Nettoie les espaces éventuels autour du token
        value = raw_value.strip()
        # Ignore les entrées vides si la chaîne contient des virgules finales
        if not value:
            # Passe à l'entrée suivante pour éviter un crash
            continue
        # Retire le préfixe S si présent pour garder l'index nu
        numeric = value[1:] if value.upper().startswith("S") else value
        # Vérifie que l'entrée restante est numérique
        if not numeric.isdigit():
            # Signale un format invalide pour éviter un sujet incohérent
            raise ValueError(f"Sujet invalide: {value}")
        # Convertit l'index en entier pour le formatter
        subject_index = int(numeric)
        # Ajoute l'identifiant formaté à la liste finale
        subjects.append(subject_identifier(subject_index))
    # Retourne la liste telle que fournie par l'utilisateur
    return subjects


# Construit la configuration pipeline depuis les arguments CLI
def build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Construit la PipelineConfig alignée sur scripts/train.py."""

    # Convertit l'option scaler "none" en None attendu par la pipeline
    scaler = None if args.scaler == "none" else args.scaler
    # Calcule la normalisation via l'opt-out du flag
    normalize = not args.no_normalize_features
    # Récupère la valeur optionnelle des composantes
    n_components = getattr(args, "n_components", None)
    # Retourne la configuration de pipeline complète
    return PipelineConfig(
        sfreq=args.sfreq,
        feature_strategy=args.feature_strategy,
        normalize_features=normalize,
        dim_method=args.dim_method,
        n_components=n_components,
        classifier=args.classifier,
        scaler=scaler,
    )


# Entraîne puis score un run pour un sujet donné
def train_and_score_run(
    subject: str,
    run: str,
    config: ExperimentTrainingConfig,
) -> float:
    """Entraîne la pipeline puis calcule l'accuracy d'un run."""

    # Construit la requête d'entraînement pour le run ciblé
    request = TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=config.pipeline_config,
        data_dir=config.data_dir,
        artifacts_dir=config.artifacts_dir,
        raw_dir=config.raw_dir,
        enable_grid_search=config.enable_grid_search,
        grid_search_splits=config.grid_search_splits,
    )
    # Déclenche l'entraînement pour matérialiser les artefacts
    run_training(request)
    # Évalue l'accuracy du run entraîné
    result = evaluate_run(
        subject=subject,
        run=run,
        data_dir=config.data_dir,
        artifacts_dir=config.artifacts_dir,
        raw_dir=config.raw_dir,
    )
    # Retourne l'accuracy calculée pour l'agrégation
    return float(result["accuracy"])


# Calcule le rapport agrégé par type d'expérience
def compute_experiment_type_report(
    subjects: Iterable[str],
    experiment_types: dict[str, tuple[str, ...]],
    score_lookup: Callable[[str, str], float],
) -> ExperimentTypeReport:
    """Retourne les moyennes par sujet, par type, et la moyenne globale."""

    # Prépare le stockage des moyennes par sujet
    subject_means: dict[str, dict[str, float]] = {}
    # Prépare le stockage des scores par type pour la moyenne globale
    type_scores: dict[str, list[float]] = {
        exp_type: [] for exp_type in experiment_types
    }
    # Parcourt chaque sujet fourni
    for subject in subjects:
        # Prépare le dictionnaire des moyennes par type pour ce sujet
        per_type: dict[str, float] = {}
        # Parcourt chaque type d'expérience pour agréger les runs
        for exp_type, runs in experiment_types.items():
            # Calcule les scores pour chaque run du type
            run_scores = [score_lookup(subject, run) for run in runs]
            # Calcule la moyenne pour ce sujet et ce type
            mean_score = safe_mean(run_scores)
            # Stocke la moyenne pour ce sujet
            per_type[exp_type] = mean_score
            # Accumule la moyenne pour la moyenne globale du type
            type_scores[exp_type].append(mean_score)
        # Stocke les moyennes pour le sujet courant
        subject_means[subject] = per_type
    # Calcule la moyenne par type d'expérience
    type_means = {
        exp_type: safe_mean(scores) for exp_type, scores in type_scores.items()
    }
    # Calcule la moyenne globale des quatre types
    overall_mean = safe_mean(type_means.values())
    # Calcule les points bonus au-delà de 75 %
    bonus_points = compute_bonus_points(overall_mean)
    # Retourne le rapport agrégé avec indicateur de seuil
    return {
        "subjects": subject_means,
        "by_type": type_means,
        "overall_mean": overall_mean,
        "meets_target": overall_mean >= TARGET_ACCURACY,
        "bonus_points": bonus_points,
    }


# Écrit un rapport CSV des moyennes calculées
def write_csv(report: ExperimentTypeReport, csv_path: Path) -> None:
    """Sérialise le rapport au format CSV."""

    # Crée le répertoire cible avant écriture
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier CSV en écriture texte
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        # Définit les colonnes de sortie pour le CSV
        fieldnames = ["subject", "experiment_type", "mean_accuracy"]
        # Instancie le writer CSV avec en-têtes
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Écrit l'en-tête pour les outils de lecture
        writer.writeheader()
        # Parcourt les sujets et leurs moyennes par type
        for subject, per_type in report["subjects"].items():
            # Parcourt chaque type pour écrire une ligne CSV
            for exp_type, accuracy in per_type.items():
                # Écrit la ligne CSV avec précision fixe
                writer.writerow(
                    {
                        "subject": subject,
                        "experiment_type": exp_type,
                        "mean_accuracy": f"{accuracy:.6f}",
                    }
                )


# Écrit un rapport JSON des moyennes calculées
def write_json(report: ExperimentTypeReport, json_path: Path) -> None:
    """Sérialise le rapport au format JSON."""

    # Crée le répertoire cible avant écriture
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Ouvre le fichier JSON en écriture texte
    with json_path.open("w", encoding="utf-8") as handle:
        # Sérialise le rapport en JSON indenté
        json.dump(report, handle, indent=2)


# Point d'entrée principal de la commande
def main(argv: list[str] | None = None) -> int:
    """Lance l'entraînement complet et calcule les moyennes par type."""

    # Construit le parser CLI
    parser = build_parser()
    # Parse les arguments utilisateur
    args = parser.parse_args(argv)
    # Construit la configuration de pipeline
    pipeline_config = build_pipeline_config(args)
    # Construit la configuration d'entraînement partagée
    training_config = ExperimentTrainingConfig(
        pipeline_config=pipeline_config,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        raw_dir=args.raw_dir,
        enable_grid_search=args.grid_search,
        grid_search_splits=args.grid_search_splits,
    )
    # Construit la liste des sujets à traiter
    try:
        # Interprète la liste des sujets depuis la CLI
        subjects = parse_subjects(args.subjects, args.subject_start, args.subject_end)
    except ValueError as error:
        # Affiche une erreur lisible en cas de sujet mal formé
        print(f"ERREUR: {error}")
        # Retourne un code d'échec explicite
        return 1
    # Construit la cartographie des types d'expériences
    experiment_types = build_experiment_types()
    # Initialise un cache pour éviter les doubles évaluations
    score_cache: dict[tuple[str, str], float] = {}

    # Définit la fonction de scoring réutilisable avec cache
    def score_lookup(subject: str, run: str) -> float:
        """Retourne l'accuracy en entraînant si nécessaire."""

        # Définit la clé de cache pour ce couple
        cache_key = (subject, run)
        # Retourne l'accuracy si déjà calculée
        if cache_key in score_cache:
            # Évite un recalcul coûteux en réutilisant le cache
            return score_cache[cache_key]
        # Calcule l'accuracy en entraînant le run
        accuracy = train_and_score_run(subject, run, training_config)
        # Mémorise la valeur pour les appels suivants
        score_cache[cache_key] = accuracy
        # Retourne l'accuracy calculée
        return accuracy

    # Calcule le rapport agrégé pour tous les sujets
    report = compute_experiment_type_report(
        subjects=subjects,
        experiment_types=experiment_types,
        score_lookup=score_lookup,
    )
    # Affiche les moyennes par type d'expérience
    print("Moyennes par type d'expérience:")
    # Parcourt les types pour afficher les moyennes
    for exp_type, accuracy in report["by_type"].items():
        # Affiche la moyenne avec quatre décimales
        print(f"- {exp_type}: {accuracy:.4f}")
    # Affiche la moyenne globale des quatre types
    print(f"Moyenne globale (4 types): {report['overall_mean']:.4f}")
    # Affiche l'état du seuil cible
    print(f"Seuil {TARGET_ACCURACY:.2f} atteint: {report['meets_target']}")
    # Affiche les points bonus calculés
    print(f"Points bonus (>=75% +3%): {report['bonus_points']}")
    # Écrit le CSV si demandé par l'utilisateur
    if args.csv_output is not None:
        # Sérialise le rapport au format CSV
        write_csv(report, args.csv_output)
    # Écrit le JSON si demandé par l'utilisateur
    if args.json_output is not None:
        # Sérialise le rapport au format JSON
        write_json(report, args.json_output)
    # Retourne 0 si le seuil est respecté, 1 sinon
    return 0 if report["meets_target"] else 1


# Active l'exécution CLI explicite
if __name__ == "__main__":
    # Lance la CLI et propage le code de sortie
    raise SystemExit(main())
