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

# Rassemble les structures de configuration sans mutabilité implicite
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

# Garantit la manipulation de chemins indépendants du système
from pathlib import Path

# Cast fixe les contrats issus des imports dynamiques nécessaires au script direct.
from typing import Any, cast

# Assure des moyennes numériques stables pour les agrégations
import numpy as np

# Localise la racine du dépôt pour construire un sys.path stable
REPO_ROOT = Path(__file__).resolve().parents[1]
# Évite les doublons pour stabiliser l'ordre de résolution
if str(REPO_ROOT) not in sys.path:
    # Priorise la résolution locale des modules du repo
    sys.path.insert(0, str(REPO_ROOT))

# Centralise les choix de pipeline après avoir rendu src importable.
tpv_pipeline = importlib.import_module("tpv.pipeline")
# Charge le protocole FBCSP groupé sans dupliquer son implémentation scientifique.
tpv_evaluation = importlib.import_module("tpv.evaluation")
# Centralise les quatre expériences et le périmètre officiel EEGMMIDB.
tpv_protocol = importlib.import_module("tpv.protocol")
# Centralise également les conventions de chemins du projet.
tpv_utils = importlib.import_module("tpv.utils")

# Charge le module predict après l'insertion du repo dans sys.path
predict_cli = importlib.import_module("scripts.predict")
# Charge le module train pour déclencher un auto-train piloté
train_cli = importlib.import_module("scripts.train")

# Réutilise la racine numpy configurable sans créer une convention concurrente
DEFAULT_DATA_DIR = tpv_utils.DEFAULT_DATA_DIR
# Réutilise la racine EDF canonique pour l'auto-entraînement des runs manquants
DEFAULT_RAW_DIR = tpv_utils.DEFAULT_RAW_DIR
# Réutilise la racine canonique des modèles, matrices et rapports persistés
DEFAULT_ARTIFACTS_DIR = tpv_utils.DEFAULT_ARTIFACTS_DIR

# Préserve les noms publics historiques en déléguant au protocole partagé.
EXPERIENCE_RUNS = cast(dict[str, tuple[str, ...]], tpv_protocol.EXPERIENCE_RUNS)
# L'ordre d'affichage provient de la même source que les runs d'expérience.
EXPERIENCE_ORDER = cast(tuple[str, ...], tpv_protocol.EXPERIENCE_ORDER)

# Définit le seuil de score pour déclencher des points bonus
BONUS_THRESHOLD = 0.75
# Définit le pas de progression des points bonus au-delà du seuil
BONUS_STEP = 0.03
# Une validation croisée exige au minimum deux plis.
MIN_CV_SPLITS = 2
# La grille globale porte sur l'ensemble des sujets EEGMMIDB.
EXPECTED_SUBJECT_COUNT = int(tpv_protocol.EXPECTED_SUBJECT_COUNT)
SCORE_TARGETS = (0.75, 0.81, 0.87, 0.90)


# Regroupe les options de scoring pour limiter les signatures trop longues
@dataclass
class AggregationOptions:
    """Configure l'entraînement et l'agrégation des scores par expérience."""

    # Autorise l'auto-train si des runs manquent
    allow_auto_train: bool
    # Force le ré-entraînement même si un modèle est présent
    force_retrain: bool
    # Active la grid search pendant les auto-train
    enable_grid_search: bool
    # Fixe le nombre de splits pour la grid search
    grid_search_splits: int | None
    # Renseigne le répertoire des EDF bruts pour l'auto-train
    raw_dir: Path
    # Choisit l'extraction de features pour les modèles manquants.
    feature_strategy: str = "fft"
    # Choisit la réduction maison pour les modèles manquants.
    dim_method: str = "pca"
    # Choisit le classifieur final pour les modèles manquants.
    classifier: str = "lda"
    # Choisit éventuellement un scaler dans la pipeline complète.
    scaler: str | None = None
    # Regroupe les trois runs d'un type pour disposer de 45 essais par sujet.
    pooled_experiment_cv: bool = False
    # Fixe le nombre de plis de la preuve groupée.
    cv_splits: int = 5
    # Limite facultative réservée aux diagnostics rapides.
    subject_limit: int | None = None
    # Active la sélection d'hyperparamètres uniquement dans une CV interne.
    nested_cv: bool = True
    # Produit une preuve séparée où chaque run est tenu hors entraînement.
    strict_run_report: bool = True
    # Étend la grille à ERD/ERS, CAR et Laplacien pour l'ablation FBCSP.
    enable_preprocessing_ablation: bool = False
    # Choisit FBCSP ou la branche covariance tangentielle secondaire.
    model_family: str = "fbcsp"
    # Fige une campagne progressive sans modifier la cohorte finale de 109 sujets.
    campaign_size: int | None = None


# Regroupe les informations nécessaires à l'auto-train
@dataclass
class TrainingContext:
    """Encapsule les chemins et options pour l'entraînement automatique."""

    # Porte le répertoire de données numpy
    data_dir: Path
    # Porte le répertoire d'artefacts cibles
    artifacts_dir: Path
    # Porte les options d'agrégation pour l'auto-train
    options: AggregationOptions


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
    # Partage les trois racines utilisées pour découvrir et entraîner les runs
    tpv_utils.add_storage_arguments(parser, include_raw=True)
    # Ajoute une option pour sérialiser les résultats en CSV
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport CSV par sujet",
    )
    # Ajoute une preuve JSON contenant commits, configuration, splits et scores.
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Chemin de sortie du rapport JSON reproductible",
    )
    # Ajoute un flag pour autoriser l'auto-train en l'absence d'artefacts
    parser.add_argument(
        "--auto-train-missing",
        action="store_true",
        help="Autorise le scan data/ et l'auto-train des runs manquants",
    )
    # Ajoute un flag pour réentraîner même si un artefact existe
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force le ré-entraînement même si un modèle existe déjà",
    )
    # Ajoute un flag pour activer la recherche d'hyperparamètres en auto-train
    parser.add_argument(
        "--auto-train-grid-search",
        action="store_true",
        help="Active la grid search lors des auto-train (plus lent, souvent +score)",
    )
    # Ajoute une option pour forcer le nombre de splits en grid search
    parser.add_argument(
        "--grid-search-splits",
        type=int,
        default=None,
        help="Nombre de splits CV dédié à la grid search en auto-train",
    )
    # Réserve les moyennes partielles au diagnostic local, jamais à la note finale.
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Autorise un rapport incomplet uniquement pour le diagnostic",
    )
    parser.add_argument(
        "--pooled-experiment-cv",
        action="store_true",
        help="Valide FBCSP sur les trois runs regroupés de chaque expérience",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Nombre de plis de validation croisée du protocole groupé",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        default=None,
        help="Limite de sujets, uniquement pour un diagnostic partiel",
    )
    parser.add_argument(
        "--campaign-size",
        type=int,
        choices=(10, 30, 109),
        default=None,
        help="Cohorte reproductible 10, 30 ou 109 sujets",
    )
    parser.add_argument(
        "--disable-nested-cv",
        action="store_true",
        help="Désactive la CV imbriquée (diagnostic uniquement)",
    )
    parser.add_argument(
        "--skip-strict-run-report",
        action="store_true",
        help="Ignore le rapport leave-one-run-out (diagnostic uniquement)",
    )
    parser.add_argument(
        "--preprocessing-ablation",
        action="store_true",
        help="Compare baseline ERD/ERS, CAR et Laplacien dans la CV interne",
    )
    parser.add_argument(
        "--model-family",
        choices=("fbcsp", "riemannian"),
        default="fbcsp",
        help="Famille évaluée ; riemannian reste une ablation secondaire",
    )
    # Expose les bonus de features sans créer une seconde boucle globale.
    parser.add_argument(
        "--feature-strategy",
        choices=tpv_pipeline.FEATURE_STRATEGIES,
        default="fft",
        help="Extraction utilisée lors de l'auto-entraînement",
    )
    # Expose la réduction maison depuis le moteur d'agrégation unique.
    parser.add_argument(
        "--dim-method",
        choices=tpv_pipeline.DIMENSIONALITY_METHODS,
        default="pca",
        help="Réduction utilisée lors de l'auto-entraînement",
    )
    # Expose les classifieurs déjà disponibles dans la pipeline.
    parser.add_argument(
        "--classifier",
        choices=tpv_pipeline.CLASSIFIER_CHOICES,
        default="lda",
        help="Classifieur utilisé lors de l'auto-entraînement",
    )
    # Expose le scaling optionnel sans dupliquer la construction du pipeline.
    parser.add_argument(
        "--scaler",
        choices=tpv_pipeline.SCALER_CHOICES,
        default="none",
        help="Scaler utilisé lors de l'auto-entraînement",
    )
    # Retourne le parser prêt à interpréter les arguments utilisateur
    return parser


# Fournit le parcours trié commun aux deux sources de découverte de runs
def _iter_subject_directories(root_directory: Path) -> tuple[Path, ...]:
    """Retourne les dossiers sujets triés ou un tuple vide si la racine manque."""

    # Une source absente signifie simplement qu'aucun run n'y est découvrable
    if not root_directory.exists():
        # Le tuple vide permet aux deux appelants de conserver leur boucle nominale
        return ()
    # Seuls les dossiers peuvent porter les fichiers et sous-dossiers d'un sujet
    subject_directories = (
        # Le filtre protège la découverte contre les fichiers placés à la racine
        path
        for path in root_directory.iterdir()
        if path.is_dir()
    )
    # Le tri garantit des rapports reproductibles entre systèmes de fichiers
    return tuple(sorted(subject_directories))


# Inventorie les runs disponibles en inspectant les artefacts présents
def _discover_runs_from_artifacts(artifacts_dir: Path) -> list[tuple[str, str]]:
    """Liste les couples (sujet, run) disposant d'un modèle sauvegardé."""

    # Initialise la liste de sortie pour conserver l'ordre déterministe
    runs: list[tuple[str, str]] = []
    # Parcourt uniquement les dossiers sujets fournis par le helper commun
    for subject_dir in _iter_subject_directories(artifacts_dir):
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
    edf_path, _event_path = tpv_utils.resolve_recording_paths(data_dir, subject, run)
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
    # Aplatis la liste des runs attendus pour les expériences T1-T4
    expected_runs = [run for runs in EXPERIENCE_RUNS.values() for run in runs]
    # Parcourt uniquement les dossiers sujets fournis par le helper commun
    for subject_dir in _iter_subject_directories(data_dir):
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

    # Tente d'abord de réutiliser les artefacts persistés.
    artifact_runs = _discover_runs_from_artifacts(artifacts_dir)
    # Refuse le scan des données si l'auto-train est désactivé
    if not allow_data_scan:
        # N'affiche le diagnostic d'absence que si aucun modèle n'est disponible.
        if not artifact_runs:
            # Explique comment autoriser la découverte et l'entraînement manquant.
            print(
                "INFO: aucun artefact trouvé, "
                "scan data désactivé (utilisez --auto-train-missing pour activer)."
            )
        # Retourne les artefacts présents sans déclencher d'entraînement coûteux.
        return artifact_runs
    # Complète les artefacts partiels avec les runs réellement disponibles.
    data_runs = _discover_runs_from_data(data_dir)
    # Un set supprime les doublons, puis le tri garantit un ordre reproductible.
    return sorted(set(artifact_runs).union(data_runs))


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
    score = Decimal(str(mean_score))
    threshold = Decimal(str(BONUS_THRESHOLD))
    step = Decimal(str(BONUS_STEP))
    # Ignore le bonus tant qu'un palier complet de trois points n'est pas atteint.
    if score < threshold + step:
        # Retourne zéro pour refléter l'absence de bonus
        return 0
    # Calcule l'écart au seuil pour dimensionner les points
    delta = score - threshold
    # Decimal évite que 0.81 ou 0.87 tombe du mauvais côté du palier binaire.
    earned_points = int((delta / step).to_integral_value(rounding=ROUND_FLOOR))
    # La checklist borne explicitement cette partie de la note à cinq points.
    return min(earned_points, 5)


# Construit les chemins d'artefacts attendus pour un run
def _artifact_paths(artifacts_dir: Path, subject: str, run: str) -> tuple[Path, Path]:
    """Retourne les chemins du modèle et de la matrice W attendus."""

    # Préserve l'ancien tuple public depuis le contrat de chemins partagé.
    paths = tpv_utils.resolve_artifact_paths(artifacts_dir, subject, run)
    return paths.model, paths.w_matrix


# Lance l'entraînement d'un run avec configuration contrôlée
def _train_run(subject: str, run: str, context: TrainingContext) -> None:
    """Entraîne un run en construisant une requête alignée sur scripts.train."""

    # Résout la fréquence d'échantillonnage à partir des EDF si disponibles
    resolved_sfreq = train_cli.resolve_sampling_rate(
        subject,
        run,
        context.options.raw_dir,
        train_cli.DEFAULT_SAMPLING_RATE,
        train_cli.DEFAULT_EEG_REFERENCE,
    )
    # Construit une configuration de pipeline cohérente pour l'auto-train
    pipeline_config = train_cli.PipelineConfig(
        sfreq=resolved_sfreq,
        feature_strategy=context.options.feature_strategy,
        normalize_features=True,
        dim_method=context.options.dim_method,
        n_components=None,
        classifier=context.options.classifier,
        scaler=context.options.scaler,
    )
    # Prépare la requête d'entraînement pour scripts.train.run_training
    request = train_cli.TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=pipeline_config,
        data_dir=context.data_dir,
        artifacts_dir=context.artifacts_dir,
        raw_dir=context.options.raw_dir,
        eeg_reference=train_cli.DEFAULT_EEG_REFERENCE,
        enable_grid_search=context.options.enable_grid_search,
        grid_search_splits=context.options.grid_search_splits,
    )
    # Déclenche l'entraînement et la persistance des artefacts
    train_cli.run_training(request)


# Charge une accuracy persistée uniquement si sa validation est prouvée
def _load_cached_accuracy(artifacts_dir: Path, subject: str, run: str) -> float | None:
    """Retourne exclusivement le score CV du manifeste d'entraînement."""

    # Détermine le chemin du manifeste contenant les folds de validation.
    manifest_path = tpv_utils.resolve_artifact_paths(
        artifacts_dir, subject, run
    ).manifest
    # Un ancien report.json seul ne constitue jamais une preuve scientifique.
    if not manifest_path.exists():
        # Signale l'absence de manifeste pour déclencher le contrôle complet.
        return None
    # Charge le manifeste produit lors du fit et de la validation croisée.
    manifest = tpv_utils.load_json_object(manifest_path)
    # Récupère uniquement la moyenne des folds via le contrat partagé.
    return cast(float | None, tpv_utils.extract_cv_mean(manifest))


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


def _needs_training(
    cached_accuracy: float | None,
    model_path: Path,
    w_matrix_path: Path,
    options: AggregationOptions,
) -> bool:
    """Indique si les artefacts doivent être régénérés avant évaluation."""

    return (
        options.force_retrain
        or (options.allow_auto_train and cached_accuracy is None)
        or not model_path.exists()
        or not w_matrix_path.exists()
    )


# Agrège les scores par sujet en parcourant les runs disponibles
def _collect_subject_scores(
    runs: list[tuple[str, str]],
    data_dir: Path,
    artifacts_dir: Path,
    options: AggregationOptions,
) -> dict[str, dict[str, list[float]]]:
    """Regroupe les accuracies par sujet et type d'expérience."""

    # Initialise le conteneur des scores par sujet et par type
    subject_scores: dict[str, dict[str, list[float]]] = {}
    # Prépare le contexte d'entraînement partagé pour les runs
    training_context = TrainingContext(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        options=options,
    )
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
        # Ajoute la valeur persistée si elle existe et si aucun retrain n'est forcé
        if cached_accuracy is not None and not options.force_retrain:
            # Enregistre l'accuracy dans l'agrégation par sujet
            _append_subject_score(subject_scores, subject, experience, cached_accuracy)
            # Passe au run suivant pour éviter un recalcul
            continue
        # Résout les chemins d'artefacts nécessaires à l'évaluation
        model_path, w_matrix_path = _artifact_paths(artifacts_dir, subject, run)
        # Détermine si un entraînement est requis pour ce run
        needs_training = _needs_training(
            cached_accuracy,
            model_path,
            w_matrix_path,
            options,
        )
        # Saute l'auto-train si les artefacts sont incomplets et interdits
        if not options.allow_auto_train and needs_training:
            # Informe l'utilisateur que le run est ignoré par manque d'artefact
            # Passe au run suivant sans entraîner
            continue
        # Lance l'auto-train si requis par l'absence ou le retrain forcé
        if needs_training:
            # Lance l'entraînement piloté pour assurer un modèle à jour
            _train_run(
                subject,
                run,
                training_context,
            )
        # Calcule l'accuracy en rechargeant le modèle et les données
        # Construit les options de prédiction avec le raw_dir demandé
        prediction_options = predict_cli.PredictionOptions(raw_dir=options.raw_dir)
        # Calcule l'accuracy en rechargeant le modèle et les données
        result = predict_cli.evaluate_run(
            # Transmet l'identifiant du sujet évalué
            subject,
            # Transmet l'identifiant du run évalué
            run,
            # Transmet le répertoire des données numpy
            data_dir,
            # Transmet le répertoire des artefacts
            artifacts_dir,
            # Transmet les options de prédiction
            prediction_options,
        )
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
        # Détermine si la moyenne atteint le seuil de 0.75
        meets_threshold = mean_of_means is not None and mean_of_means >= BONUS_THRESHOLD
        # Ajoute l'entrée structurée pour ce sujet
        subject_entries.append(
            {
                "subject": subject,
                "means": means_by_experience,
                "eligible": eligible,
                "meets_threshold": meets_threshold,
                "mean_of_means": mean_of_means,
            }
        )
    # Retourne les entrées prêtes pour l'affichage
    return subject_entries


# Calcule les moyennes globales par type d'expérience
def _build_global_experience_means(
    subject_scores: dict[str, dict[str, list[float]]],
) -> dict[str, float | None]:
    """Retourne la moyenne par expérience sur tous les sujets."""

    # Initialise un conteneur vide pour agréger les scores par expérience
    aggregated_scores: dict[str, list[float]] = {}
    # Parcourt les expériences attendues pour initialiser les listes
    for experience in EXPERIENCE_ORDER:
        # Prépare une liste vide pour chaque expérience attendue
        aggregated_scores[experience] = []
    # Parcourt les scores par sujet pour accumuler les valeurs disponibles
    for experience_scores in subject_scores.values():
        # Parcourt chaque expérience attendue pour préserver l'ordre
        for experience in EXPERIENCE_ORDER:
            # Récupère les scores du sujet pour l'expérience courante
            scores = experience_scores.get(experience, [])
            # Ignore les expériences sans score pour éviter des moyennes vides
            if not scores:
                # Passe à l'expérience suivante pour préserver les données fiables
                continue
            # Ajoute les scores pour construire la moyenne globale
            aggregated_scores[experience].extend(scores)
    # Initialise le dictionnaire des moyennes par expérience
    experience_means: dict[str, float | None] = {}
    # Parcourt les expériences agrégées pour calculer les moyennes
    for experience, scores in aggregated_scores.items():
        # Calcule la moyenne sur tous les sujets si la liste n'est pas vide
        mean_value = float(np.mean(scores)) if scores else None
        # Stocke la moyenne calculée pour l'expérience
        experience_means[experience] = mean_value
    # Retourne les moyennes globales par expérience
    return experience_means


# Calcule la moyenne globale à partir des moyennes par expérience
def _compute_global_mean(
    global_experience_means: dict[str, float | None],
) -> float | None:
    """Calcule la moyenne des quatre moyennes T1..T4."""

    # Construit la liste des moyennes par expérience dans l'ordre attendu
    experience_values = [global_experience_means.get(exp) for exp in EXPERIENCE_ORDER]
    # Retourne None si une moyenne est manquante
    if any(value is None for value in experience_values):
        # Signale l'impossibilité de calculer une moyenne globale
        return None
    # Convertit les valeurs en float pour la moyenne
    numeric_values = [float(value) for value in experience_values if value is not None]
    # Retourne la moyenne globale des quatre expériences
    return float(np.mean(numeric_values))


# Sélectionne les sujets les moins performants selon la moyenne
def _build_worst_subjects(
    subject_entries: list[dict],
    limit: int = 10,
) -> list[dict]:
    """Retourne les sujets les plus faibles selon la moyenne des expériences."""

    # Filtre les sujets disposant d'une moyenne calculable
    scored_entries = [
        entry for entry in subject_entries if entry["mean_of_means"] is not None
    ]
    # Trie les sujets par moyenne croissante pour isoler les pires
    scored_entries.sort(key=lambda entry: float(entry["mean_of_means"]))
    # Retourne uniquement les sujets dans la limite demandée
    return scored_entries[:limit]


# Construit un rapport agrégé à partir des scores par sujet
def build_report_from_scores(subject_scores: dict[str, dict[str, list[float]]]) -> dict:
    """Construit le rapport d'agrégation à partir des scores par sujet."""

    # Construit les entrées prêtes pour affichage ou export
    subject_entries = _build_subject_entries(subject_scores)
    # Calcule les moyennes globales par expérience sur tous les sujets
    global_experience_means = _build_global_experience_means(subject_scores)
    # Calcule la moyenne globale conformément à la checklist TPV
    global_mean = _compute_global_mean(global_experience_means)
    # Calcule le bonus associé à la moyenne globale
    bonus_points = compute_bonus_points(global_mean)
    # Compte les sujets disposant des quatre expériences
    eligible_subjects = sum(1 for entry in subject_entries if entry["eligible"])
    # Sélectionne les pires sujets pour l'affichage CLI
    worst_subjects = _build_worst_subjects(subject_entries)
    # Regroupe les données dans une structure de rapport dédiée
    report = {
        # Rend explicite la provenance scientifique de toutes les accuracies.
        "evaluation_method": "cross_validation",
        # Fige la taille officielle du dataset attendue pour un rapport complet.
        "expected_subjects": EXPECTED_SUBJECT_COUNT,
        # Indique immédiatement si le rapport peut servir de preuve finale.
        "complete": eligible_subjects == EXPECTED_SUBJECT_COUNT,
        # Fournit les entrées par sujet pour l'affichage
        "subjects": subject_entries,
        # Fournit les moyennes globales par expérience
        "global_experience_means": global_experience_means,
        # Fournit la moyenne globale pour compatibilité historique
        "global_experience_mean": global_mean,
        # Fournit la moyenne globale basée sur T1..T4
        "global_mean": global_mean,
        # Fournit le nombre de sujets éligibles (4/4 expériences)
        "eligible_subjects": eligible_subjects,
        # Fournit le bonus calculé sur la moyenne globale
        "bonus_points": bonus_points,
        # Fournit la liste des pires sujets pour l'affichage CLI
        "worst_subjects": worst_subjects,
    }
    # Retourne la structure complète prête pour l'affichage ou export
    return report


def _resolve_pooled_subjects(options: AggregationOptions) -> list[str]:
    """Valide les bornes puis construit la cohorte demandée."""

    if options.cv_splits < MIN_CV_SPLITS:
        raise ValueError("cv_splits must be at least 2")
    if options.subject_limit is not None and options.subject_limit < 1:
        raise ValueError("subject_limit must be positive")
    if options.subject_limit is not None and options.campaign_size is not None:
        raise ValueError("subject_limit and campaign_size are mutually exclusive")
    subjects = cast(
        list[str],
        tpv_evaluation.campaign_subjects(
            options.campaign_size or EXPECTED_SUBJECT_COUNT
        ),
    )
    if options.subject_limit is not None:
        subjects = subjects[: options.subject_limit]
    return subjects


def _build_evaluation_config(
    options: AggregationOptions,
) -> Any:
    """Convertit les options CLI en configuration scientifique figée."""

    return tpv_evaluation.ExperimentEvaluationConfig(
        cv_splits=options.cv_splits,
        epoch_tmin=-1.0 if options.enable_preprocessing_ablation else 0.0,
        nested_cv=options.nested_cv,
        strict_run_report=options.strict_run_report,
        enable_preprocessing_ablation=options.enable_preprocessing_ablation,
        model_family=options.model_family,
    )


def _build_pooled_configuration(
    raw_dir: Path,
    subjects: list[str],
    evaluation_config: Any,
) -> dict[str, object]:
    """Sérialise la configuration complète du rapport v3."""

    config = evaluation_config
    return {
        "raw_dir": str(raw_dir),
        "pooled_experiment_cv": True,
        "cv_splits": config.cv_splits,
        "random_state": config.random_state,
        "epoch_window": [config.epoch_tmin, config.epoch_tmax],
        "eeg_reference": config.eeg_reference,
        "algorithm_version": config.algorithm_version,
        "nested_cv": config.nested_cv,
        "inner_cv_splits": config.inner_cv_splits,
        "strict_run_report": config.strict_run_report,
        "preprocessing_ablation": config.enable_preprocessing_ablation,
        "model_family": config.model_family,
        "campaign_size": len(subjects),
        "run_adaptive_filters": config.run_adaptive_filters,
        "filter_mode": config.filter_mode,
        "baseline_window": config.baseline_window,
        "task_pipeline_configs": {
            experience: {
                "motor_roi": list(task.motor_roi),
                "filter_bank_bands": [list(band) for band in task.bands],
                "filter_bank_windows": [list(window) for window in task.windows],
                "selected_features": task.selected_features,
                "spatial_reference": task.spatial_reference,
            }
            for experience, task in tpv_evaluation.TASK_PIPELINE_CONFIGS.items()
        },
        "classifier": "lda_lsqr_auto_shrinkage",
    }


def _strict_global_mean(evidence: list[dict[str, object]]) -> float | None:
    """Agrège séparément les scores des runs entièrement tenus à l'écart."""

    strict_means = [
        float(cast(float, result["strict_run_mean"]))
        for subject in evidence
        for result in cast(
            dict[str, dict[str, object]], subject["experiences"]
        ).values()
        if result.get("strict_run_mean") is not None
    ]
    return float(np.mean(strict_means)) if strict_means else None


def _highest_score_target(global_mean: object) -> float | None:
    """Retourne le jalon atteint sans modifier les quatre seuils figés."""

    if not isinstance(global_mean, (float, int)):
        return None
    return max(
        (target for target in SCORE_TARGETS if global_mean >= target),
        default=None,
    )


def _aggregate_pooled_experiment_scores(
    raw_dir: Path,
    artifacts_dir: Path,
    options: AggregationOptions,
) -> dict:
    """Évalue FBCSP sur 45 essais par sujet et type d'expérience."""

    subjects = _resolve_pooled_subjects(options)
    evaluation_config = _build_evaluation_config(options)

    def report_progress(subject: str, results: dict) -> None:
        means = [
            float(results[experience]["cv_mean"]) for experience in EXPERIENCE_ORDER
        ]
        print(f"INFO: {subject} FBCSP mean={float(np.mean(means)):.3f}", flush=True)

    subject_scores, evidence = tpv_evaluation.evaluate_all_subjects(
        raw_dir,
        subjects,
        evaluation_config,
        progress=report_progress,
        cache_dir=artifacts_dir / "evaluation" / "fbcsp_subjects",
    )
    report = build_report_from_scores(subject_scores)
    report["schema_version"] = 3
    validation_name = "nested_stratified_cv" if options.nested_cv else "stratified_cv"
    if options.strict_run_report:
        validation_name += "_with_leave_one_run_out"
    report["evaluation_method"] = f"{validation_name}_{options.model_family}"
    report["expected_subject_ids"] = [
        f"S{subject_index:03d}"
        for subject_index in range(1, EXPECTED_SUBJECT_COUNT + 1)
    ]
    report["requested_configuration"] = _build_pooled_configuration(
        raw_dir, subjects, evaluation_config
    )
    report["evaluated_subjects"] = evidence
    report["strict_run_global_mean"] = _strict_global_mean(evidence)
    global_mean = report.get("global_mean")
    report["score_targets"] = list(SCORE_TARGETS)
    report["highest_target_reached"] = _highest_score_target(global_mean)
    report["source_commits"] = [train_cli._get_git_commit()]
    return report


# Calcule les moyennes par type d'expérience pour chaque sujet
def aggregate_experience_scores(
    data_dir: Path,
    artifacts_dir: Path,
    options: AggregationOptions,
) -> dict:
    """Produit un rapport agrégé par sujet et type d'expérience (WBS 7.4)."""

    if options.pooled_experiment_cv:
        return _aggregate_pooled_experiment_scores(
            options.raw_dir,
            artifacts_dir,
            options,
        )

    # Récupère la liste des runs éligibles à l'agrégation
    runs = _discover_runs(data_dir, artifacts_dir, options.allow_auto_train)
    # Regroupe les accuracies par sujet et type d'expérience
    subject_scores = _collect_subject_scores(runs, data_dir, artifacts_dir, options)
    # Construit le rapport agrégé à partir des scores collectés
    report = build_report_from_scores(subject_scores)
    # Versionne la structure pour permettre de relire durablement la preuve.
    report["schema_version"] = 1
    # Énumère explicitement les 109 sujets requis par l'évaluation officielle.
    report["expected_subject_ids"] = [
        f"S{subject_index:03d}"
        for subject_index in range(1, EXPECTED_SUBJECT_COUNT + 1)
    ]
    # Conserve la configuration demandée sans objets Path non sérialisables.
    report["requested_configuration"] = {
        "allow_auto_train": options.allow_auto_train,
        "force_retrain": options.force_retrain,
        "enable_grid_search": options.enable_grid_search,
        "grid_search_splits": options.grid_search_splits,
        "raw_dir": str(options.raw_dir),
        "feature_strategy": options.feature_strategy,
        "dim_method": options.dim_method,
        "classifier": options.classifier,
        "scaler": options.scaler,
    }
    # Charge pour chaque run la preuve CV et le commit de son entraînement.
    evaluated_runs = _build_run_evidence(runs, artifacts_dir)
    # Rend les folds et scores consultables dans un rapport JSON unique.
    report["evaluated_runs"] = evaluated_runs
    # Signale immédiatement si plusieurs commits ont produit les artefacts.
    report["source_commits"] = sorted(
        {
            str(entry["git_commit"])
            for entry in evaluated_runs
            if entry["git_commit"] is not None
        }
    )
    # Retourne la preuve complète prête pour affichage ou sérialisation.
    return report


# Rassemble les manifests de run dans une preuve globale auditable.
def _build_run_evidence(
    runs: list[tuple[str, str]],
    artifacts_dir: Path,
) -> list[dict]:
    """Retourne commits, configuration, folds et scores de chaque run moteur."""

    # Prépare une liste stable pour faciliter les comparaisons entre exécutions.
    evidence: list[dict] = []
    # Parcourt les runs dans un ordre indépendant du système de fichiers.
    for subject, run in sorted(runs):
        # Ignore les baselines qui n'entrent pas dans les quatre expériences.
        experience = _map_run_to_experience(run)
        # Passe au run suivant lorsqu'il est hors protocole moteur.
        if experience is None:
            # Évite d'introduire une ligne sans signification dans la preuve.
            continue
        # Localise le manifeste écrit par le seul flux d'entraînement.
        manifest_path = tpv_utils.resolve_artifact_paths(
            artifacts_dir, subject, run
        ).manifest
        # Un run ignoré faute d'artefact ne doit pas devenir une fausse preuve.
        if not manifest_path.exists():
            # Conserve uniquement les évaluations effectivement matérialisées.
            continue
        # Charge la configuration et les scores persistés par train.py.
        manifest = tpv_utils.load_json_object(manifest_path)
        # Isole les sections optionnelles sans supposer un manifeste parfait.
        dataset = manifest.get("dataset", {})
        # Isole les hyperparamètres utilisés par le fit correspondant.
        hyperparams = manifest.get("hyperparams", {})
        # Isole les folds qui justifient l'accuracy officielle.
        scores = manifest.get("scores", {})
        # Ajoute une entrée compacte mais suffisante pour reproduire le run.
        evidence.append(
            {
                "subject": subject,
                "run": run,
                "experience": experience,
                "git_commit": manifest.get("git_commit"),
                "epoch_window": dataset.get("epoch_window"),
                "hyperparams": hyperparams,
                "cv_scores": scores.get("cv_scores"),
                "cv_mean": scores.get("cv_mean"),
            }
        )
    # Retourne toutes les preuves disponibles pour le rapport global.
    return evidence


# Formate une moyenne optionnelle en chaîne lisible
def _format_mean_value(value: float | None) -> str:
    """Retourne une moyenne formatée ou 'n/a' si absente."""

    # Retourne n/a si la valeur est absente
    if value is None:
        return "n/a"
    # Formate la moyenne sur trois décimales
    return "{:.3f}".format(value)


# Formate les moyennes par expérience pour une ligne donnée
def _format_experience_values(means: dict[str, float | None]) -> list[str]:
    """Retourne les valeurs formatées par expérience."""

    # Prépare la liste des valeurs formatées
    values: list[str] = []
    # Parcourt les expériences dans l'ordre attendu
    for experience in EXPERIENCE_ORDER:
        # Ajoute la moyenne formatée pour l'expérience courante
        values.append(_format_mean_value(means.get(experience)))
    # Retourne les valeurs formatées
    return values


# Construit la ligne d'un sujet pour le tableau d'agrégation
def _format_subject_row(entry: dict) -> str:
    """Construit une ligne tabulée pour un sujet."""

    # Récupère les moyennes formatées par expérience
    values = _format_experience_values(entry["means"])
    # Formate la moyenne globale du sujet
    mean_of_means = _format_mean_value(entry["mean_of_means"])
    # Détermine le libellé d'éligibilité
    eligible_label = "yes" if entry["eligible"] else "no"
    # Détermine le libellé de seuil
    threshold_label = "yes" if entry["meets_threshold"] else "no"
    # Assemble la ligne tabulée pour le sujet
    return "\t".join(
        [
            entry["subject"],
            *values,
            mean_of_means,
            eligible_label,
            threshold_label,
        ]
    )


# Calcule le libellé de seuil global pour la ligne de synthèse
def _format_global_threshold_label(global_mean_value: float | None) -> str:
    """Retourne yes/no/n/a pour le seuil global."""

    # Retourne n/a si la moyenne globale est absente
    if global_mean_value is None:
        return "n/a"
    # Retourne yes si la moyenne atteint le seuil
    if global_mean_value >= BONUS_THRESHOLD:
        return "yes"
    # Retourne no si la moyenne est disponible mais insuffisante
    return "no"


# Construit la ligne globale du tableau d'agrégation
def _format_global_row(report: dict) -> str:
    """Construit la ligne globale pour le tableau d'agrégation."""

    # Récupère les moyennes globales par expérience
    global_experience_means = report.get("global_experience_means", {})
    # Récupère les valeurs formatées par expérience
    global_experience_values = _format_experience_values(global_experience_means)
    # Récupère la moyenne globale depuis le rapport
    global_mean_value = report.get("global_mean")
    # Normalise la moyenne globale si elle n'est pas numérique
    if not isinstance(global_mean_value, (float, int)):
        global_mean_value = None
    # Prépare l'affichage de la moyenne globale
    global_mean_label = _format_mean_value(
        float(global_mean_value) if global_mean_value is not None else None
    )

    # Prépare le libellé du nombre de sujets éligibles et du bonus
    eligible_label = (
        f"{report['eligible_subjects']} subjects, bonus {report['bonus_points']}"
    )

    # Détermine le libellé de seuil global
    global_threshold_label = _format_global_threshold_label(
        float(global_mean_value) if global_mean_value is not None else None
    )

    # Assemble la ligne globale tabulée
    return "\t".join(
        [
            "Global",
            *global_experience_values,
            global_mean_label,
            eligible_label,
            global_threshold_label,
        ]
    )


# Formate un tableau texte des moyennes par expérience
def format_experience_table(report: dict) -> str:
    """Construit un tableau lisible des moyennes par type d'expérience."""

    # Définit l'en-tête du tableau pour l'affichage CLI
    header = [
        # Ajoute la colonne du sujet pour identifier la ligne
        "Subject",
        # Insère les expériences dans l'ordre standardisé
        *EXPERIENCE_ORDER,
        # Ajoute la colonne de moyenne par sujet
        "Mean",
        # Ajoute la colonne d'éligibilité (4/4 expériences)
        "Eligible(4/4)",
        # Ajoute la colonne de seuil global pour chaque sujet
        "MeetsThreshold_0p75",
    ]
    # Initialise les lignes avec l'en-tête tabulé
    lines = ["\t".join(header)]
    # Parcourt les entrées pour chaque sujet
    for entry in report["subjects"]:
        # Ajoute la ligne formatée pour le sujet courant
        lines.append(_format_subject_row(entry))
    # Ajoute la ligne globale pour la transparence du rapport
    lines.append(_format_global_row(report))
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
            "meets_threshold_0p75",
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
                "meets_threshold_0p75": entry["meets_threshold"],
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


# Sérialise la preuve scientifique complète sans perte de précision.
def write_json(report: dict, json_path: Path) -> None:
    """Écrit le rapport reproductible au format JSON."""

    # Crée le dossier cible pour autoriser un chemin dans artifacts/.
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Sérialise avec un ordre stable et une indentation relisible en soutenance.
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# Centralise les sorties optionnelles pour garder le point d'entrée lisible.
def _write_requested_reports(
    report: dict,
    csv_path: Path | None,
    json_path: Path | None,
) -> None:
    """Écrit les formats explicitement demandés par l'utilisateur."""

    # Produit le tableau compact seulement lorsqu'un chemin CSV est fourni.
    if csv_path is not None:
        # Délègue le formatage CSV au helper déjà testé séparément.
        write_csv(report, csv_path)
    # Produit la preuve complète seulement lorsqu'un chemin JSON est fourni.
    if json_path is not None:
        # Délègue la sérialisation reproductible au helper dédié.
        write_json(report, json_path)


# Point d'entrée principal pour l'usage en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments puis affiche le tableau par expérience."""

    # Construit le parser pour interpréter les options fournies
    parser = build_parser()
    # Parse les arguments utilisateurs pour récupérer les chemins
    args = parser.parse_args(argv)
    # Calcule le rapport d'accuracy par expérience
    # Rassemble les options pour éviter de multiplier les arguments
    options = AggregationOptions(
        allow_auto_train=args.auto_train_missing,
        force_retrain=args.force_retrain,
        enable_grid_search=args.auto_train_grid_search,
        grid_search_splits=args.grid_search_splits,
        raw_dir=args.raw_dir,
        feature_strategy=args.feature_strategy,
        dim_method=args.dim_method,
        classifier=args.classifier,
        scaler=None if args.scaler == "none" else args.scaler,
        pooled_experiment_cv=args.pooled_experiment_cv,
        cv_splits=args.cv_splits,
        subject_limit=args.subject_limit,
        nested_cv=not args.disable_nested_cv,
        strict_run_report=not args.skip_strict_run_report,
        enable_preprocessing_ablation=args.preprocessing_ablation,
        model_family=args.model_family,
        campaign_size=args.campaign_size,
    )
    report = aggregate_experience_scores(
        args.data_dir,
        args.artifacts_dir,
        options,
    )
    # Formate le tableau lisible pour l'utilisateur CLI
    table = format_experience_table(report)
    # Imprime le tableau pour inspection ou redirection
    print(table)
    # Récupère la liste des pires sujets pour l'affichage

    worst_subjects = report.get("worst_subjects", [])
    # Affiche les pires sujets si la liste est disponible
    if worst_subjects:
        # Ajoute un titre pour la lisibilité du rapport
        print("Worst subjects by Mean (mean_of_means):")
        # Parcourt les sujets triés par moyenne croissante
        for entry in worst_subjects:
            # Récupère la moyenne calculée pour le sujet
            mean_of_means = entry.get("mean_of_means")
            # Ignore les sujets sans moyenne calculable
            if mean_of_means is None:
                # Passe au sujet suivant si la moyenne est absente
                continue
            # Affiche le sujet et sa moyenne formatée
            print(f"- {entry['subject']}: {float(mean_of_means):.3f}")
    # Écrit les sorties optionnelles sans alourdir le routeur CLI.
    _write_requested_reports(report, args.csv_output, args.json_output)
    # Récupère la moyenne globale pour vérifier le seuil
    global_mean = report.get("global_mean")
    # Retourne un code d'erreur si la moyenne globale est sous le seuil
    if isinstance(global_mean, (float, int)) and global_mean < BONUS_THRESHOLD:
        # Signale explicitement l'échec de la contrainte de score
        print(
            "ERROR: GlobalMean inférieur au seuil 0.75 " f"({float(global_mean):.3f})."
        )
        # Retourne un code d'erreur pour signaler l'échec
        return 1
    # Compte uniquement les sujets possédant les quatre expériences requises.
    eligible_subjects = int(report.get("eligible_subjects", 0))
    # Refuse qu'une moyenne partielle soit présentée comme résultat global.
    if not args.allow_partial and eligible_subjects != EXPECTED_SUBJECT_COUNT:
        # Le diagnostic donne le nombre exact de sujets restant à produire.
        print(
            "ERROR: évaluation globale incomplète: "
            f"{eligible_subjects}/{EXPECTED_SUBJECT_COUNT} sujets éligibles."
        )
        # Un code non nul empêche la CI ou Make de valider silencieusement ce rapport.
        return 1
    # Retourne 0 pour signaler un succès standard
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
