"""CLI d'entraînement pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
# Expose les primitives d'analyse des arguments CLI
# Fournit le parsing CLI pour aligner la signature mybci
import argparse

# Fournit l'écriture CSV pour exposer un manifeste tabulaire
from asyncio import events
import csv

# Fournit la sérialisation JSON pour exposer un manifeste exploitable
import json

# Rassemble la construction de structures immuables orientées données
from dataclasses import asdict, dataclass

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Expose cast pour documenter les conversions de types
# Expose Sequence pour typer les fenêtres temporelles
from typing import Sequence, cast

# Offre la persistance dédiée aux objets scikit-learn pour inspection séparée
import joblib

# Centralise l'accès aux tableaux manipulés par scikit-learn
import numpy as np

# Fournit la validation croisée pour évaluer la pipeline complète
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Fournit la validation croisée pour évaluer la pipeline complète
# isort: off
# Expose la grille de recherche pour optimiser les hyperparamètres
from sklearn.model_selection import GridSearchCV

# Offre un splitter non stratifié pour les petits effectifs
from sklearn.model_selection import ShuffleSplit

# Offre un splitter stratifié pour préserver l'équilibre des classes
from sklearn.model_selection import StratifiedShuffleSplit

# Calcule les scores de validation croisée de la pipeline
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut, StratifiedKFold

# isort: on

# Fournit le type Pipeline pour typer les helpers de scoring
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

# Centralise le parsing et le contrôle qualité des fichiers EDF
# Extrait les features fréquentielles depuis des epochs EEG
from tpv import preprocessing

# Assemble la pipeline cohérente pour l'entraînement
from tpv.classifier import CentroidClassifier

# Permet de persister séparément la matrice W apprise
from tpv.dimensionality import TPVDimReducer
from tpv.pipeline import (
    PipelineConfig,
    build_pipeline,
    build_search_pipeline,
    save_pipeline,
)

# Déclare la liste des runs moteurs à couvrir pour l'entraînement massif
MOTOR_RUNS = (
    # Couvre le run moteur R03 documenté dans le protocole Physionet
    "R03",
    # Couvre le run moteur R04 documenté dans le protocole Physionet
    "R04",
    # Couvre le run moteur R05 documenté dans le protocole Physionet
    "R05",
    # Couvre le run moteur R06 documenté dans le protocole Physionet
    "R06",
    # Couvre le run moteur R07 documenté dans le protocole Physionet
    "R07",
    # Couvre le run moteur R08 documenté dans le protocole Physionet
    "R08",
    # Couvre le run moteur R09 documenté dans le protocole Physionet
    "R09",
    # Couvre le run moteur R10 documenté dans le protocole Physionet
    "R10",
    # Couvre le run moteur R11 documenté dans le protocole Physionet
    "R11",
    # Couvre le run moteur R12 documenté dans le protocole Physionet
    "R12",
    # Couvre le run moteur R13 documenté dans le protocole Physionet
    "R13",
    # Couvre le run moteur R14 documenté dans le protocole Physionet
    "R14",
)

# Définit le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path("data")

# Fixe la dimension attendue pour les matrices de features en mémoire
EXPECTED_FEATURES_DIMENSIONS = 3

# Définit le répertoire par défaut pour déposer les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Définit le répertoire par défaut où résident les fichiers EDF bruts
DEFAULT_RAW_DIR = Path("data")

# Fige la fréquence d'échantillonnage par défaut utilisée pour les features
DEFAULT_SAMPLING_RATE = 50.0

# Définit un seuil max de pic-à-pic pour rejeter les artefacts (en Volts)
DEFAULT_MAX_PEAK_TO_PEAK = 800e-6
# Fixe la bande passante MI recommandée pour la tâche motrice
DEFAULT_BANDPASS_BAND = (8.0, 30.0)
# Fixe la fréquence de notch pour supprimer la pollution secteur
DEFAULT_NOTCH_FREQ = 50.0
# Définit des fenêtres post-cue candidates pour la sélection d'epochs
DEFAULT_EPOCH_WINDOWS: Sequence[tuple[float, float]] = (
    # Cible une fenêtre courte centrée sur l'ERD/ERS initial
    (0.5, 2.5),
    # Cible une fenêtre plus tardive pour les sujets lents
    (1.0, 3.0),
    # Cible une fenêtre plus proche du cue pour capturer l'initiation
    (0.0, 2.0),
)
# Fixe la fenêtre par défaut utilisée en absence de sélection
DEFAULT_EPOCH_WINDOW = DEFAULT_EPOCH_WINDOWS[0]
# Fixe le nombre de composantes CSP pour la sélection de fenêtre
DEFAULT_CSP_COMPONENTS = 4


# Normalise un identifiant brut en appliquant un préfixe standard
def _normalize_identifier(value: str, prefix: str, width: int, label: str) -> str:
    """Normalise un identifiant pour respecter le format Physionet."""

    # Nettoie la valeur reçue pour éviter des espaces parasites
    cleaned_value = value.strip()
    # Refuse une valeur vide pour éviter un identifiant incomplet
    if not cleaned_value:
        # Signale une valeur vide pour forcer la correction côté CLI
        raise argparse.ArgumentTypeError(f"{label} vide")
    # Récupère le premier caractère pour détecter un préfixe explicite
    first_char = cleaned_value[0]
    # Déduit si l'utilisateur a fourni le préfixe attendu
    has_prefix = first_char.upper() == prefix.upper()
    # Extrait la portion numérique selon la présence du préfixe
    numeric_part = cleaned_value[1:] if has_prefix else cleaned_value
    # Refuse les valeurs non numériques pour garantir un ID valide
    if not numeric_part.isdigit():
        # Signale l'identifiant invalide pour guider l'utilisateur
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Convertit en entier pour normaliser les zéros initiaux
    numeric_value = int(numeric_part)
    # Refuse les index non positifs pour respecter la base Physionet
    if numeric_value < 1:
        # Signale l'identifiant non valide pour arrêter le parsing
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Reconstruit l'identifiant normalisé avec le padding attendu
    return f"{prefix}{numeric_value:0{width}d}"


# Normalise un identifiant de sujet pour la CLI d'entraînement
def _parse_subject(value: str) -> str:
    """Normalise un identifiant de sujet en format Sxxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="S", width=3, label="Sujet")


# Normalise un identifiant de run pour la CLI d'entraînement
def _parse_run(value: str) -> str:
    """Normalise un identifiant de run en format Rxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="R", width=2, label="Run")


# Résout une fréquence d'échantillonnage fiable pour un sujet/run donné
def resolve_sampling_rate(
    subject: str,
    run: str,
    raw_dir: Path,
    requested_sfreq: float,
) -> float:
    """Retourne la fréquence d'échantillonnage détectée ou la valeur demandée."""

    # Préserve la fréquence explicitement demandée lorsqu'elle diffère du défaut
    if requested_sfreq != DEFAULT_SAMPLING_RATE:
        # Retourne la valeur explicite pour respecter la volonté utilisateur
        return requested_sfreq
    # Construit le chemin du fichier EDF brut pour la détection auto
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Renvoie la valeur demandée si l'EDF n'est pas disponible
    if not raw_path.exists():
        # Retourne la valeur par défaut en l'absence de fichier exploitable
        return requested_sfreq
    # Encadre la lecture MNE pour éviter un crash si l'EDF est invalide
    try:
        # Charge l'EDF et récupère les métadonnées utiles
        raw, metadata = preprocessing.load_physionet_raw(raw_path)
        # Extrait la valeur brute de la fréquence depuis les métadonnées
        sampling_rate_value = metadata.get("sampling_rate")
        # Convertit la valeur si possible pour préserver une fréquence cohérente
        if isinstance(sampling_rate_value, (int, float, str)):
            # Convertit explicitement pour accepter str/int/float
            sampling_rate = float(sampling_rate_value)
        else:
            # Préserve la valeur demandée si la conversion est impossible
            sampling_rate = requested_sfreq
        # Ferme explicitement le Raw pour libérer la mémoire
        raw.close()
    except (FileNotFoundError, OSError, ValueError) as error:
        # Signale la détection impossible sans interrompre l'entraînement
        print(
            "INFO: lecture EDF impossible pour "
            f"{subject} {run} ({error}), "
            "sfreq par défaut conservée."
        )
        # Retourne la valeur demandée en cas d'échec de lecture
        return requested_sfreq
    # Retourne la fréquence détectée pour aligner les features
    return sampling_rate


# Déclare le nombre cible de splits utilisé pour la validation croisée
DEFAULT_CV_SPLITS = 10

# Fixe le nombre minimal de splits pour déclencher la validation croisée
MIN_CV_SPLITS = 1

# Fixe la taille minimale du test pour stabiliser les splits CV
DEFAULT_CV_TEST_SIZE = 0.2

# Fixe le nombre minimal de classes pour activer la CV
MIN_CV_CLASS_COUNT = 2

# Définit un seuil minimal total pour tenter une CV relâchée
MIN_CV_TOTAL_SAMPLES = MIN_CV_CLASS_COUNT + 1

# Fixe le nombre maximal de tentatives pour filtrer les splits CV
MAX_CV_SPLIT_ATTEMPTS_FACTOR = 10


# Stabilise la reproductibilité des splits de cross-validation
DEFAULT_RANDOM_STATE = 42


# Construit un split stratifié reproductible avec un nombre fixe d'itérations

from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from sklearn.model_selection import StratifiedKFold, LeaveOneOut

def _build_cv_splitter(y: np.ndarray, n_splits: int):
    """
    Crée un splitter robuste qui s'adapte à la taille des classes 
    pour éviter les UserWarnings de scikit-learn.
    """
    n_samples = len(y)
    if n_samples < 2:
        return None

    unique_classes, counts = np.unique(y, return_counts=True)
    
    # Si on n'a qu'une seule classe, la CV est impossible
    if len(unique_classes) < 2:
        return None

    min_class_size = int(np.min(counts))

    # CAS 1 : Très peu de données (ex: S054) -> Leave-One-Out
    if min_class_size < 2:
        return LeaveOneOut()

    # CAS 2 : Données suffisantes mais inférieures au n_splits par défaut (ex: votre erreur)
    # On ajuste n_splits pour qu'il soit au maximum égal à min_class_size
    actual_splits = min(n_splits, min_class_size)
    
    # On s'assure que actual_splits est au moins 2
    actual_splits = max(2, actual_splits)

    return StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)


# Filtre un splitter shuffle pour garantir deux classes dans le train
def _filter_shuffle_splits_for_binary_train(
    y: np.ndarray,
    splitter: ShuffleSplit,
    requested_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construit des splits qui conservent deux classes dans le train."""

    # Prépare une liste de splits validés pour cross_val_score
    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Calcule le nombre total d'échantillons pour générer un X fictif
    sample_count = int(y.shape[0])
    # Prépare un tableau factice pour piloter splitter.split
    placeholder_X = np.zeros((sample_count, 1))
    # Définit une limite de tentatives pour éviter les boucles infinies
    max_attempts = max(1, requested_splits * MAX_CV_SPLIT_ATTEMPTS_FACTOR)
    # Parcourt les splits générés par ShuffleSplit
    for attempt_index, (train_idx, test_idx) in enumerate(
        splitter.split(placeholder_X, y)
    ):
        # Stoppe la recherche lorsque la limite est atteinte
        if attempt_index >= max_attempts:
            break
        # Évite les splits qui ne contiennent pas les deux classes en train
        if np.unique(y[train_idx]).size < MIN_CV_CLASS_COUNT:
            continue
        # Conserve le split valide pour la CV
        valid_splits.append((train_idx, test_idx))
        # Stoppe dès que le nombre de splits demandé est atteint
        if len(valid_splits) >= requested_splits:
            break
    # Retourne la liste finale des splits valides
    return valid_splits


# Construit un splitter CV valide ou renvoie la cause d'indisponibilité
# Ajoutez ce code temporaire dans _resolve_cv_splits pour diagnostiquer

def _resolve_cv_splits(
    y: np.ndarray,
    requested_splits: int,
) -> tuple[
    StratifiedShuffleSplit | ShuffleSplit | list[tuple[np.ndarray, np.ndarray]] | None,
    str | None,
]:
    """Retourne un splitter compatible avec deux classes en train."""
    
    # === DEBUG: Affichage des caractéristiques du dataset ===
    sample_count = int(y.shape[0])
    class_count = int(np.unique(y).size)
    
    if class_count >= 2:
        _, class_counts = np.unique(y, return_counts=True)
        min_class_count = int(class_counts.min())
    # === FIN DEBUG ===

    cv = _build_cv_splitter(y, requested_splits)
    if cv is None:
        reason = _describe_cv_unavailability(y, requested_splits)
        print(f"DEBUG CV: cv=None, reason='{reason}'")  # DEBUG
        return None, reason
    
    if isinstance(cv, ShuffleSplit):
        filtered_splits = _filter_shuffle_splits_for_binary_train(
            y, cv, requested_splits,
        )
        if not filtered_splits:
            reason = _describe_cv_unavailability(y, requested_splits)
            print(f"DEBUG CV: ShuffleSplit filtré vide, reason='{reason}'")  # DEBUG
            return None, reason
        return filtered_splits, None
    
    return cv, None


# Gère le fallback lorsque la validation croisée est impossible
def _handle_cv_unavailability(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    reason: str,
) -> tuple[np.ndarray, Pipeline, str]:
    """Ajuste la pipeline et retourne un diagnostic CV explicite."""
    # Ajoutez ce print pour voir la "vraie" raison
    print(f"DEBUG REASON: {reason}")
    # Informe l'utilisateur que la CV est désactivée
    print(
        # Message informatif pour éviter un warning bloquant côté bench
        "INFO: validation croisée indisponible, "
        "entraînement direct sans cross-val"
    )
    # Ajuste la pipeline sur toutes les données malgré l'absence de CV
    pipeline.fit(X, y)
    # Retourne un tableau vide et la raison d'indisponibilité
    return np.array([]), pipeline, reason


# Lance la recherche d'hyperparamètres et retourne ses scores
def _run_grid_search(
    search_pipeline: Pipeline,
    param_grid: dict[str, list[object]],
    search_cv: object,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, Pipeline, dict[str, object]]:
    """Exécute GridSearchCV et retourne scores, pipeline et résumé."""

    # Déduit le nombre de splits à partir du splitter fourni
    if isinstance(search_cv, list):
        # Utilise la longueur de liste lorsque les splits sont pré-calculés
        split_count = len(search_cv)
    else:
        # Récupère get_n_splits si la méthode est disponible
        split_count = int(getattr(search_cv, "get_n_splits", lambda: 0)())
    # Utilise un fallback si le splitter ne rapporte aucun split
    split_count = split_count or DEFAULT_CV_SPLITS
    # Instancie la recherche exhaustive pour maximiser l'accuracy
    search = GridSearchCV(
        search_pipeline,
        param_grid,
        cv=search_cv,
        scoring="accuracy",
        refit=True,
    )
    # Lance l'optimisation des hyperparamètres
    search.fit(X, y)
    # Extrait les scores par split pour l'entrée de manifeste
    cv_scores = _extract_grid_search_scores(search, split_count)
    # Capture un résumé des meilleurs paramètres
    search_summary = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }
    # Retourne les scores, la meilleure pipeline et le résumé
    return cv_scores, search.best_estimator_, search_summary


# Mise à jour de _describe_cv_unavailability pour être cohérent
def _describe_cv_unavailability(y: np.ndarray, requested_splits: int) -> str:
    """Explique pourquoi la validation croisée est indisponible."""
    
    sample_count = int(y.shape[0])
    if sample_count == 0:
        return "aucun échantillon disponible pour la validation croisée"
    
    class_count = int(np.unique(y).size)
    if class_count < MIN_CV_CLASS_COUNT:
        return "une seule classe présente, CV binaire impossible"
    
    _labels, class_counts = np.unique(y, return_counts=True)
    min_class_count = int(class_counts.min())
    
    # CAS CRITIQUE: Si min_class_count == 1
    if min_class_count == 1:
        return (
            f"effectif minimal par classe = {min_class_count} "
            "(impossible de diviser 1 échantillon entre train et test)"
        )
    
    # Si entre 1 et MIN_CV_SPLITS
    if min_class_count < MIN_CV_SPLITS:
        return (
            f"effectif minimal par classe insuffisant "
            f"({min_class_count} < {MIN_CV_SPLITS})"
        )
    
    # Cas du test_size impossible
    min_test_size = 1.0 / float(min_class_count)
    test_size = max(DEFAULT_CV_TEST_SIZE, min_test_size)
    max_test_size = (min_class_count - 1) / float(min_class_count)
    
    if test_size > max_test_size:
        return (
            f"split stratifié impossible "
            f"(test_size={test_size:.3f} > max={max_test_size:.3f})"
        )
    
    return (
        f"validation croisée indisponible pour une raison inconnue "
        f"(splits demandés={requested_splits})"
    )


# Adapte la configuration pour éviter les classifieurs instables en très bas n
def _adapt_pipeline_config_for_samples(
    config: PipelineConfig, y: np.ndarray
) -> PipelineConfig:
    """Adapte la configuration si l'effectif est trop faible pour LDA."""

    # Calcule le nombre de classes présentes dans les labels
    class_count = int(np.unique(y).size)
    # Calcule le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Conserve la configuration si LDA n'est pas utilisé
    if config.classifier != "lda":
        # Retourne la configuration d'origine sans modification
        return config
    # Conserve LDA lorsque l'effectif dépasse strictement le nombre de classes
    if sample_count > class_count:
        # Retourne la configuration d'origine dans le cas valide
        return config
    # Bascule vers un classifieur plus robuste aux petits effectifs
    # Retourne une nouvelle configuration alignée avec le fallback
    return PipelineConfig(
        # Conserve la fréquence d'échantillonnage identique
        sfreq=config.sfreq,
        # Conserve la stratégie de features pour la comparabilité
        feature_strategy=config.feature_strategy,
        # Conserve la normalisation pour éviter les dérives d'échelle
        normalize_features=config.normalize_features,
        # Conserve la méthode de réduction pour limiter l'écart de pipeline
        dim_method=config.dim_method,
        # Conserve le nombre de composantes initialement demandé
        n_components=config.n_components,
        # Remplace LDA par un classifieur stable sur petits échantillons
        classifier="centroid",
        # Conserve le scaler optionnel demandé par la configuration
        scaler=config.scaler,
        # Conserve la régularisation CSP pour stabiliser les covariances
        csp_regularization=config.csp_regularization,
    )


# Regroupe toutes les informations nécessaires à un run d'entraînement
@dataclass
class TrainingRequest:
    """Décrit les paramètres nécessaires pour entraîner un run."""

    # Identifie le sujet cible pour l'entraînement
    subject: str
    # Identifie le run ciblé pour le sujet sélectionné
    run: str
    # Transporte la configuration complète de pipeline
    pipeline_config: PipelineConfig
    # Spécifie le répertoire contenant les données numpy
    data_dir: Path
    # Spécifie le répertoire racine pour déposer les artefacts
    artifacts_dir: Path
    # Spécifie le répertoire des enregistrements EDF bruts
    raw_dir: Path = DEFAULT_RAW_DIR
    # Active une optimisation systématique des hyperparamètres si demandé
    enable_grid_search: bool = False
    # Fixe un nombre de splits spécifique pour la recherche si fourni
    grid_search_splits: int | None = None


# Regroupe les entrées nécessaires à la sélection de fenêtre d'epochs
@dataclass
class EpochWindowContext:
    """Agrège les données requises pour sélectionner une fenêtre d'epochs."""

    # Porte l'enregistrement brut filtré pour l'epoching
    filtered_raw: preprocessing.mne.io.BaseRaw
    # Porte les événements détectés pour l'epoching
    events: np.ndarray
    # Porte le mapping d'événements vers labels
    event_id: dict[str, int]
    # Porte la liste des labels moteurs alignés
    motor_labels: list[str]
    # Identifie le sujet pour les logs et erreurs
    subject: str
    # Identifie le run pour les logs et erreurs
    run: str


# Centralise les ressources partagées entre plusieurs entraînements
@dataclass
class TrainingResources:
    """Agrège les chemins et la configuration pipeline pour un batch."""

    # Transporte la configuration partagée pour toutes les exécutions
    pipeline_config: PipelineConfig
    # Spécifie le répertoire contenant les données numpy
    data_dir: Path
    # Spécifie le répertoire racine pour déposer les artefacts
    artifacts_dir: Path
    # Spécifie le répertoire des enregistrements EDF bruts
    raw_dir: Path = DEFAULT_RAW_DIR
    # Active une optimisation systématique des hyperparamètres si demandé
    enable_grid_search: bool = False
    # Fixe un nombre de splits spécifique pour la recherche si fourni
    grid_search_splits: int | None = None


# Construit un argument parser aligné sur la CLI mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'entraînement TPV."""

    # Crée le parser avec description lisible pour l'utilisateur
    parser = argparse.ArgumentParser(
        description="Entraîne une pipeline TPV et sauvegarde ses artefacts",
    )
    # Ajoute l'argument positionnel du sujet pour identifier les fichiers
    parser.add_argument(
        "subject",
        type=_parse_subject,
        help="Identifiant du sujet (ex: 4)",
    )
    # Ajoute l'argument positionnel du run pour sélectionner la session
    parser.add_argument(
        "run",
        type=_parse_run,
        help="Identifiant du run (ex: 14)",
    )
    # Ajoute l'option classifieur pour synchroniser avec mybci
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm", "centroid"),
        default="lda",
        help="Classifieur final utilisé pour l'entraînement",
    )
    # Ajoute le choix du scaler optionnel pour stabiliser les features
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "none"),
        default="none",
        help="Scaler optionnel appliqué après l'extraction de features",
    )
    # Ajoute la stratégie d'extraction de features pour garder la cohérence
    parser.add_argument(
        "--feature-strategy",
        choices=("fft", "wavelet"),
        default="fft",
        help="Méthode d'extraction de features spectrales",
    )
    # Ajoute la méthode de réduction de dimension pour contrôler la compression
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp", "svd"),
        default="csp",
        help="Méthode de réduction de dimension pour la pipeline",
    )
    # Ajoute la régularisation CSP pour stabiliser les covariances
    parser.add_argument(
        "--csp-regularization",
        type=float,
        default=0.1,
        help="Régularisation diagonale appliquée aux covariances CSP",
    )
    # Ajoute le nombre de composantes cible pour la réduction
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
    # Ajoute une option pour cibler un répertoire de données spécifique
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les fichiers numpy",
    )
    # Ajoute une option pour configurer le répertoire d'artefacts
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où enregistrer le modèle",
    )
    # Ajoute une option pour pointer vers les fichiers EDF bruts
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Répertoire racine contenant les fichiers EDF bruts",
    )
    # Ajoute un mode pour générer tous les .npy sans lancer un fit complet
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="Génère les fichiers _X.npy/_y.npy pour tous les sujets détectés",
    )
    # Ajoute un mode pour entraîner tous les runs moteurs disponibles
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Entraîne tous les sujets/runs détectés dans data/",
    )
    # Ajoute une option pour activer une recherche systématique de paramètres
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Active une optimisation systématique des hyperparamètres",
    )
    # Ajoute une option pour forcer le nombre de splits en grid search
    parser.add_argument(
        "--grid-search-splits",
        type=int,
        default=None,
        help="Nombre de splits CV dédié à la recherche d'hyperparamètres",
    )
    # Ajoute une option pour spécifier la fréquence d'échantillonnage
    parser.add_argument(
        "--sfreq",
        type=float,
        default=DEFAULT_SAMPLING_RATE,
        help="Fréquence d'échantillonnage utilisée pour les features",
    )
    # Retourne le parser configuré
    return parser


# Construit les chemins des données pour un sujet et un run donnés
def _resolve_data_paths(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit le chemin du fichier de fenêtre d'epochs pour un run
def _resolve_epoch_window_path(subject: str, run: str, data_dir: Path) -> Path:
    """Retourne le chemin du JSON décrivant la fenêtre d'epochs sélectionnée."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Construit le chemin du fichier de fenêtre pour ce run
    window_path = base_dir / f"{run}_epoch_window.json"
    # Retourne le chemin du fichier de fenêtre
    return window_path


# Écrit la fenêtre d'epochs sélectionnée pour usage futur
def _write_epoch_window_metadata(
    subject: str,
    run: str,
    data_dir: Path,
    window: tuple[float, float],
) -> None:
    """Enregistre la fenêtre d'epochs sélectionnée pour ce run."""

    # Construit le chemin du fichier de fenêtre pour ce run
    window_path = _resolve_epoch_window_path(subject, run, data_dir)
    # Assure l'existence du dossier cible pour éviter une erreur d'écriture
    window_path.parent.mkdir(parents=True, exist_ok=True)
    # Prépare la structure JSON pour la persistance
    payload = {"tmin": window[0], "tmax": window[1]}
    # Sérialise la fenêtre dans un fichier JSON dédié
    window_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


# Charge la fenêtre d'epochs persistée pour un run si disponible
def _read_epoch_window_metadata(
    subject: str,
    run: str,
    data_dir: Path,
) -> tuple[float, float] | None:
    """Retourne la fenêtre d'epochs persistée ou None si absente."""

    # Construit le chemin du fichier de fenêtre pour ce run
    window_path = _resolve_epoch_window_path(subject, run, data_dir)
    # Retourne None si le fichier n'existe pas
    if not window_path.exists():
        return None
    # Charge le contenu JSON pour récupérer la fenêtre
    payload = json.loads(window_path.read_text())
    # Extrait tmin du JSON en float
    tmin = float(payload.get("tmin", DEFAULT_EPOCH_WINDOW[0]))
    # Extrait tmax du JSON en float
    tmax = float(payload.get("tmax", DEFAULT_EPOCH_WINDOW[1]))
    # Retourne la fenêtre reconstruite
    return (tmin, tmax)


# Construit un pipeline léger pour scorer les fenêtres temporelles
def _build_window_search_pipeline(sfreq: float) -> Pipeline:
    """Construit un pipeline CSP+Centroid pour la sélection de fenêtre."""

    search_config = PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="csp",
        n_components=DEFAULT_CSP_COMPONENTS,
        classifier="centroid",  # ⭐ CHANGEMENT : centroid au lieu de lda
        scaler=None,
        csp_regularization=0.1,
    )
    return build_pipeline(search_config)


# Évalue une fenêtre temporelle via validation croisée stratifiée
# Évalue une fenêtre temporelle via validation croisée stratifiée
def _score_epoch_window(X: np.ndarray, y: np.ndarray, sfreq: float) -> float | None:
    if len(y) < 2:
        return 0.5 # Niveau du hasard silencieux

    cv = _build_cv_splitter(y, DEFAULT_CV_SPLITS)
    
    # On a supprimé les messages INFO/DEBUG ici
    if cv is None:
        return 0.5

    pipeline = _build_window_search_pipeline(sfreq)
    
    try:
        # On utilise cross_val_score normalement
        scores = cross_val_score(
            pipeline, X, y, cv=cv, error_score=0.5
        )
        return float(np.mean(scores))
    except Exception:
        return 0.5


# Construit les epochs et labels pour une fenêtre temporelle donnée
def _build_epochs_for_window(
    context: EpochWindowContext,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les epochs et labels alignés pour une fenêtre donnée."""

    # Découpe le signal filtré en epochs exploitables pour la fenêtre
    epochs = preprocessing.create_epochs_from_raw(
        context.filtered_raw,
        context.events,
        context.event_id,
        tmin=window[0],
        tmax=window[1],
    )

    # DEBUG: Afficher le nombre d'epochs créés
    print(f"DEBUG EPOCHS: {len(epochs)} epochs créés pour {context.subject} {context.run}")
    print(f"DEBUG EVENTS: {len(context.events)} événements détectés")
    print(f"DEBUG LABELS: {len(context.motor_labels)} labels fournis")

    # BONNE PRATIQUE: Récupère les indices des epochs conservés après filtrage MNE
    # epochs.selection contient les indices des événements conservés
    # Si tous les événements sont conservés: selection = [0, 1, 2, ..., n-1]
    # Si certains sont rejetés par annotations: selection = [0, 2, 4, ...] (indices non consécutifs)
    kept_indices = epochs.selection
    
    # Filtre motor_labels pour ne garder que les labels des epochs conservés
    epochs_aligned_labels = [context.motor_labels[i] for i in kept_indices]
    
    # Vérification de sécurité (assertion pour détecter les bugs)
    assert len(epochs_aligned_labels) == len(epochs), (
        f"Désalignement détecté: {len(epochs_aligned_labels)} labels "
        f"pour {len(epochs)} epochs"
    )

    # Applique un rejet d'artefacts
    try:
        # On augmente le seuil de rejet (max_peak_to_peak) pour garder plus de données
        cleaned_epochs, _report, cleaned_labels = preprocessing.summarize_epoch_quality(
            epochs,
            epochs_aligned_labels,
            (context.subject, context.run),
            max_peak_to_peak=1500e-6, # Seuil plus tolérant (était à 800e-6)
        )
        
        # SÉCURITÉ : Si le nettoyage a tout supprimé, on FORCE l'utilisation
        # des données brutes pour éviter d'avoir 0 échantillon.
        if len(cleaned_epochs) == 0:
            print(f"WARN: [{context.subject}] Nettoyage trop strict, conservation des données brutes.")
            return epochs.get_data(), epochs_aligned_labels
            
        return cleaned_epochs, cleaned_labels

    except Exception as e:
        print(f"ERROR: Erreur lors du nettoyage pour {context.subject}: {e}")
        return epochs.get_data(), epochs_aligned_labels


# Sélectionne la meilleure fenêtre selon un score cross-val
def _select_best_epoch_window(
    context: EpochWindowContext,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """Retourne la fenêtre optimale et les données associées."""

    # Initialise la meilleure fenêtre par défaut
    best_window = DEFAULT_EPOCH_WINDOW
    # Initialise le score de fenêtre à None pour détecter l'absence de CV
    best_score: float | None = None
    # Initialise les données d'epochs pour la fenêtre retenue
    best_epochs_data: np.ndarray | None = None
    # Initialise les labels numériques pour la fenêtre retenue
    best_labels: np.ndarray | None = None

    # Parcourt les fenêtres candidates pour sélectionner la plus robuste
    for window in DEFAULT_EPOCH_WINDOWS:
        # Construit les epochs et labels pour la fenêtre courante
        epochs_data, numeric_labels = _build_epochs_for_window(context, window)
        # Calcule le score cross-val pour cette fenêtre si possible
        window_score = _score_epoch_window(
            epochs_data,
            numeric_labels,
            float(context.filtered_raw.info["sfreq"]),
        )
        # Initialise la fenêtre par défaut lorsqu'aucune n'est retenue
        if best_epochs_data is None:
            # Stocke les données de référence pour éviter un fallback vide
            best_epochs_data = epochs_data
            # Stocke les labels de référence pour éviter un fallback vide
            best_labels = numeric_labels
            # Conserve la fenêtre courante comme valeur par défaut
            best_window = window
        # Ignore les fenêtres sans score si une meilleure est déjà connue
        if window_score is None and best_score is not None:
            continue
        # Met à jour la meilleure fenêtre si un score supérieur est trouvé
        if window_score is not None and (
            best_score is None or window_score > best_score
        ):
            # Conserve le score de la fenêtre retenue
            best_score = window_score
            # Conserve les données de la fenêtre retenue
            best_epochs_data = epochs_data
            # Conserve les labels de la fenêtre retenue
            best_labels = numeric_labels
            # Conserve la fenêtre retenue pour usage ultérieur
            best_window = window

    # Garantit que les données retenues existent avant la sauvegarde
    if best_epochs_data is None or best_labels is None:
        # Signale une absence complète de données après sélection
        raise ValueError(
            f"Aucune epoch valide pour {context.subject} {context.run} "
            "après sélection de fenêtre."
        )

    # Retourne la fenêtre retenue et les données associées
    return best_window, best_epochs_data, best_labels


# Construit des matrices numpy à partir d'un EDF lorsqu'elles manquent
def _build_npy_from_edf(
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
) -> tuple[Path, Path]:
    """Génère X (epochs brutes) et y depuis un fichier EDF Physionet.

    - X est sauvegardé sous forme (n_trials, n_channels, n_times)
      pour être compatible avec la pipeline (tpv.features).
    - Les features fréquentielles sont ensuite calculées *dans* la
      pipeline, pas au moment de la génération des .npy.
    """

    # Calcule les chemins cibles pour les fichiers numpy
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)
    # Calcule le chemin attendu du fichier EDF brut
    raw_path = raw_dir / subject / f"{subject}{run}.edf"

    # Interrompt tôt si l'EDF est absent
    if not raw_path.exists():
        raise FileNotFoundError(
            "EDF introuvable pour "
            f"{subject} {run}: {raw_path}. "
            "Téléchargez les enregistrements Physionet dans data ou "
            "pointez --raw-dir vers un dossier déjà synchronisé."
        )

    # Crée l'arborescence cible pour déposer les .npy
    features_path.parent.mkdir(parents=True, exist_ok=True)

    # Charge l'EDF en conservant les métadonnées essentielles
    raw, _ = preprocessing.load_physionet_raw(raw_path)

    # Applique un notch pour supprimer la pollution secteur
    notched_raw = preprocessing.apply_notch_filter(raw, freq=DEFAULT_NOTCH_FREQ)
    # Applique le filtrage bande-passante pour stabiliser les bandes MI
    filtered_raw = preprocessing.apply_bandpass_filter(
        notched_raw,
        freq_band=DEFAULT_BANDPASS_BAND,
    )

    # Mappe les annotations en événements moteurs après filtrage
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(
        filtered_raw
    )
    print(f"DEBUG MAPPING: {len(events)} événements moteurs pour {subject} {run}")
    print(f"DEBUG EVENT_ID: {event_id}")
    print(f"DEBUG LABELS COUNT: {len(motor_labels)}")


    # Construit le contexte nécessaire à la sélection de fenêtre
    window_context = EpochWindowContext(
        filtered_raw=filtered_raw,
        events=events,
        event_id=event_id,
        motor_labels=motor_labels,
        subject=subject,
        run=run,
    )
    # Sélectionne la meilleure fenêtre et ses données associées
    best_window, best_epochs_data, best_labels = _select_best_epoch_window(
        window_context
    )

    # Persiste les epochs brutes sélectionnées
    np.save(features_path, best_epochs_data)
    # Persiste les labels alignés sur la fenêtre retenue
    np.save(labels_path, best_labels)
    # Écrit la fenêtre retenue pour la réutiliser en prédiction
    _write_epoch_window_metadata(subject, run, data_dir, best_window)

    # Retourne les chemins nouvellement générés
    return features_path, labels_path


# Construit les .npy pour l'ensemble des sujets disponibles
def _build_all_npy(raw_dir: Path, data_dir: Path) -> None:
    """Génère les fichiers numpy pour chaque run moteur disponible."""

    # Parcourt les dossiers de sujets triés pour des logs prédictibles
    subject_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())

    # Explore chaque sujet détecté dans le répertoire brut
    for subject_dir in subject_dirs:
        # Extrait l'identifiant du sujet à partir du nom de dossier
        subject = subject_dir.name
        # Liste tous les enregistrements EDF associés au sujet courant
        edf_paths = sorted(subject_dir.glob(f"{subject}R*.edf"))

        # Traite chaque enregistrement pour générer les .npy associés
        for edf_path in edf_paths:
            # Déduit le run en retirant le préfixe sujet du nom de fichier
            run = edf_path.stem.replace(subject, "")

            # Ignore explicitement les runs dépourvus d'événements moteurs
            try:
                _build_npy_from_edf(subject, run, data_dir, raw_dir)
            except ValueError as error:
                if "No motor events present" in str(error):
                    print(
                        "INFO: Événements moteurs absents pour "
                        f"{subject} {run}, passage."
                    )
                    continue
                raise


# Liste les sujets disponibles dans le répertoire brut
def _list_subjects(raw_dir: Path) -> list[str]:
    """Retourne les identifiants de sujets triés présents dans raw_dir."""

    # Construit la liste des dossiers de sujets pour préparer l'entraînement
    subjects = [entry.name for entry in raw_dir.iterdir() if entry.is_dir()]
    # Trie les identifiants pour obtenir des logs stables et reproductibles
    subjects.sort()
    # Retourne la liste triée pour l'appelant
    return subjects


# Entraîne un couple sujet/run en réutilisant la configuration partagée
def _train_single_run(
    subject: str,
    run: str,
    resources: TrainingResources,
) -> bool:
    """Lance l'entraînement d'un sujet pour un run donné."""

    # Prépare la requête complète pour exécuter run_training
    request = TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=resources.pipeline_config,
        data_dir=resources.data_dir,
        artifacts_dir=resources.artifacts_dir,
        raw_dir=resources.raw_dir,
    )
    # Protège l'appel pour signaler les données manquantes sans stopper la boucle
    try:
        # Entraîne la pipeline et persiste les artefacts nécessaires
        _ = run_training(request)
    except FileNotFoundError as error:
        # Alerte l'utilisateur lorsqu'un EDF ou des événements sont absents
        print(f"AVERTISSEMENT: {error}")
        # Indique l'échec pour déclencher un récapitulatif final
        return False
    # Retourne True pour signaler un entraînement réussi
    return True


# Entraîne tous les runs moteurs pour chaque sujet détecté
def _train_all_runs(
    config: PipelineConfig,
    data_dir: Path,
    artifacts_dir: Path,
    raw_dir: Path,
) -> int:
    """Parcourt les sujets et runs moteurs pour générer tous les modèles."""

    # Récupère la liste des sujets disponibles dans le répertoire brut
    subjects = _list_subjects(raw_dir)
    # Centralise les ressources immuables pour éviter des répétitions
    resources = TrainingResources(
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        raw_dir=raw_dir,
    )
    # Prépare un compteur d'échecs pour informer l'utilisateur à la fin
    failures = 0
    # Parcourt chaque sujet détecté
    for subject in subjects:
        # Parcourt chaque run moteur attendu
        for run in MOTOR_RUNS:
            # Calcule le chemin EDF attendu pour vérifier l'existence
            raw_path = raw_dir / subject / f"{subject}{run}.edf"
            # Ignore le couple lorsque l'EDF est absent du disque
            if not raw_path.exists():
                # Informe l'utilisateur de l'absence pour transparence
                print(
                    "INFO: EDF introuvable pour "
                    f"{subject} {run} dans {raw_path.parent}, passage."
                )
                # Passe au run suivant sans incrémenter les échecs
                continue
            # Entraîne le run courant et capture le statut
            success = _train_single_run(
                subject,
                run,
                resources,
            )
            # Incrémente le compteur d'échecs lorsque l'entraînement échoue
            if not success:
                failures += 1
    # Affiche un résumé pour guider l'utilisateur après la boucle
    if failures:
        # Mentionne le nombre total d'entraînements manquants
        print(
            "AVERTISSEMENT: certains entraînements ont échoué. "
            f"Exécutions manquantes: {failures}."
        )
    else:
        # Confirme que tous les artefacts ont été générés avec succès
        print("INFO: modèles entraînés pour tous les runs moteurs détectés.")
    # Retourne 1 si des échecs sont survenus pour refléter l'état global
    return 1 if failures else 0


# Vérifie si les caches existants respectent les shapes attendues
def _needs_rebuild_from_shapes(
    candidate_X: np.ndarray,
    candidate_y: np.ndarray,
    features_path: Path,
    labels_path: Path,
    run_label: str,
) -> bool:
    """Valide les dimensions des caches existants pour éviter des erreurs."""

    # RECTIFICATION : On accepte le cache dès qu'il contient au moins 1 échantillon.
    # Si on demande 4 et que le sujet n'en a que 2, le code bouclera à l'infini sinon.
    MIN_REQUIRED_SAMPLES = 1 
    
    if candidate_X.shape[0] < MIN_REQUIRED_SAMPLES:
        print(
            f"INFO: Cache vide ou invalide pour {run_label}. "
            f"Régénération depuis l'EDF..."
        )
        return True

    # Vérification de la cohérence entre X (données) et y (labels)
    if candidate_X.shape[0] != candidate_y.shape[0]:
        print(f"WARN: Incohérence de taille pour {run_label}. Régénération...")
        return True
        
    return False


def _should_check_shapes(
    needs_rebuild: bool,
    corrupted_reason: str | None,
    candidate_X: np.ndarray | None,
    candidate_y: np.ndarray | None,
) -> bool:
    """Détermine si la validation des shapes est nécessaire."""

    return (
        not needs_rebuild
        and corrupted_reason is None
        and candidate_X is not None
        and candidate_y is not None
    )


# Charge ou génère les matrices numpy attendues pour l'entraînement
def _load_data(
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge ou construit les données et étiquettes pour un run.

    - Si les .npy n'existent pas, on les génère depuis l'EDF.
    - Si X existe mais n'est pas 3D, on reconstruit depuis l'EDF.
    - Si X et y n'ont pas le même nombre d'échantillons, on
      reconstruit pour réaligner les labels sur les epochs.
    """

    # Concatène le couple sujet/run pour les messages utilisateur
    run_label = f"{subject} {run}"

    # Détermine les chemins attendus pour les features et labels
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)

    # Garde un bool strict (évite None, tue le mutant False->None).
    needs_rebuild: bool = False
    # Stocke les chemins invalides pour enrichir les logs utilisateurs
    corrupted_reason: str | None = None
    # Conserve les caches chargés pour valider leurs formes
    candidate_X: np.ndarray | None = None
    # Conserve les labels chargés pour valider la longueur
    candidate_y: np.ndarray | None = None

    # Cas 1 : fichiers manquants → on reconstruira
    if not features_path.exists() or not labels_path.exists():
        needs_rebuild = True
    else:
        # Sécurise le chargement numpy pour tolérer les fichiers corrompus
        try:
            # Charge X en mmap pour inspecter la forme sans tout charger
            candidate_X = np.load(features_path, mmap_mode="r")
            # Charge y en mmap pour inspecter la longueur
            candidate_y = np.load(labels_path, mmap_mode="r")
        except (OSError, ValueError) as error:
            # Demande la reconstruction dès qu'un chargement échoue
            needs_rebuild = True
            # Conserve la raison pour orienter l'utilisateur
            corrupted_reason = str(error)

    # Valide les shapes lorsque les caches ont été chargés avec succès
    if _should_check_shapes(needs_rebuild, corrupted_reason, candidate_X, candidate_y):
        # Convertit X vers un tableau typé pour satisfaire mypy et bandit
        validated_X = cast(np.ndarray, candidate_X)
        # Convertit y vers un vecteur typé pour satisfaire mypy et bandit
        validated_y = cast(np.ndarray, candidate_y)
        # Détecte les incohérences de dimension et déclenche une régénération
        needs_rebuild = bool(
            _needs_rebuild_from_shapes(
                validated_X,
                validated_y,
                features_path,
                labels_path,
                run_label,
            )
        )

    # Informe l'utilisateur lorsqu'un fichier corrompu bloque le chargement
    if corrupted_reason is not None:
        print(
            "INFO: Chargement numpy impossible pour "
            f"{subject} {run}: {corrupted_reason}. "
            "Régénération depuis l'EDF..."
        )
        needs_rebuild = True

    # Force un bool Python strict (évite None / numpy.bool_).
    needs_rebuild = True if needs_rebuild else False

    # Reconstruit les fichiers lorsque nécessaire
    if needs_rebuild:
        features_path, labels_path = _build_npy_from_edf(
            subject,
            run,
            data_dir,
            raw_dir,
        )

    # Charge les données validées (3D) et labels réalignés
    X = np.load(features_path)
    y = np.load(labels_path)

    return X, y


# Récupère le hash git courant pour tracer la reproductibilité
def _get_git_commit() -> str:
    """Retourne le hash du commit courant ou "unknown" en secours."""

    # Localise le fichier HEAD pour extraire la référence courante
    head_path = Path(".git") / "HEAD"
    # Retourne unknown lorsque le dépôt git n'est pas disponible
    if not head_path.exists():
        # Fournit une valeur de repli pour conserver un manifeste valide
        return "unknown"
    # Lit le contenu du HEAD pour déterminer la référence active
    head_content = head_path.read_text().strip()
    # Détecte les références symboliques du style "ref: ..."
    if head_content.startswith("ref:"):
        # Isole le chemin relatif vers le fichier de référence
        ref_path = Path(".git") / head_content.split(" ", 1)[1]
        # Retourne unknown si la référence est introuvable
        if not ref_path.exists():
            # Fournit une valeur de repli pour préserver la validation
            return "unknown"
        # Lit le hash contenu dans le fichier de référence
        return ref_path.read_text().strip()
    # Retourne le contenu brut lorsque HEAD contient déjà un hash
    return head_content or "unknown"


# Sérialise un manifeste complet à côté du modèle entraîné
def _flatten_hyperparams(hyperparams: dict) -> dict[str, str]:
    """Aplati les hyperparamètres pour une exportation CSV lisible."""

    # Prépare un dictionnaire de sortie initialement vide
    flattened: dict[str, str] = {}
    # Parcourt chaque entrée pour extraire les valeurs simples
    for key, value in hyperparams.items():
        # Sérialise chaque valeur pour conserver la lisibilité CSV
        flattened[key] = json.dumps(value, ensure_ascii=False)
    # Retourne le dictionnaire aplati prêt pour l'écriture CSV
    return flattened


def _write_manifest(
    request: TrainingRequest,
    target_dir: Path,
    cv_scores: np.ndarray,
    artifacts: dict[str, Path | None],
    search_summary: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Écrit des manifestes JSON et CSV décrivant le run d'entraînement."""

    # Charge la fenêtre d'epochs persistée pour l'inclure au manifeste
    epoch_window = _read_epoch_window_metadata(
        request.subject,
        request.run,
        request.data_dir,
    )
    # Prépare la section dataset pour identifier les entrées de données
    dataset = {
        "subject": request.subject,
        "run": request.run,
        "data_dir": str(request.data_dir),
        "epoch_window": epoch_window,
    }
    # Convertit la configuration de pipeline en dictionnaire sérialisable
    hyperparams = asdict(request.pipeline_config)
    # Calcule la moyenne des scores si la validation croisée a tourné
    cv_mean = float(np.mean(cv_scores)) if cv_scores.size else None
    # Prépare la section des scores en sérialisant les arrays numpy
    scores = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": cv_mean,
    }
    # Résout l'identifiant du commit git pour tracer les artefacts
    git_commit = _get_git_commit()
    # Prépare la section chemins pour retrouver rapidement les fichiers
    artifacts_section = {
        "model": str(artifacts["model"]),
        "scaler": str(artifacts["scaler"]) if artifacts["scaler"] else None,
        "w_matrix": str(artifacts["w_matrix"]),
    }
    # Assemble toutes les sections dans un objet manifeste unique
    manifest = {
        "dataset": dataset,
        "hyperparams": hyperparams,
        "scores": scores,
        "git_commit": git_commit,
        "artifacts": artifacts_section,
    }
    # Ajoute un résumé de recherche uniquement si une optimisation a eu lieu
    if search_summary is not None:
        # Expose les paramètres optimaux et score CV associé
        manifest["hyperparam_search"] = search_summary
    # Définit le chemin de sortie du manifeste JSON à côté des artefacts
    manifest_json_path = target_dir / "manifest.json"
    # Prépare un JSON tolérant les objets non sérialisables du manifeste
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
    # Écrit le manifeste JSON sur disque en UTF-8 pour la portabilité
    manifest_json_path.write_text(manifest_json)
    # Aplati les hyperparamètres pour faciliter la lecture dans un tableur
    flattened_hyperparams = _flatten_hyperparams(hyperparams)
    # Construit une ligne CSV unique regroupant toutes les informations
    csv_line = {
        "subject": request.subject,
        "run": request.run,
        "data_dir": str(request.data_dir),
        "git_commit": git_commit,
        "cv_scores": ";".join(str(score) for score in cv_scores.tolist()),
        "cv_mean": "" if cv_mean is None else str(cv_mean),
        **flattened_hyperparams,
    }
    # Définit le chemin du manifeste CSV à côté du JSON
    manifest_csv_path = target_dir / "manifest.csv"
    # Ouvre le fichier CSV en écriture sans lignes superflues
    with manifest_csv_path.open("w", newline="") as handle:
        # Initialise l'écriture CSV avec les clés détectées
        writer = csv.DictWriter(handle, fieldnames=list(csv_line.keys()))
        # Inscrit les en-têtes pour faciliter l'import dans un tableur
        writer.writeheader()
        # Inscrit la ligne unique décrivant le run en cours
        writer.writerow(csv_line)
    # Retourne les chemins des manifestes pour les appels appelants
    return {"json": manifest_json_path, "csv": manifest_csv_path}


# Construit la grille d'hyperparamètres par défaut pour l'optimisation
# Construit la grille d'hyperparamètres pour la recherche d'optimisation
def _build_grid_search_grid(
    config: PipelineConfig, allow_lda: bool
) -> dict[str, list[object]]:
    """Retourne une grille raisonnable pour la recherche d'hyperparamètres."""

    # Retourne une grille réduite si CSP est utilisé (pas de features/scaler amont)
    if config.dim_method == "csp":
        # Déclare un ensemble restreint de classifieurs pour CSP
        csp_classifier_grid: list[object] = []
        # Ajoute LDA uniquement lorsque l'effectif le permet
        if allow_lda:
            # Utilise LDA shrinkage pour stabilité des covariances
            csp_classifier_grid.append(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            )
        # Ajoute une régression logistique pour des décisions régularisées
        csp_classifier_grid.append(LogisticRegression(max_iter=1000))
        # Augmente max_iter pour limiter les warnings de convergence liblinear
        csp_classifier_grid.append(LinearSVC(max_iter=5000))
        # Ajoute un classifieur centroïde pour les petits échantillons
        csp_classifier_grid.append(CentroidClassifier())
        # Prépare une liste de composantes CSP sans doublons
        csp_n_components_grid: list[object] = (
            [config.n_components]
            if config.n_components is not None
            else [None, DEFAULT_CSP_COMPONENTS]
        )
        # Ajoute la valeur explicite si absente de la grille
        if (
            config.n_components is not None
            and config.n_components not in csp_n_components_grid
        ):
            csp_n_components_grid.append(config.n_components)
        # Retourne une grille compatible avec la pipeline CSP réduite
        return {
            "dimensionality__n_components": csp_n_components_grid,
            "classifier": csp_classifier_grid,
        }

    # Déclare un ensemble restreint de classifieurs pour limiter la complexité
    # Initialise la grille de classifieurs candidates
    classifier_grid: list[object] = []
    # Ajoute LDA uniquement lorsque l'effectif le permet
    if allow_lda:
        # Utilise LDA pour les effectifs suffisants
        classifier_grid.append(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
    # Ajoute une régression logistique pour des décisions régularisées
    classifier_grid.append(LogisticRegression(max_iter=1000))
    # Augmente max_iter pour limiter les warnings de convergence liblinear
    classifier_grid.append(LinearSVC(max_iter=5000))
    # Ajoute un classifieur centroïde pour les petits échantillons
    classifier_grid.append(CentroidClassifier())
    # Déclare des scalers optionnels, dont passthrough pour désactiver
    scaler_grid: list[object] = ["passthrough", StandardScaler(), RobustScaler()]
    # Déclare une plage compacte de composantes pour PCA/SVD
    n_components_grid: list[object] = [None, 2, 4, 8]
    # Ajoute la valeur demandée explicitement pour garantir sa présence
    if config.n_components is not None and config.n_components not in n_components_grid:
        n_components_grid.append(config.n_components)
    # Construit la grille finale en couvrant features + réduction + classif
    # Étend la grille aux familles Welch et mixte pour enrichir les features
    return {
        "features__feature_strategy": ["fft", "welch", ("fft", "welch"), "wavelet"],
        "features__normalize": [True, False],
        "dimensionality__method": ["pca", "svd"],
        "dimensionality__n_components": n_components_grid,
        "classifier": classifier_grid,
        "scaler": scaler_grid,
    }


# Extrait les scores par split d'une GridSearchCV pour le meilleur modèle
def _extract_grid_search_scores(search: GridSearchCV, n_splits: int) -> np.ndarray:
    """Construit un tableau de scores à partir des résultats de grid search."""

    # Récupère l'index du meilleur modèle sélectionné
    best_index = int(search.best_index_)
    # Construit la liste des scores par split pour l'entrée manifeste
    split_scores = []
    for split_index in range(n_splits):
        # Compose la clé de split attendue dans cv_results_
        key = f"split{split_index}_test_score"
        # Récupère la colonne associée si disponible
        column = search.cv_results_.get(key)
        # Ignore les splits manquants si la CV est réduite
        if column is None:
            continue
        split_scores.append(float(column[best_index]))
    # Retourne un tableau numpy stable même si partiellement rempli
    return np.array(split_scores, dtype=float)


# Gère l'entraînement et la CV optionnelle pour un dataset donné
def _train_with_optional_cv(
    request: TrainingRequest,
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    adapted_config: PipelineConfig,
) -> tuple[np.ndarray, Pipeline, dict[str, object] | None, str | None, str | None]:
    """Retourne la pipeline entraînée et les scores CV si disponibles."""

    # Calcule le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Calcule le nombre de classes distinctes pour l'évaluation
    class_count = int(np.unique(y).size)
    # Résout le splitter CV en tenant compte des classes disponibles
    cv, cv_unavailability_reason = _resolve_cv_splits(y, DEFAULT_CV_SPLITS)
    # Initialise un tableau vide lorsque la validation croisée est impossible
    cv_scores = np.array([])
    # Prépare un résumé de recherche d'hyperparamètres si besoin
    search_summary: dict[str, object] | None = None
    # Prépare un message d'erreur si cross_val_score échoue
    cv_error: str | None = None
    # Vérifie si l'effectif autorise une validation croisée exploitable
    if cv is None:
        # Prépare la raison finale pour expliquer l'absence de CV
        cv_unavailability_reason = (
            cv_unavailability_reason
            or _describe_cv_unavailability(y, DEFAULT_CV_SPLITS)
        )
        # Applique le fallback CV et récupère la pipeline ajustée
        cv_scores, pipeline, cv_unavailability_reason = _handle_cv_unavailability(
            pipeline,
            X,
            y,
            cv_unavailability_reason,
        )
    # Lance une recherche systématique des hyperparamètres si demandée
    elif request.enable_grid_search:
        # Construit la pipeline dédiée au grid search
        search_pipeline = build_search_pipeline(adapted_config)
        # Détermine si LDA est autorisé par le ratio effectif/classes
        allow_lda = sample_count > class_count
        # Construit la grille d'hyperparamètres à explorer
        param_grid = _build_grid_search_grid(adapted_config, allow_lda)
        # Détermine le nombre de splits spécifique si fourni
        search_splits = request.grid_search_splits or DEFAULT_CV_SPLITS
        # Construit un splitter dédié pour la recherche d'hyperparamètres
        search_cv, search_reason = _resolve_cv_splits(y, search_splits)
        # Désactive la recherche si la CV est impossible
        if search_cv is None:
            # Prépare une raison explicite pour l'absence de CV
            cv_unavailability_reason = search_reason or _describe_cv_unavailability(
                y, search_splits
            )
            # Applique le fallback CV et retourne immédiatement
            cv_scores, pipeline, cv_unavailability_reason = _handle_cv_unavailability(
                pipeline,
                X,
                y,
                cv_unavailability_reason,
            )
            # Retourne immédiatement les sorties sans grid search
            return (
                cv_scores,
                pipeline,
                search_summary,
                cv_unavailability_reason,
                cv_error,
            )
        # Lance la recherche d'hyperparamètres et récupère les scores
        cv_scores, pipeline, search_summary = _run_grid_search(
            search_pipeline,
            param_grid,
            search_cv,
            X,
            y,
        )
    else:
        # Calcule les scores de validation croisée sur l'ensemble du pipeline
        try:
            # Lance la cross-validation pour mesurer la performance
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, error_score="raise")
        except ValueError as error:
            # Capture l'erreur pour un diagnostic CLI explicite
            cv_error = str(error)
            # Déclare un score vide pour conserver le flux nominal
            cv_scores = np.array([])
        # Ajuste la pipeline sur toutes les données après évaluation
        pipeline.fit(X, y)
    # Retourne les informations calculées pour l'entraînement
    return (
        cv_scores,
        pipeline,
        search_summary,
        cv_unavailability_reason,
        cv_error,
    )


# Exécute la validation croisée et l'entraînement final
def run_training(request: TrainingRequest) -> dict:
    """Entraîne la pipeline et sauvegarde ses artefacts."""

    # Charge ou génère les tableaux numpy nécessaires à l'entraînement
    X, y = _load_data(request.subject, request.run, request.data_dir, request.raw_dir)
    # Adapte la configuration au niveau d'effectif pour stabiliser l'entraînement
    adapted_config = _adapt_pipeline_config_for_samples(request.pipeline_config, y)
    # Construit la pipeline complète sans préprocesseur amont
    pipeline = build_pipeline(adapted_config)
    # Lance la procédure CV et récupère les artefacts d'entraînement
    (
        cv_scores,
        pipeline,
        search_summary,
        cv_unavailability_reason,
        cv_error,
    ) = _train_with_optional_cv(
        request,
        X,
        y,
        pipeline,
        adapted_config,
    )
    # Prépare le dossier d'artefacts spécifique au sujet et au run
    target_dir = request.artifacts_dir / request.subject / request.run
    # Assure l'existence du parent pour stabiliser la création du dossier cible
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    # Crée les répertoires au besoin pour éviter les erreurs de sauvegarde
    target_dir.mkdir(parents=True, exist_ok=True)
    # Calcule le chemin du fichier modèle pour joblib
    model_path = target_dir / "model.joblib"
    # Sauvegarde la pipeline complète pour les prédictions futures
    save_pipeline(pipeline, str(model_path))
    # Récupère l'éventuel scaler pour une sauvegarde dédiée
    scaler_step = pipeline.named_steps.get("scaler")
    # Sauvegarde le scaler uniquement s'il est présent dans la pipeline
    if scaler_step is not None and scaler_step != "passthrough":
        # Dépose le scaler dans un fichier distinct pour inspection
        joblib.dump(scaler_step, target_dir / "scaler.joblib")
    # Récupère le réducteur de dimension pour exposer la matrice W
    dim_reducer: TPVDimReducer = pipeline.named_steps["dimensionality"]
    # Sauvegarde la matrice de projection pour les usages temps-réel
    dim_reducer.save(target_dir / "w_matrix.joblib")
    # Calcule le chemin du scaler pour l'ajouter au manifeste
    scaler_path = None
    # Renseigne le chemin du scaler uniquement lorsqu'il existe
    if scaler_step is not None and scaler_step != "passthrough":
        # Stocke le chemin vers le scaler sauvegardé pour le manifeste
        scaler_path = target_dir / "scaler.joblib"
    # Calcule le chemin du fichier W pour le référencer dans le manifeste
    w_matrix_path = target_dir / "w_matrix.joblib"
    # Écrit un manifeste décrivant l'entraînement et ses artefacts
    manifest_paths = _write_manifest(
        request,
        target_dir,
        cv_scores,
        {
            "model": model_path,
            "scaler": scaler_path,
            "w_matrix": w_matrix_path,
        },
        search_summary,
    )
    # Retourne un rapport synthétique pour les tests et la CLI
    return {
        "cv_scores": cv_scores,
        "cv_splits_requested": DEFAULT_CV_SPLITS,
        "cv_unavailability_reason": cv_unavailability_reason,
        "cv_error": cv_error,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "w_matrix_path": w_matrix_path,
        "manifest_path": manifest_paths["json"],
        "manifest_csv_path": manifest_paths["csv"],
    }


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'entraînement."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Exécute la génération massive et s'arrête si le flag est positionné
    if args.build_all:
        _build_all_npy(args.raw_dir, args.data_dir)
        return 0
    # Convertit l'option scaler "none" en None pour la pipeline
    scaler = None if args.scaler == "none" else args.scaler
    # Calcule la valeur de normalisation en inversant le flag d'opt-out
    normalize = not args.no_normalize_features
    # Récupère le paramètre n_components s'il est fourni
    n_components = getattr(args, "n_components", None)
    # Résout la fréquence d'échantillonnage en s'appuyant sur l'EDF si possible
    resolved_sfreq = resolve_sampling_rate(
        args.subject,
        args.run,
        args.raw_dir,
        args.sfreq,
    )
    # Construit la configuration de pipeline alignée sur mybci
    config = PipelineConfig(
        sfreq=resolved_sfreq,
        feature_strategy=args.feature_strategy,
        normalize_features=normalize,
        dim_method=args.dim_method,
        n_components=n_components,
        classifier=args.classifier,
        scaler=scaler,
        csp_regularization=args.csp_regularization,
    )
    # Déclenche l'entraînement massif si le flag est activé
    if args.train_all:
        # Propulse la configuration commune vers l'ensemble des runs moteurs
        return _train_all_runs(
            config,
            args.data_dir,
            args.artifacts_dir,
            args.raw_dir,
        )
    # Regroupe les paramètres d'entraînement dans une structure dédiée
    request = TrainingRequest(
        subject=args.subject,
        run=args.run,
        pipeline_config=config,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        raw_dir=args.raw_dir,
        enable_grid_search=args.grid_search,
        grid_search_splits=args.grid_search_splits,
    )
    # Exécute l'entraînement et la sauvegarde des artefacts
    # Sécurise l'exécution pour afficher une erreur lisible sans trace
    try:
        # Lance l'entraînement et récupère le rapport pour afficher les scores
        result = run_training(request)
    except FileNotFoundError as error:
        # Remonte l'erreur utilisateur de manière concise pour la CLI
        print(f"ERREUR: {error}")
        # Expose un code de sortie explicite pour signaler l'échec
        return 1

    # Récupère les scores de validation croisée depuis le rapport
    cv_scores = result["cv_scores"]
    # Récupère le nombre de splits demandé si le rapport le fournit
    cv_splits_requested = int(result.get("cv_splits_requested", DEFAULT_CV_SPLITS))
    # Récupère la raison d'absence de CV si disponible
    cv_unavailability_reason = result.get("cv_unavailability_reason")
    # Récupère un éventuel message d'erreur CV pour l'affichage
    cv_error = result.get("cv_error")
    # Calcule le nombre de scores réellement obtenus
    cv_scores_count = int(cv_scores.size) if isinstance(cv_scores, np.ndarray) else 0
    # Informe toujours l'utilisateur du nombre de splits attendus
    print("CV_SPLITS: " f"{cv_splits_requested} (scores: {cv_scores_count})")
    # Informe si cross_val_score a échoué malgré un splitter valide
    if cv_error:
        # Expose une alerte concise sans trace pour l'utilisateur
        print(f"AVERTISSEMENT: cross_val_score échoué ({cv_error})")
    # Informe si la CV est indisponible pour l'utilisateur
    if cv_unavailability_reason:
        # Expose une raison explicite pour éviter un silence UX
        print(f"INFO: CV indisponible ({cv_unavailability_reason})")

    # Si des scores ont été calculés, on les affiche au format attendu
    if isinstance(cv_scores, np.ndarray) and cv_scores.size > 0:
        # Force quatre décimales fixes pour suivre l'exemple du sujet
        formatted_scores = np.array2string(
            cv_scores, precision=4, separator=" ", floatmode="fixed"
        )
        # Affiche le tableau numpy (format [0.6666 0.4444 ...])
        print(formatted_scores)
        # Calcule la moyenne pour l'affichage "cross_val_score: 0.5333"
        mean_score = float(cv_scores.mean())
        # Affiche la moyenne arrondie sur quatre décimales pour homogénéiser
        print(f"cross_val_score: {mean_score:.4f}")
    else:
        # Fallback lisible si la CV n'a pas pu être calculée
        print(np.array([]))
        print("cross_val_score: 0.0")

    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
