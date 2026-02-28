"""CLI de prédiction pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
# Garantit l'accès aux chemins portables pour données et artefacts
# Fournit le parsing CLI pour aligner la signature mybci
import argparse

# Fournit l'écriture CSV pour exposer les prédictions individuelles
import csv

# Permet de charger dynamiquement scripts.train depuis un fichier
import importlib.util

# Fournit la sérialisation JSON pour tracer les rapports générés
import json
import os

# Fournit dataclass pour regrouper des options de prédiction
# Fournit field pour définir des factories de dataclass
from dataclasses import dataclass, field

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Garantit l'accès au type ModuleType pour typer le chargement dynamique
from types import ModuleType

# Garantit l'accès aux annotations Any et Mapping pour les overrides
# Garantit l'accès à Iterable pour les stratégies multi-entries
# Garantit l'accès à cast pour typer les sorties numpy
from typing import Any, Iterable, Mapping, cast

# Centralise l'accès aux tableaux numpy pour l'évaluation
import numpy as np

# Calcule les métriques de classification pour le rapport
from sklearn.metrics import confusion_matrix

# Centralise le parsing et le contrôle qualité des fichiers EDF
from tpv import preprocessing

# Permet de restaurer la matrice W pour des usages temps-réel
from tpv.dimensionality import TPVDimReducer

# Expose le découpage glissant et l'agrégation pour l'inférence Welch
# Expose l'extracteur de features pour inspecter la stratégie Welch
from tpv.features import (
    ExtractFeatures,
    aggregate_window_probabilities,
    build_sliding_windows,
)

# Expose la configuration de pipeline pour déclencher un auto-train
from tpv.pipeline import PipelineConfig, load_pipeline

# Centralise la fenêtre d'epoching par défaut
from tpv.utils import (
    DEFAULT_EPOCH_WINDOW,
    HANDLED_CLI_ERROR_EXIT_CODE,
    render_cli_error_lines,
)


# Charge dynamiquement scripts.train pour l'exécution directe
def _load_train_module() -> ModuleType:
    """Charge scripts.train depuis le chemin local du dépôt."""

    # Construit le chemin du script train pour ce module
    train_path = Path(__file__).resolve().with_name("train.py")
    # Construit la spec d'import depuis le fichier local
    spec = importlib.util.spec_from_file_location("scripts.train", train_path)
    # Refuse la spec vide pour éviter un import silencieux
    if spec is None or spec.loader is None:
        # Signale l'échec de chargement pour diagnostiquer rapidement
        raise ImportError(f"Impossible de charger {train_path}")
    # Construit un module vierge à partir de la spec
    module = importlib.util.module_from_spec(spec)
    # Exécute le module pour charger ses symboles
    spec.loader.exec_module(module)
    # Retourne le module prêt à l'emploi
    return module


# Expose l'entraînement programmatique pour générer un modèle manquant
train_module: Any = _load_train_module()

# Définit le nom de la variable d'environnement pour la racine dataset
DATA_DIR_ENV_VAR = "EEGMMIDB_DATA_DIR"

# Définit le volume attendu des données EEG brutes (trials, canaux, temps)
EXPECTED_FEATURES_DIMENSIONS = 3

# Définit le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path(os.environ.get(DATA_DIR_ENV_VAR, "data")).expanduser()

# Définit le répertoire par défaut pour récupérer les artefacts
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Définit le répertoire par défaut pour les fichiers EDF bruts
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR

# Définit la référence EEG par défaut pour le re-référencement
DEFAULT_EEG_REFERENCE = "average"

# Définit la durée de fenêtre glissante en secondes pour la stratégie Welch
DEFAULT_WELCH_WINDOW_SECONDS = 1.0
# Définit l'overlap des fenêtres glissantes pour la stratégie Welch
DEFAULT_WELCH_WINDOW_OVERLAP = 0.5
# Définit l'agrégation probabiliste par défaut pour la stratégie Welch
DEFAULT_WELCH_AGGREGATION = "mean_proba"


# Regroupe les paramètres de fenêtrage Welch en prédiction
@dataclass
class WelchWindowConfig:
    """Paramètres de fenêtrage Welch pour l'inférence."""

    # Stocke la durée de fenêtre en secondes
    window_seconds: float
    # Stocke l'overlap entre fenêtres successives
    overlap: float
    # Stocke la stratégie d'agrégation par fenêtre
    aggregation: str


# Extrait le premier extracteur de features d'une pipeline scikit-learn
def _find_feature_extractor(pipeline: object) -> ExtractFeatures | None:
    """Retourne l'extracteur de features Welch depuis une pipeline."""

    # Récupère les étapes d'une pipeline scikit-learn si disponibles
    steps = getattr(pipeline, "steps", [])
    # Parcourt les étapes pour identifier l'extracteur de features
    for _name, step in steps:
        # Retourne l'extracteur lorsqu'il est détecté
        if isinstance(step, ExtractFeatures):
            return step
    # Retourne None si aucune étape de features n'est trouvée
    return None


# Normalise une stratégie de features en ensemble de libellés simples
def _normalize_feature_strategy(strategy: str | Iterable[str]) -> set[str]:
    """Normalise une stratégie de features en set de labels."""

    # Supporte une stratégie unique exprimée en chaîne
    if isinstance(strategy, str):
        # Normalise la casse et l'espace pour faciliter les comparaisons
        return {strategy.strip().lower()}
    # Normalise une stratégie multi-entries via un set de labels
    return {str(item).strip().lower() for item in strategy}


# Agrège des prédictions issues de fenêtres glissantes Welch
def _predict_with_welch_windows(
    pipeline: Any,
    X: np.ndarray,
    sfreq: float,
    window_config: WelchWindowConfig,
) -> np.ndarray:
    """Prédit des labels en agrégeant des fenêtres Welch."""

    # Normalise la fréquence d'échantillonnage pour le découpage glissant
    resolved_sfreq = float(sfreq)
    # Normalise la durée de fenêtre pour un découpage cohérent
    resolved_window = float(window_config.window_seconds)
    # Normalise l'overlap pour éviter des pas irréguliers
    resolved_overlap = float(window_config.overlap)
    # Alias la fonction pour limiter la longueur des lignes de découpage
    build_windows = build_sliding_windows
    # Applique le découpage glissant pour enrichir l'analyse temporelle
    windowed = build_windows(X, resolved_sfreq, resolved_window, resolved_overlap)
    # Récupère les dimensions pour replier ensuite les fenêtres
    n_epochs, n_windows, n_channels, n_times = windowed.shape
    # Aplatie les fenêtres pour une prédiction batch via scikit-learn
    flattened = windowed.reshape(n_epochs * n_windows, n_channels, n_times)

    # Prédit des probabilités si le classifieur le permet
    if hasattr(pipeline, "predict_proba"):
        # Récupère les probabilités par fenêtre pour chaque classe
        proba = pipeline.predict_proba(flattened)
        # Recompose en (epochs, fenêtres, classes) pour l'agrégation
        proba = proba.reshape(n_epochs, n_windows, -1)
        # Agrège les probabilités selon la stratégie choisie
        aggregated = aggregate_window_probabilities(proba, window_config.aggregation)
        # Récupère les classes connues du pipeline
        classes = getattr(pipeline, "classes_", None)
        # Refuse l'absence de classes pour éviter un indexage invalide
        if classes is None:
            # Signale l'absence d'attribut classes_ pour l'agrégation
            raise ValueError("Pipeline has no classes_ attribute for aggregation.")
        # Normalise les classes en ndarray pour un indexage typé
        class_labels = cast(np.ndarray, classes)
        # Sélectionne l'index maximal pour chaque epoch
        predicted_indices = np.argmax(aggregated, axis=1)
        # Convertit les indices en labels réels
        predicted_labels = cast(np.ndarray, class_labels[predicted_indices])
        # Retourne les labels prédits avec un type numpy stable
        return predicted_labels

    # Prédit des labels par fenêtre si les probabilités sont indisponibles
    window_predictions = pipeline.predict(flattened)
    # Recompose les labels fenêtre par fenêtre pour chaque epoch
    window_predictions = window_predictions.reshape(n_epochs, n_windows)
    # Récupère les classes connues pour construire les votes
    classes = getattr(pipeline, "classes_", None)
    # Refuse l'absence de classes pour conserver un mapping stable
    if classes is None:
        # Signale l'absence d'attribut classes_ pour l'agrégation
        raise ValueError("Pipeline has no classes_ attribute for aggregation.")
    # Normalise les classes en ndarray pour un indexage typé
    class_labels = cast(np.ndarray, classes)
    # Construit un mapping label -> index pour encoder les votes
    class_index = {label: idx for idx, label in enumerate(class_labels)}
    # Prépare un tenseur de votes one-hot pour chaque fenêtre
    proba = np.zeros((n_epochs, n_windows, len(classes)), dtype=float)
    # Parcourt les epochs pour encoder les votes par fenêtre
    for epoch_index in range(n_epochs):
        # Parcourt les fenêtres de l'epoch courant
        for window_index in range(n_windows):
            # Récupère le label prédit pour cette fenêtre
            label = window_predictions[epoch_index, window_index]
            # Convertit le label en index de classe stable
            label_index = class_index[label]
            # Déclare un vote unitaire pour ce label
            proba[epoch_index, window_index, label_index] = 1.0
    # Agrège les votes par epoch pour obtenir une pseudo-probabilité
    aggregated = aggregate_window_probabilities(proba, "majority_vote")
    # Sélectionne l'index maximal pour chaque epoch
    predicted_indices = np.argmax(aggregated, axis=1)
    # Convertit les indices en labels réels
    predicted_labels = cast(np.ndarray, class_labels[predicted_indices])
    # Retourne les labels prédits avec un type numpy stable
    return predicted_labels


# Regroupe les options de prédiction et d'auto-train
@dataclass
class PredictionOptions:
    """Regroupe les chemins et overrides pour l'évaluation."""

    # Stocke le répertoire des fichiers EDF bruts à utiliser
    raw_dir: Path = DEFAULT_RAW_DIR
    # Stocke la référence EEG à appliquer au chargement EDF
    eeg_reference: str | None = DEFAULT_EEG_REFERENCE
    # Stocke la configuration de prétraitement pour filtrage/normalisation
    preprocess_config: preprocessing.PreprocessingConfig = field(
        # Utilise une factory pour éviter le partage d'instance
        default_factory=preprocessing.PreprocessingConfig
    )
    # Stocke les overrides de pipeline pour l'auto-train éventuel
    pipeline_overrides: Mapping[str, str] | None = None


# Regroupe les chemins et réglages nécessaires à la génération des numpy
@dataclass
class NpyBuildContext:
    """Encapsule les paramètres nécessaires à la génération des .npy."""

    # Spécifie le répertoire contenant les données numpy
    data_dir: Path
    # Spécifie le répertoire des enregistrements EDF bruts
    raw_dir: Path
    # Définit la référence EEG à appliquer lors du chargement EDF
    eeg_reference: str | None
    # Regroupe les réglages de filtrage et de normalisation
    preprocess_config: preprocessing.PreprocessingConfig


# Regroupe les overrides résolus pour l'auto-train
@dataclass
class ResolvedOverrides:
    """Expose une configuration de pipeline prête à l'emploi."""

    # Stocke la stratégie de features effectivement utilisée
    feature_strategy: str
    # Stocke la méthode de réduction effectivement utilisée
    dim_method: str
    # Stocke le classifieur effectivement utilisé
    classifier: str
    # Stocke le scaler effectivement utilisé
    scaler: str | None


# Définit un seuil max de pic-à-pic pour rejeter les artefacts (en Volts)
DEFAULT_MAX_PEAK_TO_PEAK = 200e-6


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


# Normalise un identifiant de sujet pour la CLI de prédiction
def _parse_subject(value: str) -> str:
    """Normalise un identifiant de sujet en format Sxxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="S", width=3, label="Sujet")


# Normalise un identifiant de run pour la CLI de prédiction
def _parse_run(value: str) -> str:
    """Normalise un identifiant de run en format Rxx."""

    # Délègue la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="R", width=2, label="Run")


# Normalise la référence EEG demandée via CLI
def _parse_eeg_reference(value: str) -> str | None:
    """Retourne la référence EEG normalisée ou None."""

    # Nettoie la valeur reçue pour éviter les espaces parasites
    cleaned_value = value.strip()
    # Refuse une valeur vide pour éviter une référence ambiguë
    if not cleaned_value:
        # Signale une référence vide pour guider l'utilisateur
        raise argparse.ArgumentTypeError("Référence EEG vide")
    # Interprète l'alias "none" comme une désactivation explicite
    if cleaned_value.lower() == "none":
        # Retourne None pour indiquer l'absence de re-référencement
        return None
    # Retourne la valeur brute pour passer à MNE
    return cleaned_value


# Construit un argument parser aligné sur l'appel mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la prédiction TPV."""

    # Crée le parser avec description explicite pour l'utilisateur
    parser = argparse.ArgumentParser(
        description="Charge une pipeline TPV entraînée et produit un rapport",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument(
        "subject",
        type=_parse_subject,
        help="Identifiant du sujet (ex: 4)",
    )
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument(
        "run",
        type=_parse_run,
        help="Identifiant du run (ex: 14)",
    )

    # ------------------------------------------------------------------
    # Options de compatibilité avec la CLI mybci (train/predict)
    # Ces options sont acceptées mais *ignorées* côté prédiction, car
    # le modèle déjà entraîné porte la vraie configuration.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm", "centroid"),
        default="lda",
        help="Classifieur final (ignoré en prédiction, pour compatibilité CLI)",
    )
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "none"),
        default="none",
        help="Scaler appliqué en entraînement (ignoré en prédiction)",
    )
    parser.add_argument(
        "--feature-strategy",
        # Aligne les choix sur ceux de scripts.train pour les alias
        choices=train_module.FEATURE_STRATEGY_CHOICES,
        # Conserve FFT par défaut pour rester aligné sur l'entraînement
        default="fft",
        # Décrit la stratégie attendue pour compatibilité CLI
        help="Stratégie de features utilisée à l'entraînement (ignorée ici)",
    )
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp", "cssp", "svd"),
        default="pca",
        help="Méthode de réduction de dimension (ignorée en prédiction)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=argparse.SUPPRESS,
        help="Nombre de composantes (ignoré en prédiction)",
    )
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Flag de normalisation (ignoré en prédiction)",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence utilisée en features (ignorée ici)",
    )

    # ------------------------------------------------------------------
    # Options réellement utilisées par scripts/predict
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Répertoire racine contenant les fichiers numpy",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Répertoire racine où lire le modèle",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Répertoire racine contenant les fichiers EDF bruts",
    )
    # Ajoute l'option de re-référencement EEG lors du chargement EDF
    parser.add_argument(
        "--eeg-reference",
        type=_parse_eeg_reference,
        default=DEFAULT_EEG_REFERENCE,
        help="Référence EEG appliquée au chargement (ex: average, none)",
    )
    # Ajoute la borne basse du filtre passe-bande MI
    parser.add_argument(
        # Déclare le nom du flag CLI pour la borne basse
        "--bandpass-low",
        # Type float pour accepter des fréquences décimales
        type=float,
        # Fixe la valeur par défaut alignée sur la bande MI
        default=preprocessing.DEFAULT_BANDPASS_BAND[0],
        # Décrit l'usage utilisateur pour éviter les confusions
        help="Fréquence basse du passe-bande MI (ex: 8.0)",
    )
    # Ajoute la borne haute du filtre passe-bande MI
    parser.add_argument(
        # Déclare le nom du flag CLI pour la borne haute
        "--bandpass-high",
        # Type float pour accepter des fréquences décimales
        type=float,
        # Fixe la valeur par défaut alignée sur la bande MI
        default=preprocessing.DEFAULT_BANDPASS_BAND[1],
        # Décrit l'usage utilisateur pour éviter les confusions
        help="Fréquence haute du passe-bande MI (ex: 30.0)",
    )
    # Ajoute la fréquence de notch pour supprimer le bruit secteur
    parser.add_argument(
        # Déclare le nom du flag CLI pour le notch
        "--notch-freq",
        # Type float pour autoriser 50 ou 60 Hz
        type=float,
        # Fixe la valeur par défaut compatible Europe
        default=preprocessing.DEFAULT_NOTCH_FREQ,
        # Décrit l'usage utilisateur pour éviter les confusions
        help="Fréquence de notch pour le bruit secteur (50 ou 60 Hz)",
    )
    # Ajoute la méthode de normalisation par canal des epochs
    parser.add_argument(
        # Déclare le nom du flag CLI pour la normalisation canal
        "--normalize-channels",
        # Liste les méthodes acceptées pour sécuriser les entrées
        choices=("zscore", "robust", "none"),
        # Fixe la méthode par défaut alignée sur preprocessing
        default=preprocessing.DEFAULT_NORMALIZE_METHOD,
        # Décrit l'usage utilisateur pour éviter les confusions
        help="Normalisation par canal appliquée aux epochs (zscore/robust/none)",
    )
    # Ajoute l'epsilon de stabilisation pour la normalisation
    parser.add_argument(
        # Déclare le nom du flag CLI pour l'epsilon
        "--normalize-epsilon",
        # Type float pour accepter des epsilon personnalisés
        type=float,
        # Fixe la valeur par défaut alignée sur preprocessing
        default=preprocessing.DEFAULT_NORMALIZE_EPSILON,
        # Décrit l'usage utilisateur pour éviter les confusions
        help="Epsilon de stabilité pour la normalisation par canal",
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
    """Retourne le chemin du JSON décrivant la fenêtre d'epochs."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Construit le chemin du fichier de fenêtre pour ce run
    window_path = base_dir / f"{run}_epoch_window.json"
    # Retourne le chemin du fichier de fenêtre
    return window_path


# Vérifie la présence des fichiers PhysioNet requis avant le chargement
# pour éviter un échec tardif pendant le parsing EDF
def _ensure_physionet_files_exist(
    subject: str,
    run: str,
    raw_path: Path,
    event_path: Path,
) -> None:
    """Valide que l'EDF et son fichier d'événement existent et sont non vides."""

    # Arrête l'exécution si l'EDF est introuvable ou vide
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        # Signale explicitement le chemin absent pour guider l'utilisateur
        raise FileNotFoundError(
            f"EDF introuvable pour {subject} {run}: {raw_path}. "
            "Lancez `make download_dataset` ou pointez --raw-dir vers "
            "un dataset EEGMMIDB complet."
        )
    # Arrête l'exécution si le fichier d'événements est absent ou vide
    if not event_path.exists() or event_path.stat().st_size == 0:
        # Signale explicitement l'absence du fichier .edf.event
        raise FileNotFoundError(
            f"Fichier événement introuvable pour {subject} {run}: {event_path}. "
            "Le dataset semble incomplet: relancez `make download_dataset` "
            f"ou définissez {DATA_DIR_ENV_VAR} vers un dossier valide."
        )


# Applique le contrôle qualité des epochs avec un fallback dédié
# pour préserver la prédiction quand le QC supprime une classe
def _summarize_epochs_with_missing_labels_fallback(
    epochs: Any,
    motor_labels: list[str],
    subject: str,
    run: str,
) -> tuple[Any, list[str]]:
    """Retourne les epochs nettoyés ou la version brute si labels manquants."""

    # Applique un rejet d'artefacts pour limiter les essais aberrants
    try:
        # Utilise le filtrage par qualité pour supprimer les segments cassés
        cleaned_epochs, _report, cleaned_labels = preprocessing.summarize_epoch_quality(
            epochs,
            motor_labels,
            (subject, run),
            max_peak_to_peak=DEFAULT_MAX_PEAK_TO_PEAK,
        )
    except ValueError as error:
        # Détecte l'absence de classe après filtrage pour éviter un crash
        if "Missing labels" not in str(error):
            # Relance l'erreur originale si elle ne concerne pas les labels
            raise
        # Conserve les epochs filtrées par annotations uniquement
        cleaned_epochs = epochs
        # Conserve les labels moteurs initiaux pour aligner les données
        cleaned_labels = motor_labels
    # Retourne les sorties d'epochs retenues pour la suite de la pipeline
    return cleaned_epochs, cleaned_labels


# Charge la fenêtre d'epochs persistée par l'entraînement
def _read_epoch_window_metadata(
    subject: str,
    run: str,
    data_dir: Path,
) -> tuple[float, float]:
    """Retourne la fenêtre d'epochs à utiliser en prédiction."""

    # Construit le chemin du fichier de fenêtre pour ce run
    window_path = _resolve_epoch_window_path(subject, run, data_dir)
    # Utilise la fenêtre par défaut si aucun fichier n'est trouvé
    if not window_path.exists():
        return DEFAULT_EPOCH_WINDOW
    # Charge le contenu JSON pour récupérer la fenêtre
    payload = json.loads(window_path.read_text())
    # Extrait tmin du JSON en float
    tmin = float(payload.get("tmin", DEFAULT_EPOCH_WINDOW[0]))
    # Extrait tmax du JSON en float
    tmax = float(payload.get("tmax", DEFAULT_EPOCH_WINDOW[1]))
    # Retourne la fenêtre reconstruite
    return (tmin, tmax)


# Construit des matrices numpy à partir d'un EDF lorsque nécessaire
def _build_npy_from_edf(
    subject: str,
    run: str,
    build_context: NpyBuildContext,
) -> tuple[Path, Path]:
    """Génère X (epochs brutes) et y depuis un fichier EDF Physionet."""

    # Calcule les chemins cibles pour les fichiers numpy
    features_path, labels_path = _resolve_data_paths(
        # Transmet l'identifiant de sujet pour les chemins
        subject,
        # Transmet l'identifiant de run pour les chemins
        run,
        # Transmet le répertoire de base des numpy
        build_context.data_dir,
    )
    # Calcule les chemins attendus des fichiers bruts PhysioNet
    raw_path = build_context.raw_dir / subject / f"{subject}{run}.edf"
    event_path = raw_path.with_suffix(".edf.event")
    # Valide les fichiers bruts PhysioNet avant tout traitement de signal
    _ensure_physionet_files_exist(subject, run, raw_path, event_path)
    # Crée l'arborescence cible pour déposer les .npy
    features_path.parent.mkdir(parents=True, exist_ok=True)
    # Charge l'EDF en conservant les métadonnées essentielles
    raw, _ = preprocessing.load_physionet_raw(
        # Transmet le chemin EDF brut
        raw_path,
        # Transmet la référence EEG configurée
        reference=build_context.eeg_reference,
    )
    # Résout la fenêtre d'epochs alignée avec l'entraînement
    epoch_window = _read_epoch_window_metadata(
        # Transmet l'identifiant de sujet pour le chemin JSON
        subject,
        # Transmet l'identifiant de run pour le chemin JSON
        run,
        # Transmet le répertoire de base des numpy
        build_context.data_dir,
    )
    # Applique un notch si une fréquence valide est fournie
    if build_context.preprocess_config.notch_freq > 0.0:
        # Applique le notch pour supprimer la pollution secteur
        notched_raw = preprocessing.apply_notch_filter(
            # Transmet le signal brut chargé
            raw,
            # Transmet la fréquence de notch configurée
            freq=build_context.preprocess_config.notch_freq,
        )
    else:
        # Conserve le signal brut si le notch est désactivé
        notched_raw = raw
    # Applique le filtrage bande-passante pour stabiliser les bandes MI
    filtered_raw = preprocessing.apply_bandpass_filter(
        # Transmet le signal après notch (ou brut si désactivé)
        notched_raw,
        # Transmet la bande passante configurée pour la MI
        freq_band=build_context.preprocess_config.bandpass_band,
    )
    # Mappe les annotations en événements moteurs après filtrage
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(
        filtered_raw
    )
    # Découpe le signal filtré en epochs exploitables
    epochs = preprocessing.create_epochs_from_raw(
        filtered_raw,
        events,
        event_id,
        tmin=epoch_window[0],
        tmax=epoch_window[1],
    )
    # Sécurise le QC avec un fallback lorsque les labels manquent
    cleaned_epochs, cleaned_labels = _summarize_epochs_with_missing_labels_fallback(
        # Transmet les epochs découpés pour le contrôle qualité
        epochs,
        # Transmet les labels moteurs associés aux epochs
        motor_labels,
        # Transmet le sujet pour enrichir les messages d'erreur
        subject,
        # Transmet le run pour enrichir les messages d'erreur
        run,
    )
    # Récupère les données brutes des epochs (n_trials, n_channels, n_times)
    epochs_data = cleaned_epochs.get_data(copy=True)
    # Normalise les epochs par canal si la méthode est activée
    if build_context.preprocess_config.normalize_method != "none":
        # Applique la normalisation par canal sur chaque epoch
        epochs_data = preprocessing.normalize_epoch_data(
            # Transmet les epochs nettoyés pour la normalisation
            epochs_data,
            # Transmet la méthode de normalisation choisie
            method=build_context.preprocess_config.normalize_method,
            # Transmet l'epsilon de stabilité pour éviter les divisions nulles
            epsilon=build_context.preprocess_config.normalize_epsilon,
        )
    # Définit un mapping stable label → entier
    label_mapping = {
        label: idx for idx, label in enumerate(sorted(set(cleaned_labels)))
    }
    # Convertit les labels symboliques en entiers
    numeric_labels = np.array([label_mapping[label] for label in cleaned_labels])
    # Persiste les epochs brutes pour déléguer l'extraction des features
    np.save(features_path, epochs_data)
    # Persiste les labels alignés
    np.save(labels_path, numeric_labels)
    # Retourne les chemins nouvellement générés
    return features_path, labels_path


# Charge ou génère les matrices numpy attendues pour la prédiction
def _load_data(
    subject: str,
    run: str,
    build_context: NpyBuildContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge ou construit les données et étiquettes pour un run."""

    # Détermine les chemins attendus pour les features et labels
    features_path, labels_path = _resolve_data_paths(
        # Transmet l'identifiant de sujet pour les chemins
        subject,
        # Transmet l'identifiant de run pour les chemins
        run,
        # Transmet le répertoire de base des numpy
        build_context.data_dir,
    )

    # Indique si nous devons régénérer les .npy
    needs_rebuild = False

    # Construit les .npy depuis l'EDF si l'un d'eux manque
    if not features_path.exists() or not labels_path.exists():
        # Force une reconstruction complète pour retrouver les tensors bruts
        needs_rebuild = True
    else:
        # Charge X en mmap pour inspecter la forme sans tout charger
        candidate_X = np.load(features_path, mmap_mode="r")
        # Charge y en mmap pour inspecter la longueur
        candidate_y = np.load(labels_path, mmap_mode="r")

        # Vérifie que X est bien un tenseur 3D attendu par la pipeline
        if candidate_X.ndim != EXPECTED_FEATURES_DIMENSIONS:
            # Relance la génération si l'ancien format tabulaire est détecté
            needs_rebuild = True
        # Vérifie l'alignement entre le nombre d'epochs et de labels
        elif candidate_X.shape[0] != candidate_y.shape[0]:
            # Relance la génération pour réaligner les données et labels
            needs_rebuild = True

    # Reconstruit les fichiers lorsque nécessaire
    if needs_rebuild:
        # Convertit l'EDF associé en fichiers numpy persistés
        features_path, labels_path = _build_npy_from_edf(
            # Transmet l'identifiant de sujet pour reconstruire les numpy
            subject,
            # Transmet l'identifiant de run pour reconstruire les numpy
            run,
            # Transmet le contexte de génération des numpy
            build_context,
        )

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour le scoring
    return X, y


# Restaure le réducteur de dimension pour valider l'artefact W
def _load_w_matrix(path: Path) -> TPVDimReducer:
    """Recharge la matrice W persistée lors de l'entraînement."""

    # Instancie un réducteur de dimension vierge pour recharger la matrice
    reducer = TPVDimReducer()
    # Charge le contenu sérialisé pour restaurer les attributs internes
    reducer.load(path)
    # Retourne le réducteur prêt pour une projection éventuelle
    return reducer


# Normalise un label pour l'écriture des rapports et l'affichage CLI
def _stringify_label(label: object) -> str:
    """Retourne une version textuelle stable d'un label de classe."""

    # Garantit un rendu stable pour les entiers numpy ou Python
    if isinstance(label, (np.integer, int)):
        # Conserve un affichage entier pour les classes numériques
        return str(int(label))
    # Convertit les floats qui représentent un entier logique
    if isinstance(label, (np.floating, float)) and float(label).is_integer():
        # Évite d'afficher un suffixe .0 dans les rapports
        return str(int(label))
    # Préserve les labels symboliques pour le reporting lisible
    return str(label)


# Sérialise les rapports JSON et CSV pour un run donné
def _write_reports(
    target_dir: Path,
    identifiers: dict[str, str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
) -> dict:
    """Écrit les rapports de prédiction et retourne les chemins créés."""

    # Agrège les classes vues et prédites pour éviter les labels manquants
    labels = sorted(np.unique(np.concatenate((y_true, y_pred))).tolist())
    # Calcule la matrice de confusion en préservant l'ordre des labels
    confusion_array = confusion_matrix(y_true, y_pred, labels=labels)
    # Convertit la matrice en liste pour la sérialisation JSON
    confusion = confusion_array.tolist()
    # Prépare la structure d'accuracy par classe pour la CLI
    per_class_accuracy: dict[str, float] = {}
    # Calcule l'accuracy pour chaque classe en utilisant la diagonale
    for index, label in enumerate(labels):
        # Calcule le nombre total d'échantillons pour la classe courante
        class_total = int(confusion_array[index].sum())
        # Calcule le nombre de prédictions correctes pour la classe courante
        correct = int(confusion_array[index][index])
        # Enregistre l'accuracy en évitant la division par zéro
        per_class_accuracy[str(label)] = correct / class_total if class_total else 0.0
    # Prépare un rapport JSON synthétique pour la CLI et la CI
    report = {
        "subject": identifiers["subject"],
        "run": identifiers["run"],
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy,
        "samples": len(y_true),
    }
    # Définit le chemin du rapport JSON dans les artefacts
    report_path = target_dir / "report.json"
    # Écrit le rapport JSON en UTF-8 avec indentation pour inspection
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # Définit le chemin du rapport CSV par classe pour les diagnostics
    class_report_path = target_dir / "class_report.csv"
    # Ouvre le fichier CSV pour enregistrer accuracy et support
    with class_report_path.open("w", newline="") as handle:
        # Définit les en-têtes pour l'accuracy par classe
        fieldnames = ["class", "accuracy", "support"]
        # Construit le writer CSV prêt à écrire chaque classe
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Inscrit les en-têtes pour faciliter la lecture
        writer.writeheader()
        # Parcourt chaque classe pour détailler les métriques associées
        for label in labels:
            # Récupère l'accuracy calculée pour la classe ciblée
            class_accuracy = per_class_accuracy[str(label)]
            # Calcule le support en comptant les occurrences de la classe
            support = int(confusion_array[labels.index(label)].sum())
            # Écrit la ligne de métriques dédiée à la classe
            writer.writerow(
                {
                    "class": label,
                    "accuracy": class_accuracy,
                    "support": support,
                }
            )
    # Définit le chemin du CSV listant chaque prédiction
    csv_path = target_dir / "predictions.csv"
    # Ouvre le fichier CSV en écriture sans lignes superflues
    with csv_path.open("w", newline="") as handle:
        # Prépare les en-têtes pour aligner vérité terrain et prédiction
        fieldnames = ["subject", "run", "index", "y_true", "y_pred"]
        # Crée un writer dict pour simplifier l'écriture ligne par ligne
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Inscrit l'en-tête pour faciliter la lecture
        writer.writeheader()
        # Parcourt chaque échantillon pour exposer la prédiction individuelle
        for idx, (true_label, pred_label) in enumerate(
            zip(y_true, y_pred, strict=False)
        ):
            # Normalise la valeur de vérité pour garantir un CSV lisible
            true_value = _stringify_label(true_label)
            # Normalise la prédiction pour conserver les classes symboliques
            pred_value = _stringify_label(pred_label)
            # Écrit la ligne CSV pour l'index courant
            writer.writerow(
                {
                    "subject": identifiers["subject"],
                    "run": identifiers["run"],
                    "index": idx,
                    "y_true": true_value,
                    "y_pred": pred_value,
                }
            )
    # Retourne les chemins créés pour validation amont
    return {
        "json_report": report_path,
        "csv_report": csv_path,
        "class_report": class_report_path,
        "confusion": confusion,
        "per_class_accuracy": per_class_accuracy,
    }


# Résout un alias de features vers une méthode de réduction
def _resolve_feature_strategy_alias(
    feature_strategy: str,
    dim_method: str,
    dim_method_explicit: bool,
) -> tuple[str, str] | None:
    """Résout un alias de features en méthode de réduction."""

    # Quitte si aucune stratégie alias n'est demandée
    if feature_strategy not in train_module.FEATURE_STRATEGY_ALIASES:
        # Indique l'absence d'alias à résoudre
        return None
    # Force la stratégie FFT pour conserver une extraction valide
    resolved_feature_strategy = "fft"
    # Conserve la méthode explicite si elle est fournie
    if dim_method_explicit:
        # Informe que l'alias est ignoré au profit du dim_method explicite
        print(
            # Décrit la résolution de l'alias pour l'utilisateur
            "INFO: feature_strategy interprété comme alias de dim_method, "
            # Précise la conservation de la stratégie FFT
            "feature_strategy='fft' conservée car --dim-method explicite."
        )
        # Retourne la stratégie et la méthode explicite
        return resolved_feature_strategy, dim_method
    # Informe que l'alias est utilisé comme méthode de réduction
    print(
        # Décrit la résolution de l'alias pour l'utilisateur
        "INFO: feature_strategy interprété comme alias de dim_method, "
        # Précise la conservation de la stratégie FFT
        "feature_strategy='fft' appliquée."
    )
    # Retourne la stratégie FFT et l'alias comme méthode
    return resolved_feature_strategy, feature_strategy


# Ajuste la méthode de réduction lorsque les features sont tabulaires
def _adjust_dim_method_for_tabular_features(
    feature_strategy: str,
    dim_method: str,
    dim_method_explicit: bool,
) -> str:
    """Ajuste le dim_method lorsque CSP ignore les features."""

    # Ignore l'ajustement si la stratégie n'est pas spectrale
    if feature_strategy not in {"wavelet", "welch"}:
        # Retourne la méthode inchangée
        return dim_method
    # Retourne la méthode inchangée si CSP/CSSP est explicitement voulu
    if dim_method in {"csp", "cssp"}:
        # Informe sur l'usage de CSP avant l'extraction des features
        if not dim_method_explicit:
            # Documente la configuration pour la compatibilité CLI
            print(
                "INFO: dim_method='csp/cssp' appliqué avant "
                "l'extraction des features."
            )
        # Retourne la méthode CSP/CSSP pour Welch+CSP
        return dim_method
    # Retourne la méthode inchangée pour les autres cas
    return dim_method


# Résout les overrides de pipeline transmis à l'auto-train
def _resolve_pipeline_overrides(
    pipeline_overrides: Mapping[str, str] | None,
) -> ResolvedOverrides:
    """Résout les overrides CLI appliqués à l'auto-train."""

    # Fixe la stratégie de features par défaut pour l'auto-train
    feature_strategy = "fft"
    # Fixe la méthode de réduction par défaut pour l'auto-train
    dim_method = "csp"
    # Fixe le classifieur par défaut pour l'auto-train
    classifier = "lda"
    # Fixe l'absence de scaler par défaut pour l'auto-train
    scaler: str | None = None
    # Suit la présence explicite d'une méthode de réduction
    dim_method_explicit = False

    # Applique les overrides si la CLI en fournit
    if pipeline_overrides:
        # Applique la stratégie de features explicitement demandée
        if "feature_strategy" in pipeline_overrides:
            # Cast en str pour sécuriser les valeurs sérialisées
            feature_strategy = str(pipeline_overrides["feature_strategy"])
        # Applique la méthode de réduction explicitement demandée
        if "dim_method" in pipeline_overrides:
            # Cast en str pour sécuriser les valeurs sérialisées
            dim_method = str(pipeline_overrides["dim_method"])
            # Signale la présence explicite pour la logique d'alias
            dim_method_explicit = True
        # Applique le classifieur explicitement demandé
        if "classifier" in pipeline_overrides:
            # Cast en str pour sécuriser les valeurs sérialisées
            classifier = str(pipeline_overrides["classifier"])
        # Applique le scaler explicitement demandé
        if "scaler" in pipeline_overrides:
            # Cast en str pour sécuriser les valeurs sérialisées
            scaler_value = str(pipeline_overrides["scaler"])
            # Interprète "none" comme l'absence de scaler
            scaler = None if scaler_value == "none" else scaler_value

    # Résout l'alias éventuel de feature_strategy
    alias_resolution = _resolve_feature_strategy_alias(
        feature_strategy,
        dim_method,
        dim_method_explicit,
    )
    # Applique l'alias si présent
    if alias_resolution is not None:
        # Déstructure la stratégie et la méthode résolues
        feature_strategy, dim_method = alias_resolution
        # Retourne les overrides résolus
        return ResolvedOverrides(
            feature_strategy=feature_strategy,
            dim_method=dim_method,
            classifier=classifier,
            scaler=scaler,
        )

    # Ajuste la méthode de réduction pour les features tabulaires
    dim_method = _adjust_dim_method_for_tabular_features(
        feature_strategy,
        dim_method,
        dim_method_explicit,
    )
    # Retourne la configuration résolue pour l'auto-train
    return ResolvedOverrides(
        feature_strategy=feature_strategy,
        dim_method=dim_method,
        classifier=classifier,
        scaler=scaler,
    )


# Entraîne un modèle par défaut lorsqu'aucun artefact n'est disponible
def _train_missing_pipeline(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    # Transporte les options de prédiction pour l'auto-train
    options: PredictionOptions | None = None,
) -> None:
    """Construit un pipeline CSP/LDA shrinkage lorsque le modèle manque."""

    # Applique les options par défaut si aucune n'est fournie
    resolved_options = options or PredictionOptions()
    # Extrait le répertoire EDF depuis les options
    raw_dir = resolved_options.raw_dir
    # Extrait la référence EEG depuis les options
    eeg_reference = resolved_options.eeg_reference
    # Extrait la configuration de prétraitement depuis les options
    preprocess_config = resolved_options.preprocess_config
    # Extrait les overrides de pipeline depuis les options
    pipeline_overrides = resolved_options.pipeline_overrides
    # Résout la fréquence d'échantillonnage à partir de l'EDF si disponible
    resolved_sfreq = train_module.resolve_sampling_rate(
        subject,
        run,
        raw_dir,
        train_module.DEFAULT_SAMPLING_RATE,
        eeg_reference,
    )
    # Résout les overrides CLI pour calibrer l'auto-train
    resolved_overrides = _resolve_pipeline_overrides(pipeline_overrides)
    # Utilise la fréquence de référence pour aligner extraction et entraînement
    pipeline_config = PipelineConfig(
        # Propage la fréquence d'échantillonnage résolue
        sfreq=resolved_sfreq,
        # Propage la stratégie de features résolue
        feature_strategy=resolved_overrides.feature_strategy,
        # Conserve la normalisation pour stabiliser les features
        normalize_features=True,
        # Propage la méthode de réduction résolue
        dim_method=resolved_overrides.dim_method,
        # Conserve un nombre de composantes robuste pour CSP/PCA
        n_components=train_module.DEFAULT_CSP_COMPONENTS,
        # Propage le classifieur résolu
        classifier=resolved_overrides.classifier,
        # Propage le scaler résolu
        scaler=resolved_overrides.scaler,
        # Conserve la régularisation CSP pour stabiliser les covariances
        csp_regularization=0.1,
    )
    # Prépare la requête pour déléguer l'entraînement à scripts.train
    request = train_module.TrainingRequest(
        # Renseigne le sujet pour associer les artefacts au bon dossier
        subject=subject,
        # Renseigne le run pour distinguer les expérimentations
        run=run,
        # Transmet la configuration du pipeline de base
        pipeline_config=pipeline_config,
        # Utilise le répertoire de données fourni par l'appelant
        data_dir=data_dir,
        # Utilise le répertoire d'artefacts pour stocker le modèle
        artifacts_dir=artifacts_dir,
        # Transmet le répertoire des EDF bruts pour les métadonnées
        raw_dir=raw_dir,
        # Transmet la référence EEG pour aligner train et predict
        eeg_reference=eeg_reference,
        # Transmet la configuration de prétraitement
        preprocess_config=preprocess_config,
        # Désactive la recherche exhaustive pour accélérer l'auto-train
        enable_grid_search=False,
        # Fixe un nombre de splits raisonnable si la recherche est réactivée
        grid_search_splits=5,
    )
    # Lance l'entraînement pour matérialiser model.joblib et w_matrix.joblib
    train_module.run_training(request)


# Évalue un run donné et produit un rapport structuré
def evaluate_run(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    # Transporte les options de prédiction pour l'auto-train
    options: PredictionOptions | None = None,
) -> dict:
    """Évalue l'accuracy d'un run en rechargeant le pipeline entraîné."""

    # Applique les options par défaut si aucune n'est fournie
    resolved_options = options or PredictionOptions()
    # Extrait le répertoire EDF depuis les options
    raw_dir = resolved_options.raw_dir
    # Extrait la référence EEG depuis les options
    eeg_reference = resolved_options.eeg_reference
    # Extrait la configuration de prétraitement depuis les options
    preprocess_config = resolved_options.preprocess_config
    # Construit le contexte de génération des numpy pour ce run
    build_context = NpyBuildContext(
        # Transmet le répertoire des numpy
        data_dir=data_dir,
        # Transmet le répertoire des EDF bruts
        raw_dir=raw_dir,
        # Transmet la référence EEG configurée
        eeg_reference=eeg_reference,
        # Transmet la configuration de prétraitement
        preprocess_config=preprocess_config,
    )
    # Charge ou génère les tableaux numpy nécessaires au scoring
    X, y = _load_data(
        # Transmet l'identifiant de sujet pour le chargement
        subject,
        # Transmet l'identifiant de run pour le chargement
        run,
        # Transmet le contexte de génération des numpy
        build_context,
    )
    # Construit le dossier d'artefacts spécifique au sujet et au run
    target_dir = artifacts_dir / subject / run
    # Assure la présence du dossier pour pouvoir écrire les rapports
    target_dir.mkdir(parents=True, exist_ok=True)
    # Calcule les chemins des artefacts attendus pour détecter les absences
    model_path = target_dir / "model.joblib"
    # Vérifie la présence de la matrice W utilisée par le temps-réel
    w_matrix_path = target_dir / "w_matrix.joblib"
    # Déclenche un entraînement si le modèle ou la matrice sont manquants
    if not model_path.exists() or not w_matrix_path.exists():
        # Informe l'utilisateur que l'auto-train est lancé faute d'artefacts
        print(
            f"INFO: modèle absent pour {subject} {run}, "
            "entraînement automatique en cours..."
        )
        # Génère les artefacts de base pour permettre l'évaluation
        _train_missing_pipeline(
            # Relaye le sujet pour matérialiser les artefacts
            subject,
            # Relaye le run pour isoler l'expérience
            run,
            # Relaye le répertoire de données numpy
            data_dir,
            # Relaye le répertoire d'artefacts pour l'écriture
            artifacts_dir,
            # Relaye les options de prédiction pour l'auto-train
            resolved_options,
        )
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(model_path))
    # Récupère l'extracteur de features pour détecter la stratégie Welch
    extractor = _find_feature_extractor(pipeline)
    # Initialise la stratégie normalisée pour la comparaison
    normalized_strategy: set[str] = set()
    # Normalise la stratégie si un extracteur est présent
    if extractor is not None:
        # Normalise la stratégie déclarée pour simplifier la détection
        normalized_strategy = _normalize_feature_strategy(extractor.feature_strategy)
    # Détecte si la stratégie Welch est active dans la pipeline
    uses_welch = "welch" in normalized_strategy

    # Applique la logique glissante Welch uniquement pour des epochs bruts
    if uses_welch and X.ndim == EXPECTED_FEATURES_DIMENSIONS and extractor is not None:
        # Extrait la fréquence d'échantillonnage depuis l'extracteur
        sfreq = float(extractor.sfreq)
        # Capture la durée de fenêtre par défaut pour l'inférence Welch
        window_seconds = DEFAULT_WELCH_WINDOW_SECONDS
        # Capture l'overlap par défaut pour l'inférence Welch
        overlap = DEFAULT_WELCH_WINDOW_OVERLAP
        # Capture la stratégie d'agrégation par défaut pour l'inférence Welch
        aggregation = DEFAULT_WELCH_AGGREGATION
        # Regroupe les paramètres de fenêtre pour la prédiction
        window_config = WelchWindowConfig(window_seconds, overlap, aggregation)
        # Alias la fonction pour limiter la longueur des lignes de prédiction
        predict_with_windows = _predict_with_welch_windows
        # Prépare les arguments de prédiction pour éviter une ligne trop longue
        predict_args = (pipeline, X, sfreq, window_config)
        # Prédit via fenêtres Welch et agrégation probabiliste
        y_pred = predict_with_windows(*predict_args)
        # Calcule l'accuracy à partir des labels agrégés
        accuracy = float(np.mean(y_pred == y))
    else:
        # Génère les prédictions individuelles pour le rapport
        y_pred = pipeline.predict(X)
        # Calcule l'accuracy du pipeline sur les données fournies
        accuracy = float(pipeline.score(X, y))
    # Recharge la matrice W pour confirmer sa présence
    w_matrix = _load_w_matrix(w_matrix_path)
    # Écrit les rapports JSON et CSV dans le dossier d'artefacts
    reports = _write_reports(
        target_dir,
        {"subject": subject, "run": run},
        y,
        y_pred,
        accuracy,
    )
    # Retourne le rapport local incluant la matrice pour les tests
    return {
        "run": run,
        "subject": subject,
        "accuracy": accuracy,
        "w_matrix": w_matrix,
        "reports": reports,
        "predictions": y_pred,
        "truth": y,  # <--- clé attendue par mybci
        "y_true": y,  # Aligne la clé avec les attentes des tests CLI
    }


# Construit un rapport agrégé par run, sujet et global
def build_report(result: dict) -> dict:
    """Structure les métriques d'accuracy pour exploitation CLI."""

    # Extrait l'accuracy du run pour construire l'agrégation
    accuracy = result["accuracy"]
    # Calcule l'accuracy par run sous forme de mapping
    by_run = {result["run"]: accuracy}
    # Calcule l'accuracy par sujet en regroupant le run unique
    by_subject = {result["subject"]: accuracy}
    # Calcule l'accuracy globale sur le run fourni
    global_accuracy = accuracy
    # Récupère la matrice de confusion construite lors de l'évaluation
    confusion = result["reports"]["confusion"]
    # Retourne une structure prête à être sérialisée
    return {
        "by_run": by_run,
        "by_subject": by_subject,
        "global": global_accuracy,
        "confusion_matrix": confusion,
        "reports": result["reports"],
    }


# Construit les overrides de pipeline depuis les arguments CLI
def _build_pipeline_overrides_from_args(
    args: argparse.Namespace,
) -> dict[str, str]:
    """Transforme les arguments CLI en overrides d'auto-train."""

    # Prépare un dictionnaire vide pour les overrides explicites
    overrides: dict[str, str] = {}
    # Enregistre la stratégie de features demandée en CLI
    overrides["feature_strategy"] = str(args.feature_strategy)
    # Enregistre la méthode de réduction demandée en CLI
    overrides["dim_method"] = str(args.dim_method)
    # Enregistre le classifieur demandé en CLI
    overrides["classifier"] = str(args.classifier)
    # Enregistre le scaler demandé en CLI
    overrides["scaler"] = str(args.scaler)
    # Retourne les overrides construits pour l'auto-train
    return overrides


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'évaluation."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Refuse une bande passante incohérente pour éviter un filtrage invalide
    if args.bandpass_low >= args.bandpass_high:
        # Informe l'utilisateur de l'erreur de configuration
        print("ERREUR: bandpass_low doit être inférieur à bandpass_high.")
        # Retourne un code d'erreur explicite pour la CLI
        return HANDLED_CLI_ERROR_EXIT_CODE
    # Construit la configuration de prétraitement à partir des arguments
    preprocess_config = preprocessing.PreprocessingConfig(
        # Définit la bande passante MI configurée
        bandpass_band=(args.bandpass_low, args.bandpass_high),
        # Définit la fréquence de notch configurée
        notch_freq=args.notch_freq,
        # Définit la méthode de normalisation par canal
        normalize_method=args.normalize_channels,
        # Définit l'epsilon de stabilité pour la normalisation
        normalize_epsilon=args.normalize_epsilon,
    )
    # Évalue le run demandé et récupère la matrice W
    # Construit les overrides de pipeline à partir des arguments CLI
    pipeline_overrides = _build_pipeline_overrides_from_args(args)
    # Construit les options de prédiction à partir des arguments CLI
    options = PredictionOptions(
        # Transmet le répertoire EDF brut pour l'auto-train
        raw_dir=args.raw_dir,
        # Transmet la référence EEG pour re-référencer au chargement
        eeg_reference=args.eeg_reference,
        # Transmet la configuration de prétraitement
        preprocess_config=preprocess_config,
        # Transmet les overrides pour l'auto-train en prédiction
        pipeline_overrides=pipeline_overrides,
    )
    # Évalue le run demandé et récupère la matrice W
    # Sécurise l'évaluation pour éviter un traceback brut côté CLI
    try:
        # Évalue le run demandé et récupère la matrice W
        result = evaluate_run(
            # Relaye l'identifiant de sujet fourni par la CLI
            args.subject,
            # Relaye l'identifiant de run fourni par la CLI
            args.run,
            # Relaye le répertoire des données numpy
            args.data_dir,
            # Relaye le répertoire d'artefacts
            args.artifacts_dir,
            # Relaye les options de prédiction construites
            options,
        )
    except (FileNotFoundError, PermissionError, ValueError) as error:
        # Affiche une erreur courte et actionnable sans traceback brut
        for line in render_cli_error_lines(
            error,
            subject=args.subject,
            run=args.run,
        ):
            print(line)
        # Retourne un code d'échec explicite pour la CLI
        return HANDLED_CLI_ERROR_EXIT_CODE
    # Construit le rapport structuré attendu par les tests
    _ = build_report(result)

    # Récupère les prédictions et la vérité terrain pour l'affichage CLI
    y_pred = result["predictions"]
    y_true = result["y_true"]

    # En-tête identique à la version de référence
    print("epoch nb: [prediction] [truth] equal?")

    # Affiche une ligne par epoch : numéro, prédiction, vérité, égalité
    for idx, (pred, true) in enumerate(zip(y_pred, y_true, strict=True)):
        # Normalise la prédiction pour l'affichage CLI
        pred_value = _stringify_label(pred)
        # Normalise la vérité terrain pour l'affichage CLI
        true_value = _stringify_label(true)
        # Compare les labels bruts pour garder la logique métier
        equal = bool(pred == true)
        print(f"epoch {idx:02d}: [{pred_value}] [{true_value}] {equal}")

    # Affiche l'accuracy avec 4 décimales comme dans l'exemple
    print(f"Accuracy: {result['accuracy']:.4f}")

    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
