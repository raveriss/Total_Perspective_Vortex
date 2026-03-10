"""CLI d'entraînement pour le pipeline TPV."""

# Pour exposer une CLI stable et testable pour le script.
import argparse

# Pour exporter un manifeste plat lisible dans un tableur.
import csv

# Pour sérialiser les métadonnées de run sans dépendance externe.
import json

# Pour lire une racine dataset configurable via l’environnement.
import os

# Pour distinguer proprement erreurs CLI et sorties métier.
import sys

# Pour regrouper la configuration sans multiplier les tuples fragiles.
from dataclasses import asdict, dataclass, field

# Pour manipuler les chemins sans dépendre du shell courant.
from pathlib import Path

# Pour garder un contrat de types exploitable par mypy.
from typing import Sequence, TypeAlias, cast

# Pour persister la pipeline complète sans dissocier ses étapes.
import joblib

# Pour porter les tableaux et validations numériques du pipeline.
import numpy as np

# Pour centraliser l’accès aux stratégies de CV via un seul import stable.
import sklearn.model_selection as sklearn_model_selection

# Pour proposer un classifieur simple et interprétable en baseline.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Pour disposer d’un fallback robuste sur petits effectifs.
from sklearn.linear_model import LogisticRegression

# Pour préserver un entraînement cohérent entre fit et predict.
from sklearn.pipeline import Pipeline

# Pour laisser le scaling configurable sans changer le pipeline.
from sklearn.preprocessing import RobustScaler, StandardScaler

# Pour fournir une alternative linéaire stable sur features EEG.
from sklearn.svm import LinearSVC

# Pour centraliser les helpers pipeline via un seul import compatible isort.
import tpv.pipeline as tpv_pipeline

# Pour centraliser les helpers de fenêtres via un seul import compatible isort.
import tpv.utils as tpv_utils

# Pour réutiliser une logique EEG factorisée entre train et predict.
from tpv import preprocessing

# Pour garder un classifieur très simple disponible en fallback.
from tpv.classifier import CentroidClassifier

# Pour centraliser la réduction de dimension maison du projet.
from tpv.dimensionality import TPVDimReducer

# Pour préserver une API de module stable pour les appels externes et les tests.
PipelineConfig: TypeAlias = tpv_pipeline.PipelineConfig

# Pour préserver une API de module stable pour les appels externes et les tests.
ShuffleSplit: TypeAlias = sklearn_model_selection.ShuffleSplit

# Pour préserver une API de module stable pour les appels externes et les tests.
StratifiedShuffleSplit: TypeAlias = sklearn_model_selection.StratifiedShuffleSplit

# Pour préserver le point d’entrée public attendu pour la recherche exhaustive.
GridSearchCV = sklearn_model_selection.GridSearchCV

# Pour préserver le point d’entrée public attendu pour la mesure de CV.
cross_val_score = sklearn_model_selection.cross_val_score

# Pour préserver le point d’entrée public attendu pour construire la pipeline.
build_pipeline = tpv_pipeline.build_pipeline

# Pour préserver le point d’entrée public attendu pour la recherche interne.
build_search_pipeline = tpv_pipeline.build_search_pipeline

# Pour préserver le point d’entrée public attendu pour la persistance du modèle.
save_pipeline = tpv_pipeline.save_pipeline

# Pour préserver le point d’entrée public attendu pour les fenêtres par sujet.
resolve_epoch_windows = tpv_utils.resolve_epoch_windows

# Pour préserver le code retour public attendu par la CLI et les tests.
HANDLED_CLI_ERROR_EXIT_CODE: int = int(tpv_utils.HANDLED_CLI_ERROR_EXIT_CODE)

# Pour fixer la liste des runs moteurs à couvrir pour l'entraînement massif
MOTOR_RUNS = (
    # Pour inclure le run moteur R03 documenté dans le protocole Physionet
    "R03",
    # Pour inclure le run moteur R04 documenté dans le protocole Physionet
    "R04",
    # Pour inclure le run moteur R05 documenté dans le protocole Physionet
    "R05",
    # Pour inclure le run moteur R06 documenté dans le protocole Physionet
    "R06",
    # Pour inclure le run moteur R07 documenté dans le protocole Physionet
    "R07",
    # Pour inclure le run moteur R08 documenté dans le protocole Physionet
    "R08",
    # Pour inclure le run moteur R09 documenté dans le protocole Physionet
    "R09",
    # Pour inclure le run moteur R10 documenté dans le protocole Physionet
    "R10",
    # Pour inclure le run moteur R11 documenté dans le protocole Physionet
    "R11",
    # Pour inclure le run moteur R12 documenté dans le protocole Physionet
    "R12",
    # Pour inclure le run moteur R13 documenté dans le protocole Physionet
    "R13",
    # Pour inclure le run moteur R14 documenté dans le protocole Physionet
    "R14",
)

# Pour fixer le nom de la variable d'environnement pour la racine dataset
DATA_DIR_ENV_VAR = "EEGMMIDB_DATA_DIR"

# Pour fixer le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path(os.environ.get(DATA_DIR_ENV_VAR, "data")).expanduser()

# Pour stabiliser la dimension attendue pour les matrices de features en mémoire
EXPECTED_FEATURES_DIMENSIONS = 3

# Pour fixer le répertoire par défaut pour déposer les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Pour fixer le répertoire par défaut où résident les fichiers EDF bruts
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR

# Pour fixer la référence EEG par défaut pour le re-référencement
DEFAULT_EEG_REFERENCE = "average"

# Pour figer la fréquence d'échantillonnage par défaut utilisée pour les features
DEFAULT_SAMPLING_RATE = 50.0

# Pour fixer un seuil max de pic-à-pic pour rejeter les artefacts (en Volts)
DEFAULT_MAX_PEAK_TO_PEAK = 3000e-6
# Pour stabiliser le nombre de composantes CSP pour la sélection de fenêtre
DEFAULT_CSP_COMPONENTS = 4
# Pour centraliser les stratégies de features supportées par la pipeline
FEATURE_STRATEGIES = ("fft", "welch", "wavelet")
# Pour centraliser les méthodes de réduction de dimension supportées par la pipeline
DIM_METHODS = ("pca", "csp", "cssp", "svd")
# Pour accepter un alias CLI pour rediriger vers la réduction de dimension
FEATURE_STRATEGY_ALIASES = DIM_METHODS
# Pour réunir les valeurs autorisées pour l'argument --feature-strategy
FEATURE_STRATEGY_CHOICES = FEATURE_STRATEGIES + FEATURE_STRATEGY_ALIASES


# Pour éviter d’exposer un global mutable directement au reste du module.
@dataclass
# Pour porter la config active sans exposer un global mutable nu.
class EpochWindowState:
    """Conteneur mutable pour la configuration des fenêtres d'epochs."""

    # Pour mémoriser la configuration active utilisée par les helpers
    config: tpv_utils.EpochWindowConfig


# Pour partager une config de fenêtres unique entre tous les helpers du module.
ACTIVE_EPOCH_WINDOW_CONFIG = EpochWindowState(
    # Pour charger le repli canonique dès l’initialisation du module.
    config=tpv_utils.default_epoch_window_config()
)


# Pour harmoniser un identifiant brut en appliquant un préfixe standard
def _normalize_identifier(value: str, prefix: str, width: int, label: str) -> str:
    """Normalise un identifiant pour respecter le format Physionet."""

    # Pour neutraliser la valeur reçue pour éviter des espaces parasites
    cleaned_value = value.strip()
    # Pour rejeter une valeur vide pour éviter un identifiant incomplet
    if not cleaned_value:
        # Pour rendre explicite une valeur vide pour forcer la correction côté CLI
        raise argparse.ArgumentTypeError(f"{label} vide")
    # Pour isoler le premier caractère pour détecter un préfixe explicite
    first_char = cleaned_value[0]
    # Pour distinguer si l'utilisateur a fourni le préfixe attendu
    has_prefix = first_char.upper() == prefix.upper()
    # Pour isoler la portion numérique selon la présence du préfixe
    numeric_part = cleaned_value[1:] if has_prefix else cleaned_value
    # Pour rejeter les valeurs non numériques pour garantir un ID valide
    if not numeric_part.isdigit():
        # Pour rendre explicite l'identifiant invalide pour guider l'utilisateur
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Pour supprimer les zéros parasites avant de reconstruire l’identifiant.
    numeric_value = int(numeric_part)
    # Pour rejeter les index non positifs pour respecter la base Physionet
    if numeric_value < 1:
        # Pour rendre explicite l'identifiant non valide pour arrêter le parsing
        raise argparse.ArgumentTypeError(f"{label} invalide: {value}")
    # Pour rétablir un identifiant canonique compatible avec PhysioNet.
    return f"{prefix}{numeric_value:0{width}d}"


# Pour harmoniser un identifiant de sujet pour la CLI d'entraînement
def _parse_subject(value: str) -> str:
    """Normalise un identifiant de sujet en format Sxxx."""

    # Pour réutiliser la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="S", width=3, label="Sujet")


# Pour harmoniser un identifiant de run pour la CLI d'entraînement
def _parse_run(value: str) -> str:
    """Normalise un identifiant de run en format Rxx."""

    # Pour réutiliser la normalisation au helper générique
    return _normalize_identifier(value=value, prefix="R", width=2, label="Run")


# Pour harmoniser la référence EEG demandée via CLI
def _parse_eeg_reference(value: str) -> str | None:
    """Retourne la référence EEG normalisée ou None."""

    # Pour neutraliser la valeur reçue pour éviter les espaces parasites
    cleaned_value = value.strip()
    # Pour rejeter une valeur vide pour éviter une référence ambiguë
    if not cleaned_value:
        # Pour rendre explicite une référence vide pour guider l'utilisateur
        raise argparse.ArgumentTypeError("Référence EEG vide")
    # Pour réserver l'alias "none" comme une désactivation explicite
    if cleaned_value.lower() == "none":
        # Pour garder None pour indiquer l'absence de re-référencement
        return None
    # Pour préserver les références EEG custom sans les restreindre artificiellement.
    return cleaned_value


# Pour harmoniser un choix CLI en minuscules pour les comparaisons
def _normalize_choice(value: str, label: str) -> str:
    """Normalise un choix CLI et refuse les valeurs vides."""

    # Pour neutraliser la valeur reçue pour éviter les espaces parasites
    cleaned_value = value.strip()
    # Pour rejeter une valeur vide pour éviter un choix ambigu
    if not cleaned_value:
        # Pour rendre explicite un choix vide pour guider l'utilisateur
        raise argparse.ArgumentTypeError(f"{label} vide")
    # Pour harmoniser en minuscules pour accepter des variantes de casse
    return cleaned_value.lower()


# Pour harmoniser une stratégie de features pour le parsing CLI
def _parse_feature_strategy(value: str) -> str:
    """Normalise la stratégie de features en minuscules."""

    # Pour réutiliser la normalisation au helper générique
    return _normalize_choice(value=value, label="Stratégie de features")


# Pour harmoniser une méthode de réduction pour le parsing CLI
def _parse_dim_method(value: str) -> str:
    """Normalise la méthode de réduction en minuscules."""

    # Pour réutiliser la normalisation au helper générique
    return _normalize_choice(value=value, label="Méthode de réduction")


# Pour privilégier la fréquence réelle du fichier quand aucun override CLI n’existe.
def resolve_sampling_rate(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour dissocier la source EDF des artefacts générés.
    raw_dir: Path,
    # Pour respecter un override CLI quand il existe.
    requested_sfreq: float,
    # Pour préserver le re-référencement jusqu’au chargement EDF.
    eeg_reference: str | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> float:
    """Retourne la fréquence d'échantillonnage détectée ou la valeur demandée."""

    # Pour préserver la fréquence explicitement demandée lorsqu'elle diffère du défaut
    if requested_sfreq != DEFAULT_SAMPLING_RATE:
        # Pour garder la valeur explicite pour respecter la volonté utilisateur
        return requested_sfreq
    # Pour construire le chemin du fichier EDF brut pour la détection auto
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Pour conserver la valeur demandée si l'EDF n'est pas disponible
    if not raw_path.exists():
        # Pour garder la valeur par défaut en l'absence de fichier exploitable
        return requested_sfreq
    # Pour borner la lecture MNE pour éviter un crash si l'EDF est invalide
    try:
        # Pour lire l’EDF tout en récupérant les métadonnées utiles au fallback.
        raw, metadata = preprocessing.load_physionet_raw(
            # Pour conserver la provenance du fichier EDF visé.
            raw_path,
            # Pour transmettre explicitement la référence EEG retenue.
            reference=eeg_reference,
        )
        # Pour isoler la valeur brute de la fréquence depuis les métadonnées
        sampling_rate_value = metadata.get("sampling_rate")
        # Pour normaliser la valeur si possible pour préserver une fréquence cohérente
        if isinstance(sampling_rate_value, (int, float, str)):
            # Pour normaliser explicitement pour accepter str/int/float
            sampling_rate = float(sampling_rate_value)
        # Pour conserver un fallback explicite quand la branche nominale échoue.
        else:
            # Pour préserver la valeur demandée si la conversion est impossible
            sampling_rate = requested_sfreq
        # Pour libérer vite les ressources MNE sur un chemin de simple inspection.
        raw.close()
    # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
    except (FileNotFoundError, OSError, ValueError):
        # Pour signaler seulement le fallback de fréquence et éviter la redondance.
        # Pour garder la valeur demandée en cas d'échec de lecture
        return requested_sfreq
    # Pour garder la fréquence détectée pour aligner les features
    return sampling_rate


# Pour fixer le nombre cible de splits utilisé pour la validation croisée
DEFAULT_CV_SPLITS = 10

# Pour stabiliser le nombre minimal de splits pour déclencher la validation croisée
MIN_CV_SPLITS = 1

# Pour stabiliser la taille minimale du test pour stabiliser les splits CV
DEFAULT_CV_TEST_SIZE = 0.2

# Pour stabiliser le nombre minimal de classes pour activer la CV
MIN_CV_CLASS_COUNT = 2

# Pour fixer un seuil minimal total pour tenter une CV relâchée
MIN_CV_TOTAL_SAMPLES = MIN_CV_CLASS_COUNT + 1

# Pour stabiliser le nombre maximal de tentatives pour filtrer les splits CV
MAX_CV_SPLIT_ATTEMPTS_FACTOR = 10

# Pour stabiliser le seuil d'epochs sous lequel on désactive le nettoyage
MIN_EPOCHS_DISABLE_CLEANING = 6

# Pour stabiliser le seuil d'epochs pour appliquer un nettoyage plus tolérant
MIN_EPOCHS_LOW_THRESHOLD = 10

# Pour stabiliser le seuil d'epochs pour appliquer un nettoyage intermédiaire
MIN_EPOCHS_MEDIUM_THRESHOLD = 20


# Pour stabiliser la reproductibilité des splits de cross-validation
DEFAULT_RANDOM_STATE = 42

# Pour stabiliser le nombre de splits par défaut pour la sélection interne
DEFAULT_HYPERPARAM_SPLITS = 3

# Pour stabiliser la taille de test pour la sélection interne des hyperparamètres
DEFAULT_HYPERPARAM_TEST_SIZE = 0.2

# Pour fixer les valeurs de C explorées pour SVM/LogReg
DEFAULT_HYPERPARAM_C_VALUES = (0.1, 1.0, 10.0)

# Pour fixer les tailles de fenêtres Welch testées par défaut
DEFAULT_HYPERPARAM_WELCH_NPERSEG = (64, 128)


# Pour construire un split stratifié reproductible avec un nombre fixe d'itérations
def _build_cv_splitter(y: np.ndarray, n_splits: int):
    """
    Crée un splitter robuste qui s'adapte à la taille des classes
    pour éviter les UserWarnings de scikit-learn.
    """
    # Pour déterminer le nombre total d'échantillons pour valider la CV
    n_samples = len(y)
    # Pour rejeter la CV si l'échantillon est trop petit pour séparer les classes
    if n_samples < MIN_CV_TOTAL_SAMPLES:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None

    # Pour déterminer les classes et leurs effectifs pour définir la stratégie
    unique_classes, counts = np.unique(y, return_counts=True)

    # Pour refuser une CV sans séparation binaire exploitable.
    if len(unique_classes) < MIN_CV_CLASS_COUNT:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None

    # Pour borner la taille minimale pour adapter la stratégie de split
    min_class_size = int(np.min(counts))

    # Pour basculer vers un split tolérant quand une classe devient trop rare.
    if min_class_size < MIN_CV_CLASS_COUNT:
        # Pour garder un splitter shuffle pour éviter l'instabilité de la CV
        return sklearn_model_selection.ShuffleSplit(
            # Pour rendre explicite l’intensité de la validation utilisée.
            n_splits=n_splits,
            # Pour borner explicitement la part réservée au test.
            test_size=DEFAULT_CV_TEST_SIZE,
            # Pour conserver des splits reproductibles entre exécutions.
            random_state=DEFAULT_RANDOM_STATE,
        )

    # Pour préserver la stratification sans exiger des folds impossibles.
    if min_class_size < n_splits:
        # Pour déterminer une taille de test alignée sur l'effectif minimal
        test_size = 1.0 / float(min_class_size)
        # Pour garder un splitter stratifié shuffle pour stabiliser la CV
        return StratifiedShuffleSplit(
            # Pour rendre explicite l’intensité de la validation utilisée.
            n_splits=n_splits,
            # Pour borner explicitement la part réservée au test.
            test_size=test_size,
            # Pour conserver des splits reproductibles entre exécutions.
            random_state=DEFAULT_RANDOM_STATE,
        )

    # Pour conserver la stratification stricte quand les effectifs le permettent.
    # Pour stabiliser le nombre de splits au maximum permis par les classes
    actual_splits = max(MIN_CV_CLASS_COUNT, n_splits)

    # Pour garder un splitter stratifié ajusté pour la taille des classes
    return sklearn_model_selection.StratifiedKFold(
        # Pour rendre explicite l’intensité de la validation utilisée.
        n_splits=actual_splits,
        # Pour éviter un ordre de labels trop dépendant du dataset source.
        shuffle=True,
        # Pour conserver des splits reproductibles entre exécutions.
        random_state=DEFAULT_RANDOM_STATE,
    )


# Pour réaligner un splitter shuffle pour garantir deux classes dans le train
def _filter_shuffle_splits_for_binary_train(
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    splitter: sklearn_model_selection.ShuffleSplit,
    # Pour garder ce paramètre explicite dans le contrat.
    requested_splits: int,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construit des splits qui conservent deux classes dans le train."""

    # Pour centraliser une liste de splits validés pour la CV finale.
    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Pour déterminer le nombre total d'échantillons pour générer un X fictif
    sample_count = int(y.shape[0])
    # Pour centraliser un tableau factice pour piloter splitter.split
    placeholder_X = np.zeros((sample_count, 1))
    # Pour fixer une limite de tentatives pour éviter les boucles infinies
    max_attempts = max(1, requested_splits * MAX_CV_SPLIT_ATTEMPTS_FACTOR)
    # Pour couvrir les splits générés par sklearn_model_selection.ShuffleSplit
    for attempt_index, (train_idx, test_idx) in enumerate(
        # Pour rendre explicite ce point de décision ou de contrat.
        splitter.split(placeholder_X, y)
        # Pour figer un contrat exploitable par les appels et l’outillage de types.
    ):
        # Pour borner la recherche lorsque la limite est atteinte
        if attempt_index >= max_attempts:
            # Pour stopper la boucle dès que l’objectif de robustesse est atteint.
            break
        # Pour écarter les splits qui ne contiennent pas les deux classes en train
        if np.unique(y[train_idx]).size < MIN_CV_CLASS_COUNT:
            # Pour ignorer ce cas sans mélanger le flux nominal et le cas écarté.
            continue
        # Pour préserver le split valide pour la CV
        valid_splits.append((train_idx, test_idx))
        # Pour borner dès que le nombre de splits demandé est atteint
        if len(valid_splits) >= requested_splits:
            # Pour stopper la boucle dès que l’objectif de robustesse est atteint.
            break
    # Pour garder la liste finale des splits valides
    return valid_splits


# Pour centraliser le choix du splitter et la raison d’indisponibilité.
# Pour supprimer un reliquat de debug qui n’explique rien au lecteur.


# Pour isoler la logique de CV et son diagnostic dans un seul point.
def _resolve_cv_splits(
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    requested_splits: int,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[
    # Pour détailler la signature sans compacter un contrat difficile à relire.
    sklearn_model_selection.StratifiedShuffleSplit
    # Pour couvrir aussi le fallback shuffle retenu sur petits effectifs.
    | sklearn_model_selection.ShuffleSplit
    # Pour autoriser aussi une liste de splits déjà filtrés en amont.
    | list[tuple[np.ndarray, np.ndarray]]
    # Pour garder le cas d’indisponibilité explicite dans le contrat.
    | None,
    # Pour détailler la signature sans compacter un contrat difficile à relire.
    str | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
]:
    """Retourne un splitter compatible avec deux classes en train."""

    # Pour déléguer la stratégie de split à un point unique de robustesse.
    cv = _build_cv_splitter(y, requested_splits)
    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if cv is None:
        # Pour propager un diagnostic lisible au lieu d’un simple booléen.
        reason = _describe_cv_unavailability(y, requested_splits)
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None, reason

    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if isinstance(cv, sklearn_model_selection.ShuffleSplit):
        # Pour préparer explicitement cet objet intermédiaire avant usage.
        filtered_splits = _filter_shuffle_splits_for_binary_train(
            # Pour garder les labels explicites dans le contrat de la fonction.
            y,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            cv,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            requested_splits,
        )
        # Pour court-circuiter tôt un état qui casserait le contrat aval.
        if not filtered_splits:
            # Pour propager un diagnostic lisible au lieu d’un simple booléen.
            reason = _describe_cv_unavailability(y, requested_splits)
            # Pour rendre l’état du traitement visible dans un contexte CLI long.
            print(f"DEBUG CV: ShuffleSplit filtré vide, reason='{reason}'")
            # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
            return None, reason
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return filtered_splits, None

    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return cv, None


# Pour isoler la CV interne et éviter un réglage dispersé.
def _resolve_hyperparam_splits(
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    requested_splits: int,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[sklearn_model_selection.StratifiedShuffleSplit | None, str | None]:
    """Retourne un splitter stratifié pour la sélection interne."""

    # Pour déterminer le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Pour rejeter la sélection si l'effectif est trop faible
    if sample_count < MIN_CV_TOTAL_SAMPLES:
        # Pour garder une raison explicite pour le diagnostic
        return None, _describe_cv_unavailability(y, requested_splits)
    # Pour borner le nombre de classes pour valider la stratification
    class_count = int(np.unique(y).size)
    # Pour rejeter la sélection si une seule classe est présente
    if class_count < MIN_CV_CLASS_COUNT:
        # Pour garder une raison explicite pour le diagnostic
        return None, _describe_cv_unavailability(y, requested_splits)
    # Pour isoler l'effectif minimal par classe pour ajuster test_size
    _labels, class_counts = np.unique(y, return_counts=True)
    # Pour déterminer le nombre minimal d'échantillons par classe
    min_class_count = int(class_counts.min())
    # Pour rejeter la sélection si l'effectif minimal est insuffisant
    if min_class_count < MIN_CV_CLASS_COUNT:
        # Pour garder une raison explicite pour le diagnostic
        return None, _describe_cv_unavailability(y, requested_splits)
    # Pour déterminer la taille minimale pour garantir une classe en test
    min_test_size = 1.0 / float(min_class_count)
    # Pour adapter la taille de test pour rester valide et stable
    test_size = max(DEFAULT_HYPERPARAM_TEST_SIZE, min_test_size)
    # Pour déterminer la taille maximale autorisée pour le split
    max_test_size = (min_class_count - 1) / float(min_class_count)
    # Pour rejeter la sélection si le split devient impossible
    if test_size > max_test_size:
        # Pour garder une raison explicite pour le diagnostic
        return None, _describe_cv_unavailability(y, requested_splits)
    # Pour garder un splitter stratifié pour la sélection interne
    return (
        # Pour préparer explicitement cet objet intermédiaire avant usage.
        StratifiedShuffleSplit(
            # Pour rendre explicite l’intensité de la validation utilisée.
            n_splits=requested_splits,
            # Pour borner explicitement la part réservée au test.
            test_size=test_size,
            # Pour conserver des splits reproductibles entre exécutions.
            random_state=DEFAULT_RANDOM_STATE,
        ),
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        None,
    )


# Pour centraliser le fallback lorsque la validation croisée est impossible
def _handle_cv_unavailability(
    # Pour manipuler la pipeline complète sans état global caché.
    pipeline: Pipeline,
    # Pour garder les features explicites dans le contrat de la fonction.
    X: np.ndarray,
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    reason: str,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[np.ndarray, Pipeline, str]:
    """Ajuste la pipeline et retourne un diagnostic CV explicite."""
    # Pour écarter un fit lorsque les données sont absentes pour prévenir une erreur
    if X.size == 0 or y.size == 0:
        # Pour garder un score vide sans ajustement lorsque l'échantillon est nul
        return np.array([]), pipeline, reason
    # Pour adapter la pipeline sur toutes les données malgré l'absence de CV
    pipeline.fit(X, y)
    # Pour garder un tableau vide et la raison d'indisponibilité
    return np.array([]), pipeline, reason


# Pour déclencher la recherche d'hyperparamètres et retourne ses scores
def _run_grid_search(
    # Pour séparer la pipeline de recherche de la pipeline finale.
    search_pipeline: Pipeline,
    # Pour rendre l’espace de recherche explicite et testable.
    param_grid: dict[str, list[object]],
    # Pour injecter une stratégie de CV déjà validée en amont.
    search_cv: object,
    # Pour garder les features explicites dans le contrat de la fonction.
    X: np.ndarray,
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[np.ndarray, Pipeline, dict[str, object]]:
    """Exécute une GridSearchCV et retourne scores, pipeline et résumé."""

    # Pour distinguer le nombre de splits à partir du splitter fourni
    if isinstance(search_cv, list):
        # Pour conserver la longueur de liste lorsque les splits sont pré-calculés
        split_count = len(search_cv)
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour isoler get_n_splits si la méthode est disponible
        split_count = int(getattr(search_cv, "get_n_splits", lambda: 0)())
    # Pour conserver un fallback si le splitter ne rapporte aucun split
    split_count = split_count or DEFAULT_CV_SPLITS
    # Pour confier l’exploration exhaustive à un composant standard et fiable.
    search = GridSearchCV(
        # Pour séparer la pipeline de recherche de la pipeline finale.
        search_pipeline,
        # Pour rendre l’espace de recherche explicite et testable.
        param_grid,
        # Pour injecter une stratégie de validation déjà sécurisée en amont.
        cv=search_cv,
        # Pour garder une métrique optimisée cohérente avec le projet.
        scoring="accuracy",
        # Pour récupérer une pipeline finale déjà réentraînée sur le meilleur réglage.
        refit=True,
    )
    # Pour déclencher l'optimisation des hyperparamètres
    search.fit(X, y)
    # Pour isoler les scores par split pour l'entrée de manifeste
    cv_scores = _extract_grid_search_scores(search, split_count)
    # Pour conserver un résumé des meilleurs paramètres
    search_summary = {
        # Pour tracer la configuration gagnante dans le résumé de recherche.
        "best_params": search.best_params_,
        # Pour exposer le score de référence retenu après recherche.
        "best_score": float(search.best_score_),
        # Pour conserver les scores de recherche dans le manifeste de diagnostic.
        "search_scores": cv_scores.tolist(),
    }
    # Pour garder les scores, la meilleure pipeline et le résumé
    return cv_scores, search.best_estimator_, search_summary


# Pour garder un diagnostic CV homogène avec les autres fallbacks.
def _describe_cv_unavailability(y: np.ndarray, requested_splits: int) -> str:
    """Explique pourquoi la validation croisée est indisponible."""

    # Pour éviter de recalculer un effectif utilisé par plusieurs garde-fous.
    sample_count = int(y.shape[0])
    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if sample_count == 0:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return "aucun échantillon disponible pour la validation croisée"

    # Pour décider tôt si le problème reste bien binaire.
    class_count = int(np.unique(y).size)
    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if class_count < MIN_CV_CLASS_COUNT:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return "une seule classe présente, CV binaire impossible"

    # Pour rendre explicite ce point de décision ou de contrat.
    _labels, class_counts = np.unique(y, return_counts=True)
    # Pour raisonner sur la classe la plus fragile avant toute CV.
    min_class_count = int(class_counts.min())

    # Pour expliquer le cas où aucun split valide n’existe plus du tout.
    if min_class_count == 1:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return (
            # Pour garder un message complet sans casser la lisibilité.
            f"effectif minimal par classe = {min_class_count} "
            # Pour garder un message complet sans casser la lisibilité.
            "(impossible de diviser 1 échantillon entre train et test)"
        )

    # Pour détailler le cas où la rareté empêche la CV demandée.
    if min_class_count < MIN_CV_SPLITS:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return (
            # Pour garder un message complet sans casser la lisibilité.
            f"effectif minimal par classe insuffisant "
            # Pour garder un message complet sans casser la lisibilité.
            f"({min_class_count} < {MIN_CV_SPLITS})"
        )

    # Pour expliciter le cas où la taille de test devient mathématiquement invalide.
    min_test_size = 1.0 / float(min_class_count)
    # Pour stabiliser la taille de test avec une borne compatible.
    test_size = max(DEFAULT_CV_TEST_SIZE, min_test_size)
    # Pour détecter le point où un split devient mathématiquement invalide.
    max_test_size = (min_class_count - 1) / float(min_class_count)

    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if test_size > max_test_size:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return (
            # Pour garder un message complet sans casser la lisibilité.
            f"split stratifié impossible "
            # Pour garder un message complet sans casser la lisibilité.
            f"(test_size={test_size:.3f} > max={max_test_size:.3f})"
        )

    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return (
        # Pour garder un message complet sans casser la lisibilité.
        f"validation croisée indisponible pour une raison inconnue "
        # Pour garder un message complet sans casser la lisibilité.
        f"(splits demandés={requested_splits})"
    )


# Pour durcir la config quand l’effectif rend certains classifieurs instables.
def _adapt_pipeline_config_for_samples(
    # Pour faire circuler la configuration sans la re-décomposer.
    config: tpv_pipeline.PipelineConfig,
    # Pour décider ce fallback à partir de la distribution réelle des labels.
    y: np.ndarray,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tpv_pipeline.PipelineConfig:
    """Adapte la configuration si l'effectif est trop faible pour LDA."""

    # Pour déterminer le nombre de classes présentes dans les labels
    class_count = int(np.unique(y).size)
    # Pour déterminer le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Pour préserver la configuration si LDA n'est pas utilisé
    if config.classifier != "lda":
        # Pour garder la configuration d'origine sans modification
        return config
    # Pour préserver LDA lorsque l'effectif dépasse strictement le nombre de classes
    if sample_count > class_count:
        # Pour garder la configuration d'origine dans le cas valide
        return config
    # Pour éviter qu’un LDA trop fragile casse sur un très petit effectif.
    # Pour garder une nouvelle configuration alignée avec le fallback
    return tpv_pipeline.PipelineConfig(
        # Pour préserver la fréquence d'échantillonnage identique
        sfreq=config.sfreq,
        # Pour préserver la stratégie de features pour la comparabilité
        feature_strategy=config.feature_strategy,
        # Pour préserver la normalisation pour éviter les dérives d'échelle
        normalize_features=config.normalize_features,
        # Pour préserver la méthode de réduction pour limiter l'écart de pipeline
        dim_method=config.dim_method,
        # Pour préserver le nombre de composantes initialement demandé
        n_components=config.n_components,
        # Pour substituer un classifieur plus tolérant lorsque LDA devient fragile.
        classifier="centroid",
        # Pour préserver le scaler optionnel demandé par la configuration
        scaler=config.scaler,
        # Pour préserver la régularisation CSP pour stabiliser les covariances
        csp_regularization=config.csp_regularization,
    )


# Pour adapter la méthode de réduction en fonction des features demandées
def _resolve_dim_method_for_features(
    # Pour laisser explicite la stratégie de features demandée.
    feature_strategy: str,
    # Pour laisser explicite la réduction de dimension demandée.
    dim_method: str,
    # Pour tester la CLI sans dépendre implicitement de sys.argv.
    argv: list[str] | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> str:
    """Retourne la méthode de réduction adaptée à la stratégie de features."""

    # Pour isoler la liste brute d'arguments pour détecter un override explicite
    raw_args = argv if argv is not None else sys.argv[1:]
    # Pour détecter si --dim-method a été fourni par l'utilisateur
    dim_method_explicit = "--dim-method" in raw_args
    # Pour valider si la stratégie impose des features spectrales
    if feature_strategy in {"wavelet", "welch"} and dim_method in {"csp", "cssp"}:
        # Pour rendre explicite l'utilisateur d'un enchaînement CSP suivi des features
        if not dim_method_explicit:
            # Pour rendre explicite un couplage implicite sinon surprenant.
            print(
                # Pour rendre le diagnostic exploitable sans ouvrir le code.
                "INFO: dim_method='csp/cssp' appliqué avant "
                # Pour garder un message complet sans casser la lisibilité.
                "l'extraction des features."
            )
        # Pour garder la méthode sans modification pour permettre Welch+CSP
        return dim_method
    # Pour garder la méthode inchangée si aucune adaptation n'est requise
    return dim_method


# Pour adapter la stratégie de features si un alias de réduction est fourni
def _resolve_feature_strategy_and_dim_method(
    # Pour laisser explicite la stratégie de features demandée.
    feature_strategy: str,
    # Pour laisser explicite la réduction de dimension demandée.
    dim_method: str,
    # Pour tester la CLI sans dépendre implicitement de sys.argv.
    argv: list[str] | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[str, str]:
    """Retourne une stratégie de features valide et le dim_method associé."""

    # Pour isoler la liste brute d'arguments pour détecter un override explicite
    raw_args = argv if argv is not None else sys.argv[1:]
    # Pour détecter si --dim-method a été fourni par l'utilisateur
    dim_method_explicit = "--dim-method" in raw_args
    # Pour lever l'ambiguïté quand CSP/PCA/SVD arrivent via --feature-strategy.
    if feature_strategy in FEATURE_STRATEGY_ALIASES:
        # Pour préserver la stratégie FFT pour garantir une extraction valide
        resolved_feature_strategy = "fft"
        # Pour préserver le dim_method explicite si l'utilisateur l'a fourni
        if dim_method_explicit:
            # Pour rendre explicite de l'alias ignoré au profit de la valeur explicite
            print(
                # Pour rendre le diagnostic exploitable sans ouvrir le code.
                "INFO: feature_strategy interprété comme alias de dim_method, "
                # Pour garder un message complet sans casser la lisibilité.
                "feature_strategy='fft' conservée car --dim-method explicite."
            )
            # Pour garder la stratégie FFT et le dim_method explicite
            return resolved_feature_strategy, dim_method
        # Pour rendre explicite que l'alias est interprété comme dim_method
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            "INFO: feature_strategy interprété comme alias de dim_method, "
            # Pour garder un message complet sans casser la lisibilité.
            "feature_strategy='fft' appliquée."
        )
        # Pour garder la stratégie FFT et le dim_method dérivé
        return resolved_feature_strategy, feature_strategy
    # Pour garder les paramètres inchangés si aucune correction n'est requise
    return feature_strategy, dim_method


# Pour fournir les fenêtres d'epochs par défaut pour le contexte
def _default_epoch_windows() -> Sequence[tuple[float, float]]:
    """Retourne les fenêtres d'epochs par défaut."""

    # Pour isoler les fenêtres par défaut via la configuration centralisée
    return cast(
        # Pour figer le type public attendu malgré une source plus lâche.
        Sequence[tuple[float, float]],
        # Pour réutiliser la config centrale sans dupliquer cette valeur.
        tpv_utils.default_epoch_window_config().default_windows,
    )


# Pour centraliser toutes les informations nécessaires à un run d'entraînement
@dataclass
# Pour regrouper toutes les entrées d’un entraînement atomique.
class TrainingRequest:
    """Décrit les paramètres nécessaires pour entraîner un run."""

    # Pour tracer le sujet cible pour l'entraînement
    subject: str
    # Pour tracer le run ciblé pour le sujet sélectionné
    run: str
    # Pour regrouper la configuration complète de pipeline
    pipeline_config: tpv_pipeline.PipelineConfig
    # Pour fixer le répertoire contenant les données numpy
    data_dir: Path
    # Pour fixer le répertoire racine pour déposer les artefacts
    artifacts_dir: Path
    # Pour fixer le répertoire des enregistrements EDF bruts
    raw_dir: Path = DEFAULT_RAW_DIR
    # Pour fixer la référence EEG à appliquer lors du chargement EDF
    eeg_reference: str | None = DEFAULT_EEG_REFERENCE
    # Pour centraliser les réglages de filtrage et de normalisation
    preprocess_config: preprocessing.PreprocessingConfig = field(
        # Pour conserver une factory pour éviter le partage d'instance
        default_factory=preprocessing.PreprocessingConfig
    )
    # Pour permettre une optimisation systématique des hyperparamètres si demandé
    enable_grid_search: bool = False
    # Pour stabiliser un nombre de splits spécifique pour la recherche si fourni
    grid_search_splits: int | None = None


# Pour centraliser les entrées nécessaires à la sélection de fenêtre d'epochs
@dataclass
# Pour partager un contexte stable entre essais de fenêtres.
class EpochWindowContext:
    """Agrège les données requises pour sélectionner une fenêtre d'epochs."""

    # Pour transporter l'enregistrement brut filtré pour l'epoching
    filtered_raw: preprocessing.mne.io.BaseRaw
    # Pour transporter les événements détectés pour l'epoching
    events: np.ndarray
    # Pour transporter le mapping d'événements vers labels
    event_id: dict[str, int]
    # Pour transporter la liste des labels moteurs alignés
    motor_labels: list[str]
    # Pour tracer le sujet pour les logs et erreurs
    subject: str
    # Pour tracer le run pour les logs et erreurs
    run: str
    # Pour regrouper les fenêtres candidates à évaluer
    windows: Sequence[tuple[float, float]] = field(
        # Pour matérialiser une valeur intermédiaire utile au diagnostic.
        default_factory=_default_epoch_windows
    )


# Pour conserver l'état courant de la sélection de fenêtre
@dataclass
# Pour mémoriser le meilleur candidat sans état global diffus.
class WindowSelectionState:
    """Stocke l'état de sélection d'une fenêtre d'epochs."""

    # Pour mémoriser la fenêtre candidate courante
    best_window: tuple[float, float]
    # Pour mémoriser le meilleur score connu ou None
    best_score: float | None
    # Pour mémoriser les données d'epochs associées
    best_epochs_data: np.ndarray | None
    # Pour mémoriser les labels associés
    best_labels: np.ndarray | None


# Pour centraliser les ressources partagées entre plusieurs entraînements
@dataclass
# Pour mutualiser la config commune aux entraînements batch.
class TrainingResources:
    """Agrège les chemins et la configuration pipeline pour un batch."""

    # Pour regrouper la configuration partagée pour toutes les exécutions
    pipeline_config: tpv_pipeline.PipelineConfig
    # Pour fixer le répertoire contenant les données numpy
    data_dir: Path
    # Pour fixer le répertoire racine pour déposer les artefacts
    artifacts_dir: Path
    # Pour fixer le répertoire des enregistrements EDF bruts
    raw_dir: Path = DEFAULT_RAW_DIR
    # Pour fixer la référence EEG à appliquer lors du chargement EDF
    eeg_reference: str | None = DEFAULT_EEG_REFERENCE
    # Pour centraliser les réglages de filtrage et de normalisation
    preprocess_config: preprocessing.PreprocessingConfig = field(
        # Pour conserver une factory pour éviter le partage d'instance
        default_factory=preprocessing.PreprocessingConfig
    )
    # Pour permettre une optimisation systématique des hyperparamètres si demandé
    enable_grid_search: bool = False
    # Pour stabiliser un nombre de splits spécifique pour la recherche si fourni
    grid_search_splits: int | None = None


# Pour centraliser les chemins et réglages nécessaires à la génération des numpy
@dataclass
# Pour éviter une signature trop longue lors de la génération.
class NpyBuildContext:
    """Encapsule les paramètres nécessaires à la génération des .npy."""

    # Pour fixer le répertoire contenant les données numpy
    data_dir: Path
    # Pour fixer le répertoire des enregistrements EDF bruts
    raw_dir: Path
    # Pour fixer la référence EEG à appliquer lors du chargement EDF
    eeg_reference: str | None
    # Pour centraliser les réglages de filtrage et de normalisation
    preprocess_config: preprocessing.PreprocessingConfig


# Pour construire un argument parser aligné sur la CLI mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'entraînement TPV."""

    # Pour garantir le parser avec description lisible pour l'utilisateur
    parser = argparse.ArgumentParser(
        # Pour fixer explicitement ce réglage dans l’objet construit.
        description="Entraîne une pipeline TPV et sauvegarde ses artefacts",
    )
    # Pour exposer l'argument positionnel du sujet pour identifier les fichiers
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "subject",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=_parse_subject,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Identifiant du sujet (ex: 4)",
    )
    # Pour exposer l'argument positionnel du run pour sélectionner la session
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "run",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=_parse_run,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Identifiant du run (ex: 14)",
    )
    # Pour exposer l'option classifieur pour synchroniser avec mybci
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--classifier",
        # Pour borner l’entrée aux valeurs réellement supportées.
        choices=("lda", "logistic", "svm", "centroid"),
        # Pour garantir un comportement stable sans override utilisateur.
        default="lda",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Classifieur final utilisé pour l'entraînement",
    )
    # Pour exposer le choix du scaler optionnel pour stabiliser les features
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--scaler",
        # Pour borner l’entrée aux valeurs réellement supportées.
        choices=("standard", "robust", "none"),
        # Pour garantir un comportement stable sans override utilisateur.
        default="none",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Scaler optionnel appliqué après l'extraction de features",
    )
    # Pour exposer la stratégie d'extraction de features pour garder la cohérence
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--feature-strategy",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=_parse_feature_strategy,
        # Pour borner l’entrée aux valeurs réellement supportées.
        choices=FEATURE_STRATEGY_CHOICES,
        # Pour garantir un comportement stable sans override utilisateur.
        default="fft",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Méthode d'extraction de features spectrales",
    )
    # Pour exposer la méthode de réduction de dimension pour contrôler la compression
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--dim-method",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=_parse_dim_method,
        # Pour borner l’entrée aux valeurs réellement supportées.
        choices=DIM_METHODS,
        # Pour garantir un comportement stable sans override utilisateur.
        default="csp",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Méthode de réduction de dimension pour la pipeline",
    )
    # Pour exposer la régularisation CSP pour stabiliser les covariances
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--csp-regularization",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=float,
        # Pour garantir un comportement stable sans override utilisateur.
        default=0.1,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Régularisation diagonale appliquée aux covariances CSP",
    )
    # Pour exposer le nombre de composantes cible pour la réduction
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--n-components",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=int,
        # Pour garantir un comportement stable sans override utilisateur.
        default=argparse.SUPPRESS,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Nombre de composantes conservées par le réducteur",
    )
    # Pour exposer un flag pour désactiver la normalisation des features
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--no-normalize-features",
        # Pour représenter ce flag comme un booléen sans valeur parasite.
        action="store_true",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Désactive la normalisation des features extraites",
    )
    # Pour exposer la borne basse du filtre passe-bande MI
    parser.add_argument(
        # Pour fixer le nom du flag CLI pour la borne basse
        "--bandpass-low",
        # Pour accepter float pour accepter des fréquences décimales
        type=float,
        # Pour stabiliser la valeur par défaut alignée sur la bande MI
        default=preprocessing.DEFAULT_BANDPASS_BAND[0],
        # Pour rendre la borne basse immédiatement compréhensible en CLI.
        help="Fréquence basse du passe-bande MI (ex: 8.0)",
    )
    # Pour exposer la borne haute du filtre passe-bande MI
    parser.add_argument(
        # Pour fixer le nom du flag CLI pour la borne haute
        "--bandpass-high",
        # Pour accepter float pour accepter des fréquences décimales
        type=float,
        # Pour stabiliser la valeur par défaut alignée sur la bande MI
        default=preprocessing.DEFAULT_BANDPASS_BAND[1],
        # Pour rendre la borne haute immédiatement compréhensible en CLI.
        help="Fréquence haute du passe-bande MI (ex: 30.0)",
    )
    # Pour exposer la fréquence de notch pour supprimer le bruit secteur
    parser.add_argument(
        # Pour fixer le nom du flag CLI pour le notch
        "--notch-freq",
        # Pour accepter float pour autoriser 50 ou 60 Hz
        type=float,
        # Pour stabiliser la valeur par défaut compatible Europe
        default=preprocessing.DEFAULT_NOTCH_FREQ,
        # Pour rendre le notch immédiatement compréhensible en CLI.
        help="Fréquence de notch pour le bruit secteur (50 ou 60 Hz)",
    )
    # Pour exposer la méthode de normalisation par canal des epochs
    parser.add_argument(
        # Pour fixer le nom du flag CLI pour la normalisation canal
        "--normalize-channels",
        # Pour recenser les méthodes acceptées pour sécuriser les entrées
        choices=("zscore", "robust", "none"),
        # Pour stabiliser la méthode par défaut alignée sur preprocessing
        default=preprocessing.DEFAULT_NORMALIZE_METHOD,
        # Pour rendre la méthode de normalisation explicite dès l'aide CLI.
        help="Normalisation par canal appliquée aux epochs (zscore/robust/none)",
    )
    # Pour exposer l'epsilon de stabilisation pour la normalisation
    parser.add_argument(
        # Pour fixer le nom du flag CLI pour l'epsilon
        "--normalize-epsilon",
        # Pour accepter float pour accepter des epsilon personnalisés
        type=float,
        # Pour stabiliser la valeur par défaut alignée sur preprocessing
        default=preprocessing.DEFAULT_NORMALIZE_EPSILON,
        # Pour rendre l'epsilon de normalisation explicite dès l'aide CLI.
        help="Epsilon de stabilité pour la normalisation par canal",
    )
    # Pour exposer une option pour cibler un répertoire de données spécifique
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--data-dir",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=Path,
        # Pour garantir un comportement stable sans override utilisateur.
        default=DEFAULT_DATA_DIR,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Répertoire racine contenant les fichiers numpy",
    )
    # Pour exposer une option pour configurer le répertoire d'artefacts
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--artifacts-dir",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=Path,
        # Pour garantir un comportement stable sans override utilisateur.
        default=DEFAULT_ARTIFACTS_DIR,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Répertoire racine où enregistrer le modèle",
    )
    # Pour exposer une option pour charger une configuration de fenêtres par sujet
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--epoch-window-config",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=Path,
        # Pour garantir un comportement stable sans override utilisateur.
        default=None,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Chemin d'un JSON définissant les fenêtres d'epochs par sujet",
    )
    # Pour exposer une option pour pointer vers les fichiers EDF bruts
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--raw-dir",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=Path,
        # Pour garantir un comportement stable sans override utilisateur.
        default=DEFAULT_RAW_DIR,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Répertoire racine contenant les fichiers EDF bruts",
    )
    # Pour exposer l'option de re-référencement EEG lors du chargement EDF
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--eeg-reference",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=_parse_eeg_reference,
        # Pour garantir un comportement stable sans override utilisateur.
        default=DEFAULT_EEG_REFERENCE,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Référence EEG appliquée au chargement (ex: average, none)",
    )
    # Pour exposer un mode pour générer tous les .npy sans lancer un fit complet
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--build-all",
        # Pour représenter ce flag comme un booléen sans valeur parasite.
        action="store_true",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Génère les fichiers _X.npy/_y.npy pour tous les sujets détectés",
    )
    # Pour exposer un mode pour entraîner tous les runs moteurs disponibles
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--train-all",
        # Pour représenter ce flag comme un booléen sans valeur parasite.
        action="store_true",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Entraîne tous les sujets/runs détectés dans data/",
    )
    # Pour exposer une option pour activer une recherche systématique de paramètres
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--grid-search",
        # Pour représenter ce flag comme un booléen sans valeur parasite.
        action="store_true",
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Active une optimisation systématique des hyperparamètres",
    )
    # Pour exposer une option pour forcer le nombre de splits en grid search
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--grid-search-splits",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=int,
        # Pour garantir un comportement stable sans override utilisateur.
        default=None,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Nombre de splits CV dédié à la recherche d'hyperparamètres",
    )
    # Pour exposer une option pour spécifier la fréquence d'échantillonnage
    parser.add_argument(
        # Pour exposer ce point d’entrée CLI avec un nom stable et documenté.
        "--sfreq",
        # Pour valider tôt l’entrée et éviter un état ambigu plus loin.
        type=float,
        # Pour garantir un comportement stable sans override utilisateur.
        default=DEFAULT_SAMPLING_RATE,
        # Pour rendre l’aide CLI immédiatement exploitable.
        help="Fréquence d'échantillonnage utilisée pour les features",
    )
    # Pour garder le parser configuré
    return parser


# Pour construire les chemins des données pour un sujet et un run donnés
def _resolve_data_paths(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Pour pointer vers le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Pour stabiliser le chemin du le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Pour stabiliser le chemin du le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Pour garder les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Pour construire le chemin du fichier de fenêtre d'epochs pour un run
def _resolve_epoch_window_path(subject: str, run: str, data_dir: Path) -> Path:
    """Retourne le chemin du JSON décrivant la fenêtre d'epochs sélectionnée."""

    # Pour pointer vers le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Pour construire le chemin du fichier de fenêtre pour ce run
    window_path = base_dir / f"{run}_epoch_window.json"
    # Pour garder le chemin du fichier de fenêtre
    return window_path


# Pour persister la fenêtre d'epochs sélectionnée pour usage futur
def _write_epoch_window_metadata(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour rendre la racine de données injectable et testable.
    data_dir: Path,
    # Pour garder la fenêtre temporelle explicite dans le contrat.
    window: tuple[float, float],
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> None:
    """Enregistre la fenêtre d'epochs sélectionnée pour ce run."""

    # Pour construire le chemin du fichier de fenêtre pour ce run
    window_path = _resolve_epoch_window_path(subject, run, data_dir)
    # Pour garantir l'existence du dossier cible pour éviter une erreur d'écriture
    window_path.parent.mkdir(parents=True, exist_ok=True)
    # Pour centraliser la structure JSON pour la persistance
    payload = {"tmin": window[0], "tmax": window[1]}
    # Pour préserver la fenêtre dans un fichier JSON dédié
    window_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


# Pour récupérer la fenêtre d'epochs persistée pour un run si disponible
def _read_epoch_window_metadata(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour rendre la racine de données injectable et testable.
    data_dir: Path,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[float, float] | None:
    """Retourne la fenêtre d'epochs persistée ou None si absente."""

    # Pour construire le chemin du fichier de fenêtre pour ce run
    window_path = _resolve_epoch_window_path(subject, run, data_dir)
    # Pour garder None si le fichier n'existe pas
    if not window_path.exists():
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None
    # Pour récupérer le contenu JSON pour récupérer la fenêtre
    payload = json.loads(window_path.read_text())
    # Pour isoler tmin du JSON en float
    tmin = float(payload.get("tmin", tpv_utils.DEFAULT_EPOCH_WINDOW[0]))
    # Pour isoler tmax du JSON en float
    tmax = float(payload.get("tmax", tpv_utils.DEFAULT_EPOCH_WINDOW[1]))
    # Pour garder la fenêtre reconstruite
    return (tmin, tmax)


# Pour garder une sélection de fenêtre rapide et cohérente avec le pipeline final.
def _build_window_search_pipeline(sfreq: float) -> Pipeline:
    """Construit un pipeline CSP+Centroid pour la sélection de fenêtre."""

    # Pour figer une configuration complète avant construction du pipeline.
    search_config = tpv_pipeline.PipelineConfig(
        # Pour propager la fréquence effective dans la configuration construite.
        sfreq=sfreq,
        # Pour figer la stratégie de features dans la configuration.
        feature_strategy="fft",
        # Pour figer la politique de normalisation des features.
        normalize_features=True,
        # Pour figer la réduction de dimension dans la configuration.
        dim_method="csp",
        # Pour fixer explicitement le degré de compression retenu.
        n_components=DEFAULT_CSP_COMPONENTS,
        # Pour figer explicitement le classifieur de sortie.
        classifier="lda",
        # Pour figer explicitement le scaling optionnel de la pipeline.
        scaler=None,
        # Pour rendre explicite la stabilisation des covariances CSP.
        csp_regularization=0.1,
    )
    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return build_pipeline(search_config)


# Pour mesurer chaque fenêtre avec une métrique comparable entre essais.
def _score_epoch_window(X: np.ndarray, y: np.ndarray, sfreq: float) -> float | None:
    # Pour rejeter la CV si le nombre d'échantillons est insuffisant
    if len(y) < MIN_CV_CLASS_COUNT:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None

    # Pour déterminer le nombre de classes distinctes pour évaluer la faisabilité
    unique_classes = np.unique(y)
    # Pour rejeter la CV si une classe est manquante
    if unique_classes.size < MIN_CV_CLASS_COUNT:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None
    # Pour déterminer l'effectif minimal pour juger de la stabilité des splits
    min_class_size = int(np.min([np.sum(y == label) for label in unique_classes]))
    # Pour rejeter la CV si une classe est trop rare
    if min_class_size < MIN_CV_CLASS_COUNT:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None

    # Pour construire un splitter cohérent avec la taille des classes
    cv = _build_cv_splitter(y, DEFAULT_CV_SPLITS)

    # Pour éviter un bruit CLI inutile quand la fenêtre est simplement non exploitable.
    if cv is None:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None

    # Pour manipuler la pipeline complète sans état global caché.
    pipeline = _build_window_search_pipeline(sfreq)

    # Pour borner un échec attendu sans polluer le flux nominal.
    try:
        # Pour s’appuyer sur une CV standard dès que les préconditions sont réunies.
        scores = cross_val_score(
            # Pour évaluer exactement la pipeline candidate de cette fenêtre.
            pipeline,
            # Pour transmettre les features sans reconstruction intermédiaire.
            X,
            # Pour transmettre les labels alignés sur la fenêtre testée.
            y,
            # Pour imposer le splitter déjà sécurisé en amont.
            cv=cv,
            # Pour éviter qu’un échec ponctuel casse toute l’évaluation de fenêtre.
            error_score=0.5,
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return float(np.mean(scores))
    # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
    except Exception:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None


# Pour normaliser une liste de labels moteurs en indices entiers stables
def _map_motor_labels_to_ints(labels: Sequence[str]) -> np.ndarray:
    """Transforme des labels moteurs en indices ordonnés."""

    # Pour construire un mapping ordonné pour stabiliser les conversions
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    # Pour normaliser chaque label via le mapping pour obtenir un tableau numérique
    return np.array([label_mapping[label] for label in labels])


# Pour construire les epochs et labels pour une fenêtre temporelle donnée
def _build_epochs_for_window(
    # Pour regrouper les dépendances métier sans allonger la signature.
    context: EpochWindowContext,
    # Pour garder la fenêtre temporelle explicite dans le contrat.
    window: tuple[float, float],
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les epochs et labels alignés pour une fenêtre donnée."""

    # Pour évaluer chaque fenêtre sur les epochs réellement utilisables.
    epochs = preprocessing.create_epochs_from_raw(
        # Pour transporter un signal déjà stabilisé pour la suite du flux.
        context.filtered_raw,
        # Pour garder les événements alignés sur le signal traité.
        context.events,
        # Pour préserver le mapping des événements utile à l’epoching.
        context.event_id,
        # Pour figer la borne basse de la fenêtre dans l’appel.
        tmin=window[0],
        # Pour figer la borne haute de la fenêtre dans l’appel.
        tmax=window[1],
    )
    # Pour conserver les données d'epochs pour contrôler les tailles
    epochs_data = epochs.get_data(copy=True)

    # Pour réaligner les labels sur les epochs réellement conservées par MNE.
    kept_indices = getattr(epochs, "selection", np.arange(epochs_data.shape[0]))

    # Pour réaligner motor_labels pour ne garder que les labels des epochs conservés
    epochs_aligned_labels = [context.motor_labels[i] for i in kept_indices]
    # Pour normaliser les labels en indices numériques pour le pipeline
    epochs_aligned_array = _map_motor_labels_to_ints(epochs_aligned_labels)

    # Pour valider l'alignement pour éviter des labels décalés
    if len(epochs_aligned_array) != epochs_data.shape[0]:
        # Pour rendre l’échec explicite au point exact où le contrat est violé.
        raise ValueError(
            # Pour garder un message complet sans casser la lisibilité.
            f"Désalignement détecté: {len(epochs_aligned_array)} labels "
            # Pour garder un message complet sans casser la lisibilité.
            f"pour {len(epochs)} epochs"
        )

    # Pour regrouper la logique de robustesse dédiée aux petits effectifs.
    # Pour éviter qu’un nettoyage agressif supprime tout sur petit effectif.
    # Pour regrouper la logique de robustesse dédiée aux petits effectifs.
    initial_epoch_count = epochs_data.shape[0]

    # Pour éviter qu’un nettoyage agressif supprime tout sur un très petit lot.
    if initial_epoch_count < MIN_EPOCHS_DISABLE_CLEANING:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            f"INFO: [{context.subject}] Trop peu d'epochs ({initial_epoch_count}), "
            # Pour garder un message complet sans casser la lisibilité.
            "nettoyage désactivé pour préserver les données."
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return epochs_data, epochs_aligned_array

    # Pour garder un rejet proportionné à la quantité d’epochs disponible.
    if initial_epoch_count < MIN_EPOCHS_LOW_THRESHOLD:
        # Pour proportionner le nettoyage au risque de vider un petit lot.
        threshold = 5000e-6
    # Pour distinguer un cas intermédiaire avec un traitement dédié.
    elif initial_epoch_count < MIN_EPOCHS_MEDIUM_THRESHOLD:
        # Pour proportionner le nettoyage au risque de vider un petit lot.
        threshold = 3000e-6
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour proportionner le nettoyage au risque de vider un petit lot.
        threshold = 1500e-6
    # Pour regrouper la logique de robustesse dédiée aux petits effectifs.

    # Pour aligner un rejet d'artefacts
    try:
        # Pour préparer explicitement cet objet intermédiaire avant usage.
        cleaned_epochs, _report, cleaned_labels = preprocessing.summarize_epoch_quality(
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            epochs,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            epochs_aligned_labels,
            # Pour garder ce fragment explicite malgré le format multi-ligne.
            (context.subject, context.run),
            # Pour contrôler explicitement l’agressivité du nettoyage.
            max_peak_to_peak=threshold,
        )

        # Pour empêcher un run vide après nettoyage excessif.
        # Pour préserver au moins un signal exploitable quand le nettoyage dérive.
        if len(cleaned_epochs) == 0 or len(cleaned_epochs) < MIN_CV_TOTAL_SAMPLES:
            # Pour rendre l’état du traitement visible dans un contexte CLI long.
            print(
                # Pour rendre visible une dégradation sans interrompre tout le flux.
                f"WARN: [{context.subject}] Nettoyage trop strict "
                # Pour garder un message complet sans casser la lisibilité.
                f"({len(cleaned_epochs)} epochs restants), "
                # Pour garder un message complet sans casser la lisibilité.
                "conservation des données brutes."
            )
            # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
            return epochs_data, epochs_aligned_array

        # Pour garder les epochs nettoyées pour l'apprentissage
        cleaned_numeric_labels = _map_motor_labels_to_ints(cleaned_labels)
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return cleaned_epochs.get_data(), cleaned_numeric_labels

    # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
    except Exception as e:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(f"ERROR: Erreur lors du nettoyage pour {context.subject}: {e}")
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return epochs_data, epochs_aligned_array


# Pour préparer l'état de sélection pour une liste de fenêtres
def _initialize_window_selection(context: EpochWindowContext) -> WindowSelectionState:
    """Construit l'état initial de sélection des fenêtres."""

    # Pour rejeter la sélection si aucune fenêtre n'est disponible
    if not context.windows:
        # Pour rendre explicite l'absence de fenêtres pour stopper la sélection
        raise ValueError("Aucune fenêtre d'epochs disponible pour la sélection.")
    # Pour garder l'état initialisé sur la première fenêtre
    return WindowSelectionState(
        # Pour fixer explicitement ce réglage dans l’objet construit.
        best_window=context.windows[0],
        # Pour fixer explicitement ce réglage dans l’objet construit.
        best_score=None,
        # Pour fixer explicitement ce réglage dans l’objet construit.
        best_epochs_data=None,
        # Pour fixer explicitement ce réglage dans l’objet construit.
        best_labels=None,
    )


# Pour comparer chaque fenêtre sans disperser la logique de décision.
def _update_window_selection(
    # Pour garder ce paramètre explicite dans le contrat.
    state: WindowSelectionState,
    # Pour garder la fenêtre temporelle explicite dans le contrat.
    window: tuple[float, float],
    # Pour garder ce paramètre explicite dans le contrat.
    epochs_data: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    numeric_labels: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    window_score: float | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> WindowSelectionState:
    """Retourne un état de sélection mis à jour."""

    # Pour préparer l'état de base si aucune fenêtre n'est encore retenue
    if state.best_epochs_data is None:
        # Pour garder un état initialisé avec la fenêtre courante
        return WindowSelectionState(
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_window=window,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_score=window_score,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_epochs_data=epochs_data,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_labels=numeric_labels,
        )
    # Pour écarter les fenêtres sans score si un score existe déjà
    if window_score is None and state.best_score is not None:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return state
    # Pour valider si le score courant est meilleur que le précédent
    if window_score is not None and (
        # Pour rendre explicite ce point de décision ou de contrat.
        state.best_score is None
        # Pour ne remplacer l’état que sur amélioration effective du score.
        or window_score > state.best_score
        # Pour figer un contrat exploitable par les appels et l’outillage de types.
    ):
        # Pour garder un état mis à jour avec la fenêtre courante
        return WindowSelectionState(
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_window=window,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_score=window_score,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_epochs_data=epochs_data,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            best_labels=numeric_labels,
        )
    # Pour garder l'état inchangé si aucune amélioration n'est trouvée
    return state


# Pour retenir la meilleure fenêtre selon un score cross-val
def _select_best_epoch_window(
    # Pour regrouper les dépendances métier sans allonger la signature.
    context: EpochWindowContext,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """Retourne la fenêtre optimale et les données associées."""

    # Pour préparer l'état de sélection des fenêtres
    selection_state = _initialize_window_selection(context)

    # Pour couvrir les fenêtres candidates pour sélectionner la plus robuste
    for window in context.windows:
        # Pour construire les epochs et labels pour la fenêtre courante
        epochs_data, numeric_labels = _build_epochs_for_window(context, window)
        # Pour déterminer le score cross-val pour cette fenêtre si possible
        window_score = _score_epoch_window(
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            epochs_data,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            numeric_labels,
            # Pour garder ce fragment explicite malgré le format multi-ligne.
            float(context.filtered_raw.info["sfreq"]),
        )
        # Pour retenir la fenêtre courante seulement si elle améliore l’état connu.
        selection_state = _update_window_selection(
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            selection_state,
            # Pour garder la fenêtre temporelle explicite dans le contrat.
            window,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            epochs_data,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            numeric_labels,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            window_score,
        )

    # Pour refuser une sélection vide avant toute persistance sur disque.
    if selection_state.best_epochs_data is None or selection_state.best_labels is None:
        # Pour rendre explicite une absence complète de données après sélection
        raise ValueError(
            # Pour garder un message complet sans casser la lisibilité.
            f"Aucune epoch valide pour {context.subject} {context.run} "
            # Pour garder un message complet sans casser la lisibilité.
            "après sélection de fenêtre."
        )

    # Pour garder la fenêtre retenue et les données associées
    return (
        # Pour conserver la fenêtre retenue après comparaison.
        selection_state.best_window,
        # Pour conserver les données liées à la meilleure fenêtre.
        selection_state.best_epochs_data,
        # Pour conserver les labels liés à la meilleure fenêtre.
        selection_state.best_labels,
    )


# Pour construire des matrices numpy à partir d'un EDF lorsqu'elles manquent
def _build_npy_from_edf(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour centraliser les dépendances de génération des caches numpy.
    build_context: NpyBuildContext,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[Path, Path]:
    """Génère X (epochs brutes) et y depuis un fichier EDF Physionet.

    - X est sauvegardé sous forme (n_trials, n_channels, n_times)
      pour être compatible avec la pipeline (tpv.features).
    - Les features fréquentielles sont ensuite calculées *dans* la
      pipeline, pas au moment de la génération des .npy.
    """

    # Pour déterminer les chemins cibles pour les fichiers numpy
    features_path, labels_path = _resolve_data_paths(
        # Pour propager l'identifiant de sujet pour le chemin
        subject,
        # Pour propager l'identifiant de run pour le chemin
        run,
        # Pour centraliser la racine locale des caches numpy.
        build_context.data_dir,
    )
    # Pour déterminer les chemins attendus des fichiers bruts PhysioNet
    raw_path = build_context.raw_dir / subject / f"{subject}{run}.edf"
    # Pour conserver la provenance du fichier d’événements visé.
    event_path = raw_path.with_suffix(".edf.event")

    # Pour échouer tôt si tôt si l'EDF est absent ou vide
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        # Pour rendre l’échec explicite au point exact où le contrat est violé.
        raise FileNotFoundError(
            # Pour garder un message complet sans casser la lisibilité.
            "EDF introuvable pour "
            # Pour garder un message complet sans casser la lisibilité.
            f"{subject} {run}: {raw_path}. "
            # Pour garder un message complet sans casser la lisibilité.
            "Lancez `make download_dataset` ou pointez --raw-dir vers "
            # Pour garder un message complet sans casser la lisibilité.
            "un dataset EEGMMIDB complet."
        )

    # Pour échouer tôt si tôt si le fichier .edf.event est absent ou vide
    if not event_path.exists() or event_path.stat().st_size == 0:
        # Pour rendre l’échec explicite au point exact où le contrat est violé.
        raise FileNotFoundError(
            # Pour garder un message complet sans casser la lisibilité.
            "Fichier événement introuvable pour "
            # Pour garder un message complet sans casser la lisibilité.
            f"{subject} {run}: {event_path}. "
            # Pour garder un message complet sans casser la lisibilité.
            "Le dataset semble incomplet: relancez `make download_dataset` "
            # Pour garder un message complet sans casser la lisibilité.
            f"ou définissez {DATA_DIR_ENV_VAR} vers un dossier valide."
        )

    # Pour garantir l'arborescence cible pour déposer les .npy
    features_path.parent.mkdir(parents=True, exist_ok=True)

    # Pour récupérer l'EDF en conservant les métadonnées essentielles
    raw, _ = preprocessing.load_physionet_raw(
        # Pour propager le chemin EDF brut
        raw_path,
        # Pour propager la référence EEG configurée
        reference=build_context.eeg_reference,
    )

    # Pour aligner un notch si une fréquence valide est fournie
    if build_context.preprocess_config.notch_freq > 0.0:
        # Pour aligner le notch pour supprimer la pollution secteur
        notched_raw = preprocessing.apply_notch_filter(
            # Pour propager le signal brut chargé
            raw,
            # Pour propager la fréquence de notch configurée
            freq=build_context.preprocess_config.notch_freq,
        )
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour préserver le signal brut si le notch est désactivé
        notched_raw = raw
    # Pour aligner le filtrage bande-passante pour stabiliser les bandes MI
    filtered_raw = preprocessing.apply_bandpass_filter(
        # Pour propager le signal après notch (ou brut si désactivé)
        notched_raw,
        # Pour propager la bande passante configurée pour la MI
        freq_band=build_context.preprocess_config.bandpass_band,
    )

    # Pour aligner les annotations en événements moteurs après filtrage
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(
        # Pour transporter un signal déjà stabilisé pour la suite du flux.
        filtered_raw
    )

    # Pour adapter les fenêtres candidates en fonction du sujet
    candidate_windows = resolve_epoch_windows(
        # Pour rendre explicite ce point de décision ou de contrat.
        subject,
        # Pour propager la configuration active réellement retenue par le module.
        ACTIVE_EPOCH_WINDOW_CONFIG.config,
    )
    # Pour construire le contexte nécessaire à la sélection de fenêtre
    window_context = EpochWindowContext(
        # Pour transmettre le signal filtré déjà prêt pour l’epoching.
        filtered_raw=filtered_raw,
        # Pour transmettre les événements alignés sur le signal courant.
        events=events,
        # Pour transmettre le mapping d’événements cohérent avec les labels.
        event_id=event_id,
        # Pour transmettre les labels avant encodage numérique.
        motor_labels=motor_labels,
        # Pour fixer explicitement ce réglage dans l’objet construit.
        subject=subject,
        # Pour fixer explicitement ce réglage dans l’objet construit.
        run=run,
        # Pour rendre explicites les fenêtres réellement comparées.
        windows=candidate_windows,
    )
    # Pour retenir la meilleure fenêtre et ses données associées
    best_window, best_epochs_data, best_labels = _select_best_epoch_window(
        # Pour regrouper les dépendances de sélection de fenêtre.
        window_context
    )
    # Pour harmoniser les epochs par canal si la méthode est activée
    if build_context.preprocess_config.normalize_method != "none":
        # Pour aligner la normalisation par canal sur chaque epoch
        normalized_epochs = preprocessing.normalize_epoch_data(
            # Pour propager les epochs sélectionnées pour la normalisation
            best_epochs_data,
            # Pour propager la méthode de normalisation choisie
            method=build_context.preprocess_config.normalize_method,
            # Pour propager l'epsilon de stabilité pour éviter les divisions nulles
            epsilon=build_context.preprocess_config.normalize_epsilon,
        )
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour préserver les epochs brutes lorsque la normalisation est désactivée
        normalized_epochs = best_epochs_data

    # Pour préserver les epochs (brutes ou normalisées) sélectionnées
    np.save(features_path, normalized_epochs)
    # Pour préserver les labels alignés sur la fenêtre retenue
    np.save(labels_path, best_labels)
    # Pour persister la fenêtre retenue pour la réutiliser en prédiction
    _write_epoch_window_metadata(
        # Pour propager l'identifiant de sujet pour le chemin JSON
        subject,
        # Pour propager l'identifiant de run pour le chemin JSON
        run,
        # Pour centraliser la racine locale des caches numpy.
        build_context.data_dir,
        # Pour propager la fenêtre retenue pour la persistance
        best_window,
    )

    # Pour garder les chemins nouvellement générés
    return features_path, labels_path


# Pour construire les .npy pour l'ensemble des sujets disponibles
def _build_all_npy(
    # Pour centraliser les dépendances de génération des caches numpy.
    build_context: NpyBuildContext,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> None:
    """Génère les fichiers numpy pour chaque run moteur disponible."""

    # Pour couvrir les dossiers de sujets triés pour des logs prédictibles
    subject_dirs = sorted(
        # Pour parcourir sur les entrées du répertoire brut
        path
        # Pour réaligner les entrées pour ne garder que des dossiers
        for path in build_context.raw_dir.iterdir()
        # Pour préserver uniquement les répertoires sujets
        if path.is_dir()
    )

    # Pour couvrir tous les sujets détectés sans hypothèse sur leur nombre.
    for subject_dir in subject_dirs:
        # Pour isoler l'identifiant du sujet à partir du nom de dossier
        subject = subject_dir.name
        # Pour recenser tous les enregistrements EDF associés au sujet courant
        edf_paths = sorted(subject_dir.glob(f"{subject}R*.edf"))

        # Pour convertir chaque EDF éligible en cache numpy réutilisable.
        for edf_path in edf_paths:
            # Pour distinguer le run en retirant le préfixe sujet du nom de fichier
            run = edf_path.stem.replace(subject, "")

            # Pour écarter explicitement les runs dépourvus d'événements moteurs
            try:
                # Pour préparer explicitement cet objet intermédiaire avant usage.
                _build_npy_from_edf(
                    # Pour propager l'identifiant de sujet courant
                    subject,
                    # Pour propager l'identifiant de run courant
                    run,
                    # Pour propager la configuration de génération des numpy
                    build_context,
                )
            # Pour conserver l'erreur de runs sans événements moteurs pour continuer
            except ValueError as error:
                # Pour laisser le batch avancer si un run n'apporte aucun signal moteur.
                if "No motor events present" in str(error):
                    # Pour rendre explicite l'utilisateur que le run est ignoré
                    print(
                        # Pour rendre le diagnostic exploitable sans ouvrir le code.
                        "INFO: Événements moteurs absents pour "
                        # Pour garder un message complet sans casser la lisibilité.
                        f"{subject} {run}, passage."
                    )
                    # Pour continuer au run suivant après l'information
                    continue
                # Pour ne pas masquer un échec inattendu derrière un simple skip métier.
                raise


# Pour recenser les sujets disponibles dans le répertoire brut
def _list_subjects(raw_dir: Path) -> list[str]:
    """Retourne les identifiants de sujets triés présents dans raw_dir."""

    # Pour construire la liste des dossiers de sujets pour préparer l'entraînement
    subjects = [entry.name for entry in raw_dir.iterdir() if entry.is_dir()]
    # Pour stabiliser les identifiants pour obtenir des logs stables et reproductibles
    subjects.sort()
    # Pour garder la liste triée pour l'appelant
    return subjects


# Pour isoler l’entraînement unitaire tout en réutilisant les ressources batch.
def _train_single_run(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour mutualiser les dépendances communes aux entraînements batch.
    resources: TrainingResources,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> bool:
    """Lance l'entraînement d'un sujet pour un run donné."""

    # Pour centraliser la requête complète pour exécuter run_training
    request = TrainingRequest(
        # Pour propager le sujet cible
        subject=subject,
        # Pour propager le run cible
        run=run,
        # Pour propager la configuration pipeline
        pipeline_config=resources.pipeline_config,
        # Pour propager le répertoire de données numpy
        data_dir=resources.data_dir,
        # Pour propager le répertoire d'artefacts
        artifacts_dir=resources.artifacts_dir,
        # Pour propager le répertoire des EDF bruts
        raw_dir=resources.raw_dir,
        # Pour propager la référence EEG pour le chargement
        eeg_reference=resources.eeg_reference,
        # Pour propager la configuration de prétraitement
        preprocess_config=resources.preprocess_config,
    )
    # Pour préserver l'appel pour signaler les données manquantes sans stopper la boucle
    try:
        # Pour produire les artefacts finaux à partir de la requête préparée.
        _ = run_training(request)
    # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
    except FileNotFoundError as error:
        # Pour rendre le skip explicite quand les données sources sont incomplètes.
        print(f"AVERTISSEMENT: {error}")
        # Pour remonter un statut exploitable au batch appelant.
        return False
    # Pour garder True pour signaler un entraînement réussi
    return True


# Pour industrialiser l’entraînement complet sur tout le dataset disponible.
def _train_all_runs(
    # Pour mutualiser les dépendances communes aux entraînements batch.
    resources: TrainingResources,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> int:
    """Parcourt les sujets et runs moteurs pour générer tous les modèles."""

    # Pour isoler la liste des sujets disponibles dans le répertoire brut
    subjects = _list_subjects(resources.raw_dir)
    # Pour centraliser un compteur d'échecs pour informer l'utilisateur à la fin
    failures = 0
    # Pour couvrir chaque sujet détecté
    for subject in subjects:
        # Pour couvrir chaque run moteur attendu
        for run in MOTOR_RUNS:
            # Pour déterminer le chemin EDF attendu pour vérifier l'existence
            raw_path = resources.raw_dir / subject / f"{subject}{run}.edf"
            # Pour écarter le couple lorsque l'EDF est absent du disque
            if not raw_path.exists():
                # Pour rendre explicite l'utilisateur de l'absence pour transparence
                print(
                    # Pour rendre le diagnostic exploitable sans ouvrir le code.
                    "INFO: EDF introuvable pour "
                    # Pour garder un message complet sans casser la lisibilité.
                    f"{subject} {run} dans {raw_path.parent}, passage."
                )
                # Pour continuer au run suivant sans incrémenter les échecs
                continue
            # Pour distinguer proprement succès et échec de chaque run.
            success = _train_single_run(
                # Pour garder l’identité du sujet explicite dans le contrat.
                subject,
                # Pour garder l’identité du run explicite dans le contrat.
                run,
                # Pour mutualiser les dépendances communes aux entraînements batch.
                resources,
            )
            # Pour comptabiliser le compteur d'échecs lorsque l'entraînement échoue
            if not success:
                # Pour rendre explicite ce point de décision ou de contrat.
                failures += 1
    # Pour rendre visible un résumé pour guider l'utilisateur après la boucle
    if failures:
        # Pour donner un bilan immédiatement actionnable en fin de batch.
        print(
            # Pour donner un bilan actionnable sans masquer le détail utile.
            "AVERTISSEMENT: certains entraînements ont échoué. "
            # Pour garder un message complet sans casser la lisibilité.
            f"Exécutions manquantes: {failures}."
        )
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour signaler explicitement qu’aucun run n’a été perdu en chemin.
        print("INFO: modèles entraînés pour tous les runs moteurs détectés.")
    # Pour garder 1 si des échecs sont survenus pour refléter l'état global
    return 1 if failures else 0


# Pour valider si les caches existants respectent les shapes attendues
def _needs_rebuild_from_shapes(
    # Pour valider le cache features avant de le réutiliser.
    candidate_X: np.ndarray,
    # Pour valider le cache labels avant de le réutiliser.
    candidate_y: np.ndarray,
    # Pour conserver la provenance du cache dans le diagnostic.
    features_path: Path,
    # Pour conserver la provenance des labels dans le diagnostic.
    labels_path: Path,
    # Pour rendre le diagnostic lisible dans les logs batch.
    run_label: str,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> bool:
    """Valide les dimensions des caches existants pour éviter des erreurs."""

    # Pour éviter une régénération infinie sur des sujets à faible effectif.
    # Pour accepter un cache partiel dès lors qu’il reste exploitable.
    MIN_REQUIRED_SAMPLES = 1

    # Pour court-circuiter tôt un état qui casserait le contrat aval.
    if candidate_X.shape[0] < MIN_REQUIRED_SAMPLES:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            f"INFO: Cache vide ou invalide pour {run_label}. "
            # Pour garder un message complet sans casser la lisibilité.
            f"Régénération depuis l'EDF..."
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return True

    # Pour valider que X respecte la dimension attendue pour les epochs
    if candidate_X.ndim != EXPECTED_FEATURES_DIMENSIONS:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            f"INFO: X chargé depuis '{features_path}' a "
            # Pour garder un message complet sans casser la lisibilité.
            f"ndim={candidate_X.ndim} au lieu de "
            # Pour garder un message complet sans casser la lisibilité.
            f"{EXPECTED_FEATURES_DIMENSIONS}, régénération depuis l'EDF..."
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return True

    # Pour valider que y reste un vecteur 1D pour les labels
    if candidate_y.ndim != 1:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            f"INFO: y chargé depuis '{labels_path}' a "
            # Pour garder un message complet sans casser la lisibilité.
            f"ndim={candidate_y.ndim} au lieu de 1, régénération depuis l'EDF..."
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return True

    # Pour détecter un désalignement qui rendrait l’entraînement incohérent.
    if candidate_X.shape[0] != candidate_y.shape[0]:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            "INFO: Désalignement détecté pour "
            # Pour garder un message complet sans casser la lisibilité.
            f"{run_label}: X.shape[0]={candidate_X.shape[0]} != "
            # Pour garder un message complet sans casser la lisibilité.
            f"y.shape[0]={candidate_y.shape[0]}. Régénération..."
        )
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return True

    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return False


# Pour rendre la décision de rebuild isolée et facilement testable.
def _should_check_shapes(
    # Pour garder ce paramètre explicite dans le contrat.
    needs_rebuild: bool,
    # Pour garder ce paramètre explicite dans le contrat.
    corrupted_reason: str | None,
    # Pour valider le cache features avant de le réutiliser.
    candidate_X: np.ndarray | None,
    # Pour valider le cache labels avant de le réutiliser.
    candidate_y: np.ndarray | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> bool:
    """Détermine si la validation des shapes est nécessaire."""

    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return (
        # Pour rendre explicite ce point de décision ou de contrat.
        not needs_rebuild
        # Pour n’activer cette branche que si la précondition complémentaire tient.
        and corrupted_reason is None
        # Pour n’activer cette branche que si la précondition complémentaire tient.
        and candidate_X is not None
        # Pour n’activer cette branche que si la précondition complémentaire tient.
        and candidate_y is not None
    )


# Pour récupérer ou génère les matrices numpy attendues pour l'entraînement
def _load_data(
    # Pour garder l’identité du sujet explicite dans le contrat.
    subject: str,
    # Pour garder l’identité du run explicite dans le contrat.
    run: str,
    # Pour centraliser les dépendances de génération des caches numpy.
    build_context: NpyBuildContext,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[np.ndarray, np.ndarray]:
    """Charge ou construit les données et étiquettes pour un run.

    - Si les .npy n'existent pas, on les génère depuis l'EDF.
    - Si X existe mais n'est pas 3D, on reconstruit depuis l'EDF.
    - Si X et y n'ont pas le même nombre d'échantillons, on
      reconstruit pour réaligner les labels sur les epochs.
    """

    # Pour stabiliser les messages autour d’un identifiant de run unique.
    run_label = f"{subject} {run}"

    # Pour centraliser les deux caches attendus avant lecture ou rebuild.
    features_path, labels_path = _resolve_data_paths(
        # Pour garder les chemins dérivés alignés sur le sujet demandé.
        subject,
        # Pour garder les chemins dérivés alignés sur le run demandé.
        run,
        # Pour centraliser la racine locale des caches numpy.
        build_context.data_dir,
    )

    # Pour figer un bool Python strict et éviter une ambiguïté de contrat.
    needs_rebuild: bool = False
    # Pour mémoriser les chemins invalides pour enrichir les logs utilisateurs
    corrupted_reason: str | None = None
    # Pour préserver les caches chargés pour valider leurs formes
    candidate_X: np.ndarray | None = None
    # Pour préserver les labels chargés pour valider la longueur
    candidate_y: np.ndarray | None = None

    # Pour déclencher une reconstruction dès qu’un cache manque au couple demandé.
    if not features_path.exists() or not labels_path.exists():
        # Pour matérialiser une valeur intermédiaire utile au diagnostic.
        needs_rebuild = True
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour tolérer le chargement numpy pour tolérer les fichiers corrompus
        try:
            # Pour récupérer X en mmap pour inspecter la forme sans tout charger
            candidate_X = np.load(features_path, mmap_mode="r")
            # Pour récupérer y en mmap pour inspecter la longueur
            candidate_y = np.load(labels_path, mmap_mode="r")
        # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
        except (OSError, ValueError) as error:
            # Pour déclencher la reconstruction dès qu'un chargement échoue
            needs_rebuild = True
            # Pour préserver la raison pour orienter l'utilisateur
            corrupted_reason = str(error)

    # Pour valider les shapes lorsque les caches ont été chargés avec succès
    if _should_check_shapes(needs_rebuild, corrupted_reason, candidate_X, candidate_y):
        # Pour normaliser X vers un tableau typé pour satisfaire mypy et bandit
        validated_X = cast(np.ndarray, candidate_X)
        # Pour normaliser y vers un vecteur typé pour satisfaire mypy et bandit
        validated_y = cast(np.ndarray, candidate_y)
        # Pour détecter les incohérences de dimension et déclenche une régénération
        needs_rebuild = bool(
            # Pour préparer explicitement cet objet intermédiaire avant usage.
            _needs_rebuild_from_shapes(
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                validated_X,
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                validated_y,
                # Pour conserver la provenance du cache dans le diagnostic.
                features_path,
                # Pour conserver la provenance des labels dans le diagnostic.
                labels_path,
                # Pour rendre le diagnostic lisible dans les logs batch.
                run_label,
            )
        )

    # Pour rendre explicite un cache corrompu avant la phase de rebuild.
    if corrupted_reason is not None:
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            "INFO: Chargement numpy impossible pour "
            # Pour garder un message complet sans casser la lisibilité.
            f"{subject} {run}: {corrupted_reason}. "
            # Pour garder un message complet sans casser la lisibilité.
            "Régénération depuis l'EDF..."
        )
        # Pour matérialiser une valeur intermédiaire utile au diagnostic.
        needs_rebuild = True

    # Pour empêcher qu’un booléen numpy brouille le contrat de retour.
    needs_rebuild = True if needs_rebuild else False

    # Pour régénérer les caches seulement quand la validation l’exige.
    if needs_rebuild:
        # Pour déclencher la reconstruction avec la configuration active
        features_path, labels_path = _build_npy_from_edf(
            # Pour propager l'identifiant de sujet pour reconstruire les numpy
            subject,
            # Pour propager l'identifiant de run pour reconstruire les numpy
            run,
            # Pour propager la configuration de génération des numpy
            build_context,
        )

    # Pour récupérer les données validées (3D) et labels réalignés
    X = np.load(features_path)
    # Pour garder les labels explicites dans le contrat de la fonction.
    y = np.load(labels_path)

    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return X, y


# Pour isoler le hash git courant pour tracer la reproductibilité
def _get_git_commit() -> str:
    """Retourne le hash du commit courant ou "unknown" en secours."""

    # Pour pointer vers le fichier HEAD pour extraire la référence courante
    head_path = Path(".git") / "HEAD"
    # Pour garder unknown lorsque le dépôt git n'est pas disponible
    if not head_path.exists():
        # Pour fournir une valeur de repli pour conserver un manifeste valide
        return "unknown"
    # Pour distinguer un HEAD symbolique d’un hash déjà résolu.
    head_content = head_path.read_text().strip()
    # Pour détecter les références symboliques du style "ref: ..."
    if head_content.startswith("ref:"):
        # Pour rejoindre la vraie référence git quand HEAD pointe vers une ref.
        ref_path = Path(".git") / head_content.split(" ", 1)[1]
        # Pour garder unknown si la référence est introuvable
        if not ref_path.exists():
            # Pour fournir une valeur de repli pour préserver la validation
            return "unknown"
        # Pour récupérer le commit exact derrière la référence courante.
        return ref_path.read_text().strip()
    # Pour garder le contenu brut lorsque HEAD contient déjà un hash
    return head_content or "unknown"


# Pour préserver un manifeste complet à côté du modèle entraîné
def _flatten_hyperparams(hyperparams: dict) -> dict[str, str]:
    """Aplati les hyperparamètres pour une exportation CSV lisible."""

    # Pour centraliser un dictionnaire de sortie initialement vide
    flattened: dict[str, str] = {}
    # Pour couvrir chaque entrée pour extraire les valeurs simples
    for key, value in hyperparams.items():
        # Pour préserver chaque valeur pour conserver la lisibilité CSV
        flattened[key] = json.dumps(value, ensure_ascii=False)
    # Pour garder le dictionnaire aplati prêt pour l'écriture CSV
    return flattened


# Pour centraliser la traçabilité d’un run dans un seul point.
def _write_manifest(
    # Pour transmettre un contrat d’entraînement complet et stable.
    request: TrainingRequest,
    # Pour garder ce paramètre explicite dans le contrat.
    target_dir: Path,
    # Pour garder ce paramètre explicite dans le contrat.
    cv_scores: np.ndarray,
    # Pour garder ce paramètre explicite dans le contrat.
    artifacts: dict[str, Path | None],
    # Pour garder ce paramètre explicite dans le contrat.
    search_summary: dict[str, object] | None = None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> dict[str, Path]:
    """Écrit des manifestes JSON et CSV décrivant le run d'entraînement."""

    # Pour récupérer la fenêtre d'epochs persistée pour l'inclure au manifeste
    epoch_window = _read_epoch_window_metadata(
        # Pour garder l’identité du sujet explicite dans le contrat.
        request.subject,
        # Pour garder l’identité du run explicite dans le contrat.
        request.run,
        # Pour rendre la racine de données injectable et testable.
        request.data_dir,
    )
    # Pour centraliser la section dataset pour identifier les entrées de données
    dataset = {
        # Pour rattacher explicitement le manifeste au sujet traité.
        "subject": request.subject,
        # Pour rattacher explicitement le manifeste au run traité.
        "run": request.run,
        # Pour conserver la provenance locale des données utilisées.
        "data_dir": str(request.data_dir),
        # Pour relier l’artefact à la fenêtre effectivement retenue.
        "epoch_window": epoch_window,
    }
    # Pour normaliser la configuration de pipeline en dictionnaire sérialisable
    hyperparams = asdict(request.pipeline_config)
    # Pour déterminer la moyenne des scores si la validation croisée a tourné
    cv_mean = float(np.mean(cv_scores)) if cv_scores.size else None
    # Pour centraliser la section des scores en sérialisant les arrays numpy
    scores = {
        # Pour garder le détail complet des scores de validation.
        "cv_scores": cv_scores.tolist(),
        # Pour fournir une synthèse rapide de la qualité observée.
        "cv_mean": cv_mean,
    }
    # Pour adapter l'identifiant du commit git pour tracer les artefacts
    git_commit = _get_git_commit()
    # Pour centraliser la section chemins pour retrouver rapidement les fichiers
    artifacts_section = {
        # Pour pointer vers l’artefact principal réutilisable en prédiction.
        "model": str(artifacts["model"]),
        # Pour tracer l’artefact optionnel nécessaire au même pipeline.
        "scaler": str(artifacts["scaler"]) if artifacts["scaler"] else None,
        # Pour conserver la matrice utile à l’analyse de la réduction.
        "w_matrix": str(artifacts["w_matrix"]) if artifacts["w_matrix"] else None,
    }
    # Pour regrouper la traçabilité du run dans une seule structure sérialisable.
    manifest = {
        # Pour regrouper l’identité des données sources dans le manifeste.
        "dataset": dataset,
        # Pour tracer les réglages exacts de la pipeline entraînée.
        "hyperparams": hyperparams,
        # Pour exposer les métriques produites pendant l’entraînement.
        "scores": scores,
        # Pour relier l’artefact à un état précis du dépôt.
        "git_commit": git_commit,
        # Pour retrouver rapidement les fichiers matériels générés.
        "artifacts": artifacts_section,
    }
    # Pour exposer un résumé de recherche uniquement si une optimisation a eu lieu
    if search_summary is not None:
        # Pour rendre le résultat de la recherche lisible sans relire GridSearchCV.
        manifest["hyperparam_search"] = search_summary
    # Pour fixer le chemin de sortie du manifeste JSON à côté des artefacts
    manifest_json_path = target_dir / "manifest.json"
    # Pour centraliser un JSON tolérant les objets non sérialisables du manifeste
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
    # Pour persister le manifeste JSON sur disque en UTF-8 pour la portabilité
    manifest_json_path.write_text(manifest_json)
    # Pour rendre les hyperparamètres exploitables dans une vue CSV plate.
    flattened_hyperparams = _flatten_hyperparams(hyperparams)
    # Pour construire une ligne CSV unique regroupant toutes les informations
    csv_line = {
        # Pour rattacher explicitement le manifeste au sujet traité.
        "subject": request.subject,
        # Pour rattacher explicitement le manifeste au run traité.
        "run": request.run,
        # Pour conserver la provenance locale des données utilisées.
        "data_dir": str(request.data_dir),
        # Pour relier l’artefact à un état précis du dépôt.
        "git_commit": git_commit,
        # Pour garder le détail complet des scores de validation.
        "cv_scores": ";".join(str(score) for score in cv_scores.tolist()),
        # Pour fournir une synthèse rapide de la qualité observée.
        "cv_mean": "" if cv_mean is None else str(cv_mean),
        # Pour garder ce fragment explicite malgré le format multi-ligne.
        **flattened_hyperparams,
    }
    # Pour fixer le chemin du manifeste CSV à côté du JSON
    manifest_csv_path = target_dir / "manifest.csv"
    # Pour garder un CSV portable entre plateformes sans lignes vides parasites.
    with manifest_csv_path.open("w", newline="") as handle:
        # Pour préparer l'écriture CSV avec les clés détectées
        writer = csv.DictWriter(handle, fieldnames=list(csv_line.keys()))
        # Pour rendre le CSV auto-descriptif dès l’ouverture.
        writer.writeheader()
        # Pour conserver un enregistrement plat du run courant dans le CSV.
        writer.writerow(csv_line)
    # Pour garder les chemins des manifestes pour les appels appelants
    return {"json": manifest_json_path, "csv": manifest_csv_path}


# Pour construire la grille d'hyperparamètres par défaut pour l'optimisation
# Pour construire la grille d'hyperparamètres pour la recherche d'optimisation
def _build_classifier_grid(allow_lda: bool) -> list[object]:
    """Construit une grille compacte de classifieurs."""

    # Pour préparer la grille pour les classifieurs candidats
    classifier_grid: list[object] = []
    # Pour exposer LDA uniquement lorsque l'effectif le permet
    if allow_lda:
        # Pour conserver LDA shrinkage pour stabilité des covariances
        classifier_grid.append(
            # Pour rendre explicite ce point de décision ou de contrat.
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
    # Pour couvrir les valeurs de C pour limiter la complexité des modèles
    for c_value in DEFAULT_HYPERPARAM_C_VALUES:
        # Pour exposer une régression logistique régularisée
        classifier_grid.append(LogisticRegression(C=c_value, max_iter=1000))
        # Pour exposer un SVM linéaire régularisé
        classifier_grid.append(LinearSVC(C=c_value, max_iter=5000))
    # Pour exposer un classifieur centroïde pour les petits échantillons
    classifier_grid.append(CentroidClassifier())
    # Pour garder la grille compacte des classifieurs
    return classifier_grid


# Pour construire la grille des paramètres Welch pour la sélection interne
def _build_welch_config_grid(
    # Pour garder ce paramètre explicite dans le contrat.
    base_config: dict[str, object] | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> list[object]:
    """Construit une grille compacte pour les paramètres Welch."""

    # Pour toujours inclure la configuration demandée dans la recherche.
    welch_config_grid: list[object] = [base_config]
    # Pour exposer les tailles de fenêtre Welch ciblées
    for nperseg in DEFAULT_HYPERPARAM_WELCH_NPERSEG:
        # Pour exposer la config nperseg si distincte de la base
        if {"nperseg": nperseg} not in welch_config_grid:
            # Pour élargir la recherche sans perdre la valeur de base demandée.
            welch_config_grid.append({"nperseg": nperseg})
    # Pour garder la grille Welch finale
    return welch_config_grid


# Pour éviter des grilles divergentes entre chemins d’entraînement.
def _build_grid_search_grid(
    # Pour faire circuler la configuration sans la re-décomposer.
    config: tpv_pipeline.PipelineConfig,
    # Pour adapter la grille aux contraintes numériques des petits effectifs.
    allow_lda: bool,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> dict[str, list[object]]:
    """Retourne une grille raisonnable pour la recherche d'hyperparamètres."""

    # Pour détecter les stratégies qui s'appuient sur des features de signal
    uses_signal_features = config.feature_strategy in {"welch", "wavelet"}
    # Pour garder une grille réduite si CSP/CSSP est utilisé
    if config.dim_method in {"csp", "cssp"}:
        # Pour fixer un ensemble restreint de classifieurs pour CSP
        csp_classifier_grid = _build_classifier_grid(allow_lda)
        # Pour centraliser une liste de composantes CSP sans doublons
        csp_n_components_grid: list[object] = (
            # Pour rendre explicite ce point de décision ou de contrat.
            [config.n_components]
            # Pour court-circuiter tôt un état qui casserait le contrat aval.
            if config.n_components is not None
            # Pour rendre explicite ce point de décision ou de contrat.
            else [None, DEFAULT_CSP_COMPONENTS]
        )
        # Pour exposer la valeur explicite si absente de la grille
        if (
            # Pour rendre explicite ce point de décision ou de contrat.
            config.n_components is not None
            # Pour n’activer cette branche que si la précondition complémentaire tient.
            and config.n_components not in csp_n_components_grid
            # Pour figer un contrat exploitable par les appels et l’outillage de types.
        ):
            # Pour rendre explicite ce point de décision ou de contrat.
            csp_n_components_grid.append(config.n_components)
        # Pour centraliser une grille de configuration Welch si besoin
        welch_config_grid = _build_welch_config_grid(config.feature_strategy_config)
        # Pour garder une grille compatible avec la pipeline CSP réduite
        grid = {
            # Pour conserver cette information dans le résumé produit.
            "spatial_filters__n_components": csp_n_components_grid,
            # Pour conserver cette information dans le résumé produit.
            "classifier": csp_classifier_grid,
        }
        # Pour n'ajouter Welch que lorsque les features de signal sont actives.
        if uses_signal_features:
            # Pour accepter la variation des paramètres Welch au sein de CSP
            grid["features__strategy_config"] = welch_config_grid
        # Pour garder la grille configurée pour CSP
        return grid

    # Pour fixer un ensemble restreint de classifieurs pour limiter la complexité
    # Pour préparer la grille de classifieurs candidates
    classifier_grid = _build_classifier_grid(allow_lda)
    # Pour fixer des scalers optionnels, dont passthrough pour désactiver
    scaler_grid: list[object] = ["passthrough", StandardScaler(), RobustScaler()]
    # Pour fixer une plage compacte de composantes pour PCA/SVD
    n_components_grid: list[object] = [None, 2, 4, 8]
    # Pour exposer la valeur demandée explicitement pour garantir sa présence
    if config.n_components is not None and config.n_components not in n_components_grid:
        # Pour rendre explicite ce point de décision ou de contrat.
        n_components_grid.append(config.n_components)
    # Pour centraliser une grille de configuration Welch si besoin
    welch_config_grid = _build_welch_config_grid(config.feature_strategy_config)
    # Pour construire la grille finale en couvrant features + réduction + classif
    # Pour comparer des familles de features proches sans dupliquer la logique.
    return {
        # Pour conserver cette information dans le résumé produit.
        "features__feature_strategy": ["fft", "welch", ("fft", "welch"), "wavelet"],
        # Pour conserver cette information dans le résumé produit.
        "features__strategy_config": welch_config_grid,
        # Pour conserver cette information dans le résumé produit.
        "features__normalize": [True, False],
        # Pour conserver cette information dans le résumé produit.
        "dimensionality__method": ["pca", "svd"],
        # Pour conserver cette information dans le résumé produit.
        "dimensionality__n_components": n_components_grid,
        # Pour conserver cette information dans le résumé produit.
        "classifier": classifier_grid,
        # Pour tracer l’artefact optionnel nécessaire au même pipeline.
        "scaler": scaler_grid,
    }


# Pour isoler les scores par split d'une GridSearchCV pour le meilleur modèle
def _extract_grid_search_scores(
    # Pour accéder aux résultats détaillés du meilleur modèle uniquement.
    search: sklearn_model_selection.GridSearchCV,
    # Pour borner le nombre de colonnes de score attendues dans cv_results_.
    n_splits: int,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> np.ndarray:
    """Construit un tableau de scores à partir des résultats de grid search."""

    # Pour isoler l'index du meilleur modèle sélectionné
    best_index = int(search.best_index_)
    # Pour construire la liste des scores par split pour l'entrée manifeste
    split_scores = []
    # Pour couvrir explicitement chaque élément utile sans logique implicite.
    for split_index in range(n_splits):
        # Pour stabiliser le chemin du la clé de split attendue dans cv_results_
        key = f"split{split_index}_test_score"
        # Pour isoler la colonne associée si disponible
        column = search.cv_results_.get(key)
        # Pour écarter les splits manquants si la CV est réduite
        if column is None:
            # Pour ignorer ce cas sans mélanger le flux nominal et le cas écarté.
            continue
        # Pour rendre explicite ce point de décision ou de contrat.
        split_scores.append(float(column[best_index]))
    # Pour garder un tableau numpy stable même si partiellement rempli
    return np.array(split_scores, dtype=float)


# Pour centraliser l'entraînement et la CV optionnelle pour un dataset donné
def _train_with_optional_cv(
    # Pour transmettre un contrat d’entraînement complet et stable.
    request: TrainingRequest,
    # Pour garder les features explicites dans le contrat de la fonction.
    X: np.ndarray,
    # Pour garder les labels explicites dans le contrat de la fonction.
    y: np.ndarray,
    # Pour manipuler la pipeline complète sans état global caché.
    pipeline: Pipeline,
    # Pour garder ce paramètre explicite dans le contrat.
    adapted_config: tpv_pipeline.PipelineConfig,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tuple[np.ndarray, Pipeline, dict[str, object] | None, str | None, str | None]:
    """Retourne la pipeline entraînée et les scores CV si disponibles."""

    # Pour déterminer le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Pour déterminer le nombre de classes distinctes pour l'évaluation
    class_count = int(np.unique(y).size)
    # Pour adapter le splitter CV en tenant compte des classes disponibles
    cv, cv_unavailability_reason = _resolve_cv_splits(y, DEFAULT_CV_SPLITS)
    # Pour préparer un tableau vide lorsque la validation croisée est impossible
    cv_scores = np.array([])
    # Pour centraliser un résumé de recherche d'hyperparamètres si besoin
    search_summary: dict[str, object] | None = None
    # Pour centraliser un message d'erreur si la CV échoue malgré le splitter.
    cv_error: str | None = None
    # Pour valider si l'effectif autorise une validation croisée exploitable
    if cv is None:
        # Pour centraliser la raison finale pour expliquer l'absence de CV
        cv_unavailability_reason = (
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            cv_unavailability_reason
            # Pour n’activer cette branche que si la précondition complémentaire tient.
            or _describe_cv_unavailability(y, DEFAULT_CV_SPLITS)
        )
        # Pour aligner le fallback CV et récupère la pipeline ajustée
        cv_scores, pipeline, cv_unavailability_reason = _handle_cv_unavailability(
            # Pour manipuler la pipeline complète sans état global caché.
            pipeline,
            # Pour garder les features explicites dans le contrat de la fonction.
            X,
            # Pour garder les labels explicites dans le contrat de la fonction.
            y,
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            cv_unavailability_reason,
        )
    # Pour déclencher une recherche systématique des hyperparamètres si demandée
    elif request.enable_grid_search:
        # Pour construire la pipeline dédiée au grid search
        search_pipeline = build_search_pipeline(adapted_config)
        # Pour interdire LDA quand le ratio rend son fit numériquement fragile.
        allow_lda = sample_count > class_count
        # Pour construire la grille d'hyperparamètres à explorer
        param_grid = _build_grid_search_grid(adapted_config, allow_lda)
        # Pour adapter la recherche interne à la taille réelle du dataset.
        search_splits = request.grid_search_splits or DEFAULT_HYPERPARAM_SPLITS
        # Pour construire un splitter dédié à la sélection interne
        search_cv, search_reason = _resolve_hyperparam_splits(y, search_splits)
        # Pour éviter une recherche coûteuse qui échouerait avant même de démarrer.
        if search_cv is None:
            # Pour centraliser une raison explicite pour l'absence de sélection
            cv_unavailability_reason = search_reason or _describe_cv_unavailability(
                # Pour rendre explicite ce point de décision ou de contrat.
                y,
                # Pour relier le diagnostic au nombre de splits effectivement tenté.
                search_splits,
            )
            # Pour mesurer la CV finale sur la meilleure pipeline retenue.
            try:
                # Pour déclencher la cross-validation pour mesurer la performance finale
                cv_scores = cross_val_score(
                    # Pour évaluer la pipeline finale retenue après la recherche.
                    pipeline,
                    # Pour transmettre les features sans transformation additionnelle.
                    X,
                    # Pour transmettre les labels associés à cette évaluation finale.
                    y,
                    # Pour réutiliser le splitter principal déjà validé pour ce run.
                    cv=cv,
                    # Pour faire remonter les erreurs de CV plutôt que les masquer.
                    error_score="raise",
                )
            # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
            except ValueError as error:
                # Pour conserver l'erreur pour un diagnostic CLI explicite
                cv_error = str(error)
                # Pour fixer un score vide pour conserver le flux nominal
                cv_scores = np.array([])
            # Pour adapter la pipeline sur toutes les données après évaluation
            pipeline.fit(X, y)
            # Pour garder immédiatement les sorties sans sélection interne
            return (
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                cv_scores,
                # Pour manipuler la pipeline complète sans état global caché.
                pipeline,
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                search_summary,
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                cv_unavailability_reason,
                # Pour transmettre explicitement ce contexte à l’appel encapsulé.
                cv_error,
            )
        # Pour récupérer la meilleure pipeline quand la recherche est activée.
        _search_scores, pipeline, search_summary = _run_grid_search(
            # Pour séparer la pipeline de recherche de la pipeline finale.
            search_pipeline,
            # Pour rendre l’espace de recherche explicite et testable.
            param_grid,
            # Pour injecter une stratégie de CV déjà validée en amont.
            search_cv,
            # Pour garder les features explicites dans le contrat de la fonction.
            X,
            # Pour garder les labels explicites dans le contrat de la fonction.
            y,
        )
        # Pour déterminer les scores de validation croisée sur la pipeline sélectionnée
        try:
            # Pour déclencher la cross-validation pour mesurer la performance finale
            cv_scores = cross_val_score(
                # Pour évaluer la pipeline sélectionnée après la recherche interne.
                pipeline,
                # Pour transmettre les features sans transformation additionnelle.
                X,
                # Pour transmettre les labels associés à cette évaluation finale.
                y,
                # Pour réutiliser le splitter principal déjà validé pour ce run.
                cv=cv,
                # Pour faire remonter les erreurs de CV plutôt que les masquer.
                error_score="raise",
            )
        # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
        except ValueError as error:
            # Pour conserver l'erreur pour un diagnostic CLI explicite
            cv_error = str(error)
            # Pour fixer un score vide pour conserver le flux nominal
            cv_scores = np.array([])
        # Pour adapter la pipeline sur toutes les données après évaluation
        pipeline.fit(X, y)
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour déterminer les scores de validation croisée sur l'ensemble du pipeline
        try:
            # Pour déclencher la cross-validation pour mesurer la performance
            cv_scores = cross_val_score(
                # Pour évaluer la pipeline entraînée sans recherche interne.
                pipeline,
                # Pour transmettre les features sans transformation additionnelle.
                X,
                # Pour transmettre les labels associés à cette évaluation finale.
                y,
                # Pour réutiliser le splitter principal déjà validé pour ce run.
                cv=cv,
                # Pour faire remonter les erreurs de CV plutôt que les masquer.
                error_score="raise",
            )
        # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
        except ValueError as error:
            # Pour conserver l'erreur pour un diagnostic CLI explicite
            cv_error = str(error)
            # Pour fixer un score vide pour conserver le flux nominal
            cv_scores = np.array([])
        # Pour adapter la pipeline sur toutes les données après évaluation
        pipeline.fit(X, y)
    # Pour garder les informations calculées pour l'entraînement
    return (
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_scores,
        # Pour manipuler la pipeline complète sans état global caché.
        pipeline,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        search_summary,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_unavailability_reason,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_error,
    )


# Pour retenir l'étape de réduction de dimension ou les filtres spatiaux
def _resolve_reducer_step(pipeline: Pipeline) -> object | None:
    """Retourne l'étape de réduction à utiliser pour persister W."""

    # Pour retrouver la projection sérialisable quelle que soit la pipeline retenue.
    reducer = cast(object | None, pipeline.named_steps.get("dimensionality"))
    # Pour garder la réduction si elle existe déjà
    if reducer is not None:
        # Pour préserver l'étape explicite pour les pipelines classiques
        return reducer
    # Pour garder CSP/CSSP si la pipeline n'a pas de réduction dédiée
    return cast(object | None, pipeline.named_steps.get("spatial_filters"))


# Pour préserver une matrice W en respectant le format TPVDimReducer
def _persist_w_matrix(
    # Pour garder ce paramètre explicite dans le contrat.
    reducer: object,
    # Pour garder ce paramètre explicite dans le contrat.
    path: Path,
    # Pour garder ce paramètre explicite dans le contrat.
    adapted_config: tpv_pipeline.PipelineConfig,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> None:
    """Sauvegarde la matrice W pour les prédictions temps réel."""

    # Pour conserver la sauvegarde native pour les réducteurs TPV
    if isinstance(reducer, TPVDimReducer):
        # Pour préserver la logique de sérialisation propre au réducteur maison.
        reducer.save(path)
        # Pour borner l'exécution pour éviter un double dump
        return
    # Pour isoler la matrice W exposée par un CSP/CSSP
    w_matrix = getattr(reducer, "w_matrix", None)
    # Pour échouer tôt si la sauvegarde si aucun filtre n'est disponible
    if w_matrix is None:
        # Pour éviter sans écrire de fichier pour éviter un artefact vide
        return
    # Pour construire un réducteur TPV pour sérialiser W au format attendu
    fallback = TPVDimReducer(
        # Pour préserver la méthode de réduction demandée
        method=adapted_config.dim_method,
        # Pour préserver le nombre de composantes demandé
        n_components=adapted_config.n_components,
        # Pour préserver la régularisation CSP paramétrée
        regularization=adapted_config.csp_regularization,
    )
    # Pour fournir la matrice W pour fournir une projection cohérente
    fallback.w_matrix = w_matrix
    # Pour préserver les valeurs propres si elles existent pour diagnostic
    fallback.eigenvalues_ = getattr(reducer, "eigenvalues_", None)
    # Pour réutiliser la sérialisation au format TPVDimReducer attendu par predict
    fallback.save(path)


# Pour déclencher la validation croisée et l'entraînement final
def run_training(request: TrainingRequest) -> dict:
    """Entraîne la pipeline et sauvegarde ses artefacts."""

    # Pour construire le contexte de génération des numpy à partir de la requête
    build_context = NpyBuildContext(
        # Pour propager le répertoire des numpy
        data_dir=request.data_dir,
        # Pour propager le répertoire des EDF bruts
        raw_dir=request.raw_dir,
        # Pour propager la référence EEG configurée
        eeg_reference=request.eeg_reference,
        # Pour propager la configuration de prétraitement
        preprocess_config=request.preprocess_config,
    )
    # Pour récupérer ou génère les tableaux numpy nécessaires à l'entraînement
    X, y = _load_data(
        # Pour propager le sujet demandé pour chargement
        request.subject,
        # Pour propager le run demandé pour chargement
        request.run,
        # Pour propager le contexte de génération des numpy
        build_context,
    )
    # Pour assouplir automatiquement la config sur des effectifs trop faibles.
    adapted_config = _adapt_pipeline_config_for_samples(request.pipeline_config, y)
    # Pour construire la pipeline complète sans préprocesseur amont
    pipeline = build_pipeline(adapted_config)
    # Pour déclencher la procédure CV et récupère les artefacts d'entraînement
    (
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_scores,
        # Pour manipuler la pipeline complète sans état global caché.
        pipeline,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        search_summary,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_unavailability_reason,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_error,
        # Pour préparer explicitement cet objet intermédiaire avant usage.
    ) = _train_with_optional_cv(
        # Pour transmettre un contrat d’entraînement complet et stable.
        request,
        # Pour garder les features explicites dans le contrat de la fonction.
        X,
        # Pour garder les labels explicites dans le contrat de la fonction.
        y,
        # Pour manipuler la pipeline complète sans état global caché.
        pipeline,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        adapted_config,
    )
    # Pour rendre explicite l'utilisateur lorsque la validation croisée est désactivée
    if cv_unavailability_reason and cv_scores.size == 0:
        # Pour rendre visible un message explicite pour signaler le fallback direct
        print(
            # Pour rendre le diagnostic exploitable sans ouvrir le code.
            "INFO: validation croisée indisponible, entraînement direct sans cross-val"
        )
    # Pour centraliser le dossier d'artefacts spécifique au sujet et au run
    target_dir = request.artifacts_dir / request.subject / request.run
    # Pour garantir l'existence du parent pour stabiliser la création du dossier cible
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    # Pour garantir les répertoires au besoin pour éviter les erreurs de sauvegarde
    target_dir.mkdir(parents=True, exist_ok=True)
    # Pour déterminer le chemin du fichier modèle pour joblib
    model_path = target_dir / "model.joblib"
    # Pour préserver la pipeline complète pour les prédictions futures
    save_pipeline(pipeline, str(model_path))
    # Pour isoler l'éventuel scaler pour une sauvegarde dédiée
    scaler_step = pipeline.named_steps.get("scaler")
    # Pour préserver le scaler uniquement s'il est présent dans la pipeline
    if scaler_step is not None and scaler_step != "passthrough":
        # Pour rendre inspectable le scaler dans un fichier distinct pour inspection
        joblib.dump(scaler_step, target_dir / "scaler.joblib")
    # Pour isoler le réducteur de dimension pour exposer la matrice W
    dim_reducer = _resolve_reducer_step(pipeline)
    # Pour centraliser un chemin de sauvegarde pour la matrice W si disponible
    w_matrix_path: Path | None = None
    # Pour isoler la matrice W si l'étape est disponible
    w_matrix = getattr(dim_reducer, "w_matrix", None) if dim_reducer else None
    # Pour préserver la matrice de projection seulement si elle existe
    if w_matrix is not None and dim_reducer is not None:
        # Pour stabiliser le chemin cible pour la matrice W sérialisée
        w_matrix_path = target_dir / "w_matrix.joblib"
        # Pour préserver la matrice de projection pour les usages temps-réel
        _persist_w_matrix(dim_reducer, w_matrix_path, adapted_config)
    # Pour déterminer le chemin du scaler pour l'ajouter au manifeste
    scaler_path = None
    # Pour propager le chemin du scaler uniquement lorsqu'il existe
    if scaler_step is not None and scaler_step != "passthrough":
        # Pour mémoriser le chemin vers le scaler sauvegardé pour le manifeste
        scaler_path = target_dir / "scaler.joblib"
    # Pour persister un manifeste décrivant l'entraînement et ses artefacts
    manifest_paths = _write_manifest(
        # Pour transmettre un contrat d’entraînement complet et stable.
        request,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        target_dir,
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        cv_scores,
        # Pour rendre explicite ce point de décision ou de contrat.
        {
            # Pour pointer vers l’artefact principal réutilisable en prédiction.
            "model": model_path,
            # Pour tracer l’artefact optionnel nécessaire au même pipeline.
            "scaler": scaler_path,
            # Pour conserver la matrice utile à l’analyse de la réduction.
            "w_matrix": w_matrix_path,
        },
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        search_summary,
    )
    # Pour garder un rapport synthétique pour les tests et la CLI
    return {
        # Pour garder le détail complet des scores de validation.
        "cv_scores": cv_scores,
        # Pour conserver cette information dans le résumé produit.
        "cv_splits_requested": DEFAULT_CV_SPLITS,
        # Pour conserver cette information dans le résumé produit.
        "cv_unavailability_reason": cv_unavailability_reason,
        # Pour conserver cette information dans le résumé produit.
        "cv_error": cv_error,
        # Pour conserver cette information dans le résumé produit.
        "model_path": model_path,
        # Pour conserver cette information dans le résumé produit.
        "scaler_path": scaler_path,
        # Pour conserver cette information dans le résumé produit.
        "w_matrix_path": w_matrix_path,
        # Pour conserver cette information dans le résumé produit.
        "manifest_path": manifest_paths["json"],
        # Pour conserver cette information dans le résumé produit.
        "manifest_csv_path": manifest_paths["csv"],
    }


# Pour aligner la configuration des fenêtres si un fichier est fourni
def _apply_epoch_window_config(args: argparse.Namespace) -> None:
    """Met à jour la configuration des fenêtres à partir des arguments."""

    # Pour éviter tôt si aucun fichier n'est fourni
    if args.epoch_window_config is None:
        # Pour terminer explicitement cette branche sans état supplémentaire.
        return
    # Pour récupérer et applique la configuration dédiée
    ACTIVE_EPOCH_WINDOW_CONFIG.config = tpv_utils.load_epoch_window_config(
        # Pour transmettre explicitement ce contexte à l’appel encapsulé.
        args.epoch_window_config
    )


# Pour centraliser la génération complète des fichiers .npy si demandée
def _maybe_build_all(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour garder ce paramètre explicite dans le contrat.
    preprocess_config: preprocessing.PreprocessingConfig,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> bool:
    """Construit tous les .npy et retourne True si exécuté."""

    # Pour écarter si le flag build_all est absent
    if not args.build_all:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return False
    # Pour construire les .npy avec la configuration active
    # Pour construire le contexte de génération des numpy pour la commande build-all
    build_context = NpyBuildContext(
        # Pour propager le répertoire des numpy
        data_dir=args.data_dir,
        # Pour propager le répertoire des EDF bruts
        raw_dir=args.raw_dir,
        # Pour propager la référence EEG configurée
        eeg_reference=args.eeg_reference,
        # Pour propager la configuration de prétraitement
        preprocess_config=preprocess_config,
    )
    # Pour préparer explicitement cet objet intermédiaire avant usage.
    _build_all_npy(
        # Pour propager le contexte de génération des numpy
        build_context,
    )
    # Pour distinguer un no-op complet d’une action réellement déclenchée.
    return True


# Pour centraliser l'entraînement massif si demandé
def _maybe_train_all(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour faire circuler la configuration sans la re-décomposer.
    config: tpv_pipeline.PipelineConfig,
    # Pour garder ce paramètre explicite dans le contrat.
    preprocess_config: preprocessing.PreprocessingConfig,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> int | None:
    """Lance l'entraînement massif et retourne le code si exécuté."""

    # Pour écarter si le flag train_all est absent
    if not args.train_all:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return None
    # Pour déclencher l'entraînement global avec la configuration active
    # Pour construire les ressources partagées pour l'entraînement global
    resources = TrainingResources(
        # Pour propager la configuration pipeline partagée
        pipeline_config=config,
        # Pour propager le répertoire des numpy
        data_dir=args.data_dir,
        # Pour propager le répertoire d'artefacts
        artifacts_dir=args.artifacts_dir,
        # Pour propager le répertoire des EDF bruts
        raw_dir=args.raw_dir,
        # Pour propager la référence EEG configurée
        eeg_reference=args.eeg_reference,
        # Pour propager la configuration de prétraitement
        preprocess_config=preprocess_config,
        # Pour propager l'activation du grid search si nécessaire
        enable_grid_search=args.grid_search,
        # Pour propager le nombre de splits pour la recherche
        grid_search_splits=args.grid_search_splits,
    )
    # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
    return _train_all_runs(
        # Pour propager les ressources centralisées pour l'entraînement global
        resources,
    )


# Pour construire la configuration de prétraitement à partir des arguments CLI
def _build_preprocess_config_from_args(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> preprocessing.PreprocessingConfig:
    """Construit la configuration de prétraitement validée."""

    # Pour rejeter une bande passante incohérente pour éviter un filtrage invalide
    if args.bandpass_low >= args.bandpass_high:
        # Pour rendre explicite l'incohérence pour la gestion d'erreur CLI
        raise ValueError("bandpass_low doit être inférieur à bandpass_high")
    # Pour construire la configuration de prétraitement à partir des arguments
    return preprocessing.PreprocessingConfig(
        # Pour fixer la bande passante MI configurée
        bandpass_band=(args.bandpass_low, args.bandpass_high),
        # Pour fixer la fréquence de notch configurée
        notch_freq=args.notch_freq,
        # Pour fixer la méthode de normalisation par canal
        normalize_method=args.normalize_channels,
        # Pour fixer l'epsilon de stabilité pour la normalisation
        normalize_epsilon=args.normalize_epsilon,
    )


# Pour construire la configuration pipeline à partir des arguments CLI
def _build_pipeline_config_from_args(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour tester la CLI sans dépendre implicitement de sys.argv.
    argv: list[str] | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> tpv_pipeline.PipelineConfig:
    """Construit la PipelineConfig alignée sur les arguments CLI."""

    # Pour normaliser l'option scaler "none" en None pour la pipeline
    scaler = None if args.scaler == "none" else args.scaler
    # Pour déterminer la valeur de normalisation en inversant le flag d'opt-out
    normalize = not args.no_normalize_features
    # Pour isoler le paramètre n_components s'il est fourni
    n_components = getattr(args, "n_components", None)
    # Pour adapter la fréquence d'échantillonnage en s'appuyant sur l'EDF si possible
    resolved_sfreq = resolve_sampling_rate(
        # Pour propager l'identifiant de sujet pour la résolution
        args.subject,
        # Pour propager l'identifiant de run pour la résolution
        args.run,
        # Pour propager le répertoire des EDF bruts
        args.raw_dir,
        # Pour propager la fréquence demandée en CLI
        args.sfreq,
        # Pour propager la référence EEG configurée
        args.eeg_reference,
    )
    # Pour lever l’ambiguïté CLI entre famille de features et méthode de réduction.
    feature_strategy, dim_method = _resolve_feature_strategy_and_dim_method(
        # Pour propager la stratégie de features demandée
        args.feature_strategy,
        # Pour propager la méthode de réduction demandée
        args.dim_method,
        # Pour propager argv pour tracer les choix explicites
        argv,
    )
    # Pour adapter la méthode de réduction en fonction de la stratégie de features
    dim_method = _resolve_dim_method_for_features(
        # Pour propager la stratégie de features résolue
        feature_strategy,
        # Pour propager la méthode de réduction résolue
        dim_method,
        # Pour propager argv pour tracer les choix explicites
        argv,
    )
    # Pour construire la configuration de pipeline alignée sur mybci
    return tpv_pipeline.PipelineConfig(
        # Pour propager la fréquence d'échantillonnage résolue
        sfreq=resolved_sfreq,
        # Pour propager la stratégie de features résolue
        feature_strategy=feature_strategy,
        # Pour propager le flag de normalisation des features
        normalize_features=normalize,
        # Pour propager la méthode de réduction résolue
        dim_method=dim_method,
        # Pour propager le nombre de composantes demandé
        n_components=n_components,
        # Pour propager le classifieur demandé
        classifier=args.classifier,
        # Pour propager le scaler optionnel demandé
        scaler=scaler,
        # Pour propager la régularisation CSP configurée
        csp_regularization=args.csp_regularization,
    )


# Pour construire une requête d'entraînement complète à partir des arguments CLI
def _build_training_request_from_args(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour faire circuler la configuration sans la re-décomposer.
    config: tpv_pipeline.PipelineConfig,
    # Pour garder ce paramètre explicite dans le contrat.
    preprocess_config: preprocessing.PreprocessingConfig,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> TrainingRequest:
    """Construit la TrainingRequest alignée sur les arguments CLI."""

    # Pour centraliser les paramètres d'entraînement dans une structure dédiée
    return TrainingRequest(
        # Pour propager le sujet cible
        subject=args.subject,
        # Pour propager le run cible
        run=args.run,
        # Pour propager la configuration pipeline choisie
        pipeline_config=config,
        # Pour propager le répertoire des numpy
        data_dir=args.data_dir,
        # Pour propager le répertoire d'artefacts
        artifacts_dir=args.artifacts_dir,
        # Pour propager le répertoire des EDF bruts
        raw_dir=args.raw_dir,
        # Pour propager la référence EEG configurée
        eeg_reference=args.eeg_reference,
        # Pour propager la configuration de prétraitement
        preprocess_config=preprocess_config,
        # Pour propager le flag d'activation du grid search
        enable_grid_search=args.grid_search,
        # Pour propager le nombre de splits pour la recherche si fourni
        grid_search_splits=args.grid_search_splits,
    )


# Pour déclencher l'entraînement et affiche le résumé CLI
def _execute_training_request(request: TrainingRequest) -> int:
    """Exécute run_training et imprime le résumé CLI."""

    # Pour déclencher l'entraînement et récupère le rapport pour afficher les scores
    result = run_training(request)

    # Pour isoler les scores de validation croisée depuis le rapport
    cv_scores = result["cv_scores"]
    # Pour isoler le nombre de splits demandé si le rapport le fournit
    cv_splits_requested = int(result.get("cv_splits_requested", DEFAULT_CV_SPLITS))
    # Pour déterminer le nombre de scores disponibles pour l'affichage CLI
    cv_scores_count = int(cv_scores.size) if isinstance(cv_scores, np.ndarray) else 0
    # Pour isoler la raison d'absence de CV si disponible
    cv_unavailability_reason = result.get("cv_unavailability_reason")
    # Pour isoler un éventuel message d'erreur CV pour l'affichage
    cv_error = result.get("cv_error")

    # Pour rendre explicite toujours l'utilisateur du nombre de splits attendus
    print(f"CV_SPLITS: {cv_splits_requested} (scores: {cv_scores_count})")
    # Pour rendre explicite un échec de CV malgré un splitter valide.
    if cv_error:
        # Pour garder une erreur CLI lisible sans bruit de traceback.
        print(f"AVERTISSEMENT: cross_val_score échoué ({cv_error})")
    # Pour rendre explicite si la CV est indisponible pour l'utilisateur
    if cv_unavailability_reason:
        # Pour donner une cause exploitable au lieu d’un échec silencieux.
        print(f"INFO: CV indisponible ({cv_unavailability_reason})")

    # Pour n’afficher les scores que lorsqu’une CV exploitable a réellement tourné.
    if isinstance(cv_scores, np.ndarray) and cv_scores.size > 0:
        # Pour garder un rendu aligné avec le formalisme attendu en soutenance.
        formatted_scores = np.array2string(
            # Pour propager les scores CV pour formater l'affichage
            cv_scores,
            # Pour homogénéiser l’affichage des scores split par split.
            precision=4,
            # Pour fixer le séparateur pour coller au format attendu
            separator=" ",
            # Pour éviter un rendu variable selon la précision native des floats.
            floatmode="fixed",
        )
        # Pour rendre visible le tableau numpy (format [0.6666 0.4444 ...])
        print(formatted_scores)
        # Pour préparer la moyenne affichée sous la ligne cross_val_score.
        mean_score = float(cv_scores.mean())
        # Pour rendre visible la moyenne arrondie sur quatre décimales pour homogénéiser
        print(f"cross_val_score: {mean_score:.4f}")
    # Pour conserver un fallback explicite quand la branche nominale échoue.
    else:
        # Pour rendre explicite l’absence de CV plutôt que d’afficher un tableau vide.
        print(np.array([]))
        # Pour rendre l’état du traitement visible dans un contexte CLI long.
        print("cross_val_score: 0.0")

    # Pour garder 0 pour signaler un succès CLI à mybci
    return 0


# Pour déclencher la logique CLI principale une fois les arguments parsés
def _run_from_args(
    # Pour garder ce paramètre explicite dans le contrat.
    args: argparse.Namespace,
    # Pour tester la CLI sans dépendre implicitement de sys.argv.
    argv: list[str] | None,
    # Pour figer un contrat de retour exploitable par les appels et mypy.
) -> int:
    """Orchestre les étapes CLI pour l'entraînement."""

    # Pour construire la configuration de prétraitement validée
    preprocess_config = _build_preprocess_config_from_args(args)
    # Pour aligner la configuration de fenêtres depuis les arguments
    _apply_epoch_window_config(args)
    # Pour déclencher la génération massive et s'arrête si le flag est positionné
    if _maybe_build_all(args, preprocess_config):
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return 0
    # Pour construire la configuration pipeline alignée sur les arguments
    config = _build_pipeline_config_from_args(args, argv)
    # Pour donner la priorité au batch complet quand ce mode est demandé.
    train_all_status = _maybe_train_all(args, config, preprocess_config)
    # Pour garder le code d'exécution si l'entraînement massif a été lancé
    if train_all_status is not None:
        # Pour restituer un état cohérent à l’appelant sans effet de bord caché.
        return train_all_status
    # Pour construire la requête d'entraînement complète
    request = _build_training_request_from_args(args, config, preprocess_config)
    # Pour déclencher l'entraînement et l'affichage CLI
    return _execute_training_request(request)


# Pour exposer un point d’entrée CLI stable et facilement testable.
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'entraînement."""

    # Pour construire le parser pour interpréter les arguments
    parser = build_parser()
    # Pour normaliser les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Pour déclencher l'orchestration CLI avec gestion d'erreur utilisateur
    try:
        # Pour déclencher l'orchestration des étapes CLI
        return _run_from_args(args, argv)
    # Pour garder un diagnostic maîtrisé sur cette famille d’échecs attendus.
    except (FileNotFoundError, PermissionError, ValueError) as error:
        # Pour rendre visible une erreur lisible et actionnable sans traceback
        for line in tpv_utils.render_cli_error_lines(
            # Pour transmettre explicitement ce contexte à l’appel encapsulé.
            error,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            subject=args.subject,
            # Pour fixer explicitement ce réglage dans l’objet construit.
            run=args.run,
            # Pour figer un contrat exploitable par les appels et l’outillage de types.
        ):
            # Pour rendre l’état du traitement visible dans un contexte CLI long.
            print(line)
        # Pour garder un code d'erreur explicite pour la CLI
        return int(tpv_utils.HANDLED_CLI_ERROR_EXIT_CODE)


# Pour préserver l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Pour garder l'issue du main comme code de sortie du processus
    raise SystemExit(main())
