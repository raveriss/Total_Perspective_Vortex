"""CLI d'entraînement pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
# Expose les primitives d'analyse des arguments CLI
# Fournit le parsing CLI pour aligner la signature mybci
import argparse

# Fournit l'écriture CSV pour exposer un manifeste tabulaire
import csv

# Fournit la sérialisation JSON pour exposer un manifeste exploitable
import json

# Rassemble la construction de structures immuables orientées données
from dataclasses import asdict, dataclass

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Expose cast pour documenter les conversions de types
from typing import cast

# Offre la persistance dédiée aux objets scikit-learn pour inspection séparée
import joblib

# Centralise l'accès aux tableaux manipulés par scikit-learn
import numpy as np

# Fournit la validation croisée pour évaluer la pipeline complète
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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
DEFAULT_MAX_PEAK_TO_PEAK = 200e-6


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
MIN_CV_SPLITS = 3


# Stabilise la reproductibilité des splits de cross-validation
DEFAULT_RANDOM_STATE = 42


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
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour sélectionner la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
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
        default="pca",
        help="Méthode de réduction de dimension pour la pipeline",
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

    # Applique le filtrage bande-passante pour stabiliser les bandes MI
    filtered_raw = preprocessing.apply_bandpass_filter(raw)

    # Mappe les annotations en événements moteurs après filtrage
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(
        filtered_raw
    )

    # Découpe le signal filtré en epochs exploitables
    epochs = preprocessing.create_epochs_from_raw(filtered_raw, events, event_id)

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
        # Signale un fallback pour préserver l'entraînement sur ce run
        print(
            "AVERTISSEMENT: filtrage QC a supprimé une classe pour "
            f"{subject} {run}, fallback sans QC."
        )
        # Conserve les epochs filtrées par annotations uniquement
        cleaned_epochs = epochs
        # Conserve les labels moteurs initiaux pour aligner les données
        cleaned_labels = motor_labels

    # Récupère les données brutes des epochs (n_trials, n_channels, n_times)
    epochs_data = cleaned_epochs.get_data(copy=True)

    # Définit un mapping stable label → entier
    label_mapping = {
        label: idx for idx, label in enumerate(sorted(set(cleaned_labels)))
    }

    # Convertit les labels symboliques en entiers
    numeric_labels = np.array([label_mapping[label] for label in cleaned_labels])

    # Persiste les epochs brutes
    np.save(features_path, epochs_data)
    # Persiste les labels alignés
    np.save(labels_path, numeric_labels)

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

    # Cas 2 : X n'a pas la bonne dimension → reconstruction
    if candidate_X.ndim != EXPECTED_FEATURES_DIMENSIONS:
        # Informe l'utilisateur de la dimension inattendue
        print(
            "INFO: X chargé depuis "
            f"'{features_path}' a ndim={candidate_X.ndim} au lieu de "
            f"{EXPECTED_FEATURES_DIMENSIONS}, "
            "régénération depuis l'EDF..."
        )
        # Demande une régénération pour retrouver la forme attendue
        return True

    # Cas 3 : désalignement entre n_samples de X et y → reconstruction
    if candidate_X.shape[0] != candidate_y.shape[0]:
        # Signale l'incohérence de longueur entre X et y
        print(
            "INFO: Désalignement détecté pour "
            f"{run_label}: X.shape[0]={candidate_X.shape[0]}, "
            f"y.shape[0]={candidate_y.shape[0]}. Régénération depuis l'EDF..."
        )
        # Demande une reconstruction pour réaligner les labels
        return True

    # Cas 4 : labels mal dimensionnés → reconstruction
    if candidate_y.ndim != 1:
        # Informe l'utilisateur que y possède une dimension inattendue
        print(
            "INFO: y chargé depuis "
            f"'{labels_path}' a ndim={candidate_y.ndim} au lieu de 1, "
            "régénération depuis l'EDF..."
        )
        # Demande une régénération pour retrouver un vecteur 1D
        return True

    # Confirme que les caches existants sont utilisables
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

    # Prépare la section dataset pour identifier les entrées de données
    dataset = {
        "subject": request.subject,
        "run": request.run,
        "data_dir": str(request.data_dir),
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

    # Déclare un ensemble restreint de classifieurs pour limiter la complexité
    # Initialise la grille de classifieurs candidates
    classifier_grid: list[object] = []
    # Ajoute LDA uniquement lorsque l'effectif le permet
    if allow_lda:
        # Utilise LDA pour les effectifs suffisants
        classifier_grid.append(LinearDiscriminantAnalysis())
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
    return {
        "features__feature_strategy": ["fft", "wavelet"],
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


# Exécute la validation croisée et l'entraînement final
def run_training(request: TrainingRequest) -> dict:
    """Entraîne la pipeline et sauvegarde ses artefacts."""

    # Charge ou génère les tableaux numpy nécessaires à l'entraînement
    X, y = _load_data(request.subject, request.run, request.data_dir, request.raw_dir)
    # Adapte la configuration au niveau d'effectif pour stabiliser l'entraînement
    adapted_config = _adapt_pipeline_config_for_samples(request.pipeline_config, y)
    # Construit la pipeline complète sans préprocesseur amont
    pipeline = build_pipeline(adapted_config)
    # Calcule le nombre total d'échantillons disponibles
    sample_count = int(y.shape[0])
    # Calcule le nombre de classes distinctes pour l'évaluation LDA
    class_count = int(np.unique(y).size)
    # Calcule le nombre minimal d'échantillons par classe pour calibrer la CV
    min_class_count = int(np.bincount(y).min())
    # Déclare le nombre de splits cible imposé par la consigne (10)
    requested_splits = DEFAULT_CV_SPLITS
    # Calcule le nombre de splits atteignable avec la classe minoritaire
    n_splits = min(requested_splits, min_class_count)
    # Initialise un tableau vide lorsque la validation croisée est impossible
    cv_scores = np.array([])
    # Prépare un résumé de recherche d'hyperparamètres si besoin
    search_summary: dict[str, object] | None = None
    # Vérifie si l'effectif autorise une validation croisée exploitable
    if n_splits < MIN_CV_SPLITS:
        # Signale la désactivation de la validation croisée par manque d'échantillons
        print(
            "AVERTISSEMENT: effectif par classe insuffisant pour la "
            "validation croisée, cross-val ignorée"
        )
        # Ajuste la pipeline sur toutes les données malgré l'absence de CV
        pipeline.fit(X, y)
    else:
        # Configure une StratifiedKFold stable sur le nombre de splits calculé
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE
        )
        # Lance une recherche systématique des hyperparamètres si demandée
        if request.enable_grid_search:
            # Construit la pipeline dédiée au grid search
            search_pipeline = build_search_pipeline(adapted_config)
            # Détermine si LDA est autorisé par le ratio effectif/classes
            allow_lda = sample_count > class_count
            # Construit la grille d'hyperparamètres à explorer
            param_grid = _build_grid_search_grid(adapted_config, allow_lda)
            # Détermine le nombre de splits spécifique si fourni
            search_splits = request.grid_search_splits or n_splits
            # S'assure que la recherche respecte les bornes disponibles
            search_splits = min(search_splits, min_class_count)
            search_splits = max(search_splits, MIN_CV_SPLITS)
            # Prépare la CV pour la recherche avec splits contrôlés
            search_cv = StratifiedKFold(
                n_splits=search_splits,
                shuffle=True,
                random_state=DEFAULT_RANDOM_STATE,
            )
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
            cv_scores = _extract_grid_search_scores(search, search_splits)
            # Capture un résumé des meilleurs paramètres
            search_summary = {
                "best_params": search.best_params_,
                "best_score": float(search.best_score_),
            }
            # Récupère l'estimateur entraîné sur tous les échantillons
            pipeline = search.best_estimator_
        else:
            # Calcule les scores de validation croisée sur l'ensemble du pipeline
            cv_scores = cross_val_score(pipeline, X, y, cv=cv)
            # Ajuste la pipeline sur toutes les données après évaluation
            pipeline.fit(X, y)
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
