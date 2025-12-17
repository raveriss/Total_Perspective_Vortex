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

# Offre la persistance dédiée aux objets scikit-learn pour inspection séparée
import joblib

# Centralise l'accès aux tableaux manipulés par scikit-learn
import numpy as np

# Fournit la validation croisée pour évaluer la pipeline complète
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Centralise le parsing et le contrôle qualité des fichiers EDF
# Extrait les features fréquentielles depuis des epochs EEG
from tpv import preprocessing

# Permet de persister séparément la matrice W apprise
from tpv.dimensionality import TPVDimReducer

# Assemble la pipeline cohérente pour l'entraînement
from tpv.pipeline import PipelineConfig, build_pipeline, save_pipeline

# Déclare la liste des runs moteurs à couvrir pour l'entraînement massif
MOTOR_RUNS = ("R03", "R04", "R05", "R06", "R07", "R08")

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

# Déclare le nombre cible de splits utilisé pour la validation croisée
DEFAULT_CV_SPLITS = 10

# Fixe le nombre minimal de splits pour déclencher la validation croisée
MIN_CV_SPLITS = 2


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
        choices=("pca", "csp"),
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

    # Mappe les annotations en événements moteurs
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(raw)

    # Découpe le signal en epochs exploitables
    epochs = preprocessing.create_epochs_from_raw(raw, events, event_id)

    # Récupère les données brutes des epochs (n_trials, n_channels, n_times)
    epochs_data = epochs.get_data(copy=True)

    # Définit un mapping stable label → entier
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(motor_labels)))}

    # Convertit les labels symboliques en entiers
    numeric_labels = np.array([label_mapping[label] for label in motor_labels])

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

    # Détermine les chemins attendus pour les features et labels
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)

    # Indique si nous devons régénérer les .npy
    needs_rebuild = False
    # Stocke les chemins invalides pour enrichir les logs utilisateurs
    corrupted_reason: str | None = None

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
        else:
            # Cas 2 : X n'a pas la bonne dimension → reconstruction
            if candidate_X.ndim != EXPECTED_FEATURES_DIMENSIONS:
                print(
                    "INFO: X chargé depuis "
                    f"'{features_path}' a ndim={candidate_X.ndim} au lieu de "
                    f"{EXPECTED_FEATURES_DIMENSIONS}, "
                    "régénération depuis l'EDF..."
                )
                needs_rebuild = True
            # Cas 3 : désalignement entre n_samples de X et y → reconstruction
            elif candidate_X.shape[0] != candidate_y.shape[0]:
                print(
                    "INFO: Désalignement détecté pour "
                    f"{subject} {run}: X.shape[0]={candidate_X.shape[0]}, "
                    f"y.shape[0]={candidate_y.shape[0]}. Régénération depuis l'EDF..."
                )
                needs_rebuild = True
            # Cas 4 : labels mal dimensionnés → reconstruction
            elif candidate_y.ndim != 1:
                print(
                    "INFO: y chargé depuis "
                    f"'{labels_path}' a ndim={candidate_y.ndim} au lieu de 1, "
                    "régénération depuis l'EDF..."
                )
                needs_rebuild = True

    # Informe l'utilisateur lorsqu'un fichier corrompu bloque le chargement
    if corrupted_reason is not None:
        print(
            "INFO: Chargement numpy impossible pour "
            f"{subject} {run}: {corrupted_reason}. "
            "Régénération depuis l'EDF..."
        )

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
    # Définit le chemin de sortie du manifeste JSON à côté des artefacts
    manifest_json_path = target_dir / "manifest.json"
    # Écrit le manifeste JSON sur disque en UTF-8 pour la portabilité
    manifest_json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
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


# Exécute la validation croisée et l'entraînement final
def run_training(request: TrainingRequest) -> dict:
    """Entraîne la pipeline et sauvegarde ses artefacts."""

    # Charge ou génère les tableaux numpy nécessaires à l'entraînement
    X, y = _load_data(request.subject, request.run, request.data_dir, request.raw_dir)
    # Construit la pipeline complète sans préprocesseur amont
    pipeline = build_pipeline(request.pipeline_config)
    # Calcule le nombre minimal d'échantillons par classe pour calibrer la CV
    min_class_count = int(np.bincount(y).min())
    # Choisit le nombre de splits en respectant la disponibilité par classe
    n_splits = min(DEFAULT_CV_SPLITS, min_class_count) if min_class_count > 1 else 0
    # Initialise un tableau vide lorsque la validation croisée est impossible
    cv_scores = np.array([])
    # Lance la validation croisée seulement si chaque classe dispose de deux points
    # Évite la validation croisée quand un fold manquerait de diversité
    # Garantit au moins deux échantillons par classe dans chaque ensemble d'entraînement
    if n_splits >= MIN_CV_SPLITS and min_class_count > MIN_CV_SPLITS:
        # Configure une StratifiedKFold stable sur le nombre de splits calculé
        cv = StratifiedKFold(n_splits=n_splits)
        # Calcule les scores de validation croisée sur l'ensemble du pipeline
        cv_scores = cross_val_score(pipeline, X, y, cv=cv)
    # Ajuste la pipeline sur toutes les données après évaluation
    pipeline.fit(X, y)
    # Prépare le dossier d'artefacts spécifique au sujet et au run
    target_dir = request.artifacts_dir / request.subject / request.run
    # Crée les répertoires au besoin pour éviter les erreurs de sauvegarde
    target_dir.mkdir(parents=True, exist_ok=True)
    # Calcule le chemin du fichier modèle pour joblib
    model_path = target_dir / "model.joblib"
    # Sauvegarde la pipeline complète pour les prédictions futures
    save_pipeline(pipeline, str(model_path))
    # Récupère l'éventuel scaler pour une sauvegarde dédiée
    scaler_step = pipeline.named_steps.get("scaler")
    # Sauvegarde le scaler uniquement s'il est présent dans la pipeline
    if scaler_step is not None:
        # Dépose le scaler dans un fichier distinct pour inspection
        joblib.dump(scaler_step, target_dir / "scaler.joblib")
    # Récupère le réducteur de dimension pour exposer la matrice W
    dim_reducer: TPVDimReducer = pipeline.named_steps["dimensionality"]
    # Sauvegarde la matrice de projection pour les usages temps-réel
    dim_reducer.save(target_dir / "w_matrix.joblib")
    # Calcule le chemin du scaler pour l'ajouter au manifeste
    scaler_path = None
    # Renseigne le chemin du scaler uniquement lorsqu'il existe
    if scaler_step is not None:
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
    # Construit la configuration de pipeline alignée sur mybci
    config = PipelineConfig(
        sfreq=args.sfreq,
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
        # Formate les scores sur quatre décimales pour refléter la consigne
        formatted_scores = np.array2string(cv_scores, precision=4, separator=" ")
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
