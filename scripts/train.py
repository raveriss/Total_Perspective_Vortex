"""CLI d'entraînement pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
# Expose les primitives d'analyse des arguments CLI
import argparse

# Rassemble la construction de structures immuables orientées données
from dataclasses import dataclass

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Offre la persistance dédiée aux objets scikit-learn pour inspection séparée
import joblib

# Centralise l'accès aux tableaux manipulés par scikit-learn
import numpy as np

# Fournit la validation croisée pour évaluer la pipeline complète
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Permet de persister séparément la matrice W apprise
from tpv.dimensionality import TPVDimReducer

# Assemble la pipeline cohérente pour l'entraînement
from tpv.pipeline import PipelineConfig, build_pipeline, save_pipeline

# Définit le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut pour déposer les artefacts d'entraînement
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Fige la fréquence d'échantillonnage par défaut utilisée pour les features
DEFAULT_SAMPLING_RATE = 50.0

# Déclare le seuil minimal de splits exigé pour la validation croisée
MIN_CV_SPLITS = 3


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


# Construit un argument parser aligné sur la CLI mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'entraînement TPV."""

    # Crée le parser avec description lisible pour l'utilisateur
    parser = argparse.ArgumentParser(
        description="Entraîne une pipeline TPV et sauvegarde ses artefacts",
    )
    # Ajoute l'argument positionnel du sujet pour identifier les fichiers
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    # Ajoute l'argument positionnel du run pour sélectionner la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    # Ajoute l'option classifieur pour synchroniser avec mybci
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm"),
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


# Charge les matrices numpy attendues pour l'entraînement
def _load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour scikit-learn
    return X, y


# Exécute la validation croisée et l'entraînement final
def run_training(request: TrainingRequest) -> dict:
    """Entraîne la pipeline et sauvegarde ses artefacts."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(
        request.subject, request.run, request.data_dir
    )
    # Charge les tableaux numpy nécessaires à l'entraînement
    X, y = _load_data(features_path, labels_path)
    # Construit la pipeline complète sans préprocesseur amont
    pipeline = build_pipeline(None, request.pipeline_config)
    # Calcule le nombre minimal d'échantillons par classe pour calibrer la CV
    min_class_count = int(np.bincount(y).min())
    # Choisit le nombre de splits en restant compatible avec la taille des classes
    n_splits = min(MIN_CV_SPLITS, min_class_count) if min_class_count > 0 else 0
    # Initialise un tableau vide lorsque la validation croisée est impossible
    cv_scores = np.array([])
    # Lance la validation croisée uniquement si chaque classe possède trois exemples
    if n_splits >= MIN_CV_SPLITS:
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
    # Retourne un rapport synthétique pour les tests et la CLI
    return {
        "cv_scores": cv_scores,
        "model_path": model_path,
        "scaler_path": (
            target_dir / "scaler.joblib" if scaler_step is not None else None
        ),
        "w_matrix_path": target_dir / "w_matrix.joblib",
    }


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'entraînement."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
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
    # Regroupe les paramètres d'entraînement dans une structure dédiée
    request = TrainingRequest(
        subject=args.subject,
        run=args.run,
        pipeline_config=config,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
    )
    # Exécute l'entraînement et la sauvegarde des artefacts
    run_training(request)
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
