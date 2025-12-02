"""CLI de prédiction pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
import argparse

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Centralise l'accès aux tableaux numpy pour l'évaluation
import numpy as np

# Permet de charger un pipeline entraîné pour la prédiction
from tpv.pipeline import load_pipeline

# Permet de restaurer la matrice W pour des usages temps-réel
from tpv.dimensionality import TPVDimReducer

# Définit le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut pour récupérer les artefacts
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


# Construit un argument parser aligné sur l'appel mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la prédiction TPV."""

    # Crée le parser avec description explicite pour l'utilisateur
    parser = argparse.ArgumentParser(
        description="Charge une pipeline TPV entraînée et produit un rapport",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
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
        help="Répertoire racine où lire le modèle",
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


# Charge les matrices numpy attendues pour la prédiction
def _load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

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


# Évalue un run donné et produit un rapport structuré
def evaluate_run(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
) -> dict:
    """Évalue l'accuracy d'un run en rechargeant le pipeline entraîné."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)
    # Charge les tableaux numpy nécessaires au scoring
    X, y = _load_data(features_path, labels_path)
    # Construit le dossier d'artefacts spécifique au sujet et au run
    target_dir = artifacts_dir / subject / run
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(target_dir / "model.joblib"))
    # Calcule l'accuracy du pipeline sur les données fournies
    accuracy = float(pipeline.score(X, y))
    # Recharge la matrice W pour confirmer sa présence
    w_matrix = _load_w_matrix(target_dir / "w_matrix.joblib")
    # Retourne le rapport local incluant la matrice pour les tests
    return {"run": run, "subject": subject, "accuracy": accuracy, "w_matrix": w_matrix}


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
    # Retourne une structure prête à être sérialisée
    return {"by_run": by_run, "by_subject": by_subject, "global": global_accuracy}


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'évaluation."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Évalue le run demandé et récupère la matrice W
    result = evaluate_run(args.subject, args.run, args.data_dir, args.artifacts_dir)
    # Construit le rapport structuré attendu par les tests
    _ = build_report(result)
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
