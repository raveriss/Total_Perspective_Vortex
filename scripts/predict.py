"""CLI de prédiction pour le pipeline TPV."""

# Préserve argparse pour exposer une interface CLI homogène avec mybci
# Garantit l'accès aux chemins portables pour données et artefacts
# Fournit le parsing CLI pour aligner la signature mybci
import argparse

# Fournit l'écriture CSV pour exposer les prédictions individuelles
import csv

# Fournit la sérialisation JSON pour tracer les rapports générés
import json

# Garantit l'accès aux chemins portables pour données et artefacts
from pathlib import Path

# Centralise l'accès aux tableaux numpy pour l'évaluation
import numpy as np

# Calcule les métriques de classification pour le rapport
from sklearn.metrics import confusion_matrix

# Centralise le parsing et le contrôle qualité des fichiers EDF
# Extrait les features fréquentielles depuis des epochs EEG
from tpv import features as features_extraction
from tpv import preprocessing

# Permet de restaurer la matrice W pour des usages temps-réel
from tpv.dimensionality import TPVDimReducer

# Permet de charger un pipeline entraîné pour la prédiction
from tpv.pipeline import load_pipeline

# Définit le répertoire par défaut où chercher les enregistrements
DEFAULT_DATA_DIR = Path("data")

# Définit le répertoire par défaut pour récupérer les artefacts
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# Définit le répertoire par défaut pour les fichiers EDF bruts
DEFAULT_RAW_DIR = Path("data/raw")


# Construit un argument parser aligné sur l'appel mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour la prédiction TPV."""

    # Crée le parser avec description explicite pour l'utilisateur
    parser = argparse.ArgumentParser(
        description="Charge une pipeline TPV entraînée et produit un rapport",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")

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
        choices=("fft", "wavelet"),
        default="fft",
        help="Stratégie de features utilisée à l'entraînement (ignorée ici)",
    )
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp"),
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


# Construit des matrices numpy à partir d'un EDF lorsque nécessaire
def _build_npy_from_edf(
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
) -> tuple[Path, Path]:
    """Génère X et y depuis un fichier EDF brut Physionet."""

    # Calcule les chemins cibles pour les fichiers numpy
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)
    # Calcule le chemin attendu du fichier EDF brut
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Arrête l'exécution si l'EDF est introuvable
    if not raw_path.exists():
        # Signale explicitement le chemin absent pour guider l'utilisateur
        raise FileNotFoundError(f"EDF introuvable pour {subject} {run}: {raw_path}")
    # Crée l'arborescence cible pour déposer les .npy
    features_path.parent.mkdir(parents=True, exist_ok=True)
    # Charge l'EDF en conservant les métadonnées essentielles
    raw, _ = preprocessing.load_physionet_raw(raw_path)
    # Mappe les annotations en événements moteurs
    events, event_id, motor_labels = preprocessing.map_events_to_motor_labels(raw)
    # Découpe le signal en epochs exploitables
    epochs = preprocessing.create_epochs_from_raw(raw, events, event_id)
    # Extrait des features fréquentielles par défaut
    features, _ = features_extraction.extract_features(epochs)
    # Définit un mapping stable label → entier
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(motor_labels)))}
    # Convertit les labels symboliques en entiers
    numeric_labels = np.array([label_mapping[label] for label in motor_labels])
    # Persiste les features calculées
    np.save(features_path, features)
    # Persiste les labels alignés
    np.save(labels_path, numeric_labels)
    # Retourne les chemins nouvellement générés
    return features_path, labels_path


# Charge ou génère les matrices numpy attendues pour la prédiction
def _load_data(
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Charge ou construit les données et étiquettes pour un run."""

    # Détermine les chemins attendus pour les features et labels
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)
    # Construit les .npy depuis l'EDF si l'un d'eux manque
    if not features_path.exists() or not labels_path.exists():
        # Convertit l'EDF associé en fichiers numpy persistés
        features_path, labels_path = _build_npy_from_edf(
            subject, run, data_dir, raw_dir
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


# Sérialise les rapports JSON et CSV pour un run donné
def _write_reports(
    target_dir: Path,
    identifiers: dict[str, str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accuracy: float,
) -> dict:
    """Écrit les rapports de prédiction et retourne les chemins créés."""

    # Identifie les classes présentes pour stabiliser l'ordre des rapports
    labels = sorted(np.unique(y_true).tolist())
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
            # Écrit la ligne CSV pour l'index courant
            writer.writerow(
                {
                    "subject": identifiers["subject"],
                    "run": identifiers["run"],
                    "index": idx,
                    "y_true": int(true_label),
                    "y_pred": int(pred_label),
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


# Évalue un run donné et produit un rapport structuré
def evaluate_run(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> dict:
    """Évalue l'accuracy d'un run en rechargeant le pipeline entraîné."""

    # Charge ou génère les tableaux numpy nécessaires au scoring
    X, y = _load_data(subject, run, data_dir, raw_dir)
    # Construit le dossier d'artefacts spécifique au sujet et au run
    target_dir = artifacts_dir / subject / run
    # Assure la présence du dossier pour pouvoir écrire les rapports
    target_dir.mkdir(parents=True, exist_ok=True)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(target_dir / "model.joblib"))
    # Génère les prédictions individuelles pour le rapport
    y_pred = pipeline.predict(X)
    # Calcule l'accuracy du pipeline sur les données fournies
    accuracy = float(pipeline.score(X, y))
    # Recharge la matrice W pour confirmer sa présence
    w_matrix = _load_w_matrix(target_dir / "w_matrix.joblib")
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


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'évaluation."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Évalue le run demandé et récupère la matrice W
    result = evaluate_run(
        args.subject, args.run, args.data_dir, args.artifacts_dir, args.raw_dir
    )
    # Construit le rapport structuré attendu par les tests
    _ = build_report(result)

    # Récupère les prédictions et la vérité terrain pour l'affichage CLI
    y_pred = result["predictions"]
    y_true = result["y_true"]

    # En-tête identique à la version de référence
    print("epoch nb: [prediction] [truth] equal?")

    # Affiche une ligne par epoch : numéro, prédiction, vérité, égalité
    for idx, (pred, true) in enumerate(zip(y_pred, y_true, strict=True)):
        equal = bool(pred == true)
        print(f"epoch {idx:02d}: [{int(pred)}] [{int(true)}] {equal}")

    # Affiche l'accuracy avec 4 décimales comme dans l'exemple
    print(f"Accuracy: {result['accuracy']:.4f}")

    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
