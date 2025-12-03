"""Boucle d'inférence fenêtrée pour la prédiction temps réel."""

# Fournit argparse pour exposer un parser CLI dédié
import argparse

# Mesure précisément le temps d'exécution pour les métriques de latence
import time

# Fournit dataclass pour encapsuler les événements de streaming
from dataclasses import dataclass

# Garantit l'accès aux chemins portables pour les données et artefacts
from pathlib import Path

# Spécifie les Protocol et générateurs pour typer le streaming
from typing import Generator, Protocol, TypedDict

# Centralise les opérations numériques nécessaires au fenêtrage
import numpy as np

# Récupère la restauration du pipeline entraîné pour la prédiction
from tpv.pipeline import load_pipeline


# Définit une interface minimale pour les pipelines prédictifs
class PredictablePipeline(Protocol):
    """Expose la méthode predict attendue pour une pipeline sklearn."""

    # Fournit une méthode predict compatible avec scikit-learn
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# Regroupe les hyperparamètres nécessaires à l'inférence temps réel
@dataclass
class RealtimeConfig:
    """Encapsule les paramètres fenêtrage et cadence pour le streaming."""

    # Fixe la taille de fenêtre glissante en échantillons
    window_size: int
    # Fixe le pas de glissement entre deux fenêtres
    step_size: int
    # Fixe la taille du buffer de lissage des prédictions
    buffer_size: int
    # Fixe la fréquence d'échantillonnage pour les offsets
    sfreq: float


# Décrit un événement individuel produit par la boucle temps réel
@dataclass
class RealtimeEvent:
    """Capture les métadonnées d'une fenêtre prédite en streaming."""

    # Identifie l'index séquentiel de la fenêtre traitée
    window_index: int
    # Spécifie le décalage temporel de début de fenêtre en secondes
    window_offset: float
    # Renseigne le timestamp relatif du démarrage d'inférence
    inference_started_at: float
    # Stocke la latence de prédiction observée en secondes
    latency: float
    # Conserve la prédiction brute issue du classifieur
    raw_prediction: int
    # Conserve la prédiction lissée issue du buffer circulaire
    smoothed_prediction: int


# Typedef pour décrire les métriques retournées par l'inférence
class RealtimeResult(TypedDict):
    """Structure typée des métriques retournées par le streaming."""

    # Expose la liste d'événements générés durant l'inférence
    events: list[RealtimeEvent]
    # Expose la latence moyenne observée sur la session
    latency_mean: float
    # Expose la latence maximale observée sur la session
    latency_max: float


# Génère des fenêtres glissantes sur un flux continu
def _window_stream(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Calcule une prédiction lissée à partir d'un buffer borné
def _smooth_prediction(buffer: list[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def run_realtime_inference(
    pipeline: PredictablePipeline,
    stream: np.ndarray,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Produit des prédictions fenêtrées et des métriques de latence."""

    # Initialise la liste des événements pour tracer l'ordre temporel
    events: list[RealtimeEvent] = []
    # Conserve un buffer borné pour lisser les prédictions successives
    buffer: list[int] = []
    # Capture l'instant initial pour obtenir des timestamps relatifs
    base_time = time.perf_counter()
    # Itère sur les fenêtres générées à partir du flux continu
    for index, (start, window) in enumerate(
        _window_stream(stream, config.window_size, config.step_size)
    ):
        # Calcule l'offset temporel correspondant à la fenêtre courante
        offset_seconds = float(start) / config.sfreq
        # Capture l'instant de début pour mesurer la latence d'inférence
        inference_start = time.perf_counter()
        # Lance la prédiction sur la fenêtre encapsulée dans un batch
        raw_prediction = int(pipeline.predict(window[np.newaxis, ...])[0])
        # Calcule la latence en secondes pour la fenêtre traitée
        latency = time.perf_counter() - inference_start
        # Alimente le buffer de lissage avec la prédiction obtenue
        buffer.append(raw_prediction)
        # Tronque le buffer pour respecter la taille maximale demandée
        buffer = buffer[-config.buffer_size :]
        # Calcule la prédiction lissée en fonction des valeurs récentes
        smoothed = _smooth_prediction(buffer)
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=latency,
                raw_prediction=raw_prediction,
                smoothed_prediction=smoothed,
            )
        )
    # Calcule la latence moyenne sur l'ensemble des fenêtres traitées
    mean_latency = (
        float(np.mean([event.latency for event in events])) if events else 0.0
    )
    # Calcule la latence maximale pour identifier d'éventuels pics
    max_latency = float(np.max([event.latency for event in events])) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Construit un argument parser aligné avec le mode realtime de mybci
def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    # Ajoute une option pour cibler un répertoire de données spécifique
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Répertoire racine contenant les fichiers numpy",
    )
    # Ajoute une option pour configurer le répertoire d'artefacts
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Répertoire racine où lire le modèle",
    )
    # Ajoute une option pour définir la taille de fenêtre en échantillons
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Taille de fenêtre glissante en échantillons",
    )
    # Ajoute une option pour définir le pas de glissement
    parser.add_argument(
        "--step-size",
        type=int,
        default=25,
        help="Pas entre deux fenêtres successives",
    )
    # Ajoute une option pour définir la taille du buffer de lissage
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=3,
        help="Taille du buffer pour lisser les prédictions",
    )
    # Ajoute une option pour définir la fréquence d'échantillonnage
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
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


# Charge les matrices numpy attendues pour simuler un flux
def _load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour le streaming
    return X, y


# Orchestre une session temps réel à partir d'artefacts persistés
def run_realtime_session(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, run, data_dir)
    # Charge le flux et les labels pour simuler une session continue
    X, _ = _load_data(features_path, labels_path)
    # Construit un flux continu en concaténant les essais successifs
    stream = np.concatenate(list(X), axis=1)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Point d'entrée principal pour l'exécution en ligne de commande
def main(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=args.subject,
        run=args.run,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
