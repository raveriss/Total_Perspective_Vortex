"""Boucle d'inférence fenêtrée pour la prédiction temps réel."""

# Centralise argparse pour exposer un parser CLI dédié
import argparse

# Fournit sys pour écrire les erreurs CLI sur stderr
import sys

# Mesure précisément le temps d'exécution pour les métriques de latence
import time

# Fournit deque pour gérer un buffer borné en O(1)
from collections import deque

# Fournit dataclass pour encapsuler les événements de streaming
from dataclasses import dataclass

# Garantit l'accès aux chemins portables pour les données et artefacts
from pathlib import Path

# Spécifie les Protocol, générateurs et structures typées pour le streaming
from typing import Deque, Generator, Protocol, TypedDict

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
    # Fixe la latence maximale tolérée pour chaque prédiction
    max_latency: float
    # Fixe la fréquence d'échantillonnage pour les offsets
    sfreq: float
    # Définit l'étiquette lisible associée à la classe zéro
    label_zero: str
    # Définit l'étiquette lisible associée à la classe un
    label_one: str


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
def _smooth_prediction(buffer: Deque[int]) -> int:
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


# Retourne un libellé lisible pour une prédiction numérique
def _label_prediction(value: int, config: RealtimeConfig) -> str:
    """Convertit une classe numérique en libellé utilisateur."""

    # Fournit l'étiquette configurée pour la classe zéro
    if value == 0:
        # Retourne la classe zéro lisible pour l'utilisateur
        return config.label_zero
    # Fournit l'étiquette configurée pour la classe un
    if value == 1:
        # Retourne la classe un lisible pour l'utilisateur
        return config.label_one
    # Retourne un libellé générique pour les classes inattendues
    return f"classe {value}"


# Centralise les jeux de libellés par type de tâche motrice
DEFAULT_LABEL_SETS: dict[str, tuple[str, str]] = {
    # Fournit le mapping explicite des événements T1/T2
    "t1-t2": ("T1", "T2"),
    # Fournit le mapping symbolique A/B utilisé dans le pipeline
    "a-b": ("A", "B"),
    # Fournit le mapping explicite pour main gauche/droite
    "left-right": ("main gauche", "main droite"),
    # Fournit le mapping explicite pour deux poings/deux pieds
    "fists-feet": ("deux poings", "deux pieds"),
}


# Résout les libellés finaux en combinant le set et les overrides CLI
def _resolve_label_pair(
    label_set: str,
    label_zero: str | None,
    label_one: str | None,
) -> tuple[str, str]:
    """Retourne les libellés à utiliser pour les classes 0/1."""

    # Sélectionne la paire par défaut pour le set demandé
    default_zero, default_one = DEFAULT_LABEL_SETS[label_set]
    # Priorise l'override explicite pour la classe zéro
    resolved_zero = label_zero if label_zero is not None else default_zero
    # Priorise l'override explicite pour la classe un
    resolved_one = label_one if label_one is not None else default_one
    # Retourne les libellés finaux prêts à être affichés
    return resolved_zero, resolved_one


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
    buffer: Deque[int] = deque(maxlen=config.buffer_size)
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
        # Déclenche une erreur si la contrainte temps réel est dépassée
        if latency > config.max_latency:
            # Signale une violation de SLA pour interrompre la session
            raise TimeoutError(
                f"Latence {latency:.3f}s dépasse {config.max_latency:.3f}s"
            )
        # Alimente le buffer de lissage avec la prédiction obtenue
        buffer.append(raw_prediction)
        # Calcule la prédiction lissée en fonction des valeurs récentes
        smoothed = _smooth_prediction(buffer)
        # Construit le libellé lisible pour la prédiction brute
        raw_label = _label_prediction(raw_prediction, config)
        # Construit le libellé lisible pour la prédiction lissée
        smoothed_label = _label_prediction(smoothed, config)
        # Construit le message complet pour la trace temps réel
        message = (
            # Ajoute l'en-tête constant pour identifier le mode
            "realtime prediction "
            # Ajoute l'index et le décalage de la fenêtre courante
            f"window={index} offset={offset_seconds:.3f}s "
            # Ajoute le libellé utilisateur pour la prédiction brute
            f"raw={raw_prediction} ({raw_label}) "
            # Ajoute le libellé utilisateur pour la prédiction lissée
            f"smoothed={smoothed} ({smoothed_label}) "
            # Ajoute la latence mesurée pour la fenêtre courante
            f"latency={latency:.3f}s"
        )
        # Log la prédiction courante pour fournir un suivi temps réel explicite
        print(message, flush=True)
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
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
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
    # Ajoute une option pour contrôler la latence maximale autorisée
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Latence maximale tolérée en secondes",
    )
    # Ajoute une option pour définir la fréquence d'échantillonnage
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Ajoute une option pour choisir le set de libellés par tâche
    parser.add_argument(
        "--label-set",
        choices=sorted(DEFAULT_LABEL_SETS.keys()),
        default="t1-t2",
        help="Type de libellés à afficher (T1/T2, A/B, etc.)",
    )
    # Ajoute une option pour nommer la classe zéro
    parser.add_argument(
        "--label-zero",
        default=None,
        help="Étiquette affichée pour la classe 0",
    )
    # Ajoute une option pour nommer la classe un
    parser.add_argument(
        "--label-one",
        default=None,
        help="Étiquette affichée pour la classe 1",
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


# Sélectionne un code d'erreur stable selon les fichiers absents
def _resolve_missing_files_code(features_missing: bool, labels_missing: bool) -> str:
    """Retourne un code d'erreur UX explicite pour fichiers manquants."""

    # Distingue l'absence complète des entrées nécessaires au realtime
    if features_missing and labels_missing:
        # Rend le diagnostic immédiat sans table de correspondance
        return "MISSING-FEATURES-AND-LABELS"
    # Signale un échec lié uniquement aux features
    if features_missing:
        # Oriente l'utilisateur vers la génération des features
        return "MISSING-FEATURES"
    # Signale un échec lié uniquement aux labels
    return "MISSING-LABELS"



# Charge les matrices numpy attendues pour simuler un flux
def _load_data(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Détermine si les features sont absents pour piloter le code d'erreur
    features_missing = not features_path.exists()
    # Détermine si les labels sont absents pour piloter le code d'erreur
    labels_missing = not labels_path.exists()
    # Prépare une liste des chemins manquants pour un message utilisateur clair
    missing_paths = []
    # Ajoute les features manquants à la liste de diagnostic
    if features_missing:
        # Trace le fichier manquant pour guider le diagnostic utilisateur
        missing_paths.append(str(features_path))
    # Ajoute les labels manquants à la liste de diagnostic
    if labels_missing:
        # Trace le fichier manquant pour guider le diagnostic utilisateur
        missing_paths.append(str(labels_path))
    # Interrompt l'exécution si les fichiers attendus sont absents
    if missing_paths:
        # Déduit un code d'erreur stable pour aider le support utilisateur
        error_code = _resolve_missing_files_code(features_missing, labels_missing)
        # Décrit le type de fichier manquant pour un message précis
        if features_missing and labels_missing:
            # Précise que les features et labels manquent simultanément
            missing_label = "features et labels"
        # Décrit le cas où seules les features sont absentes
        elif features_missing:
            # Précise que les features manquent sans ambiguïté
            missing_label = "features"
        # Décrit le cas où seuls les labels sont absents
        else:
            # Précise que les labels manquent sans ambiguïté
            missing_label = "labels"
        # Construit un libellé unique pour la liste des fichiers manquants
        missing_list = ", ".join(missing_paths)
        # Cible le répertoire racine des données pour guider la vérification
        data_root = features_path.parent.parent
        # Construit un message UX actionnable pour corriger l'erreur
        message = (
            # Pose l'entête UX avec code d'erreur et contexte
            f"ERROR[{error_code}]: Fichiers {missing_label} manquants pour la "
            # Ajoute une coupure pour rendre la lecture plus claire
            "session temps réel.\n"
            # Liste explicitement les fichiers attendus pour l'utilisateur
            f"- Attendus: {missing_list}\n"
            # Indique la commande de train requise pour générer les artefacts
            "- Action 1: Lancez `python mybci.py <Sxxx> <Rxx> train`.\n"
            # Invite à vérifier le répertoire de données cible
            f"- Action 2: Vérifiez le dossier {data_root}.\n"
            # Rappelle l'option de surcharge pour un autre emplacement
            "- Action 3: Utilisez --data-dir si vos données sont ailleurs."
        )
        # Lève une erreur explicite pour interrompre le flux temps réel
        raise FileNotFoundError(message)
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
    # Résout les libellés finaux avec les overrides CLI
    # Résout les libellés finaux selon le set demandé
    label_zero, label_one = _resolve_label_pair(
        # Passe le set de libellés sélectionné via la CLI
        args.label_set,
        # Passe l'override explicite pour la classe zéro
        args.label_zero,
        # Passe l'override explicite pour la classe un
        args.label_one,
    )
    # Encadre l'exécution pour éviter un traceback non actionnable
    try:
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
                max_latency=args.max_latency,
                sfreq=args.sfreq,
                # Renseigne l'étiquette utilisateur associée à la classe zéro
                label_zero=label_zero,
                # Renseigne l'étiquette utilisateur associée à la classe un
                label_one=label_one,
            ),
        )
    # Intercepte les artefacts manquants pour un message clair
    except FileNotFoundError as exc:
        # Écrit le message d'erreur sans afficher le traceback
        print(str(exc), file=sys.stderr)
        # Retourne un code d'erreur non nul pour le CLI
        return 2
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
