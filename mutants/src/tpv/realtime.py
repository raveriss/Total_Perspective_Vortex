"""Boucle d'inférence fenêtrée pour la prédiction temps réel."""

# Centralise argparse pour exposer un parser CLI dédié
import argparse

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
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg is not None:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


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
def x__window_stream__mutmut_orig(
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


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_1(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(None, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_2(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, None, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_3(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, None):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_4(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_5(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_6(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, ):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_7(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(1, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_8(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size - 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_9(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] + window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_10(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[2] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_11(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 2, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_12(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = None
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_13(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start - window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = stream[:, start:end]
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window


# Génère des fenêtres glissantes sur un flux continu
def x__window_stream__mutmut_14(
    stream: np.ndarray, window_size: int, step_size: int
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Itère sur des segments de signal pour une pipeline scikit-learn."""

    # Parcourt les indices de début compatibles avec la taille du flux
    for start in range(0, stream.shape[1] - window_size + 1, step_size):
        # Calcule l'index de fin pour extraire une fenêtre complète
        end = start + window_size
        # Extrait la portion fenêtrée du flux multidimensionnel
        window = None
        # Retourne l'offset associé à la fenêtre extraite
        yield start, window

x__window_stream__mutmut_mutants : ClassVar[MutantDict] = {
'x__window_stream__mutmut_1': x__window_stream__mutmut_1, 
    'x__window_stream__mutmut_2': x__window_stream__mutmut_2, 
    'x__window_stream__mutmut_3': x__window_stream__mutmut_3, 
    'x__window_stream__mutmut_4': x__window_stream__mutmut_4, 
    'x__window_stream__mutmut_5': x__window_stream__mutmut_5, 
    'x__window_stream__mutmut_6': x__window_stream__mutmut_6, 
    'x__window_stream__mutmut_7': x__window_stream__mutmut_7, 
    'x__window_stream__mutmut_8': x__window_stream__mutmut_8, 
    'x__window_stream__mutmut_9': x__window_stream__mutmut_9, 
    'x__window_stream__mutmut_10': x__window_stream__mutmut_10, 
    'x__window_stream__mutmut_11': x__window_stream__mutmut_11, 
    'x__window_stream__mutmut_12': x__window_stream__mutmut_12, 
    'x__window_stream__mutmut_13': x__window_stream__mutmut_13, 
    'x__window_stream__mutmut_14': x__window_stream__mutmut_14
}

def _window_stream(*args, **kwargs):
    result = _mutmut_trampoline(x__window_stream__mutmut_orig, x__window_stream__mutmut_mutants, args, kwargs)
    return result 

_window_stream.__signature__ = _mutmut_signature(x__window_stream__mutmut_orig)
x__window_stream__mutmut_orig.__name__ = 'x__window_stream'


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_orig(buffer: Deque[int]) -> int:
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


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_1(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = None
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_2(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = None
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_3(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) - 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_4(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(None, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_5(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, None) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_6(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_7(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, ) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_8(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 1) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_9(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 2
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_10(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = None
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_11(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(None, key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_12(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=None)
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_13(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(key=lambda key: counts[key])
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_14(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, )
    # Retourne la classe majoritaire pour lisser la séquence
    return majority


# Calcule une prédiction lissée à partir d'un buffer borné
def x__smooth_prediction__mutmut_15(buffer: Deque[int]) -> int:
    """Retourne la valeur majoritaire observée dans le buffer."""

    # Initialise un comptage vide pour agréger les classes récentes
    counts: dict[int, int] = {}
    # Parcourt les valeurs présentes pour incrémenter les occurrences
    for value in buffer:
        # Incrémente le compteur associé à la classe rencontrée
        counts[value] = counts.get(value, 0) + 1
    # Sélectionne la classe la plus fréquente dans le buffer courant
    majority = max(counts, key=lambda key: None)
    # Retourne la classe majoritaire pour lisser la séquence
    return majority

x__smooth_prediction__mutmut_mutants : ClassVar[MutantDict] = {
'x__smooth_prediction__mutmut_1': x__smooth_prediction__mutmut_1, 
    'x__smooth_prediction__mutmut_2': x__smooth_prediction__mutmut_2, 
    'x__smooth_prediction__mutmut_3': x__smooth_prediction__mutmut_3, 
    'x__smooth_prediction__mutmut_4': x__smooth_prediction__mutmut_4, 
    'x__smooth_prediction__mutmut_5': x__smooth_prediction__mutmut_5, 
    'x__smooth_prediction__mutmut_6': x__smooth_prediction__mutmut_6, 
    'x__smooth_prediction__mutmut_7': x__smooth_prediction__mutmut_7, 
    'x__smooth_prediction__mutmut_8': x__smooth_prediction__mutmut_8, 
    'x__smooth_prediction__mutmut_9': x__smooth_prediction__mutmut_9, 
    'x__smooth_prediction__mutmut_10': x__smooth_prediction__mutmut_10, 
    'x__smooth_prediction__mutmut_11': x__smooth_prediction__mutmut_11, 
    'x__smooth_prediction__mutmut_12': x__smooth_prediction__mutmut_12, 
    'x__smooth_prediction__mutmut_13': x__smooth_prediction__mutmut_13, 
    'x__smooth_prediction__mutmut_14': x__smooth_prediction__mutmut_14, 
    'x__smooth_prediction__mutmut_15': x__smooth_prediction__mutmut_15
}

def _smooth_prediction(*args, **kwargs):
    result = _mutmut_trampoline(x__smooth_prediction__mutmut_orig, x__smooth_prediction__mutmut_mutants, args, kwargs)
    return result 

_smooth_prediction.__signature__ = _mutmut_signature(x__smooth_prediction__mutmut_orig)
x__smooth_prediction__mutmut_orig.__name__ = 'x__smooth_prediction'


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_orig(
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_1(
    pipeline: PredictablePipeline,
    stream: np.ndarray,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Produit des prédictions fenêtrées et des métriques de latence."""

    # Initialise la liste des événements pour tracer l'ordre temporel
    events: list[RealtimeEvent] = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_2(
    pipeline: PredictablePipeline,
    stream: np.ndarray,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Produit des prédictions fenêtrées et des métriques de latence."""

    # Initialise la liste des événements pour tracer l'ordre temporel
    events: list[RealtimeEvent] = []
    # Conserve un buffer borné pour lisser les prédictions successives
    buffer: Deque[int] = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_3(
    pipeline: PredictablePipeline,
    stream: np.ndarray,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Produit des prédictions fenêtrées et des métriques de latence."""

    # Initialise la liste des événements pour tracer l'ordre temporel
    events: list[RealtimeEvent] = []
    # Conserve un buffer borné pour lisser les prédictions successives
    buffer: Deque[int] = deque(maxlen=None)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_4(
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
    base_time = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_5(
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
        None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_6(
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
        _window_stream(None, config.window_size, config.step_size)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_7(
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
        _window_stream(stream, None, config.step_size)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_8(
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
        _window_stream(stream, config.window_size, None)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_9(
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
        _window_stream(config.window_size, config.step_size)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_10(
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
        _window_stream(stream, config.step_size)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_11(
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
        _window_stream(stream, config.window_size, )
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_12(
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
        offset_seconds = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_13(
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
        offset_seconds = float(start) * config.sfreq
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_14(
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
        offset_seconds = float(None) / config.sfreq
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_15(
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
        inference_start = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_16(
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
        raw_prediction = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_17(
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
        raw_prediction = int(None)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_18(
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
        raw_prediction = int(pipeline.predict(None)[0])
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_19(
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
        raw_prediction = int(pipeline.predict(window[np.newaxis, ...])[1])
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_20(
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
        latency = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_21(
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
        latency = time.perf_counter() + inference_start
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_22(
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
        if latency >= config.max_latency:
            # Signale une violation de SLA pour interrompre la session
            raise TimeoutError(
                f"Latence {latency:.3f}s dépasse {config.max_latency:.3f}s"
            )
        # Alimente le buffer de lissage avec la prédiction obtenue
        buffer.append(raw_prediction)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_23(
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
                None
            )
        # Alimente le buffer de lissage avec la prédiction obtenue
        buffer.append(raw_prediction)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_24(
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
        buffer.append(None)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_25(
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
        smoothed = None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_26(
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
        smoothed = _smooth_prediction(None)
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_27(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            None
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_28(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_29(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_30(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_31(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_32(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=latency,
                raw_prediction=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_33(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=latency,
                raw_prediction=raw_prediction,
                smoothed_prediction=None,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_34(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_35(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_36(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_37(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_38(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=latency,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_39(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start - base_time,
                latency=latency,
                raw_prediction=raw_prediction,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_40(
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
        # Construit l'événement associé à la fenêtre courante
        events.append(
            RealtimeEvent(
                window_index=index,
                window_offset=offset_seconds,
                inference_started_at=inference_start + base_time,
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


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_41(
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
    mean_latency = None
    # Calcule la latence maximale pour identifier d'éventuels pics
    max_latency = float(np.max([event.latency for event in events])) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_42(
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
        float(None) if events else 0.0
    )
    # Calcule la latence maximale pour identifier d'éventuels pics
    max_latency = float(np.max([event.latency for event in events])) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_43(
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
        float(np.mean(None)) if events else 0.0
    )
    # Calcule la latence maximale pour identifier d'éventuels pics
    max_latency = float(np.max([event.latency for event in events])) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_44(
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
        float(np.mean([event.latency for event in events])) if events else 1.0
    )
    # Calcule la latence maximale pour identifier d'éventuels pics
    max_latency = float(np.max([event.latency for event in events])) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_45(
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
    max_latency = None
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_46(
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
    max_latency = float(None) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_47(
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
    max_latency = float(np.max(None)) if events else 0.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_48(
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
    max_latency = float(np.max([event.latency for event in events])) if events else 1.0
    # Retourne les événements et les agrégats de latence pour inspection
    return {
        "events": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_49(
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
        "XXeventsXX": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_50(
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
        "EVENTS": events,
        "latency_mean": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_51(
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
        "XXlatency_meanXX": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_52(
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
        "LATENCY_MEAN": mean_latency,
        "latency_max": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_53(
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
        "XXlatency_maxXX": max_latency,
    }


# Applique la pipeline entraînée à un flux continu en mesurant la latence
def x_run_realtime_inference__mutmut_54(
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
        "LATENCY_MAX": max_latency,
    }

x_run_realtime_inference__mutmut_mutants : ClassVar[MutantDict] = {
'x_run_realtime_inference__mutmut_1': x_run_realtime_inference__mutmut_1, 
    'x_run_realtime_inference__mutmut_2': x_run_realtime_inference__mutmut_2, 
    'x_run_realtime_inference__mutmut_3': x_run_realtime_inference__mutmut_3, 
    'x_run_realtime_inference__mutmut_4': x_run_realtime_inference__mutmut_4, 
    'x_run_realtime_inference__mutmut_5': x_run_realtime_inference__mutmut_5, 
    'x_run_realtime_inference__mutmut_6': x_run_realtime_inference__mutmut_6, 
    'x_run_realtime_inference__mutmut_7': x_run_realtime_inference__mutmut_7, 
    'x_run_realtime_inference__mutmut_8': x_run_realtime_inference__mutmut_8, 
    'x_run_realtime_inference__mutmut_9': x_run_realtime_inference__mutmut_9, 
    'x_run_realtime_inference__mutmut_10': x_run_realtime_inference__mutmut_10, 
    'x_run_realtime_inference__mutmut_11': x_run_realtime_inference__mutmut_11, 
    'x_run_realtime_inference__mutmut_12': x_run_realtime_inference__mutmut_12, 
    'x_run_realtime_inference__mutmut_13': x_run_realtime_inference__mutmut_13, 
    'x_run_realtime_inference__mutmut_14': x_run_realtime_inference__mutmut_14, 
    'x_run_realtime_inference__mutmut_15': x_run_realtime_inference__mutmut_15, 
    'x_run_realtime_inference__mutmut_16': x_run_realtime_inference__mutmut_16, 
    'x_run_realtime_inference__mutmut_17': x_run_realtime_inference__mutmut_17, 
    'x_run_realtime_inference__mutmut_18': x_run_realtime_inference__mutmut_18, 
    'x_run_realtime_inference__mutmut_19': x_run_realtime_inference__mutmut_19, 
    'x_run_realtime_inference__mutmut_20': x_run_realtime_inference__mutmut_20, 
    'x_run_realtime_inference__mutmut_21': x_run_realtime_inference__mutmut_21, 
    'x_run_realtime_inference__mutmut_22': x_run_realtime_inference__mutmut_22, 
    'x_run_realtime_inference__mutmut_23': x_run_realtime_inference__mutmut_23, 
    'x_run_realtime_inference__mutmut_24': x_run_realtime_inference__mutmut_24, 
    'x_run_realtime_inference__mutmut_25': x_run_realtime_inference__mutmut_25, 
    'x_run_realtime_inference__mutmut_26': x_run_realtime_inference__mutmut_26, 
    'x_run_realtime_inference__mutmut_27': x_run_realtime_inference__mutmut_27, 
    'x_run_realtime_inference__mutmut_28': x_run_realtime_inference__mutmut_28, 
    'x_run_realtime_inference__mutmut_29': x_run_realtime_inference__mutmut_29, 
    'x_run_realtime_inference__mutmut_30': x_run_realtime_inference__mutmut_30, 
    'x_run_realtime_inference__mutmut_31': x_run_realtime_inference__mutmut_31, 
    'x_run_realtime_inference__mutmut_32': x_run_realtime_inference__mutmut_32, 
    'x_run_realtime_inference__mutmut_33': x_run_realtime_inference__mutmut_33, 
    'x_run_realtime_inference__mutmut_34': x_run_realtime_inference__mutmut_34, 
    'x_run_realtime_inference__mutmut_35': x_run_realtime_inference__mutmut_35, 
    'x_run_realtime_inference__mutmut_36': x_run_realtime_inference__mutmut_36, 
    'x_run_realtime_inference__mutmut_37': x_run_realtime_inference__mutmut_37, 
    'x_run_realtime_inference__mutmut_38': x_run_realtime_inference__mutmut_38, 
    'x_run_realtime_inference__mutmut_39': x_run_realtime_inference__mutmut_39, 
    'x_run_realtime_inference__mutmut_40': x_run_realtime_inference__mutmut_40, 
    'x_run_realtime_inference__mutmut_41': x_run_realtime_inference__mutmut_41, 
    'x_run_realtime_inference__mutmut_42': x_run_realtime_inference__mutmut_42, 
    'x_run_realtime_inference__mutmut_43': x_run_realtime_inference__mutmut_43, 
    'x_run_realtime_inference__mutmut_44': x_run_realtime_inference__mutmut_44, 
    'x_run_realtime_inference__mutmut_45': x_run_realtime_inference__mutmut_45, 
    'x_run_realtime_inference__mutmut_46': x_run_realtime_inference__mutmut_46, 
    'x_run_realtime_inference__mutmut_47': x_run_realtime_inference__mutmut_47, 
    'x_run_realtime_inference__mutmut_48': x_run_realtime_inference__mutmut_48, 
    'x_run_realtime_inference__mutmut_49': x_run_realtime_inference__mutmut_49, 
    'x_run_realtime_inference__mutmut_50': x_run_realtime_inference__mutmut_50, 
    'x_run_realtime_inference__mutmut_51': x_run_realtime_inference__mutmut_51, 
    'x_run_realtime_inference__mutmut_52': x_run_realtime_inference__mutmut_52, 
    'x_run_realtime_inference__mutmut_53': x_run_realtime_inference__mutmut_53, 
    'x_run_realtime_inference__mutmut_54': x_run_realtime_inference__mutmut_54
}

def run_realtime_inference(*args, **kwargs):
    result = _mutmut_trampoline(x_run_realtime_inference__mutmut_orig, x_run_realtime_inference__mutmut_mutants, args, kwargs)
    return result 

run_realtime_inference.__signature__ = _mutmut_signature(x_run_realtime_inference__mutmut_orig)
x_run_realtime_inference__mutmut_orig.__name__ = 'x_run_realtime_inference'


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_orig() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_1() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = None
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_2() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_3() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="XXApplique un modèle entraîné sur un flux fenêtréXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_4() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="applique un modèle entraîné sur un flux fenêtré",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_5() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="APPLIQUE UN MODÈLE ENTRAÎNÉ SUR UN FLUX FENÊTRÉ",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_6() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument(None, help="Identifiant du sujet (ex: S001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_7() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help=None)
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_8() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument(help="Identifiant du sujet (ex: S001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_9() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", )
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_10() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("XXsubjectXX", help="Identifiant du sujet (ex: S001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_11() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("SUBJECT", help="Identifiant du sujet (ex: S001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_12() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="XXIdentifiant du sujet (ex: S001)XX")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_13() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="identifiant du sujet (ex: s001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_14() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="IDENTIFIANT DU SUJET (EX: S001)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_15() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument(None, help="Identifiant du run (ex: R01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_16() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help=None)
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_17() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument(help="Identifiant du run (ex: R01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_18() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", )
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_19() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("XXrunXX", help="Identifiant du run (ex: R01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_20() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("RUN", help="Identifiant du run (ex: R01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_21() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="XXIdentifiant du run (ex: R01)XX")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_22() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="identifiant du run (ex: r01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_23() -> argparse.ArgumentParser:
    """Construit le parser CLI pour l'inférence temps réel."""

    # Crée le parser avec description dédiée au streaming
    parser = argparse.ArgumentParser(
        description="Applique un modèle entraîné sur un flux fenêtré",
    )
    # Ajoute l'argument positionnel du sujet pour cibler les artefacts
    parser.add_argument("subject", help="Identifiant du sujet (ex: S001)")
    # Ajoute l'argument positionnel du run pour cibler la session
    parser.add_argument("run", help="IDENTIFIANT DU RUN (EX: R01)")
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_24() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_25() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_26() -> argparse.ArgumentParser:
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
        default=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_27() -> argparse.ArgumentParser:
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
        help=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_28() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_29() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_30() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_31() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_32() -> argparse.ArgumentParser:
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
        "XX--data-dirXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_33() -> argparse.ArgumentParser:
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
        "--DATA-DIR",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_34() -> argparse.ArgumentParser:
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
        default=Path(None),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_35() -> argparse.ArgumentParser:
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
        default=Path("XXdataXX"),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_36() -> argparse.ArgumentParser:
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
        default=Path("DATA"),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_37() -> argparse.ArgumentParser:
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
        help="XXRépertoire racine contenant les fichiers numpyXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_38() -> argparse.ArgumentParser:
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
        help="répertoire racine contenant les fichiers numpy",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_39() -> argparse.ArgumentParser:
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
        help="RÉPERTOIRE RACINE CONTENANT LES FICHIERS NUMPY",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_40() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_41() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_42() -> argparse.ArgumentParser:
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
        default=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_43() -> argparse.ArgumentParser:
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
        help=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_44() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_45() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_46() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_47() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_48() -> argparse.ArgumentParser:
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
        "XX--artifacts-dirXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_49() -> argparse.ArgumentParser:
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
        "--ARTIFACTS-DIR",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_50() -> argparse.ArgumentParser:
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
        default=Path(None),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_51() -> argparse.ArgumentParser:
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
        default=Path("XXartifactsXX"),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_52() -> argparse.ArgumentParser:
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
        default=Path("ARTIFACTS"),
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_53() -> argparse.ArgumentParser:
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
        help="XXRépertoire racine où lire le modèleXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_54() -> argparse.ArgumentParser:
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
        help="répertoire racine où lire le modèle",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_55() -> argparse.ArgumentParser:
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
        help="RÉPERTOIRE RACINE OÙ LIRE LE MODÈLE",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_56() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_57() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_58() -> argparse.ArgumentParser:
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
        default=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_59() -> argparse.ArgumentParser:
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
        help=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_60() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_61() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_62() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_63() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_64() -> argparse.ArgumentParser:
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
        "XX--window-sizeXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_65() -> argparse.ArgumentParser:
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
        "--WINDOW-SIZE",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_66() -> argparse.ArgumentParser:
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
        default=51,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_67() -> argparse.ArgumentParser:
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
        help="XXTaille de fenêtre glissante en échantillonsXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_68() -> argparse.ArgumentParser:
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
        help="taille de fenêtre glissante en échantillons",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_69() -> argparse.ArgumentParser:
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
        help="TAILLE DE FENÊTRE GLISSANTE EN ÉCHANTILLONS",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_70() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_71() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_72() -> argparse.ArgumentParser:
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
        default=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_73() -> argparse.ArgumentParser:
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
        help=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_74() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_75() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_76() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_77() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_78() -> argparse.ArgumentParser:
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
        "XX--step-sizeXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_79() -> argparse.ArgumentParser:
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
        "--STEP-SIZE",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_80() -> argparse.ArgumentParser:
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
        default=26,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_81() -> argparse.ArgumentParser:
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
        help="XXPas entre deux fenêtres successivesXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_82() -> argparse.ArgumentParser:
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
        help="pas entre deux fenêtres successives",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_83() -> argparse.ArgumentParser:
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
        help="PAS ENTRE DEUX FENÊTRES SUCCESSIVES",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_84() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_85() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_86() -> argparse.ArgumentParser:
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
        default=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_87() -> argparse.ArgumentParser:
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
        help=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_88() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_89() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_90() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_91() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_92() -> argparse.ArgumentParser:
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
        "XX--buffer-sizeXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_93() -> argparse.ArgumentParser:
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
        "--BUFFER-SIZE",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_94() -> argparse.ArgumentParser:
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
        default=4,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_95() -> argparse.ArgumentParser:
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
        help="XXTaille du buffer pour lisser les prédictionsXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_96() -> argparse.ArgumentParser:
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
        help="taille du buffer pour lisser les prédictions",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_97() -> argparse.ArgumentParser:
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
        help="TAILLE DU BUFFER POUR LISSER LES PRÉDICTIONS",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_98() -> argparse.ArgumentParser:
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
        None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_99() -> argparse.ArgumentParser:
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
        type=None,
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_100() -> argparse.ArgumentParser:
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
        default=None,
        help="Latence maximale tolérée en secondes",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_101() -> argparse.ArgumentParser:
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
        help=None,
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_102() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_103() -> argparse.ArgumentParser:
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_104() -> argparse.ArgumentParser:
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
        help="Latence maximale tolérée en secondes",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_105() -> argparse.ArgumentParser:
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_106() -> argparse.ArgumentParser:
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
        "XX--max-latencyXX",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_107() -> argparse.ArgumentParser:
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
        "--MAX-LATENCY",
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
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_108() -> argparse.ArgumentParser:
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
        default=3.0,
        help="Latence maximale tolérée en secondes",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_109() -> argparse.ArgumentParser:
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
        help="XXLatence maximale tolérée en secondesXX",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_110() -> argparse.ArgumentParser:
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
        help="latence maximale tolérée en secondes",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_111() -> argparse.ArgumentParser:
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
        help="LATENCE MAXIMALE TOLÉRÉE EN SECONDES",
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


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_112() -> argparse.ArgumentParser:
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
        None,
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_113() -> argparse.ArgumentParser:
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
        type=None,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_114() -> argparse.ArgumentParser:
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
        default=None,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_115() -> argparse.ArgumentParser:
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
        help=None,
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_116() -> argparse.ArgumentParser:
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
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_117() -> argparse.ArgumentParser:
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
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_118() -> argparse.ArgumentParser:
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
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_119() -> argparse.ArgumentParser:
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
        )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_120() -> argparse.ArgumentParser:
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
        "XX--sfreqXX",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_121() -> argparse.ArgumentParser:
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
        "--SFREQ",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_122() -> argparse.ArgumentParser:
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
        default=51.0,
        help="Fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_123() -> argparse.ArgumentParser:
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
        help="XXFréquence d'échantillonnage utilisée pour l'offsetXX",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_124() -> argparse.ArgumentParser:
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
        help="fréquence d'échantillonnage utilisée pour l'offset",
    )
    # Retourne le parser configuré
    return parser


# Construit un argument parser aligné avec le mode realtime de mybci
def x_build_parser__mutmut_125() -> argparse.ArgumentParser:
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
        help="FRÉQUENCE D'ÉCHANTILLONNAGE UTILISÉE POUR L'OFFSET",
    )
    # Retourne le parser configuré
    return parser

x_build_parser__mutmut_mutants : ClassVar[MutantDict] = {
'x_build_parser__mutmut_1': x_build_parser__mutmut_1, 
    'x_build_parser__mutmut_2': x_build_parser__mutmut_2, 
    'x_build_parser__mutmut_3': x_build_parser__mutmut_3, 
    'x_build_parser__mutmut_4': x_build_parser__mutmut_4, 
    'x_build_parser__mutmut_5': x_build_parser__mutmut_5, 
    'x_build_parser__mutmut_6': x_build_parser__mutmut_6, 
    'x_build_parser__mutmut_7': x_build_parser__mutmut_7, 
    'x_build_parser__mutmut_8': x_build_parser__mutmut_8, 
    'x_build_parser__mutmut_9': x_build_parser__mutmut_9, 
    'x_build_parser__mutmut_10': x_build_parser__mutmut_10, 
    'x_build_parser__mutmut_11': x_build_parser__mutmut_11, 
    'x_build_parser__mutmut_12': x_build_parser__mutmut_12, 
    'x_build_parser__mutmut_13': x_build_parser__mutmut_13, 
    'x_build_parser__mutmut_14': x_build_parser__mutmut_14, 
    'x_build_parser__mutmut_15': x_build_parser__mutmut_15, 
    'x_build_parser__mutmut_16': x_build_parser__mutmut_16, 
    'x_build_parser__mutmut_17': x_build_parser__mutmut_17, 
    'x_build_parser__mutmut_18': x_build_parser__mutmut_18, 
    'x_build_parser__mutmut_19': x_build_parser__mutmut_19, 
    'x_build_parser__mutmut_20': x_build_parser__mutmut_20, 
    'x_build_parser__mutmut_21': x_build_parser__mutmut_21, 
    'x_build_parser__mutmut_22': x_build_parser__mutmut_22, 
    'x_build_parser__mutmut_23': x_build_parser__mutmut_23, 
    'x_build_parser__mutmut_24': x_build_parser__mutmut_24, 
    'x_build_parser__mutmut_25': x_build_parser__mutmut_25, 
    'x_build_parser__mutmut_26': x_build_parser__mutmut_26, 
    'x_build_parser__mutmut_27': x_build_parser__mutmut_27, 
    'x_build_parser__mutmut_28': x_build_parser__mutmut_28, 
    'x_build_parser__mutmut_29': x_build_parser__mutmut_29, 
    'x_build_parser__mutmut_30': x_build_parser__mutmut_30, 
    'x_build_parser__mutmut_31': x_build_parser__mutmut_31, 
    'x_build_parser__mutmut_32': x_build_parser__mutmut_32, 
    'x_build_parser__mutmut_33': x_build_parser__mutmut_33, 
    'x_build_parser__mutmut_34': x_build_parser__mutmut_34, 
    'x_build_parser__mutmut_35': x_build_parser__mutmut_35, 
    'x_build_parser__mutmut_36': x_build_parser__mutmut_36, 
    'x_build_parser__mutmut_37': x_build_parser__mutmut_37, 
    'x_build_parser__mutmut_38': x_build_parser__mutmut_38, 
    'x_build_parser__mutmut_39': x_build_parser__mutmut_39, 
    'x_build_parser__mutmut_40': x_build_parser__mutmut_40, 
    'x_build_parser__mutmut_41': x_build_parser__mutmut_41, 
    'x_build_parser__mutmut_42': x_build_parser__mutmut_42, 
    'x_build_parser__mutmut_43': x_build_parser__mutmut_43, 
    'x_build_parser__mutmut_44': x_build_parser__mutmut_44, 
    'x_build_parser__mutmut_45': x_build_parser__mutmut_45, 
    'x_build_parser__mutmut_46': x_build_parser__mutmut_46, 
    'x_build_parser__mutmut_47': x_build_parser__mutmut_47, 
    'x_build_parser__mutmut_48': x_build_parser__mutmut_48, 
    'x_build_parser__mutmut_49': x_build_parser__mutmut_49, 
    'x_build_parser__mutmut_50': x_build_parser__mutmut_50, 
    'x_build_parser__mutmut_51': x_build_parser__mutmut_51, 
    'x_build_parser__mutmut_52': x_build_parser__mutmut_52, 
    'x_build_parser__mutmut_53': x_build_parser__mutmut_53, 
    'x_build_parser__mutmut_54': x_build_parser__mutmut_54, 
    'x_build_parser__mutmut_55': x_build_parser__mutmut_55, 
    'x_build_parser__mutmut_56': x_build_parser__mutmut_56, 
    'x_build_parser__mutmut_57': x_build_parser__mutmut_57, 
    'x_build_parser__mutmut_58': x_build_parser__mutmut_58, 
    'x_build_parser__mutmut_59': x_build_parser__mutmut_59, 
    'x_build_parser__mutmut_60': x_build_parser__mutmut_60, 
    'x_build_parser__mutmut_61': x_build_parser__mutmut_61, 
    'x_build_parser__mutmut_62': x_build_parser__mutmut_62, 
    'x_build_parser__mutmut_63': x_build_parser__mutmut_63, 
    'x_build_parser__mutmut_64': x_build_parser__mutmut_64, 
    'x_build_parser__mutmut_65': x_build_parser__mutmut_65, 
    'x_build_parser__mutmut_66': x_build_parser__mutmut_66, 
    'x_build_parser__mutmut_67': x_build_parser__mutmut_67, 
    'x_build_parser__mutmut_68': x_build_parser__mutmut_68, 
    'x_build_parser__mutmut_69': x_build_parser__mutmut_69, 
    'x_build_parser__mutmut_70': x_build_parser__mutmut_70, 
    'x_build_parser__mutmut_71': x_build_parser__mutmut_71, 
    'x_build_parser__mutmut_72': x_build_parser__mutmut_72, 
    'x_build_parser__mutmut_73': x_build_parser__mutmut_73, 
    'x_build_parser__mutmut_74': x_build_parser__mutmut_74, 
    'x_build_parser__mutmut_75': x_build_parser__mutmut_75, 
    'x_build_parser__mutmut_76': x_build_parser__mutmut_76, 
    'x_build_parser__mutmut_77': x_build_parser__mutmut_77, 
    'x_build_parser__mutmut_78': x_build_parser__mutmut_78, 
    'x_build_parser__mutmut_79': x_build_parser__mutmut_79, 
    'x_build_parser__mutmut_80': x_build_parser__mutmut_80, 
    'x_build_parser__mutmut_81': x_build_parser__mutmut_81, 
    'x_build_parser__mutmut_82': x_build_parser__mutmut_82, 
    'x_build_parser__mutmut_83': x_build_parser__mutmut_83, 
    'x_build_parser__mutmut_84': x_build_parser__mutmut_84, 
    'x_build_parser__mutmut_85': x_build_parser__mutmut_85, 
    'x_build_parser__mutmut_86': x_build_parser__mutmut_86, 
    'x_build_parser__mutmut_87': x_build_parser__mutmut_87, 
    'x_build_parser__mutmut_88': x_build_parser__mutmut_88, 
    'x_build_parser__mutmut_89': x_build_parser__mutmut_89, 
    'x_build_parser__mutmut_90': x_build_parser__mutmut_90, 
    'x_build_parser__mutmut_91': x_build_parser__mutmut_91, 
    'x_build_parser__mutmut_92': x_build_parser__mutmut_92, 
    'x_build_parser__mutmut_93': x_build_parser__mutmut_93, 
    'x_build_parser__mutmut_94': x_build_parser__mutmut_94, 
    'x_build_parser__mutmut_95': x_build_parser__mutmut_95, 
    'x_build_parser__mutmut_96': x_build_parser__mutmut_96, 
    'x_build_parser__mutmut_97': x_build_parser__mutmut_97, 
    'x_build_parser__mutmut_98': x_build_parser__mutmut_98, 
    'x_build_parser__mutmut_99': x_build_parser__mutmut_99, 
    'x_build_parser__mutmut_100': x_build_parser__mutmut_100, 
    'x_build_parser__mutmut_101': x_build_parser__mutmut_101, 
    'x_build_parser__mutmut_102': x_build_parser__mutmut_102, 
    'x_build_parser__mutmut_103': x_build_parser__mutmut_103, 
    'x_build_parser__mutmut_104': x_build_parser__mutmut_104, 
    'x_build_parser__mutmut_105': x_build_parser__mutmut_105, 
    'x_build_parser__mutmut_106': x_build_parser__mutmut_106, 
    'x_build_parser__mutmut_107': x_build_parser__mutmut_107, 
    'x_build_parser__mutmut_108': x_build_parser__mutmut_108, 
    'x_build_parser__mutmut_109': x_build_parser__mutmut_109, 
    'x_build_parser__mutmut_110': x_build_parser__mutmut_110, 
    'x_build_parser__mutmut_111': x_build_parser__mutmut_111, 
    'x_build_parser__mutmut_112': x_build_parser__mutmut_112, 
    'x_build_parser__mutmut_113': x_build_parser__mutmut_113, 
    'x_build_parser__mutmut_114': x_build_parser__mutmut_114, 
    'x_build_parser__mutmut_115': x_build_parser__mutmut_115, 
    'x_build_parser__mutmut_116': x_build_parser__mutmut_116, 
    'x_build_parser__mutmut_117': x_build_parser__mutmut_117, 
    'x_build_parser__mutmut_118': x_build_parser__mutmut_118, 
    'x_build_parser__mutmut_119': x_build_parser__mutmut_119, 
    'x_build_parser__mutmut_120': x_build_parser__mutmut_120, 
    'x_build_parser__mutmut_121': x_build_parser__mutmut_121, 
    'x_build_parser__mutmut_122': x_build_parser__mutmut_122, 
    'x_build_parser__mutmut_123': x_build_parser__mutmut_123, 
    'x_build_parser__mutmut_124': x_build_parser__mutmut_124, 
    'x_build_parser__mutmut_125': x_build_parser__mutmut_125
}

def build_parser(*args, **kwargs):
    result = _mutmut_trampoline(x_build_parser__mutmut_orig, x_build_parser__mutmut_mutants, args, kwargs)
    return result 

build_parser.__signature__ = _mutmut_signature(x_build_parser__mutmut_orig)
x_build_parser__mutmut_orig.__name__ = 'x_build_parser'


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_orig(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_1(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = None
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_2(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir * subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_3(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = None
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_4(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir * f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir / f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_5(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = None
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path


# Construit les chemins des données pour un sujet et un run donnés
def x__resolve_data_paths__mutmut_6(subject: str, run: str, data_dir: Path) -> tuple[Path, Path]:
    """Retourne les chemins des matrices X et y pour un sujet/run."""

    # Localise le sous-dossier spécifique au sujet
    base_dir = data_dir / subject
    # Compose le chemin du fichier de données numpy
    features_path = base_dir / f"{run}_X.npy"
    # Compose le chemin du fichier d'étiquettes numpy
    labels_path = base_dir * f"{run}_y.npy"
    # Retourne les deux chemins pour chargement ultérieur
    return features_path, labels_path

x__resolve_data_paths__mutmut_mutants : ClassVar[MutantDict] = {
'x__resolve_data_paths__mutmut_1': x__resolve_data_paths__mutmut_1, 
    'x__resolve_data_paths__mutmut_2': x__resolve_data_paths__mutmut_2, 
    'x__resolve_data_paths__mutmut_3': x__resolve_data_paths__mutmut_3, 
    'x__resolve_data_paths__mutmut_4': x__resolve_data_paths__mutmut_4, 
    'x__resolve_data_paths__mutmut_5': x__resolve_data_paths__mutmut_5, 
    'x__resolve_data_paths__mutmut_6': x__resolve_data_paths__mutmut_6
}

def _resolve_data_paths(*args, **kwargs):
    result = _mutmut_trampoline(x__resolve_data_paths__mutmut_orig, x__resolve_data_paths__mutmut_mutants, args, kwargs)
    return result 

_resolve_data_paths.__signature__ = _mutmut_signature(x__resolve_data_paths__mutmut_orig)
x__resolve_data_paths__mutmut_orig.__name__ = 'x__resolve_data_paths'


# Charge les matrices numpy attendues pour simuler un flux
def x__load_data__mutmut_orig(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour le streaming
    return X, y


# Charge les matrices numpy attendues pour simuler un flux
def x__load_data__mutmut_1(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = None
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour le streaming
    return X, y


# Charge les matrices numpy attendues pour simuler un flux
def x__load_data__mutmut_2(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(None)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(labels_path)
    # Retourne les deux tableaux prêts pour le streaming
    return X, y


# Charge les matrices numpy attendues pour simuler un flux
def x__load_data__mutmut_3(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = None
    # Retourne les deux tableaux prêts pour le streaming
    return X, y


# Charge les matrices numpy attendues pour simuler un flux
def x__load_data__mutmut_4(features_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge les données et étiquettes depuis des fichiers numpy."""

    # Utilise numpy.load pour récupérer les features en mémoire
    X = np.load(features_path)
    # Utilise numpy.load pour récupérer les labels associés
    y = np.load(None)
    # Retourne les deux tableaux prêts pour le streaming
    return X, y

x__load_data__mutmut_mutants : ClassVar[MutantDict] = {
'x__load_data__mutmut_1': x__load_data__mutmut_1, 
    'x__load_data__mutmut_2': x__load_data__mutmut_2, 
    'x__load_data__mutmut_3': x__load_data__mutmut_3, 
    'x__load_data__mutmut_4': x__load_data__mutmut_4
}

def _load_data(*args, **kwargs):
    result = _mutmut_trampoline(x__load_data__mutmut_orig, x__load_data__mutmut_mutants, args, kwargs)
    return result 

_load_data.__signature__ = _mutmut_signature(x__load_data__mutmut_orig)
x__load_data__mutmut_orig.__name__ = 'x__load_data'


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_orig(
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_1(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = None
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_2(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(None, run, data_dir)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_3(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, None, data_dir)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_4(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, run, None)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_5(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(run, data_dir)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_6(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, data_dir)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_7(
    subject: str,
    run: str,
    data_dir: Path,
    artifacts_dir: Path,
    config: RealtimeConfig,
) -> RealtimeResult:
    """Charge le modèle entraîné et lance l'inférence fenêtrée."""

    # Résout les chemins des fichiers de données pour le sujet/run
    features_path, labels_path = _resolve_data_paths(subject, run, )
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_8(
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
    X, _ = None
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_9(
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
    X, _ = _load_data(None, labels_path)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_10(
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
    X, _ = _load_data(features_path, None)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_11(
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
    X, _ = _load_data(labels_path)
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_12(
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
    X, _ = _load_data(features_path, )
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


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_13(
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
    stream = None
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_14(
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
    stream = np.concatenate(None, axis=1)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_15(
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
    stream = np.concatenate(list(X), axis=None)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_16(
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
    stream = np.concatenate(axis=1)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_17(
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
    stream = np.concatenate(list(X), )
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_18(
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
    stream = np.concatenate(list(None), axis=1)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_19(
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
    stream = np.concatenate(list(X), axis=2)
    # Charge la pipeline entraînée depuis le joblib sauvegardé
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_20(
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
    pipeline = None
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_21(
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
    pipeline = load_pipeline(None)
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_22(
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
    pipeline = load_pipeline(str(None))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_23(
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
    pipeline = load_pipeline(str(artifacts_dir / subject / run * "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_24(
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
    pipeline = load_pipeline(str(artifacts_dir / subject * run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_25(
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
    pipeline = load_pipeline(str(artifacts_dir * subject / run / "model.joblib"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_26(
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
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "XXmodel.joblibXX"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_27(
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
    pipeline = load_pipeline(str(artifacts_dir / subject / run / "MODEL.JOBLIB"))
    # Lance la boucle temps réel et retourne les métriques associées
    return run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_28(
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
        pipeline=None,
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_29(
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
        stream=None,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_30(
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
        config=None,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_31(
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
        stream=stream,
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_32(
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
        config=config,
    )


# Orchestre une session temps réel à partir d'artefacts persistés
def x_run_realtime_session__mutmut_33(
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
        )

x_run_realtime_session__mutmut_mutants : ClassVar[MutantDict] = {
'x_run_realtime_session__mutmut_1': x_run_realtime_session__mutmut_1, 
    'x_run_realtime_session__mutmut_2': x_run_realtime_session__mutmut_2, 
    'x_run_realtime_session__mutmut_3': x_run_realtime_session__mutmut_3, 
    'x_run_realtime_session__mutmut_4': x_run_realtime_session__mutmut_4, 
    'x_run_realtime_session__mutmut_5': x_run_realtime_session__mutmut_5, 
    'x_run_realtime_session__mutmut_6': x_run_realtime_session__mutmut_6, 
    'x_run_realtime_session__mutmut_7': x_run_realtime_session__mutmut_7, 
    'x_run_realtime_session__mutmut_8': x_run_realtime_session__mutmut_8, 
    'x_run_realtime_session__mutmut_9': x_run_realtime_session__mutmut_9, 
    'x_run_realtime_session__mutmut_10': x_run_realtime_session__mutmut_10, 
    'x_run_realtime_session__mutmut_11': x_run_realtime_session__mutmut_11, 
    'x_run_realtime_session__mutmut_12': x_run_realtime_session__mutmut_12, 
    'x_run_realtime_session__mutmut_13': x_run_realtime_session__mutmut_13, 
    'x_run_realtime_session__mutmut_14': x_run_realtime_session__mutmut_14, 
    'x_run_realtime_session__mutmut_15': x_run_realtime_session__mutmut_15, 
    'x_run_realtime_session__mutmut_16': x_run_realtime_session__mutmut_16, 
    'x_run_realtime_session__mutmut_17': x_run_realtime_session__mutmut_17, 
    'x_run_realtime_session__mutmut_18': x_run_realtime_session__mutmut_18, 
    'x_run_realtime_session__mutmut_19': x_run_realtime_session__mutmut_19, 
    'x_run_realtime_session__mutmut_20': x_run_realtime_session__mutmut_20, 
    'x_run_realtime_session__mutmut_21': x_run_realtime_session__mutmut_21, 
    'x_run_realtime_session__mutmut_22': x_run_realtime_session__mutmut_22, 
    'x_run_realtime_session__mutmut_23': x_run_realtime_session__mutmut_23, 
    'x_run_realtime_session__mutmut_24': x_run_realtime_session__mutmut_24, 
    'x_run_realtime_session__mutmut_25': x_run_realtime_session__mutmut_25, 
    'x_run_realtime_session__mutmut_26': x_run_realtime_session__mutmut_26, 
    'x_run_realtime_session__mutmut_27': x_run_realtime_session__mutmut_27, 
    'x_run_realtime_session__mutmut_28': x_run_realtime_session__mutmut_28, 
    'x_run_realtime_session__mutmut_29': x_run_realtime_session__mutmut_29, 
    'x_run_realtime_session__mutmut_30': x_run_realtime_session__mutmut_30, 
    'x_run_realtime_session__mutmut_31': x_run_realtime_session__mutmut_31, 
    'x_run_realtime_session__mutmut_32': x_run_realtime_session__mutmut_32, 
    'x_run_realtime_session__mutmut_33': x_run_realtime_session__mutmut_33
}

def run_realtime_session(*args, **kwargs):
    result = _mutmut_trampoline(x_run_realtime_session__mutmut_orig, x_run_realtime_session__mutmut_mutants, args, kwargs)
    return result 

run_realtime_session.__signature__ = _mutmut_signature(x_run_realtime_session__mutmut_orig)
x_run_realtime_session__mutmut_orig.__name__ = 'x_run_realtime_session'


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_orig(argv: list[str] | None = None) -> int:
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
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_1(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = None
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
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_2(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = None
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
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_3(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(None)
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
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_4(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = None
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_5(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=None,
        run=args.run,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_6(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=args.subject,
        run=None,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_7(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=args.subject,
        run=args.run,
        data_dir=None,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_8(argv: list[str] | None = None) -> int:
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
        artifacts_dir=None,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_9(argv: list[str] | None = None) -> int:
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
        config=None,
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_10(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        run=args.run,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_11(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=args.subject,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_12(argv: list[str] | None = None) -> int:
    """Parse les arguments et lance l'inférence temps réel."""

    # Construit le parser pour interpréter les arguments
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Lance une session temps réel à partir des paramètres fournis
    _ = run_realtime_session(
        subject=args.subject,
        run=args.run,
        artifacts_dir=args.artifacts_dir,
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_13(argv: list[str] | None = None) -> int:
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
        config=RealtimeConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_14(argv: list[str] | None = None) -> int:
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
        )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_15(argv: list[str] | None = None) -> int:
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
            window_size=None,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_16(argv: list[str] | None = None) -> int:
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
            step_size=None,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_17(argv: list[str] | None = None) -> int:
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
            buffer_size=None,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_18(argv: list[str] | None = None) -> int:
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
            max_latency=None,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_19(argv: list[str] | None = None) -> int:
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
            max_latency=args.max_latency,
            sfreq=None,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_20(argv: list[str] | None = None) -> int:
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
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_21(argv: list[str] | None = None) -> int:
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
            buffer_size=args.buffer_size,
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_22(argv: list[str] | None = None) -> int:
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
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_23(argv: list[str] | None = None) -> int:
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


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_24(argv: list[str] | None = None) -> int:
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
            max_latency=args.max_latency,
            ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 0


# Point d'entrée principal pour l'exécution en ligne de commande
def x_main__mutmut_25(argv: list[str] | None = None) -> int:
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
            max_latency=args.max_latency,
            sfreq=args.sfreq,
        ),
    )
    # Retourne 0 pour signaler un succès CLI à mybci
    return 1

x_main__mutmut_mutants : ClassVar[MutantDict] = {
'x_main__mutmut_1': x_main__mutmut_1, 
    'x_main__mutmut_2': x_main__mutmut_2, 
    'x_main__mutmut_3': x_main__mutmut_3, 
    'x_main__mutmut_4': x_main__mutmut_4, 
    'x_main__mutmut_5': x_main__mutmut_5, 
    'x_main__mutmut_6': x_main__mutmut_6, 
    'x_main__mutmut_7': x_main__mutmut_7, 
    'x_main__mutmut_8': x_main__mutmut_8, 
    'x_main__mutmut_9': x_main__mutmut_9, 
    'x_main__mutmut_10': x_main__mutmut_10, 
    'x_main__mutmut_11': x_main__mutmut_11, 
    'x_main__mutmut_12': x_main__mutmut_12, 
    'x_main__mutmut_13': x_main__mutmut_13, 
    'x_main__mutmut_14': x_main__mutmut_14, 
    'x_main__mutmut_15': x_main__mutmut_15, 
    'x_main__mutmut_16': x_main__mutmut_16, 
    'x_main__mutmut_17': x_main__mutmut_17, 
    'x_main__mutmut_18': x_main__mutmut_18, 
    'x_main__mutmut_19': x_main__mutmut_19, 
    'x_main__mutmut_20': x_main__mutmut_20, 
    'x_main__mutmut_21': x_main__mutmut_21, 
    'x_main__mutmut_22': x_main__mutmut_22, 
    'x_main__mutmut_23': x_main__mutmut_23, 
    'x_main__mutmut_24': x_main__mutmut_24, 
    'x_main__mutmut_25': x_main__mutmut_25
}

def main(*args, **kwargs):
    result = _mutmut_trampoline(x_main__mutmut_orig, x_main__mutmut_mutants, args, kwargs)
    return result 

main.__signature__ = _mutmut_signature(x_main__mutmut_orig)
x_main__mutmut_orig.__name__ = 'x_main'


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
