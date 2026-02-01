"""Utilitaires de configuration pour TPV."""

# Garantit l'évaluation paresseuse des annotations de type
from __future__ import annotations

# Offre la sérialisation JSON pour charger une configuration externe
import json

# Offre les validations numériques pour les bornes de fenêtres
import math

# Fournit la structure immuable pour les configurations
from dataclasses import dataclass

# Garantit l'accès aux chemins portables pour la configuration
from pathlib import Path

# Expose les types de mapping pour les signatures
from typing import Mapping

# Définit le type d'une fenêtre d'epoch en secondes
EpochWindow = tuple[float, float]

# Fixe le nombre de bornes attendues pour une fenêtre
WINDOW_BOUNDS_COUNT = 2

# Centralise les fenêtres candidates par défaut pour l'epoching
DEFAULT_EPOCH_WINDOWS: tuple[EpochWindow, ...] = (
    # Cible une fenêtre courte centrée sur l'ERD/ERS initial
    (0.5, 2.5),
    # Cible une fenêtre plus tardive pour les sujets lents
    (1.0, 3.0),
    # Cible une fenêtre plus proche du cue pour capturer l'initiation
    (0.0, 2.0),
)

# Fixe la fenêtre par défaut utilisée en dernier recours
DEFAULT_EPOCH_WINDOW = DEFAULT_EPOCH_WINDOWS[0]


# Regroupe la configuration des fenêtres d'epoching
@dataclass(frozen=True)
class EpochWindowConfig:
    """Stocke les fenêtres par défaut et les overrides par sujet."""

    # Porte la liste canonique des fenêtres candidates
    default_windows: tuple[EpochWindow, ...]
    # Mappe les overrides par identifiant de sujet normalisé
    subject_overrides: Mapping[str, tuple[EpochWindow, ...]]


# Valide et convertit une fenêtre brute en tuple de floats
def _parse_window(raw_window: object, label: str) -> EpochWindow:
    """Valide une fenêtre d'epoch et retourne un tuple (tmin, tmax)."""

    # Vérifie que la fenêtre est une séquence indexable
    if not isinstance(raw_window, (list, tuple)):
        # Signale un format invalide pour faciliter le diagnostic
        raise ValueError(f"Fenêtre invalide pour {label}: {raw_window}")
    # Refuse les fenêtres qui n'ont pas exactement deux bornes
    if len(raw_window) != WINDOW_BOUNDS_COUNT:
        # Signale un format invalide pour éviter les ambiguïtés
        raise ValueError(f"Fenêtre invalide pour {label}: {raw_window}")
    # Convertit tmin en float pour normaliser les valeurs
    tmin = float(raw_window[0])
    # Convertit tmax en float pour normaliser les valeurs
    tmax = float(raw_window[1])
    # Refuse les valeurs non finies pour éviter des erreurs MNE
    if not math.isfinite(tmin) or not math.isfinite(tmax):
        # Signale des bornes non finies pour guider la correction
        raise ValueError(f"Fenêtre invalide pour {label}: {raw_window}")
    # Refuse les bornes inversées ou nulles pour éviter des epochs vides
    if tmin >= tmax:
        # Signale des bornes incohérentes pour correction
        raise ValueError(f"Fenêtre invalide pour {label}: {raw_window}")
    # Retourne la fenêtre validée sous forme de tuple
    return (tmin, tmax)


# Valide une liste de fenêtres et retourne une séquence immuable
def _parse_window_list(raw_windows: object, label: str) -> tuple[EpochWindow, ...]:
    """Convertit une liste JSON en fenêtres d'epoch validées."""

    # Vérifie que la valeur est une liste JSON
    if not isinstance(raw_windows, list):
        # Signale un format invalide pour l'utilisateur
        raise ValueError(f"Liste de fenêtres invalide pour {label}: {raw_windows}")
    # Refuse une liste vide pour garantir au moins une fenêtre
    if not raw_windows:
        # Signale l'absence de fenêtres pour éviter un fallback ambigu
        raise ValueError(f"Liste de fenêtres vide pour {label}")
    # Prépare la liste des fenêtres validées
    parsed_windows: list[EpochWindow] = []
    # Parcourt chaque fenêtre déclarée pour la valider
    for index, raw_window in enumerate(raw_windows):
        # Construit un label précis pour l'erreur éventuelle
        item_label = f"{label}[{index}]"
        # Valide et convertit la fenêtre
        parsed_windows.append(_parse_window(raw_window, item_label))
    # Retourne la liste validée sous forme de tuple immuable
    return tuple(parsed_windows)


# Charge la configuration de fenêtres à partir d'un fichier JSON
def load_epoch_window_config(config_path: Path | None) -> EpochWindowConfig:
    """Charge la configuration des fenêtres d'epochs depuis un fichier."""

    # Retourne les defaults si aucun fichier n'est fourni
    if config_path is None:
        # Construit une configuration par défaut sans overrides
        return EpochWindowConfig(
            default_windows=DEFAULT_EPOCH_WINDOWS,
            subject_overrides={},
        )
    # Vérifie la présence du fichier pour éviter un échec silencieux
    if not config_path.exists():
        # Signale explicitement le fichier manquant
        raise FileNotFoundError(f"Config de fenêtres introuvable: {config_path}")
    # Charge le contenu JSON brut
    payload = json.loads(config_path.read_text())
    # Vérifie que la racine JSON est un objet
    if not isinstance(payload, dict):
        # Signale un format inattendu pour guider l'utilisateur
        raise ValueError("La config de fenêtres doit être un objet JSON.")
    # Récupère la section default si elle est fournie
    default_payload = payload.get("default")
    # Utilise les defaults internes si la section est absente
    if default_payload is None:
        # Garde les fenêtres par défaut déclarées dans le module
        default_windows = DEFAULT_EPOCH_WINDOWS
    else:
        # Valide la section default pour obtenir des fenêtres propres
        default_windows = _parse_window_list(default_payload, "default")
    # Récupère les overrides par sujet si présents
    subjects_payload = payload.get("subjects", {})
    # Refuse un champ subjects de type inattendu
    if not isinstance(subjects_payload, dict):
        # Signale un format invalide pour l'utilisateur
        raise ValueError("Le champ 'subjects' doit être un objet JSON.")
    # Prépare le mapping des overrides validés
    subject_overrides: dict[str, tuple[EpochWindow, ...]] = {}
    # Parcourt chaque sujet déclaré dans le JSON
    for subject, raw_windows in subjects_payload.items():
        # Normalise la clé sujet en chaîne
        subject_key = str(subject)
        # Valide la liste de fenêtres pour ce sujet
        subject_overrides[subject_key] = _parse_window_list(
            raw_windows,
            f"subjects.{subject_key}",
        )
    # Retourne la configuration consolidée
    return EpochWindowConfig(
        default_windows=default_windows,
        subject_overrides=subject_overrides,
    )


# Construit la configuration par défaut sans fichier externe
def default_epoch_window_config() -> EpochWindowConfig:
    """Retourne la configuration de fenêtres d'epochs par défaut."""

    # Délègue le chargement pour centraliser les valeurs par défaut
    return load_epoch_window_config(None)


# Résout les fenêtres à utiliser pour un sujet spécifique
def resolve_epoch_windows(
    subject: str,
    config: EpochWindowConfig,
) -> tuple[EpochWindow, ...]:
    """Retourne la liste des fenêtres d'epochs pour un sujet donné."""

    # Cherche une configuration dédiée au sujet
    override = config.subject_overrides.get(subject)
    # Retourne les overrides lorsqu'ils existent
    if override is not None:
        return override
    # Retourne les fenêtres par défaut en absence d'override
    return config.default_windows
