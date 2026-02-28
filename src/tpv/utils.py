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

# Code de sortie réservé aux erreurs utilisateur déjà rendues lisiblement
HANDLED_CLI_ERROR_EXIT_CODE = 2

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


# Regroupe un message CLI court avec une action corrective optionnelle
@dataclass(frozen=True)
class CliErrorDiagnostic:
    """Décrit une erreur CLI sous forme concise et actionnable."""

    # Résume l'erreur en une ligne sans préfixe ERREUR:
    summary: str
    # Propose une action concrète lorsque le diagnostic le permet
    action: str | None = None


# Décode une erreur JSON structurée lorsque le backend l'expose
def _load_error_payload(error: Exception) -> Mapping[str, object] | None:
    """Retourne le payload JSON d'une erreur structurée si disponible."""

    # Tente de décoder le message d'erreur en JSON pour extraire les détails
    try:
        payload = json.loads(str(error))
    except json.JSONDecodeError:
        # Retourne None lorsque l'erreur est un simple message texte
        return None
    # Refuse tout payload non dictionnaire pour garder une interface stable
    if not isinstance(payload, dict):
        return None
    # Retourne le payload structuré validé
    return payload


# Formate un chemin pour la CLI en privilégiant un chemin relatif au repo
def _format_cli_path(path: str | Path) -> str:
    """Retourne un chemin stable et lisible pour les messages CLI."""

    # Normalise le chemin quelle que soit sa représentation d'origine
    normalized_path = Path(path).expanduser()
    # Résout le chemin sans exiger son existence pour limiter les surprises
    resolved_path = normalized_path.resolve()
    # Construit la racine courante pour tenter un affichage relatif
    current_root = Path.cwd().resolve()
    # Préfère un chemin relatif lorsque le fichier est dans le dépôt courant
    try:
        return str(resolved_path.relative_to(current_root))
    except ValueError:
        # Conserve l'absolu si le chemin est externe au dépôt courant
        return str(resolved_path)


# Détermine une cible lisible pour un couple sujet/run optionnel
def _format_subject_run(subject: str | None, run: str | None) -> str | None:
    """Construit une étiquette stable `SXXX RYY` pour les messages CLI."""

    # Ne conserve que les fragments réellement fournis
    parts = [part for part in (subject, run) if part]
    # Retourne None lorsqu'aucun identifiant n'est disponible
    if not parts:
        return None
    # Construit une étiquette compacte pour la CLI
    return " ".join(parts)


# Déduit les fichiers à rendre lisibles à partir d'un chemin EDF ou event
def _build_permission_targets(path: Path) -> tuple[str, ...]:
    """Retourne les chemins à inclure dans une commande chmod actionnable."""

    # Prépare les fichiers à débloquer en priorité
    targets: list[Path] = []
    # Gère le cas standard d'un EDF principal
    if path.suffix.lower() == ".edf":
        targets.append(path)
        targets.append(path.with_suffix(".edf.event"))
    # Gère le cas où l'erreur cible déjà le fichier .edf.event
    elif path.name.endswith(".edf.event"):
        targets.append(path.with_suffix(""))
        targets.append(path)
    else:
        # Conserve tout autre chemin tel quel en dernier recours
        targets.append(path)
    # Déduplique les cibles tout en conservant leur ordre
    unique_targets = list(dict.fromkeys(targets))
    # Retourne des chemins déjà formatés pour la CLI
    return tuple(_format_cli_path(target) for target in unique_targets)


# Déduit la racine dataset à débloquer lorsqu'un chemin interne est illisible
def _find_blocking_directory(path: Path) -> Path | None:
    """Retourne le dossier le plus probable qui bloque la traversée."""

    # Développe le chemin reçu pour uniformiser les contrôles
    normalized_path = path.expanduser()
    # Détermine si le chemin est absolu ou relatif au dépôt courant
    is_absolute = normalized_path.is_absolute()
    # Définit le point de départ du parcours progressif
    current = Path(normalized_path.anchor) if is_absolute else Path.cwd().resolve()
    # Isole les segments à parcourir un par un
    path_parts = normalized_path.parts[1:] if is_absolute else normalized_path.parts
    # Mémorise le dernier dossier qui a pu être résolu sans PermissionError
    last_accessible_directory: Path | None = None

    # Parcourt les segments du chemin jusqu'au premier refus d'accès
    for index, part in enumerate(path_parts):
        # Construit le candidat courant à inspecter
        current = current / part
        try:
            # Déclenche une résolution réelle du composant courant
            current.stat()
        except PermissionError:
            # Retourne le dernier dossier accessible si disponible
            return last_accessible_directory
        except OSError:
            # Abandonne la détection stricte si le chemin ne peut être inspecté
            return None
        # Le dernier segment désigne souvent le fichier, pas le dossier bloquant
        is_last_part = index == len(path_parts) - 1
        if not is_last_part:
            # Mémorise ce dossier comme dernier niveau traversable connu
            last_accessible_directory = current
    # Retourne None si aucun composant n'a levé de PermissionError
    return None


# Déduit la racine dataset à débloquer lorsqu'un chemin interne est illisible
def _infer_data_directory_target(
    path: Path,
    *,
    subject: str | None = None,
) -> Path | None:
    """Retourne le dossier dataset probable à rendre traversable."""

    # Normalise le chemin reçu pour raisonner sur une représentation stable
    normalized_path = path.expanduser()
    # Cherche d'abord le niveau exact qui bloque réellement la traversée
    blocked_directory = _find_blocking_directory(normalized_path)
    if blocked_directory is not None:
        return blocked_directory
    # Si le sujet est connu, remonte jusqu'au dossier précédent `SXXX`
    if subject is not None:
        for parent in normalized_path.parents:
            if parent.name == subject:
                return parent
    # Garde un traitement explicite pour le répertoire `data`
    if normalized_path.name == "data":
        return normalized_path
    # Remonte au premier ancêtre nommé `data` si présent
    for parent in (normalized_path, *normalized_path.parents):
        if parent.name == "data":
            return parent
    # Retourne None lorsqu'aucune racine dataset claire n'est déductible
    return None


# Construit un diagnostic à partir d'un PermissionError brut Python
def _explain_permission_error(
    error: PermissionError,
    *,
    subject: str | None = None,
) -> CliErrorDiagnostic:
    """Traduit un PermissionError système en message CLI actionnable."""

    # Récupère le chemin éventuellement fourni par l'exception système
    filename = getattr(error, "filename", None)
    if filename is None:
        return CliErrorDiagnostic(
            summary="lecture du dataset impossible",
            action="vérifiez les droits d'accès du dataset EEGMMIDB",
        )
    # Recompose un objet Path pour déduire le bon chmod à proposer
    denied_path = Path(str(filename))
    dataset_dir = _infer_data_directory_target(denied_path, subject=subject)
    if dataset_dir is not None:
        display_dir = _format_cli_path(dataset_dir)
        return CliErrorDiagnostic(
            summary=f"lecture du dossier {display_dir} impossible",
            action=(
                "donnez les droits d'accès au dossier "
                f"{display_dir} : `chmod a+rx {display_dir}`"
            ),
        )
    # Retombe sur le chemin exact si aucune racine dataset n'est identifiable
    display_path = _format_cli_path(denied_path)
    return CliErrorDiagnostic(
        summary=f"accès au chemin {display_path} impossible",
        action=f"vérifiez les droits d'accès sur `{display_path}`",
    )


# Construit un diagnostic pour un fichier EDF Physionet manquant
def _explain_missing_recording_error(
    payload: Mapping[str, object],
    *,
    subject_run: str | None,
) -> CliErrorDiagnostic:
    """Traduit un payload `Missing recording file` en message CLI."""

    # Cible le sujet/run lorsque la CLI les connaît déjà
    missing_target = subject_run or _format_cli_path(str(payload.get("path", "")))
    return CliErrorDiagnostic(
        summary=f"données EDF introuvables pour {missing_target}",
        action=(
            "lancez `make download_dataset` ou utilisez `--raw-dir` "
            "vers un dataset EEGMMIDB complet"
        ),
    )


# Construit un diagnostic lisible pour un payload d'échec MNE structuré
def _explain_mne_parse_failure(
    payload: Mapping[str, object],
    *,
    subject_run: str | None,
) -> CliErrorDiagnostic:
    """Traduit un payload `MNE parse failure` en message CLI concis."""

    # Récupère le chemin du fichier incriminé pour dériver l'action
    path_value = payload.get("path")
    # Prépare la cible de message lisible côté utilisateur
    target = subject_run
    if target is None and path_value is not None:
        target = _format_cli_path(str(path_value))
    if target is None:
        target = "ce run"
    # Récupère le type d'exception interne pour affiner l'action proposée
    exception_name = str(payload.get("exception", ""))
    # Récupère le message brut pour détecter certains cas sans le réafficher
    error_message = str(payload.get("message", ""))
    # Rend l'erreur de permission immédiatement actionnable
    if exception_name == "PermissionError" or "read permissions" in error_message:
        if path_value is not None:
            chmod_targets = _build_permission_targets(Path(str(path_value)))
            chmod_command = " ".join(chmod_targets)
            action = (
                "donnez les droits de lecture aux fichiers nécessaires : "
                f"`chmod a+r {chmod_command}`"
            )
        else:
            action = "vérifiez les droits de lecture du dataset EEGMMIDB"
        return CliErrorDiagnostic(
            summary=f"lecture EDF impossible pour {target}",
            action=action,
        )
    # Fournit un fallback lisible pour les autres erreurs MNE
    if path_value is not None:
        display_path = _format_cli_path(str(path_value))
        action = (
            f"vérifiez l'intégrité du fichier {display_path} puis "
            "relancez `make download_dataset` si nécessaire"
        )
    else:
        action = (
            "vérifiez l'intégrité du dataset puis relancez "
            "`make download_dataset` si nécessaire"
        )
    return CliErrorDiagnostic(
        summary=f"lecture EDF impossible pour {target}",
        action=action,
    )


# Produit un diagnostic utilisateur à partir d'une erreur Python brute
def explain_cli_error(
    error: Exception,
    *,
    subject: str | None = None,
    run: str | None = None,
) -> CliErrorDiagnostic:
    """Traduit une erreur interne en message CLI court et actionnable."""

    # Tente de décoder un éventuel payload structuré produit par le backend
    payload = _load_error_payload(error)
    # Prépare une cible stable du type `S001 R06` lorsque disponible
    subject_run = _format_subject_run(subject, run)
    # Traduit les PermissionError bruts issus de Path.exists/stat sans traceback
    if isinstance(error, PermissionError):
        return _explain_permission_error(error, subject=subject)
    # Gère explicitement les erreurs de fichier brut manquant
    if payload is not None and payload.get("error") == "Missing recording file":
        return _explain_missing_recording_error(payload, subject_run=subject_run)
    # Gère explicitement les erreurs de parsing MNE encapsulées en JSON
    if payload is not None and payload.get("error") == "MNE parse failure":
        return _explain_mne_parse_failure(payload, subject_run=subject_run)
    # Récupère le message texte simple pour les erreurs non structurées
    message = str(error).strip()
    # Compacte le message de fichier EDF manquant en une ligne plus courte
    if message.startswith("EDF introuvable pour "):
        return CliErrorDiagnostic(
            summary=f"données EDF introuvables pour {subject_run or 'ce run'}",
            action=(
                "lancez `make download_dataset` ou utilisez `--raw-dir` "
                "vers un dataset EEGMMIDB complet"
            ),
        )
    # Compacte le message de fichier événement manquant en une ligne plus courte
    if message.startswith("Fichier événement introuvable pour "):
        return CliErrorDiagnostic(
            summary=f"fichier événement introuvable pour {subject_run or 'ce run'}",
            action=(
                "lancez `make download_dataset` ou utilisez `--raw-dir` "
                "vers un dataset EEGMMIDB complet"
            ),
        )
    # Retourne le message brut si aucun diagnostic spécialisé n'est disponible
    return CliErrorDiagnostic(summary=message)


# Transforme une erreur en lignes CLI prêtes à afficher sans traceback
def render_cli_error_lines(
    error: Exception,
    *,
    subject: str | None = None,
    run: str | None = None,
) -> tuple[str, ...]:
    """Retourne des lignes de sortie CLI adaptées à une erreur utilisateur."""

    # Construit le diagnostic spécialisé à partir de l'erreur brute
    diagnostic = explain_cli_error(error, subject=subject, run=run)
    # Commence toujours par une ligne ERREUR concise
    lines = [f"INFO: {diagnostic.summary}"]
    # Ajoute une ligne d'action uniquement lorsqu'elle existe
    if diagnostic.action is not None:
        lines.append(f"Action: {diagnostic.action}")
    # Retourne un tuple immuable pour simplifier les assertions de tests
    return tuple(lines)


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
