# Importe json pour écrire des fichiers de configuration
import json

# Importe Path pour construire des chemins temporaires
from pathlib import Path

# Importe pytest pour vérifier les erreurs attendues
import pytest

# Importe le module utils pour tester la configuration des fenêtres
from tpv import utils


# Vérifie que la configuration par défaut est renvoyée
def test_default_epoch_window_config_returns_defaults() -> None:
    """Confirme la configuration par défaut des fenêtres."""

    # Charge la configuration par défaut via l'helper
    config = utils.default_epoch_window_config()
    # Vérifie que les fenêtres par défaut sont conservées
    assert config.default_windows == utils.DEFAULT_EPOCH_WINDOWS
    # Vérifie qu'aucun override n'est défini par défaut
    assert config.subject_overrides == {}


# Vérifie la lecture d'un fichier avec overrides par sujet
def test_load_epoch_window_config_reads_subject_overrides(tmp_path: Path) -> None:
    """Valide la lecture des overrides par sujet."""

    # Construit le chemin du fichier de configuration
    config_path = tmp_path / "epoch_windows.json"
    # Définit la fenêtre par défaut à écrire
    default_windows = [[0.1, 0.9]]
    # Définit l'override spécifique au sujet S001
    subject_windows = {"S001": [[0.2, 0.8]]}
    # Prépare un payload de configuration minimal
    payload = {"default": default_windows, "subjects": subject_windows}
    # Écrit le JSON de configuration sur disque
    config_path.write_text(json.dumps(payload))
    # Charge la configuration via l'API utilitaire
    config = utils.load_epoch_window_config(config_path)
    # Vérifie la fenêtre par défaut chargée
    assert config.default_windows == ((0.1, 0.9),)
    # Vérifie l'override spécifique au sujet
    assert config.subject_overrides["S001"] == ((0.2, 0.8),)
    # Vérifie la résolution d'un sujet override
    assert utils.resolve_epoch_windows("S001", config) == ((0.2, 0.8),)
    # Vérifie la résolution d'un sujet non override
    assert utils.resolve_epoch_windows("S999", config) == ((0.1, 0.9),)


# Vérifie que l'absence de fichier déclenche une erreur
def test_load_epoch_window_config_missing_file_raises(tmp_path: Path) -> None:
    """Assure qu'un fichier manquant est signalé."""

    # Pointe vers un fichier inexistant
    config_path = tmp_path / "missing.json"
    # Vérifie que l'erreur FileNotFoundError est levée
    with pytest.raises(FileNotFoundError):
        utils.load_epoch_window_config(config_path)


# Vérifie que la racine JSON invalide est refusée
def test_load_epoch_window_config_rejects_invalid_root(tmp_path: Path) -> None:
    """Confirme le rejet d'une racine JSON invalide."""

    # Construit le chemin du fichier de configuration
    config_path = tmp_path / "invalid_root.json"
    # Écrit un JSON invalide pour la racine attendue
    config_path.write_text(json.dumps(["bad"]))
    # Vérifie que l'erreur de structure est levée
    with pytest.raises(ValueError, match="objet JSON"):
        utils.load_epoch_window_config(config_path)


# Vérifie le rejet d'une liste de fenêtres vide
def test_load_epoch_window_config_rejects_empty_default(tmp_path: Path) -> None:
    """Confirme le rejet d'un default vide."""

    # Construit le chemin du fichier de configuration
    config_path = tmp_path / "empty_default.json"
    # Prépare un payload avec default vide
    payload: dict[str, list[list[float]]] = {"default": []}
    # Écrit un default vide pour déclencher l'erreur
    config_path.write_text(json.dumps(payload))
    # Vérifie que l'erreur de liste vide est levée
    with pytest.raises(ValueError, match="Liste de fenêtres vide"):
        utils.load_epoch_window_config(config_path)


# Vérifie le rejet d'un champ subjects invalide
def test_load_epoch_window_config_rejects_invalid_subjects(tmp_path: Path) -> None:
    """Confirme le rejet d'un champ subjects invalide."""

    # Construit le chemin du fichier de configuration
    config_path = tmp_path / "invalid_subjects.json"
    # Prépare un payload avec subjects invalide
    payload: dict[str, list[object]] = {"subjects": []}
    # Écrit un champ subjects invalide
    config_path.write_text(json.dumps(payload))
    # Vérifie que l'erreur sur subjects est levée
    with pytest.raises(ValueError, match="subjects"):
        utils.load_epoch_window_config(config_path)


# Vérifie le rejet d'une fenêtre incohérente
def test_load_epoch_window_config_rejects_invalid_bounds(tmp_path: Path) -> None:
    """Assure le rejet d'une fenêtre non valide."""

    # Construit le chemin du fichier de configuration
    config_path = tmp_path / "invalid_bounds.json"
    # Prépare un payload avec une fenêtre incohérente
    payload = {"default": [[1.0, 1.0]]}
    # Écrit une fenêtre aux bornes incohérentes
    config_path.write_text(json.dumps(payload))
    # Vérifie que l'erreur de fenêtre invalide est levée
    with pytest.raises(ValueError, match="Fenêtre invalide"):
        utils.load_epoch_window_config(config_path)


# Vérifie le rejet d'une fenêtre non finie
def test_parse_window_rejects_non_finite_value() -> None:
    """Valide le rejet d'une borne non finie."""

    # Prépare une fenêtre contenant NaN
    raw_window = [float("nan"), 1.0]
    # Vérifie que l'erreur est levée sur la borne non finie
    with pytest.raises(ValueError, match="Fenêtre invalide"):
        utils._parse_window(raw_window, "default[0]")
