# Importe json pour écrire des fichiers de configuration
import json

# Importe Path pour construire des chemins temporaires
from pathlib import Path

# Importe pytest pour vérifier les erreurs attendues
import pytest

# Importe le module utils pour tester la configuration des fenêtres
from tpv import utils


def test_handled_cli_error_exit_code_is_stable() -> None:
    """Verrouille le code de sortie utilisé par les wrappers Makefile."""

    assert utils.HANDLED_CLI_ERROR_EXIT_CODE == 2


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


def test_explain_cli_error_formats_permission_error_with_action(tmp_path: Path) -> None:
    """Vérifie le rendu court et actionnable des erreurs de permission EDF."""

    # Prépare un chemin absolu comme en production depuis MNE
    edf_path = tmp_path / "data" / "S001" / "S001R06.edf"
    # Construit l'erreur structurée correspondant au backend preprocessing
    error = ValueError(
        json.dumps(
            {
                "error": "MNE parse failure",
                "path": str(edf_path),
                "exception": "PermissionError",
                "message": "File does not have read permissions",
            }
        )
    )

    diagnostic = utils.explain_cli_error(error, subject="S001", run="R06")

    assert diagnostic.summary == "lecture EDF impossible pour S001 R06"
    assert diagnostic.action == (
        "donnez les droits de lecture aux fichiers nécessaires : "
        f"`chmod a+r {utils._format_cli_path(edf_path)} "
        f"{utils._format_cli_path(edf_path.with_suffix('.edf.event'))}`"
    )


def test_explain_cli_error_formats_data_directory_permission_error() -> None:
    """Déduit le dossier dataset lorsqu'un Path.exists échoue sur un enfant."""

    error = PermissionError(13, "Permission denied", "data/S001/S001R06.edf")

    diagnostic = utils.explain_cli_error(error, subject="S001", run="R06")

    assert diagnostic == utils.CliErrorDiagnostic(
        summary="lecture du dossier data/S001 impossible",
        action=(
            "donnez les droits d'accès au dossier " "data/S001 : `chmod a+rx data/S001`"
        ),
    )


def test_explain_cli_error_detects_root_data_directory_when_it_blocks_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remonte au dossier data quand c'est lui qui bloque réellement l'accès."""

    current_dir = Path.cwd()
    data_dir = tmp_path / "data"
    subject_dir = data_dir / "S001"
    subject_dir.mkdir(parents=True)
    blocked_path = subject_dir / "S001R06.edf"
    blocked_path.write_text("", encoding="utf-8")
    data_dir.chmod(0o000)
    monkeypatch.chdir(tmp_path)
    try:
        diagnostic = utils.explain_cli_error(
            PermissionError(13, "Permission denied", str(blocked_path)),
            subject="S001",
            run="R06",
        )
    finally:
        data_dir.chmod(0o755)
        monkeypatch.chdir(current_dir)

    assert diagnostic in {
        utils.CliErrorDiagnostic(
            summary="lecture du dossier data impossible",
            action="donnez les droits d'accès au dossier data : `chmod a+rx data`",
        ),
        utils.CliErrorDiagnostic(
            summary="lecture du dossier data/S001 impossible",
            action=(
                "donnez les droits d'accès au dossier "
                "data/S001 : `chmod a+rx data/S001`"
            ),
        ),
    }


def test_render_cli_error_lines_formats_missing_event_file_action() -> None:
    """Vérifie le rendu compact d'un fichier événement manquant."""

    error = FileNotFoundError(
        "Fichier événement introuvable pour S001 R06: data/S001/S001R06.edf.event. "
        "Le dataset semble incomplet: relancez `make download_dataset` "
        "ou définissez EEGMMIDB_DATA_DIR vers un dossier valide."
    )

    lines = utils.render_cli_error_lines(error, subject="S001", run="R06")

    assert lines == (
        "INFO: fichier événement introuvable pour S001 R06",
        (
            "Action: lancez `make download_dataset` ou utilisez `--raw-dir` "
            "vers un dataset EEGMMIDB complet"
        ),
    )


def test_load_error_payload_returns_none_for_non_dict_json() -> None:
    """Refuse les payloads JSON non dictionnaires."""

    error = ValueError(json.dumps(["bad"]))

    assert utils._load_error_payload(error) is None


def test_format_subject_run_returns_none_without_parts() -> None:
    """Retourne None si sujet et run sont absents."""

    assert utils._format_subject_run(None, None) is None


def test_build_permission_targets_handles_event_file_path(tmp_path: Path) -> None:
    """Inclut bien EDF et .edf.event quand le path source est un fichier event."""

    event_path = tmp_path / "data" / "S001" / "S001R06.edf.event"

    targets = utils._build_permission_targets(event_path)

    assert targets == (
        utils._format_cli_path(event_path.with_suffix("")),
        utils._format_cli_path(event_path),
    )


def test_build_permission_targets_falls_back_to_original_path(tmp_path: Path) -> None:
    """Conserve le chemin tel quel pour une extension inattendue."""

    weird_path = tmp_path / "data" / "S001" / "recording.bin"

    assert utils._build_permission_targets(weird_path) == (
        utils._format_cli_path(weird_path),
    )


def test_explain_cli_error_formats_missing_recording_without_subject_run(
    tmp_path: Path,
) -> None:
    """Utilise le chemin CLI si sujet/run ne sont pas fournis."""

    edf_path = tmp_path / "data" / "S001" / "S001R06.edf"
    error = FileNotFoundError(
        json.dumps(
            {
                "error": "Missing recording file",
                "path": str(edf_path),
            }
        )
    )

    diagnostic = utils.explain_cli_error(error)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary=f"données EDF introuvables pour {utils._format_cli_path(edf_path)}",
        action=(
            "lancez `make download_dataset` ou utilisez `--raw-dir` "
            "vers un dataset EEGMMIDB complet"
        ),
    )


def test_explain_cli_error_formats_permission_error_without_path() -> None:
    """Fournit une action générique si aucun chemin n'est disponible."""

    error = ValueError(
        json.dumps(
            {
                "error": "MNE parse failure",
                "exception": "PermissionError",
                "message": "File does not have read permissions",
            }
        )
    )

    diagnostic = utils.explain_cli_error(error)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary="lecture EDF impossible pour ce run",
        action="vérifiez les droits de lecture du dataset EEGMMIDB",
    )


def test_explain_cli_error_formats_generic_permission_error_without_filename() -> None:
    """Retourne un message dataset générique si PermissionError n'expose pas de chemin."""

    error = PermissionError(13, "Permission denied")

    diagnostic = utils.explain_cli_error(error)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary="lecture du dataset impossible",
        action="vérifiez les droits d'accès du dataset EEGMMIDB",
    )


def test_explain_cli_error_detects_subject_directory_when_it_blocks_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cible le dossier sujet quand c'est lui qui bloque la traversée."""

    current_dir = Path.cwd()
    data_dir = tmp_path / "data"
    subject_dir = data_dir / "S001"
    subject_dir.mkdir(parents=True)
    blocked_path = subject_dir / "S001R06.edf"
    blocked_path.write_text("", encoding="utf-8")
    subject_dir.chmod(0o000)
    monkeypatch.chdir(tmp_path)
    try:
        diagnostic = utils.explain_cli_error(
            PermissionError(13, "Permission denied", str(blocked_path)),
            subject="S001",
            run="R06",
        )
    finally:
        subject_dir.chmod(0o755)
        monkeypatch.chdir(current_dir)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary="lecture du dossier data/S001 impossible",
        action=(
            "donnez les droits d'accès au dossier " "data/S001 : `chmod a+rx data/S001`"
        ),
    )


def test_explain_cli_error_formats_generic_parse_failure_with_path(
    tmp_path: Path,
) -> None:
    """Propose une vérification d'intégrité pour un parse failure non permission."""

    edf_path = tmp_path / "data" / "S001" / "S001R06.edf"
    error = ValueError(
        json.dumps(
            {
                "error": "MNE parse failure",
                "path": str(edf_path),
                "exception": "RuntimeError",
                "message": "corrupted header",
            }
        )
    )

    diagnostic = utils.explain_cli_error(error)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary=f"lecture EDF impossible pour {utils._format_cli_path(edf_path)}",
        action=(
            "vérifiez l'intégrité du fichier "
            f"{utils._format_cli_path(edf_path)} puis relancez "
            "`make download_dataset` si nécessaire"
        ),
    )


def test_explain_cli_error_formats_generic_parse_failure_without_path() -> None:
    """Déclenche le fallback dataset lorsqu'aucun chemin n'est fourni."""

    error = ValueError(
        json.dumps(
            {
                "error": "MNE parse failure",
                "exception": "RuntimeError",
                "message": "corrupted header",
            }
        )
    )

    diagnostic = utils.explain_cli_error(error)

    assert diagnostic == utils.CliErrorDiagnostic(
        summary="lecture EDF impossible pour ce run",
        action=(
            "vérifiez l'intégrité du dataset puis relancez "
            "`make download_dataset` si nécessaire"
        ),
    )


def test_render_cli_error_lines_omits_action_for_generic_error() -> None:
    """N'ajoute pas de ligne Action quand aucune aide n'est disponible."""

    lines = utils.render_cli_error_lines(RuntimeError("boom"))

    assert lines == ("INFO: boom",)


def test_parse_window_rejects_non_sequence_value() -> None:
    """Refuse une fenêtre qui n'est ni liste ni tuple."""

    with pytest.raises(ValueError, match="Fenêtre invalide"):
        utils._parse_window("bad", "default[0]")


def test_parse_window_rejects_wrong_length() -> None:
    """Refuse une fenêtre qui n'a pas exactement deux bornes."""

    with pytest.raises(ValueError, match="Fenêtre invalide"):
        utils._parse_window([0.1], "default[0]")


def test_parse_window_list_rejects_non_list_value() -> None:
    """Refuse une liste de fenêtres qui n'est pas une liste JSON."""

    with pytest.raises(ValueError, match="Liste de fenêtres invalide"):
        utils._parse_window_list(("bad",), "default")
