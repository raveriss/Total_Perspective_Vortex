import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
from pytest import CaptureFixture, MonkeyPatch

from scripts import train


# Récupère une action argparse via son dest pour assertions stables
def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    # Parcourt toutes les actions déclarées dans argparse
    for action in parser._actions:  # pylint: disable=protected-access
        # Sélectionne l'action correspondant au dest demandé
        if action.dest == dest:
            # Retourne l'action trouvée pour inspection
            return action
    # Signale clairement un dest manquant
    raise AssertionError(f"Action argparse introuvable: dest={dest!r}")


def test_build_parser_description_and_help_texts_are_stable() -> None:
    # Construit le parser depuis la fonction de prod
    parser = train.build_parser()

    # Verrouille la description pour tuer description=None / variantes
    assert parser.description == "Entraîne une pipeline TPV et sauvegarde ses artefacts"

    # Récupère les actions positionnelles attendues
    subject_action = _get_action(parser, "subject")
    run_action = _get_action(parser, "run")

    # Verrouille l'aide exacte de l'argument subject
    assert subject_action.help == "Identifiant du sujet (ex: S001)"
    # Verrouille l'aide exacte de l'argument run
    assert run_action.help == "Identifiant du run (ex: R01)"

    # Récupère les actions optionnelles attendues
    classifier_action = _get_action(parser, "classifier")
    scaler_action = _get_action(parser, "scaler")
    feature_strategy_action = _get_action(parser, "feature_strategy")

    # Verrouille l'aide exacte de --classifier (tue help=None/altérations)
    assert classifier_action.help == "Classifieur final utilisé pour l'entraînement"
    # Verrouille l'aide exacte de --scaler (tue help=None/altérations)
    assert (
        scaler_action.help == "Scaler optionnel appliqué après l'extraction de features"
    )
    # Verrouille l'aide exacte de --feature-strategy (tue help=None)
    assert feature_strategy_action.help == "Méthode d'extraction de features spectrales"


def test_build_parser_sets_training_defaults_and_choices() -> None:
    parser = train.build_parser()

    classifier_action = _get_action(parser, "classifier")
    scaler_action = _get_action(parser, "scaler")
    feature_action = _get_action(parser, "feature_strategy")
    dim_action = _get_action(parser, "dim_method")
    n_components_action = _get_action(parser, "n_components")
    build_all_action = _get_action(parser, "build_all")
    train_all_action = _get_action(parser, "train_all")
    sfreq_action = _get_action(parser, "sfreq")

    assert classifier_action.choices is not None
    assert tuple(classifier_action.choices) == ("lda", "logistic", "svm", "centroid")
    assert classifier_action.default == "lda"
    assert scaler_action.choices is not None
    assert tuple(scaler_action.choices) == ("standard", "robust", "none")
    assert scaler_action.default == "none"
    assert feature_action.choices is not None
    assert tuple(feature_action.choices) == ("fft", "wavelet")
    assert feature_action.default == "fft"
    assert dim_action.choices is not None
    assert tuple(dim_action.choices) == ("pca", "csp")
    assert dim_action.default == "pca"
    assert n_components_action.default is argparse.SUPPRESS
    assert n_components_action.type is int
    assert build_all_action.default is False
    assert build_all_action.option_strings == ["--build-all"]
    assert train_all_action.default is False
    assert train_all_action.option_strings == ["--train-all"]
    assert sfreq_action.type is float
    assert sfreq_action.default == train.DEFAULT_SAMPLING_RATE
    assert sfreq_action.help == "Fréquence d'échantillonnage utilisée pour les features"


def test_build_parser_parses_defaults_and_suppresses_n_components() -> None:
    parser = train.build_parser()

    args = parser.parse_args(["S123", "R02"])

    assert args.classifier == "lda"
    assert args.scaler == "none"
    assert args.feature_strategy == "fft"
    assert args.dim_method == "pca"
    assert args.build_all is False
    assert args.train_all is False
    assert "n_components" not in vars(args)
    assert args.sfreq == train.DEFAULT_SAMPLING_RATE


def test_build_parser_help_texts_and_flags_are_stable() -> None:
    parser = train.build_parser()

    feature_action = _get_action(parser, "feature_strategy")
    dim_action = _get_action(parser, "dim_method")
    n_components_action = _get_action(parser, "n_components")
    no_norm_action = _get_action(parser, "no_normalize_features")
    data_dir_action = _get_action(parser, "data_dir")
    artifacts_dir_action = _get_action(parser, "artifacts_dir")
    raw_dir_action = _get_action(parser, "raw_dir")
    build_all_action = _get_action(parser, "build_all")
    train_all_action = _get_action(parser, "train_all")
    # Verrouille l'aide de --feature-strategy (tue help retiré / variantes)
    assert feature_action.help == "Méthode d'extraction de features spectrales"

    # Verrouille l'aide de --dim-method (tue help retiré / variantes)
    assert dim_action.help == "Méthode de réduction de dimension pour la pipeline"

    # Verrouille l'aide de --n-components (tue help retiré / variantes)
    assert (
        n_components_action.help == "Nombre de composantes conservées par le réducteur"
    )

    # Verrouille l'aide de --no-normalize-features (tue help retiré / variantes)
    assert no_norm_action.help == "Désactive la normalisation des features extraites"

    # Verrouille le comportement store_true (tue action=None / action supprimé)
    assert no_norm_action.default is False
    assert no_norm_action.const is True
    assert no_norm_action.nargs == 0

    # Verrouille le type, le défaut et l'aide de --data-dir
    assert data_dir_action.type is Path
    assert data_dir_action.default == train.DEFAULT_DATA_DIR
    assert data_dir_action.help == "Répertoire racine contenant les fichiers numpy"

    # Verrouille le type, le défaut et l'aide de --artifacts-dir
    assert artifacts_dir_action.type is Path
    assert artifacts_dir_action.default == train.DEFAULT_ARTIFACTS_DIR
    assert artifacts_dir_action.help == "Répertoire racine où enregistrer le modèle"

    # Verrouille le type, le défaut et l'aide de --raw-dir
    assert raw_dir_action.type is Path
    assert raw_dir_action.default == train.DEFAULT_RAW_DIR
    assert raw_dir_action.help == "Répertoire racine contenant les fichiers EDF bruts"

    # Verrouille l'aide de --build-all (tue help retiré / variantes)
    assert (
        build_all_action.help
        == "Génère les fichiers _X.npy/_y.npy pour tous les sujets détectés"
    )

    # Verrouille l'aide de --train-all (tue help retiré / variantes)
    assert train_all_action.help == "Entraîne tous les sujets/runs détectés dans data/"


def test_build_parser_parses_no_normalize_features_flag() -> None:
    parser = train.build_parser()

    # Le flag doit se parser sans valeur (tue action=None / action manquant)
    args = parser.parse_args(["S123", "R02", "--no-normalize-features"])

    assert args.no_normalize_features is True


def test_build_parser_applies_default_data_dir_when_missing() -> None:
    parser = train.build_parser()

    # Les défauts doivent rester ceux du module (tue default=None / supprimés)
    args = parser.parse_args(["S123", "R02"])
    assert args.data_dir == train.DEFAULT_DATA_DIR
    assert args.artifacts_dir == train.DEFAULT_ARTIFACTS_DIR
    assert args.raw_dir == train.DEFAULT_RAW_DIR


def test_load_data_does_not_log_corruption_info_on_clean_files(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    subject = "S123"
    run = "R02"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    X = np.zeros((5, 2, 8), dtype=float)
    y = np.zeros((5,), dtype=int)

    np.save(subject_dir / f"{run}_X.npy", X)
    np.save(subject_dir / f"{run}_y.npy", y)

    def unexpected_rebuild(*_args: object, **_kwargs: object):
        raise AssertionError("rebuild should not be called for clean files")

    monkeypatch.setattr(train, "_build_npy_from_edf", unexpected_rebuild)

    loaded_x, loaded_y = train._load_data(subject, run, data_dir, tmp_path / "raw")
    assert loaded_x.shape == X.shape
    assert loaded_y.shape == y.shape

    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_data_rebuilds_silently_when_one_file_is_missing(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    subject = "S123"
    run = "R02"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    y = np.zeros((5,), dtype=int)
    np.save(subject_dir / f"{run}_y.npy", y)

    called: dict[str, object] = {}

    def fake_rebuild(subject_arg: str, run_arg: str, data_dir_arg: Path, raw_dir: Path):
        called["raw_dir"] = raw_dir
        x_new = np.zeros((5, 2, 8), dtype=float)
        y_new = np.zeros((5,), dtype=int)
        np.save(subject_dir / f"{run}_X.npy", x_new)
        np.save(subject_dir / f"{run}_y.npy", y_new)
        return subject_dir / f"{run}_X.npy", subject_dir / f"{run}_y.npy"

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    loaded_x, loaded_y = train._load_data(subject, run, data_dir, tmp_path / "raw")
    assert called["raw_dir"] == tmp_path / "raw"
    # CORRECTION ICI : loaded_x au lieu de loaded_X
    assert loaded_x.shape[0] == loaded_y.shape[0]

    captured = capsys.readouterr()
    assert "Chargement numpy impossible" not in captured.out


def test_load_data_uses_mmap_mode_read_for_shape_validation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    subject = "S123"
    run = "R02"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    X = np.zeros((5, 2, 8), dtype=float)
    y = np.zeros((5,), dtype=int)
    np.save(subject_dir / f"{run}_X.npy", X)
    np.save(subject_dir / f"{run}_y.npy", y)

    def unexpected_rebuild(*_args: object, **_kwargs: object):
        raise AssertionError("rebuild should not be called for valid files")

    monkeypatch.setattr(train, "_build_npy_from_edf", unexpected_rebuild)

    real_np_load = np.load
    calls: list[tuple[str, object]] = []

    def fake_np_load(path: Path, *args, **kwargs):
        calls.append((str(path), kwargs.get("mmap_mode", "__absent__")))
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(train.np, "load", fake_np_load)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    assert len(calls) == 4
    assert calls[0][1] == "r"
    assert calls[1][1] == "r"
    assert calls[2][1] == "__absent__"
    assert calls[3][1] == "__absent__"


def test_load_data_initializes_candidate_buffers_as_none_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    monkeypatch.setattr(
        train,
        "_build_npy_from_edf",
        lambda *_a, **_k: pytest.fail("_build_npy_from_edf ne doit pas être appelé"),
    )

    observed: dict[str, object] = {}

    def tracer(frame, event, arg):  # noqa: PLR0911
        if event != "line":
            return tracer
        if "snapshot" in observed:
            return tracer
        if frame.f_globals.get("__name__") != "scripts.train":
            return tracer
        if "load_data" not in frame.f_code.co_name:
            return tracer
        if "candidate_X" not in frame.f_locals or "candidate_y" not in frame.f_locals:
            return tracer
        if "needs_rebuild" not in frame.f_locals:
            return tracer
        if "corrupted_reason" not in frame.f_locals:
            return tracer

        observed["snapshot"] = True
        observed["needs_rebuild"] = frame.f_locals["needs_rebuild"]
        observed["corrupted_reason"] = frame.f_locals["corrupted_reason"]
        observed["candidate_X"] = frame.f_locals["candidate_X"]
        observed["candidate_y"] = frame.f_locals["candidate_y"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        train._load_data(subject, run, data_dir, tmp_path / "raw")
    finally:
        sys.settrace(previous_tracer)

    assert observed["needs_rebuild"] is False
    assert observed["corrupted_reason"] is None
    assert observed["candidate_X"] is None
    assert observed["candidate_y"] is None


@pytest.mark.parametrize(
    "needs_rebuild, corrupted_reason, candidate_X, candidate_y, expected",
    [
        (
            False,
            None,
            np.zeros((2, 1, 4), dtype=float),
            np.zeros((2,), dtype=int),
            True,
        ),
        (
            True,
            None,
            np.zeros((2, 1, 4), dtype=float),
            np.zeros((2,), dtype=int),
            False,
        ),
        (
            False,
            "boom",
            np.zeros((2, 1, 4), dtype=float),
            np.zeros((2,), dtype=int),
            False,
        ),
        (
            False,
            None,
            None,
            np.zeros((2,), dtype=int),
            False,
        ),
        (
            False,
            None,
            np.zeros((2, 1, 4), dtype=float),
            None,
            False,
        ),
    ],
)
def test_should_check_shapes_requires_all_preconditions(
    needs_rebuild: bool,
    corrupted_reason: str | None,
    candidate_X: np.ndarray | None,
    candidate_y: np.ndarray | None,
    expected: bool,
) -> None:
    assert (
        train._should_check_shapes(
            needs_rebuild,
            corrupted_reason,
            candidate_X,
            candidate_y,
        )
        is expected
    )


def test_load_data_forwards_needs_rebuild_bool_to_should_check_shapes_on_clean_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    monkeypatch.setattr(
        train,
        "_build_npy_from_edf",
        lambda *_a, **_k: pytest.fail("_build_npy_from_edf ne doit pas être appelé"),
    )

    captured: dict[str, object] = {}
    real_fn = train._should_check_shapes

    def spy(needs_rebuild, corrupted_reason, candidate_X, candidate_y):
        captured["args"] = (needs_rebuild, corrupted_reason, candidate_X, candidate_y)
        return real_fn(needs_rebuild, corrupted_reason, candidate_X, candidate_y)

    monkeypatch.setattr(train, "_should_check_shapes", spy)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    assert "args" in captured
    needs_rebuild, corrupted_reason, candidate_X, candidate_y = cast(
        tuple[object, object, object, object], captured["args"]
    )
    assert needs_rebuild is False
    assert corrupted_reason is None
    assert candidate_X is not None
    assert candidate_y is not None


def test_load_data_reports_real_numpy_error_reason(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    subject = "S123"
    run = "R02"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    # Pré-crée des fichiers valides pour forcer la branche "validation mmap".
    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    # La reconstruction doit s'exécuter après l'échec numpy.
    def fake_rebuild(_subject: str, _run: str, _data_dir: Path, _raw_dir: Path):
        np.save(features_path, np.zeros((5, 2, 8), dtype=float))
        np.save(labels_path, np.zeros((5,), dtype=int))
        return features_path, labels_path

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    real_np_load = np.load
    call_count: dict[str, int] = {"mmap_x": 0}

    def fake_np_load(path: Path, *args, **kwargs):
        # Casse uniquement la 1ère tentative mmap sur X, puis laisse passer.
        if kwargs.get("mmap_mode") == "r" and path == features_path:
            call_count["mmap_x"] += 1
            if call_count["mmap_x"] == 1:
                raise ValueError("boom")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(train.np, "load", fake_np_load)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "INFO: Chargement numpy impossible pour" in output
    assert f"{subject} {run}" in output
    assert "boom" in output
    assert "XXINFO:" not in output
    assert "\nNone\n" not in output


def test_load_data_logs_features_ndim_mismatch_with_stable_prefix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    # Force le cas "ndim != EXPECTED_FEATURES_DIMENSIONS" avant rebuild.
    np.save(features_path, np.zeros((2, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    def fake_rebuild(*_args, **_kwargs):
        np.save(features_path, np.zeros((2, 1, 4), dtype=float))
        np.save(labels_path, np.zeros((2,), dtype=int))
        return features_path, labels_path

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    captured = capsys.readouterr()
    output = captured.out + captured.err

    pattern = (
        r"(?m)^INFO: X chargé depuis '.*_X\.npy' a "
        r"ndim=2 au lieu de "
        rf"{train.EXPECTED_FEATURES_DIMENSIONS}, régénération depuis l'EDF\.\.\.$"
    )
    assert re.search(pattern, output) is not None


def test_load_data_logs_labels_ndim_mismatch_with_stable_prefix_and_triggers_rebuild(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2, 1), dtype=int))

    rebuild_calls: dict[str, int] = {"count": 0}

    def fake_rebuild(*_args, **_kwargs):
        rebuild_calls["count"] += 1
        np.save(features_path, np.zeros((2, 1, 4), dtype=float))
        np.save(labels_path, np.zeros((2,), dtype=int))
        return features_path, labels_path

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    loaded_x, loaded_y = train._load_data(subject, run, data_dir, tmp_path / "raw")

    captured = capsys.readouterr()
    output = captured.out + captured.err

    pattern = (
        r"(?m)^INFO: y chargé depuis '.*_y\.npy' a "
        r"ndim=2 au lieu de 1, régénération depuis l'EDF\.\.\.$"
    )
    assert re.search(pattern, output) is not None
    assert "XXINFO:" not in output

    assert rebuild_calls["count"] == 1
    assert loaded_x.ndim == train.EXPECTED_FEATURES_DIMENSIONS
    assert loaded_y.ndim == 1


def test_load_data_keeps_needs_rebuild_boolean_when_nothing_to_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    # Cas "fichiers OK" : needs_rebuild doit rester False (pas None).
    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    monkeypatch.setattr(
        train,
        "_build_npy_from_edf",
        lambda *_a, **_k: pytest.fail("_build_npy_from_edf ne doit pas être appelé"),
    )

    observed: dict[str, object] = {}

    def tracer(frame, event, arg):
        # Ignore les wrappers mutmut : on ne capture que les frames
        # qui exposent réellement la locale 'needs_rebuild'.
        if event == "return" and "needs_rebuild" not in observed:
            if frame.f_globals.get("__name__") != "scripts.train":
                return tracer
            if "load_data" not in frame.f_code.co_name:
                return tracer
            if "needs_rebuild" not in frame.f_locals:
                return tracer
            observed["needs_rebuild"] = frame.f_locals["needs_rebuild"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        train._load_data(subject, run, data_dir, tmp_path / "raw")
    finally:
        sys.settrace(previous_tracer)

    assert "needs_rebuild" in observed
    assert observed["needs_rebuild"] is False


def test_load_data_keeps_needs_rebuild_strict_false_before_bool_coercion_on_clean_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    monkeypatch.setattr(
        train,
        "_build_npy_from_edf",
        lambda *_a, **_k: pytest.fail("_build_npy_from_edf ne doit pas être appelé"),
    )

    observed: dict[str, object] = {}

    def tracer(frame, event, arg):
        if event == "line" and "pre_coercion" not in observed:
            if frame.f_globals.get("__name__") != "scripts.train":
                return tracer
            if "load_data" not in frame.f_code.co_name:
                return tracer
            if "needs_rebuild" in frame.f_locals:
                observed["pre_coercion"] = frame.f_locals["needs_rebuild"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        train._load_data(subject, run, data_dir, tmp_path / "raw")
    finally:
        sys.settrace(previous_tracer)

    assert "pre_coercion" in observed
    assert isinstance(observed["pre_coercion"], bool)
    assert observed["pre_coercion"] is False


def test_load_data_sets_needs_rebuild_true_inside_numpy_load_except_before_corrupted_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    def fake_rebuild(_s: str, _r: str, _d: Path, _raw: Path):
        np.save(features_path, np.zeros((2, 1, 4), dtype=float))
        np.save(labels_path, np.zeros((2,), dtype=int))
        return features_path, labels_path

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    real_np_load = np.load
    broke_once: dict[str, bool] = {"done": False}

    def fake_np_load(path: Path, *args, **kwargs):
        if (
            kwargs.get("mmap_mode") == "r"
            and path == features_path
            and not broke_once["done"]
        ):
            broke_once["done"] = True
            raise ValueError("boom")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(train.np, "load", fake_np_load)

    observed: dict[str, object] = {}
    debug_steps: list[tuple[int, object, object]] = []

    def tracer(frame, event, arg):  # noqa: PLR0911
        if event != "line":
            return tracer
        if frame.f_globals.get("__name__") != "scripts.train":
            return tracer
        if "load_data" not in frame.f_code.co_name:
            return tracer

        if "error" in frame.f_locals:
            debug_steps.append(
                (
                    frame.f_lineno,
                    frame.f_locals.get("needs_rebuild"),
                    frame.f_locals.get("corrupted_reason"),
                )
            )

        if "in_except" in observed:
            return tracer

        if "error" not in frame.f_locals:
            return tracer

        if frame.f_locals.get("corrupted_reason") is not None:
            return tracer

        if frame.f_locals.get("needs_rebuild") is not True:
            return tracer

        observed["in_except"] = frame.f_locals["needs_rebuild"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        train._load_data(subject, run, data_dir, tmp_path / "raw")
    finally:
        sys.settrace(previous_tracer)

    assert (
        "in_except" in observed
    ), f"Trace (lineno, needs_rebuild, corrupted_reason)={debug_steps}"
    assert isinstance(observed["in_except"], bool), f"Trace={debug_steps}"
    assert observed["in_except"] is True, f"Trace={debug_steps}"


def test_load_data_forwards_corrupted_reason_to_should_check_shapes_on_numpy_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"

    np.save(features_path, np.zeros((2, 1, 4), dtype=float))
    np.save(labels_path, np.zeros((2,), dtype=int))

    def fake_rebuild(_s: str, _r: str, _d: Path, _raw: Path):
        np.save(features_path, np.zeros((2, 1, 4), dtype=float))
        np.save(labels_path, np.zeros((2,), dtype=int))
        return features_path, labels_path

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    real_np_load = np.load
    broke_once: dict[str, bool] = {"done": False}

    def fake_np_load(path: Path, *args, **kwargs):
        if (
            kwargs.get("mmap_mode") == "r"
            and path == features_path
            and not broke_once["done"]
        ):
            broke_once["done"] = True
            raise ValueError("boom")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(train.np, "load", fake_np_load)

    captured: dict[str, object] = {}
    real_fn = train._should_check_shapes

    def spy(needs_rebuild, corrupted_reason, candidate_X, candidate_y):
        captured["args"] = (needs_rebuild, corrupted_reason, candidate_X, candidate_y)
        return real_fn(needs_rebuild, corrupted_reason, candidate_X, candidate_y)

    monkeypatch.setattr(train, "_should_check_shapes", spy)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    assert "args" in captured
    needs_rebuild, corrupted_reason, _candidate_X, _candidate_y = cast(
        tuple[object, object, object, object], captured["args"]
    )
    assert needs_rebuild is True
    assert corrupted_reason == "boom"


def test_load_data_reports_misalignment_with_correct_shape0(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    subject = "S123"
    run = "R02"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    # Force le chemin "fichiers présents" pour atteindre le np.load(..., mmap_mode="r")
    # CORRECTION : x_old (minuscule) au lieu de X_old
    x_old = np.zeros((5, 2, 8), dtype=float)
    y_old = np.zeros((5,), dtype=int)
    np.save(subject_dir / f"{run}_X.npy", x_old)
    np.save(subject_dir / f"{run}_y.npy", y_old)

    # CORRECTION : x_bad (minuscule) au lieu de X_bad
    x_bad = np.zeros((5, 2, 8), dtype=float)
    y_bad = np.zeros((4,), dtype=int)
    np.save(subject_dir / f"{run}_X.npy", x_bad)
    np.save(subject_dir / f"{run}_y.npy", y_bad)

    rebuild_calls: list[Path] = []

    def fake_rebuild(subject_arg: str, run_arg: str, data_dir_arg: Path, raw_dir: Path):
        rebuild_calls.append(raw_dir)
        # CORRECTION : x_new (minuscule) au lieu de X_new
        x_new = np.zeros((5, 2, 8), dtype=float)
        y_new = np.zeros((5,), dtype=int)

        np.save(subject_dir / f"{run}_X.npy", x_new)
        np.save(subject_dir / f"{run}_y.npy", y_new)
        return subject_dir / f"{run}_X.npy", subject_dir / f"{run}_y.npy"

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[0].startswith("INFO: Désalignement détecté pour ")
    assert f"{subject} {run}" in out_lines[0]
    assert "X.shape[0]=5" in out_lines[0]
    assert "y.shape[0]=4" in out_lines[0]


def test_flatten_hyperparams_stringifies_nested_values() -> None:
    hyperparams = {
        "feature_strategy": "fft",
        "dimensionality": {"method": "pca", "n_components": 3},
        "bands": ["alpha", "beta"],
        "notes": "découpage",
    }

    flattened = train._flatten_hyperparams(hyperparams)

    assert list(flattened.keys()) == list(hyperparams.keys())
    for key, original_value in hyperparams.items():
        expected = json.dumps(original_value, ensure_ascii=False)
        assert flattened[key] == expected
        assert json.loads(flattened[key]) == original_value
    assert "découpage" in flattened["notes"]


def test_flatten_hyperparams_preserves_string_content_for_nested_dicts() -> None:
    hyperparams = {"dimensionality": {"method": "pca", "n_components": 2}}

    flattened = train._flatten_hyperparams(hyperparams)

    assert flattened["dimensionality"] == '{"method": "pca", "n_components": 2}'


def test_write_manifest_json_uses_indent_2_and_bool_false_ensure_ascii(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "découpage"
    data_dir.mkdir()

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    target_dir = artifacts_dir / "S01" / "R01"
    target_dir.mkdir(parents=True)

    config = train.PipelineConfig(
        sfreq=50.0,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
        n_components=None,
        classifier="lda",
        scaler=None,
    )

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        raw_dir=tmp_path / "raw",
    )

    artifacts = {
        "model": target_dir / "model.joblib",
        "scaler": None,
        "w_matrix": target_dir / "w_matrix.joblib",
    }

    captured: dict[str, object] = {}
    real_dumps = train.json.dumps

    def spy_dumps(value, *args, **kwargs):
        if (
            isinstance(value, dict)
            and "dataset" in value
            and "hyperparams" in value
            and "scores" in value
            and "artifacts" in value
        ):
            captured["ensure_ascii"] = kwargs.get("ensure_ascii", "__missing__")
            captured["indent"] = kwargs.get("indent", "__missing__")
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(train.json, "dumps", spy_dumps)

    manifest_paths = train._write_manifest(
        request,
        target_dir,
        np.array([]),
        artifacts,
    )

    assert captured["ensure_ascii"] is False
    assert isinstance(captured["ensure_ascii"], bool)
    assert captured["indent"] == 2
    assert isinstance(captured["indent"], int)

    manifest_text = manifest_paths["json"].read_text()
    assert "découpage" in manifest_text
    assert "\\u00e9" not in manifest_text


def test_write_manifest_creates_csv_named_manifest_csv_with_newline_empty_and_blank_cv_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    target_dir = artifacts_dir / "S01" / "R01"
    target_dir.mkdir(parents=True)

    config = train.PipelineConfig(
        sfreq=50.0,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
        n_components=None,
        classifier="lda",
        scaler=None,
    )

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        raw_dir=tmp_path / "raw",
    )

    artifacts = {
        "model": target_dir / "model.joblib",
        "scaler": None,
        "w_matrix": target_dir / "w_matrix.joblib",
    }

    expected_csv_path = target_dir / "manifest.csv"
    open_calls: dict[str, object] = {}
    real_open = Path.open

    def spy_open(self: Path, *args, **kwargs):
        if self == expected_csv_path and args and args[0] == "w":
            open_calls["newline"] = kwargs.get("newline", "__missing__")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    manifest_paths = train._write_manifest(
        request,
        target_dir,
        np.array([]),
        artifacts,
    )

    assert manifest_paths["csv"] == expected_csv_path
    assert expected_csv_path.exists()
    assert open_calls["newline"] == ""

    with expected_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    assert "cv_mean" in rows[0]
    assert rows[0]["cv_mean"] == ""


def test_flatten_hyperparams_passes_bool_false_to_ensure_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hyperparams = {
        "notes": "découpage",
        "emoji": "☃",
    }

    calls: list[object] = []
    real_dumps = train.json.dumps

    def spy(value, *args, **kwargs):
        calls.append(kwargs.get("ensure_ascii", "__missing__"))
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(train.json, "dumps", spy)

    _ = train._flatten_hyperparams(hyperparams)

    assert calls == [False, False]
    assert all(isinstance(item, bool) for item in calls)


def test_load_data_uses_ndarray_casts_for_validated_buffers(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subject = "S01"
    run = "R01"
    subject_dir = data_dir / subject
    subject_dir.mkdir()

    # Crée des fichiers valides pour entrer dans le bloc de validation de shapes
    np.save(subject_dir / f"{run}_X.npy", np.zeros((2, 1, 4), dtype=float))
    np.save(subject_dir / f"{run}_y.npy", np.zeros((2,), dtype=int))

    captured_types: list[object] = []

    # Intercepte train.cast pour vérifier les arguments de type
    # cast(typ, val) doit retourner val pour ne pas briser la logique
    def spy_cast(typ: object, val: object) -> object:
        captured_types.append(typ)
        return val

    monkeypatch.setattr(train, "cast", spy_cast)

    train._load_data(subject, run, data_dir, tmp_path / "raw")

    # Vérifie que cast a été appelé avec np.ndarray pour X et y
    # Si mutmut remplace par cast(None, ...), ces assertions échoueront
    assert len(captured_types) >= 2
    assert captured_types[0] is np.ndarray
    assert captured_types[1] is np.ndarray


def test_main_build_all_invokes_builder(monkeypatch, tmp_path):
    called: dict[str, tuple] = {}

    def fake_build_all(raw_dir, data_dir):
        called["args"] = (raw_dir, data_dir)

    monkeypatch.setattr(train, "_build_all_npy", fake_build_all)
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"

    exit_code = train.main(
        [
            "S001",
            "R03",
            "--build-all",
            "--raw-dir",
            str(raw_dir),
            "--data-dir",
            str(data_dir),
        ]
    )

    assert exit_code == 0
    assert called["args"] == (raw_dir, data_dir)


def test_main_train_all_delegates_and_propagates_code(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_train_all_runs(config, data_dir, artifacts_dir, raw_dir):
        captured["config"] = config
        captured["dirs"] = (data_dir, artifacts_dir, raw_dir)
        return 7

    monkeypatch.setattr(train, "_train_all_runs", fake_train_all_runs)
    exit_code = train.main(
        [
            "S010",
            "R05",
            "--train-all",
            "--classifier",
            "svm",
            "--scaler",
            "standard",
            "--feature-strategy",
            "wavelet",
            "--dim-method",
            "csp",
            "--n-components",
            "5",
            "--sfreq",
            "120",
            "--data-dir",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--raw-dir",
            str(tmp_path / "raw"),
        ]
    )

    assert exit_code == 7
    config = cast(train.PipelineConfig, captured["config"])
    assert config.classifier == "svm"
    assert config.scaler == "standard"
    assert config.feature_strategy == "wavelet"
    assert config.dim_method == "csp"
    assert config.n_components == 5

    # CORRECTION : Utilisation de pytest.approx pour comparer les flottants
    assert config.sfreq == pytest.approx(120.0)

    assert config.normalize_features is True
    assert captured["dirs"] == (
        tmp_path / "data",
        tmp_path / "artifacts",
        tmp_path / "raw",
    )


def test_main_train_all_respects_no_normalize_features_flag(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_train_all_runs(config, data_dir, artifacts_dir, raw_dir):
        captured["config"] = config
        captured["dirs"] = (data_dir, artifacts_dir, raw_dir)
        return 0

    monkeypatch.setattr(train, "_train_all_runs", fake_train_all_runs)

    exit_code = train.main(
        [
            "S010",
            "R05",
            "--train-all",
            "--no-normalize-features",
            "--data-dir",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--raw-dir",
            str(tmp_path / "raw"),
        ]
    )

    assert exit_code == 0
    config = cast(train.PipelineConfig, captured["config"])
    assert config.normalize_features is False


def test_main_passes_raw_dir_to_request_and_prints_cv_scores_with_expected_format(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured: dict[str, object] = {}

    def fake_run_training(request):
        captured["request"] = request
        return {"cv_scores": np.array([0.1, 0.2], dtype=float)}

    monkeypatch.setattr(train, "run_training", fake_run_training)

    real_array2string = train.np.array2string

    def spy_array2string(*args, **kwargs):
        separator = kwargs.get("separator", "__missing__")
        if separator == "__missing__" and len(args) >= 5:
            separator = args[4]
        assert separator == " "
        return real_array2string(*args, **kwargs)

    monkeypatch.setattr(train.np, "array2string", spy_array2string)

    raw_dir = tmp_path / "raw_custom"
    exit_code = train.main(
        [
            "S001",
            "R01",
            "--data-dir",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--raw-dir",
            str(raw_dir),
        ]
    )

    assert exit_code == 0
    request = cast(train.TrainingRequest, captured["request"])
    assert request.raw_dir == raw_dir

    stdout_lines = capsys.readouterr().out.splitlines()
    assert stdout_lines == [
        "[0.1000 0.2000]",
        "cross_val_score: 0.1500",
    ]


def test_main_falls_back_when_cv_scores_is_not_an_ndarray(monkeypatch, capsys):
    def fake_run_training(_request):
        return {"cv_scores": "not-an-array"}

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R01"])

    assert exit_code == 0
    stdout_lines = capsys.readouterr().out.splitlines()
    assert stdout_lines == [
        "[]",
        "cross_val_score: 0.0",
    ]


def test_main_falls_back_for_empty_cv_scores_array(monkeypatch, capsys):
    def fake_run_training(_request):
        return {"cv_scores": np.array([])}

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R01"])

    assert exit_code == 0
    stdout_lines = capsys.readouterr().out.splitlines()
    assert stdout_lines == [
        "[]",
        "cross_val_score: 0.0",
    ]


def test_main_prints_scores_for_singleton_cv_scores_array(monkeypatch, capsys):
    def fake_run_training(_request):
        return {"cv_scores": np.array([0.5], dtype=float)}

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R01"])

    assert exit_code == 0
    stdout_lines = capsys.readouterr().out.splitlines()
    assert stdout_lines == [
        "[0.5000]",
        "cross_val_score: 0.5000",
    ]


def test_main_returns_error_code_when_training_files_missing(monkeypatch, capsys):
    def fake_run_training(_):
        raise FileNotFoundError("données manquantes pour S001 R01")

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R01"])

    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "ERREUR: données manquantes pour S001 R01" in stdout


def test_get_git_commit_returns_unknown_when_head_ref_has_double_space_separator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_dir = tmp_path / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)

    (refs_dir / "main").write_text("deadbeef\n")
    (git_dir / "HEAD").write_text("ref:  refs/heads/main\n")

    monkeypatch.chdir(tmp_path)

    assert train._get_git_commit() == "unknown"


def test_get_git_commit_returns_unknown_when_head_ref_contains_extra_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_dir = tmp_path / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)

    (refs_dir / "main").write_text("deadbeef\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/main extra\n")

    monkeypatch.chdir(tmp_path)

    assert train._get_git_commit() == "unknown"


def test_run_training_passes_raw_dir_to_load_data_and_reports_scaler_path_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Capture les arguments transmis à _load_data pour tuer raw_dir=None
    captured: dict[str, object] = {}

    def fake_load_data(subject: str, run: str, data_dir: Path, raw_dir: Path):
        captured["args"] = (subject, run, data_dir, raw_dir)
        X = np.zeros((2, 1, 4), dtype=float)
        y = np.array([0, 1], dtype=int)
        return X, y

    # Force un chargement contrôlé sans dépendre du FS
    monkeypatch.setattr(train, "_load_data", fake_load_data)

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {"dimensionality": FakeDimReducer()}

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            return self

    # Neutralise la construction scikit-learn réelle
    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())

    # Neutralise la persistance joblib / pipeline
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)

    # Neutralise l'écriture de manifeste pour isoler run_training
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    # Prépare un request explicite avec raw_dir non défaut
    raw_dir = tmp_path / "raw_custom"
    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        raw_dir=raw_dir,
    )

    report = train.run_training(request)

    assert "args" in captured
    _subject, _run, _data_dir, observed_raw_dir = cast(
        tuple[object, object, object, object], captured["args"]
    )
    assert observed_raw_dir == raw_dir

    # Verrouille le contrat: pas de scaler => scaler_path doit rester None
    assert report["scaler_path"] is None


def test_run_training_prints_exact_warning_when_cross_validation_is_disabled(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(_subject: str, _run: str, _data_dir: Path, _raw_dir: Path):
        X = np.zeros((2, 1, 4), dtype=float)
        y = np.array([0, 1], dtype=int)
        return X, y

    monkeypatch.setattr(train, "_load_data", fake_load_data)

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {"dimensionality": FakeDimReducer()}

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            return self

    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        raw_dir=tmp_path / "raw",
    )

    train.run_training(request)

    stdout = capsys.readouterr().out
    assert (
        "AVERTISSEMENT: effectif par classe insuffisant pour la "
        "validation croisée, cross-val ignorée"
    ) in stdout


def test_run_training_builds_stratified_kfold_with_stable_random_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(_subject: str, _run: str, _data_dir: Path, _raw_dir: Path):
        X = np.zeros((6, 1, 4), dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        return X, y

    monkeypatch.setattr(train, "_load_data", fake_load_data)

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {"dimensionality": FakeDimReducer()}

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            return self

    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    captured: dict[str, object] = {}

    class SpyStratifiedKFold:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    def fake_cross_val_score(_pipeline: object, _X: object, _y: object, cv: object):
        captured["cv"] = cv
        return np.array([0.5], dtype=float)

    monkeypatch.setattr(train, "StratifiedKFold", SpyStratifiedKFold)
    monkeypatch.setattr(train, "cross_val_score", fake_cross_val_score)

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        raw_dir=tmp_path / "raw",
    )

    train.run_training(request)

    assert "kwargs" in captured
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert kwargs.get("n_splits") == train.MIN_CV_SPLITS
    assert kwargs.get("shuffle") is True
    assert "random_state" in kwargs
    assert kwargs["random_state"] == train.DEFAULT_RANDOM_STATE


def test_run_training_creates_target_dir_with_exist_ok_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(_subject: str, _run: str, _data_dir: Path, _raw_dir: Path):
        X = np.zeros((2, 1, 4), dtype=float)
        y = np.array([0, 1], dtype=int)
        return X, y

    monkeypatch.setattr(train, "_load_data", fake_load_data)

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {"dimensionality": FakeDimReducer()}

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            return self

    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    artifacts_dir = tmp_path / "artifacts_root"
    expected_target_dir = artifacts_dir / "S01" / "R01"

    real_mkdir: Callable[..., None] = Path.mkdir
    mkdir_calls: list[dict[str, object]] = []

    def spy_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == expected_target_dir:
            mkdir_calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", spy_mkdir)

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
        raw_dir=tmp_path / "raw",
    )

    train.run_training(request)

    assert len(mkdir_calls) == 1
    kwargs = cast(dict[str, object], mkdir_calls[0]["kwargs"])
    assert kwargs.get("parents") is True
    assert "exist_ok" in kwargs
    assert kwargs["exist_ok"] is True


def test_run_training_dumps_scaler_step_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(_subject: str, _run: str, _data_dir: Path, _raw_dir: Path):
        X = np.zeros((2, 1, 4), dtype=float)
        y = np.array([0, 1], dtype=int)
        return X, y

    monkeypatch.setattr(train, "_load_data", fake_load_data)

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    scaler_sentinel = object()

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {
                "dimensionality": FakeDimReducer(),
                "scaler": scaler_sentinel,
            }

        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            return self

    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    dumped: dict[str, object] = {}

    def fake_joblib_dump(obj: object, path: Path) -> None:
        dumped["obj"] = obj
        dumped["path"] = path

    monkeypatch.setattr(train.joblib, "dump", fake_joblib_dump)

    request = train.TrainingRequest(
        subject="S01",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        raw_dir=tmp_path / "raw",
    )

    report = train.run_training(request)

    assert dumped.get("obj") is scaler_sentinel
    assert str(dumped.get("path")) == str(report["scaler_path"])
