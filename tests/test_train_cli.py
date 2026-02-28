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


# Construit un contexte de génération des numpy pour les tests
def _build_npy_context(
    data_dir: Path,
    raw_dir: Path,
    eeg_reference: str,
) -> train.NpyBuildContext:
    # Construit une configuration de prétraitement par défaut
    preprocess_config = train.preprocessing.PreprocessingConfig()
    # Retourne le contexte complet pour charger/générer les numpy
    return train.NpyBuildContext(
        # Transmet le répertoire de base des numpy
        data_dir=data_dir,
        # Transmet le répertoire des EDF bruts
        raw_dir=raw_dir,
        # Transmet la référence EEG configurée
        eeg_reference=eeg_reference,
        # Transmet la configuration de prétraitement
        preprocess_config=preprocess_config,
    )


# Charge les données via la nouvelle signature tout en gardant les tests lisibles
def _load_data_with_context(
    subject: str,
    run: str,
    data_dir: Path,
    raw_dir: Path,
    eeg_reference: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Construit le contexte de génération des numpy
    build_context = _build_npy_context(
        # Transmet le répertoire de base des numpy
        data_dir,
        # Transmet le répertoire des EDF bruts
        raw_dir,
        # Transmet la référence EEG configurée
        eeg_reference,
    )
    # Délègue à l'API interne avec contexte explicite
    return train._load_data(subject, run, build_context)


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
    assert subject_action.help == "Identifiant du sujet (ex: 4)"
    # Verrouille l'aide exacte de l'argument run
    assert run_action.help == "Identifiant du run (ex: 14)"

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
    csp_reg_action = _get_action(parser, "csp_regularization")
    build_all_action = _get_action(parser, "build_all")
    train_all_action = _get_action(parser, "train_all")
    sfreq_action = _get_action(parser, "sfreq")
    grid_search_action = _get_action(parser, "grid_search")
    grid_search_splits_action = _get_action(parser, "grid_search_splits")

    assert classifier_action.choices is not None
    assert tuple(classifier_action.choices) == ("lda", "logistic", "svm", "centroid")
    assert classifier_action.default == "lda"
    assert scaler_action.choices is not None
    assert tuple(scaler_action.choices) == ("standard", "robust", "none")
    assert scaler_action.default == "none"
    assert feature_action.choices is not None
    assert tuple(feature_action.choices) == (
        "fft",
        "welch",
        "wavelet",
        "pca",
        "csp",
        "cssp",
        "svd",
    )
    assert feature_action.default == "fft"
    assert dim_action.choices is not None
    assert tuple(dim_action.choices) == ("pca", "csp", "cssp", "svd")
    assert dim_action.default == "csp"
    assert n_components_action.default is argparse.SUPPRESS
    assert n_components_action.type is int
    assert csp_reg_action.default == 0.1
    assert build_all_action.default is False
    assert build_all_action.option_strings == ["--build-all"]
    assert train_all_action.default is False
    assert train_all_action.option_strings == ["--train-all"]
    assert sfreq_action.type is float
    assert sfreq_action.default == train.DEFAULT_SAMPLING_RATE
    assert sfreq_action.help == "Fréquence d'échantillonnage utilisée pour les features"
    assert grid_search_action.default is False
    assert grid_search_action.option_strings == ["--grid-search"]
    assert grid_search_splits_action.default is None
    assert grid_search_splits_action.type is int


def test_build_parser_parses_defaults_and_suppresses_n_components() -> None:
    parser = train.build_parser()

    args = parser.parse_args(["S123", "R02"])

    assert args.classifier == "lda"
    assert args.scaler == "none"
    assert args.feature_strategy == "fft"
    assert args.dim_method == "csp"
    assert args.build_all is False
    assert args.train_all is False


def test_build_preprocess_config_from_args_accepts_valid_bandpass() -> None:
    # Construit le parser pour simuler des arguments CLI
    parser = train.build_parser()
    # Parse des arguments avec une bande passante MI personnalisée
    args = parser.parse_args(
        ["S001", "R01", "--bandpass-low", "7", "--bandpass-high", "35"]
    )
    # Construit la configuration de prétraitement via le helper
    config = train._build_preprocess_config_from_args(args)
    # Vérifie que la bande passante est bien propagée
    assert config.bandpass_band == (7.0, 35.0)


def test_build_preprocess_config_from_args_rejects_invalid_bandpass() -> None:
    # Construit le parser pour simuler des arguments CLI
    parser = train.build_parser()
    # Parse des arguments avec une bande passante inversée
    args = parser.parse_args(
        ["S001", "R01", "--bandpass-low", "30", "--bandpass-high", "8"]
    )
    # Vérifie que la validation lève une erreur explicite
    with pytest.raises(ValueError):
        train._build_preprocess_config_from_args(args)


def test_build_pipeline_config_from_args_respects_no_normalize_flag() -> None:
    # Construit le parser pour simuler des arguments CLI
    parser = train.build_parser()
    # Prépare une liste d'arguments avec sfreq explicite
    argv = ["S001", "R01", "--sfreq", "120", "--no-normalize-features"]
    # Parse les arguments afin de générer un Namespace cohérent
    args = parser.parse_args(argv)
    # Construit la configuration pipeline via le helper
    config = train._build_pipeline_config_from_args(args, argv)
    # Vérifie que la normalisation des features est désactivée
    assert config.normalize_features is False
    # Vérifie que la fréquence d'échantillonnage est propagée
    assert config.sfreq == 120.0


def test_build_training_request_from_args_propagates_preprocess_config() -> None:
    # Construit le parser pour simuler des arguments CLI
    parser = train.build_parser()
    # Prépare des arguments CLI minimaux
    argv = ["S001", "R01", "--sfreq", "120"]
    # Parse les arguments pour obtenir un Namespace cohérent
    args = parser.parse_args(argv)
    # Construit une configuration de prétraitement dédiée
    preprocess_config = train.preprocessing.PreprocessingConfig(
        normalize_method="robust"
    )
    # Construit la configuration pipeline pour la requête
    config = train._build_pipeline_config_from_args(args, argv)
    # Construit la requête d'entraînement
    request = train._build_training_request_from_args(args, config, preprocess_config)
    # Vérifie que la configuration de prétraitement est propagée
    assert request.preprocess_config.normalize_method == "robust"


def test_run_from_args_delegates_to_execute_training_request(
    monkeypatch: MonkeyPatch,
) -> None:
    # Construit le parser pour simuler des arguments CLI
    parser = train.build_parser()
    # Prépare des arguments CLI minimaux avec sfreq explicite
    argv = ["S001", "R01", "--sfreq", "120"]
    # Parse les arguments pour obtenir un Namespace cohérent
    args = parser.parse_args(argv)
    # Force _maybe_build_all à rester inactif
    monkeypatch.setattr(train, "_maybe_build_all", lambda *_args, **_kwargs: False)
    # Force _maybe_train_all à rester inactif
    monkeypatch.setattr(train, "_maybe_train_all", lambda *_args, **_kwargs: None)
    # Prépare un conteneur pour la requête transmise
    captured: dict[str, object] = {}

    # Remplace l'exécution finale pour éviter un entraînement réel
    def fake_execute(request: train.TrainingRequest) -> int:
        captured["request"] = request
        return 0

    # Injecte le stub pour capturer la requête finale
    monkeypatch.setattr(train, "_execute_training_request", fake_execute)
    # Exécute le flow CLI interne
    exit_code = train._run_from_args(args, argv)
    # Vérifie que l'exécution s'est bien terminée
    assert exit_code == 0
    # Vérifie qu'une requête d'entraînement a été transmise
    assert isinstance(captured["request"], train.TrainingRequest)
    assert args.grid_search is False
    assert args.grid_search_splits is None
    assert "n_components" not in vars(args)
    assert args.sfreq == 120.0
    assert args.csp_regularization == 0.1


def test_build_parser_help_texts_and_flags_are_stable() -> None:
    parser = train.build_parser()

    feature_action = _get_action(parser, "feature_strategy")
    dim_action = _get_action(parser, "dim_method")
    n_components_action = _get_action(parser, "n_components")
    csp_reg_action = _get_action(parser, "csp_regularization")
    no_norm_action = _get_action(parser, "no_normalize_features")
    data_dir_action = _get_action(parser, "data_dir")
    artifacts_dir_action = _get_action(parser, "artifacts_dir")
    raw_dir_action = _get_action(parser, "raw_dir")
    build_all_action = _get_action(parser, "build_all")
    train_all_action = _get_action(parser, "train_all")
    grid_search_action = _get_action(parser, "grid_search")
    grid_search_splits_action = _get_action(parser, "grid_search_splits")
    # Verrouille l'aide de --feature-strategy (tue help retiré / variantes)
    assert feature_action.help == "Méthode d'extraction de features spectrales"

    # Verrouille l'aide de --dim-method (tue help retiré / variantes)
    assert dim_action.help == "Méthode de réduction de dimension pour la pipeline"

    # Verrouille l'aide de --n-components (tue help retiré / variantes)
    assert (
        n_components_action.help == "Nombre de composantes conservées par le réducteur"
    )
    # Verrouille l'aide de --csp-regularization (tue help retiré / variantes)
    assert (
        csp_reg_action.help == "Régularisation diagonale appliquée aux covariances CSP"
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

    # Verrouille l'aide de --grid-search (tue help retiré / variantes)
    assert (
        grid_search_action.help
        == "Active une optimisation systématique des hyperparamètres"
    )

    # Verrouille l'aide de --grid-search-splits
    assert (
        grid_search_splits_action.help
        == "Nombre de splits CV dédié à la recherche d'hyperparamètres"
    )


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

    loaded_x, loaded_y = _load_data_with_context(
        subject, run, data_dir, tmp_path / "raw", "average"
    )
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

    def fake_rebuild(
        subject_arg: str,
        run_arg: str,
        build_context: train.NpyBuildContext,
    ):
        called["raw_dir"] = build_context.raw_dir
        called["reference"] = build_context.eeg_reference
        x_new = np.zeros((5, 2, 8), dtype=float)
        y_new = np.zeros((5,), dtype=int)
        np.save(subject_dir / f"{run}_X.npy", x_new)
        np.save(subject_dir / f"{run}_y.npy", y_new)
        return subject_dir / f"{run}_X.npy", subject_dir / f"{run}_y.npy"

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    loaded_x, loaded_y = _load_data_with_context(
        subject, run, data_dir, tmp_path / "raw", "average"
    )
    assert called["raw_dir"] == tmp_path / "raw"
    assert called["reference"] == "average"
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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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
        _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")
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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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
    def fake_rebuild(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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

    loaded_x, loaded_y = _load_data_with_context(
        subject, run, data_dir, tmp_path / "raw", "average"
    )

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
        _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")
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
        _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")
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

    def fake_rebuild(
        _s: str,
        _r: str,
        _build_context: train.NpyBuildContext,
    ):
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
        _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")
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

    def fake_rebuild(
        _s: str,
        _r: str,
        _build_context: train.NpyBuildContext,
    ):
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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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

    rebuild_calls: list[tuple[Path, str | None]] = []

    def fake_rebuild(
        subject_arg: str,
        run_arg: str,
        build_context: train.NpyBuildContext,
    ):
        rebuild_calls.append((build_context.raw_dir, build_context.eeg_reference))
        # CORRECTION : x_new (minuscule) au lieu de X_new
        x_new = np.zeros((5, 2, 8), dtype=float)
        y_new = np.zeros((5,), dtype=int)

        np.save(subject_dir / f"{run}_X.npy", x_new)
        np.save(subject_dir / f"{run}_y.npy", y_new)
        return subject_dir / f"{run}_X.npy", subject_dir / f"{run}_y.npy"

    monkeypatch.setattr(train, "_build_npy_from_edf", fake_rebuild)

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

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


def test_build_npy_from_edf_applies_notch_and_normalization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    # Définit un sujet/run pour construire un contexte minimal
    subject = "S001"
    # Définit un run pour générer les chemins EDF
    run = "R01"
    # Prépare le répertoire data pour écrire les .npy
    data_dir = tmp_path / "data"
    # Prépare le répertoire raw pour simuler l'EDF
    raw_dir = tmp_path / "raw"
    # Construit le chemin EDF attendu par la fonction
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Crée l'arborescence de l'EDF factice
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un fichier EDF factice pour passer le check d'existence
    raw_path.write_text("stub")
    # Écrit un fichier .edf.event factice pour le contrôle d'intégrité
    raw_path.with_suffix(".edf.event").write_text("stub")

    # Prépare une configuration de prétraitement avec notch et z-score
    preprocess_config = train.preprocessing.PreprocessingConfig(
        bandpass_band=(8.0, 30.0),
        notch_freq=50.0,
        normalize_method="zscore",
        normalize_epsilon=1e-8,
    )
    # Construit le contexte nécessaire à _build_npy_from_edf
    build_context = train.NpyBuildContext(
        data_dir=data_dir,
        raw_dir=raw_dir,
        eeg_reference="average",
        preprocess_config=preprocess_config,
    )

    # Prépare des epochs factices pour la sélection de fenêtre
    epochs_data = np.arange(24, dtype=float).reshape(2, 3, 4)
    # Prépare des labels factices alignés sur les epochs
    labels = np.array([0, 1])

    # Neutralise le chargement EDF pour éviter une dépendance MNE
    monkeypatch.setattr(
        train.preprocessing,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (object(), {}),
    )
    # Neutralise le notch pour éviter un filtrage réel
    monkeypatch.setattr(
        train.preprocessing, "apply_notch_filter", lambda raw, **_kwargs: raw
    )
    # Neutralise le filtre passe-bande pour éviter un filtrage réel
    monkeypatch.setattr(
        train.preprocessing, "apply_bandpass_filter", lambda raw, **_kwargs: raw
    )
    # Fournit un mapping d'événements minimal
    monkeypatch.setattr(
        train.preprocessing,
        "map_events_to_motor_labels",
        lambda *_args, **_kwargs: (np.zeros((2, 3), dtype=int), {"A": 1}, ["A", "B"]),
    )
    # Neutralise la résolution des fenêtres pour isoler la sélection
    monkeypatch.setattr(
        train, "resolve_epoch_windows", lambda *_args, **_kwargs: [(0.0, 1.0)]
    )
    # Neutralise la sélection de fenêtre pour injecter des epochs factices
    monkeypatch.setattr(
        train,
        "_select_best_epoch_window",
        lambda *_args, **_kwargs: ((0.0, 1.0), epochs_data, labels),
    )

    # Lance la construction des .npy
    features_path, labels_path = train._build_npy_from_edf(subject, run, build_context)
    # Charge les features générés pour vérifier la normalisation
    saved_features = np.load(features_path)
    # Vérifie que la normalisation z-score centre les données
    assert np.allclose(np.mean(saved_features, axis=2), 0.0)
    # Vérifie que les labels sont bien persistés
    assert np.array_equal(np.load(labels_path), labels)


def test_build_npy_from_edf_skips_notch_and_normalization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    # Définit un sujet/run pour construire un contexte minimal
    subject = "S002"
    # Définit un run pour générer les chemins EDF
    run = "R02"
    # Prépare le répertoire data pour écrire les .npy
    data_dir = tmp_path / "data"
    # Prépare le répertoire raw pour simuler l'EDF
    raw_dir = tmp_path / "raw"
    # Construit le chemin EDF attendu par la fonction
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Crée l'arborescence de l'EDF factice
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un fichier EDF factice pour passer le check d'existence
    raw_path.write_text("stub")
    # Écrit un fichier .edf.event factice pour le contrôle d'intégrité
    raw_path.with_suffix(".edf.event").write_text("stub")

    # Prépare une configuration de prétraitement sans notch ni normalisation
    preprocess_config = train.preprocessing.PreprocessingConfig(
        bandpass_band=(8.0, 30.0),
        notch_freq=0.0,
        normalize_method="none",
        normalize_epsilon=1e-8,
    )
    # Construit le contexte nécessaire à _build_npy_from_edf
    build_context = train.NpyBuildContext(
        data_dir=data_dir,
        raw_dir=raw_dir,
        eeg_reference="average",
        preprocess_config=preprocess_config,
    )

    # Prépare des epochs factices pour la sélection de fenêtre
    epochs_data = np.ones((2, 2, 3), dtype=float)
    # Prépare des labels factices alignés sur les epochs
    labels = np.array([0, 1])

    # Neutralise le chargement EDF pour éviter une dépendance MNE
    monkeypatch.setattr(
        train.preprocessing,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (object(), {}),
    )
    # Neutralise le filtre passe-bande pour éviter un filtrage réel
    monkeypatch.setattr(
        train.preprocessing, "apply_bandpass_filter", lambda raw, **_kwargs: raw
    )
    # Fournit un mapping d'événements minimal
    monkeypatch.setattr(
        train.preprocessing,
        "map_events_to_motor_labels",
        lambda *_args, **_kwargs: (np.zeros((2, 3), dtype=int), {"A": 1}, ["A", "B"]),
    )
    # Neutralise la résolution des fenêtres pour isoler la sélection
    monkeypatch.setattr(
        train, "resolve_epoch_windows", lambda *_args, **_kwargs: [(0.0, 1.0)]
    )
    # Neutralise la sélection de fenêtre pour injecter des epochs factices
    monkeypatch.setattr(
        train,
        "_select_best_epoch_window",
        lambda *_args, **_kwargs: ((0.0, 1.0), epochs_data, labels),
    )

    # Lance la construction des .npy
    features_path, labels_path = train._build_npy_from_edf(subject, run, build_context)
    # Vérifie que les features brutes sont conservées
    assert np.array_equal(np.load(features_path), epochs_data)
    # Vérifie que les labels sont bien persistés
    assert np.array_equal(np.load(labels_path), labels)


def test_build_all_npy_calls_builder_for_edf_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    # Définit un sujet et un run pour simuler un EDF
    subject = "S010"
    # Définit un run pour construire un nom EDF valide
    run = "R03"
    # Prépare le répertoire raw contenant l'EDF
    raw_dir = tmp_path / "raw"
    # Prépare le répertoire data pour le contexte
    data_dir = tmp_path / "data"
    # Construit le chemin EDF attendu par _build_all_npy
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Crée l'arborescence et le fichier EDF factice
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu factice pour l'EDF
    raw_path.write_text("stub")

    # Prépare une configuration de prétraitement par défaut
    preprocess_config = train.preprocessing.PreprocessingConfig()
    # Construit le contexte nécessaire à _build_all_npy
    build_context = train.NpyBuildContext(
        data_dir=data_dir,
        raw_dir=raw_dir,
        eeg_reference="average",
        preprocess_config=preprocess_config,
    )

    # Trace les appels faits à _build_npy_from_edf
    calls: list[tuple[str, str]] = []

    # Remplace _build_npy_from_edf pour tracer les appels
    def fake_build_npy(
        subject_arg: str,
        run_arg: str,
        _ctx: train.NpyBuildContext,
    ) -> tuple[Path, Path]:
        calls.append((subject_arg, run_arg))
        return Path("X.npy"), Path("y.npy")

    # Injecte le stub pour éviter un traitement EDF réel
    monkeypatch.setattr(train, "_build_npy_from_edf", fake_build_npy)

    # Exécute la génération pour détecter l'appel
    train._build_all_npy(build_context)
    # Vérifie que le run EDF a bien été traité
    assert calls == [(subject, run)]


def test_build_all_npy_skips_runs_without_events(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    # Définit un sujet et un run pour simuler un EDF
    subject = "S011"
    # Définit un run pour construire un nom EDF valide
    run = "R04"
    # Prépare le répertoire raw contenant l'EDF
    raw_dir = tmp_path / "raw"
    # Prépare le répertoire data pour le contexte
    data_dir = tmp_path / "data"
    # Construit le chemin EDF attendu par _build_all_npy
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Crée l'arborescence et le fichier EDF factice
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu factice pour l'EDF
    raw_path.write_text("stub")

    # Prépare une configuration de prétraitement par défaut
    preprocess_config = train.preprocessing.PreprocessingConfig()
    # Construit le contexte nécessaire à _build_all_npy
    build_context = train.NpyBuildContext(
        data_dir=data_dir,
        raw_dir=raw_dir,
        eeg_reference="average",
        preprocess_config=preprocess_config,
    )

    # Remplace _build_npy_from_edf pour simuler l'absence d'événements
    monkeypatch.setattr(
        train,
        "_build_npy_from_edf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("No motor events present")
        ),
    )

    # Exécute la génération pour déclencher la branche de skip
    train._build_all_npy(build_context)
    # Capture la sortie standard pour vérifier le log
    captured = capsys.readouterr().out
    # Vérifie que le message d'information est bien émis
    assert "Événements moteurs absents" in captured


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

    _load_data_with_context(subject, run, data_dir, tmp_path / "raw", "average")

    # Vérifie que cast a été appelé avec np.ndarray pour X et y
    # Si mutmut remplace par cast(None, ...), ces assertions échoueront
    assert len(captured_types) >= 2
    assert captured_types[0] is np.ndarray
    assert captured_types[1] is np.ndarray


def test_main_build_all_invokes_builder(monkeypatch, tmp_path):
    called: dict[str, tuple] = {}

    def fake_build_all(build_context: train.NpyBuildContext):
        called["args"] = (
            build_context.raw_dir,
            build_context.data_dir,
            build_context.eeg_reference,
        )

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
    assert called["args"] == (raw_dir, data_dir, "average")


def test_main_reports_invalid_bandpass(capsys: CaptureFixture[str]) -> None:
    # Exécute la CLI avec une bande passante invalide
    exit_code = train.main(
        ["S001", "R01", "--bandpass-low", "30", "--bandpass-high", "8"]
    )
    # Capture la sortie standard pour vérifier le message
    captured = capsys.readouterr().out
    # Vérifie que la CLI signale une erreur
    assert exit_code == train.HANDLED_CLI_ERROR_EXIT_CODE
    # Vérifie que le message d'erreur mentionne la bande passante
    assert "bandpass_low" in captured


def test_execute_training_request_reports_cv_status(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    # Prépare un run_training factice pour simuler une CV indisponible
    def fake_run_training(_request: train.TrainingRequest) -> dict:
        return {
            "cv_scores": np.array([]),
            "cv_splits_requested": 4,
            "cv_unavailability_reason": "no cv",
            "cv_error": "boom",
        }

    # Injecte le stub de run_training pour isoler l'affichage CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)

    # Construit une requête minimale pour l'exécution de la CLI
    request = train.TrainingRequest(
        subject="S010",
        run="R05",
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
    )

    # Exécute l'affichage CLI et capture la sortie
    exit_code = train._execute_training_request(request)
    # Capture les sorties pour inspection
    captured = capsys.readouterr().out
    # Vérifie que l'exécution se termine avec succès
    assert exit_code == 0
    # Vérifie que l'erreur CV est signalée
    assert "cross_val_score échoué" in captured
    # Vérifie que l'indisponibilité de CV est signalée
    assert "CV indisponible" in captured


def test_main_train_all_delegates_and_propagates_code(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_train_all_runs(resources: train.TrainingResources):
        captured["config"] = resources.pipeline_config
        captured["dirs"] = (
            resources.data_dir,
            resources.artifacts_dir,
            resources.raw_dir,
            resources.eeg_reference,
        )
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
        "average",
    )


def test_main_train_all_respects_no_normalize_features_flag(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_train_all_runs(resources: train.TrainingResources):
        captured["config"] = resources.pipeline_config
        captured["dirs"] = (
            resources.data_dir,
            resources.artifacts_dir,
            resources.raw_dir,
            resources.eeg_reference,
        )
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
    # Prépare un conteneur pour capturer la requête
    captured: dict[str, object] = {}

    # Définit un run_training factice pour injecter des scores CV
    def fake_run_training(request):
        # Capture la requête afin de vérifier les options
        captured["request"] = request
        # Retourne un tableau de scores pour l'affichage CLI
        return {"cv_scores": np.array([0.1, 0.2], dtype=float)}

    # Remplace run_training pour isoler la logique CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)

    # Sauvegarde array2string pour restaurer l'appel réel
    real_array2string = train.np.array2string

    # Définit un spy pour contrôler le séparateur choisi
    def spy_array2string(*args, **kwargs):
        # Récupère le séparateur via kwargs si présent
        separator = kwargs.get("separator", "__missing__")
        # Récupère le séparateur positionnel si absent
        if separator == "__missing__" and len(args) >= 5:
            # Extrait le séparateur depuis les arguments positionnels
            separator = args[4]
        # Vérifie que l'espace est bien utilisé comme séparateur
        assert separator == " "
        # Délègue à la fonction réelle pour formater
        return real_array2string(*args, **kwargs)

    # Remplace array2string par le spy pour ce test
    monkeypatch.setattr(train.np, "array2string", spy_array2string)

    # Définit un répertoire raw explicite pour la requête
    raw_dir = tmp_path / "raw_custom"
    # Prépare les arguments CLI pour l'appel train
    args = [
        # Fournit le sujet attendu par le parser
        "S001",
        # Fournit le run attendu par le parser
        "R01",
        # Active l'option data-dir
        "--data-dir",
        # Fournit le chemin data dédié au test
        str(tmp_path / "data"),
        # Active l'option artifacts-dir
        "--artifacts-dir",
        # Fournit le chemin artifacts dédié au test
        str(tmp_path / "artifacts"),
        # Active l'option raw-dir
        "--raw-dir",
        # Fournit le chemin raw dédié au test
        str(raw_dir),
    ]
    # Exécute la CLI avec les arguments préparés
    exit_code = train.main(args)

    # Vérifie un code de sortie nominal
    assert exit_code == 0
    # Récupère la requête capturée pour inspection
    request = cast(train.TrainingRequest, captured["request"])
    # Vérifie que le répertoire raw est bien propagé
    assert request.raw_dir == raw_dir

    # Capture la sortie standard pour l'analyse
    stdout_lines = capsys.readouterr().out.splitlines()
    # Prépare la sortie attendue avec CV_SPLITS et scores
    expected = [
        # Vérifie la ligne des splits CV affichés
        "CV_SPLITS: 10 (scores: 2)",
        # Vérifie l'affichage des scores formatés
        "[0.1000 0.2000]",
        # Vérifie l'affichage du score moyen
        "cross_val_score: 0.1500",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


def test_main_falls_back_when_cv_scores_is_not_an_ndarray(monkeypatch, capsys):
    # Définit un run_training factice renvoyant un type inattendu
    def fake_run_training(_request):
        # Retourne une valeur non-numpy pour déclencher le fallback
        return {"cv_scores": "not-an-array"}

    # Remplace run_training pour isoler la sortie CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec les arguments minimaux
    exit_code = train.main(["S001", "R01"])

    # Vérifie un code de sortie nominal malgré le fallback
    assert exit_code == 0
    # Capture la sortie standard pour inspection
    stdout_lines = capsys.readouterr().out.splitlines()
    # Définit la sortie attendue pour le fallback
    expected = [
        # Vérifie l'affichage systématique des splits
        "CV_SPLITS: 10 (scores: 0)",
        # Vérifie le tableau vide attendu
        "[]",
        # Vérifie le score moyen nul affiché
        "cross_val_score: 0.0",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


def test_main_falls_back_for_empty_cv_scores_array(monkeypatch, capsys):
    # Définit un run_training factice avec un tableau vide
    def fake_run_training(_request):
        # Retourne un tableau vide pour simuler l'absence de CV
        return {"cv_scores": np.array([])}

    # Remplace run_training pour contrôler la sortie CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec les arguments minimaux
    exit_code = train.main(["S001", "R01"])

    # Vérifie un code de sortie nominal malgré l'absence de scores
    assert exit_code == 0
    # Capture la sortie standard pour validation
    stdout_lines = capsys.readouterr().out.splitlines()
    # Définit la sortie attendue en absence de scores
    expected = [
        # Vérifie l'affichage systématique des splits
        "CV_SPLITS: 10 (scores: 0)",
        # Vérifie le tableau vide attendu
        "[]",
        # Vérifie le score moyen nul affiché
        "cross_val_score: 0.0",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


# Vérifie le maintien de dim_method pour wavelet
def test_main_keeps_dim_method_for_wavelet(monkeypatch, capsys):
    """Vérifie que CSP reste actif quand wavelet est demandé."""

    # Capture la requête d'entraînement pour inspecter la config finale
    captured: dict[str, object] = {}

    # Définit un double pour éviter un entraînement réel
    def fake_run_training(request):
        # Mémorise la requête pour assertions ultérieures
        captured["request"] = request
        # Retourne un rapport minimal pour la CLI
        return {"cv_scores": np.array([])}

    # Injecte le double dans le module train
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec wavelet sans --dim-method explicite
    exit_code = train.main(["S001", "R01", "--feature-strategy", "wavelet"])

    # Vérifie que l'exécution est nominale
    assert exit_code == 0
    # Récupère la requête typée pour inspection
    request = cast(train.TrainingRequest, captured["request"])
    # Vérifie que la stratégie wavelet est conservée
    assert request.pipeline_config.feature_strategy == "wavelet"
    # Vérifie que CSP reste actif pour la pipeline wavelet
    assert request.pipeline_config.dim_method == "csp"

    # Capture la sortie CLI pour vérifier le message d'info
    stdout_lines = capsys.readouterr().out.splitlines()
    # Vérifie que le message d'information est affiché
    assert stdout_lines[0] == (
        "INFO: dim_method='csp/cssp' appliqué avant " "l'extraction des features."
    )


def test_main_prints_scores_for_singleton_cv_scores_array(monkeypatch, capsys):
    # Définit un run_training factice avec un score unique
    def fake_run_training(_request):
        # Retourne un score unique pour vérifier l'affichage
        return {"cv_scores": np.array([0.5], dtype=float)}

    # Remplace run_training pour isoler l'output CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec les arguments minimaux
    exit_code = train.main(["S001", "R01"])

    # Vérifie un code de sortie nominal
    assert exit_code == 0
    # Capture la sortie standard pour inspection
    stdout_lines = capsys.readouterr().out.splitlines()
    # Définit la sortie attendue pour un seul score
    expected = [
        # Vérifie l'affichage des splits demandés
        "CV_SPLITS: 10 (scores: 1)",
        # Vérifie l'affichage du score unique
        "[0.5000]",
        # Vérifie la moyenne correspondant au score unique
        "cross_val_score: 0.5000",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


def test_main_returns_error_code_when_training_files_missing(monkeypatch, capsys):
    def fake_run_training(_):
        raise FileNotFoundError("données manquantes pour S001 R01")

    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    exit_code = train.main(["S001", "R01"])

    stdout = capsys.readouterr().out
    assert exit_code == train.HANDLED_CLI_ERROR_EXIT_CODE
    assert "INFO: données manquantes pour S001 R01" in stdout


def test_main_reports_permission_error_with_action(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    # Fige la résolution sfreq pour éviter un bruit de sortie parasite
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 160.0)
    # Prépare un payload identique à celui produit par preprocessing
    error_payload = json.dumps(
        {
            "error": "MNE parse failure",
            "path": str(tmp_path / "data" / "S001" / "S001R06.edf"),
            "exception": "PermissionError",
            "message": "File does not have read permissions",
        }
    )

    def fake_run_training(_request: train.TrainingRequest) -> dict[str, object]:
        raise ValueError(error_payload)

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R06"])

    stdout_lines = capsys.readouterr().out.splitlines()
    assert exit_code == train.HANDLED_CLI_ERROR_EXIT_CODE
    assert stdout_lines == [
        "INFO: lecture EDF impossible pour S001 R06",
        (
            "Action: donnez les droits de lecture aux fichiers nécessaires : "
            "`chmod a+r "
            + f"{tmp_path / 'data' / 'S001' / 'S001R06.edf'} "
            + f"{tmp_path / 'data' / 'S001' / 'S001R06.edf.event'}`"
        ),
    ]


def test_main_reports_data_directory_permission_error(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Affiche une action concise quand le dossier sujet est illisible."""

    def fake_resolve_sampling_rate(*_args: object, **_kwargs: object) -> float:
        raise PermissionError(13, "Permission denied", "data/S001/S001R06.edf")

    monkeypatch.setattr(train, "resolve_sampling_rate", fake_resolve_sampling_rate)

    exit_code = train.main(["S001", "R06"])

    stdout_lines = capsys.readouterr().out.splitlines()
    assert exit_code == train.HANDLED_CLI_ERROR_EXIT_CODE
    assert stdout_lines == [
        "INFO: lecture du dossier data/S001 impossible",
        (
            "Action: donnez les droits d'accès au dossier "
            "data/S001 : `chmod a+rx data/S001`"
        ),
    ]


def test_resolve_sampling_rate_reports_concise_info_on_parse_error(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    # Prépare le couple sujet/run ciblé par la résolution de sfreq
    subject = "S001"
    run = "R06"
    # Prépare le chemin EDF attendu par la logique de résolution
    raw_dir = tmp_path / "data"
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("", encoding="utf-8")

    def fake_loader(
        *_args: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        raise ValueError(
            json.dumps(
                {
                    "error": "MNE parse failure",
                    "path": str(raw_path),
                    "exception": "PermissionError",
                    "message": "File does not have read permissions",
                }
            )
        )

    monkeypatch.setattr(train.preprocessing, "load_physionet_raw", fake_loader)

    resolved = train.resolve_sampling_rate(
        subject,
        run,
        raw_dir,
        train.DEFAULT_SAMPLING_RATE,
        "average",
    )

    stdout = capsys.readouterr().out.strip()
    assert resolved == train.DEFAULT_SAMPLING_RATE
    assert stdout == ""


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

    def fake_load_data(
        subject: str,
        run: str,
        build_context: train.NpyBuildContext,
    ):
        captured["args"] = (
            subject,
            run,
            build_context.data_dir,
            build_context.raw_dir,
            build_context.eeg_reference,
        )
        X = np.zeros((2, 1, 4), dtype=float)
        y = np.array([0, 0], dtype=int)
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
    _subject, _run, _data_dir, observed_raw_dir, observed_reference = cast(
        tuple[object, object, object, object, object], captured["args"]
    )
    assert observed_raw_dir == raw_dir
    assert observed_reference == "average"

    # Verrouille le contrat: pas de scaler => scaler_path doit rester None
    assert report["scaler_path"] is None


def test_run_training_persists_w_matrix_when_spatial_filters_are_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Définit un loader minimal pour fournir des données synthétiques
    def fake_load_data(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
        # Prépare un tenseur d'essais minimal compatible avec CSP
        X = np.zeros((2, 2, 4), dtype=float)
        # Prépare deux labels pour satisfaire CSP/CSSP
        y = np.array([0, 1], dtype=int)
        # Retourne les données synthétiques au pipeline
        return X, y

    # Injecte le loader minimal pour isoler run_training
    monkeypatch.setattr(train, "_load_data", fake_load_data)

    # Définit un faux CSP exposant une matrice W persistable
    class FakeSpatialFilters:
        # Initialise la matrice W attendue par la persistance
        def __init__(self) -> None:
            # Fixe une matrice identité pour simplifier la validation
            self.w_matrix = np.eye(2)
            # Fixe des valeurs propres factices pour enrichir l'artefact
            self.eigenvalues_ = np.array([1.0, 0.5])

    # Prépare une pipeline factice avec uniquement des filtres spatiaux
    class FakePipeline:
        # Construit la pipeline factice pour run_training
        def __init__(self, spatial_filters: FakeSpatialFilters) -> None:
            # Injecte les filtres spatiaux à la place de dimensionality
            self.named_steps = {"spatial_filters": spatial_filters}

        # Expose une signature fit compatible scikit-learn
        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            # Retourne self pour simuler l'entraînement
            return self

    # Instancie les filtres factices pour la pipeline
    spatial_filters = FakeSpatialFilters()
    # Instancie la pipeline factice pour le flux d'entraînement
    fake_pipeline = FakePipeline(spatial_filters)

    # Neutralise la construction scikit-learn réelle
    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: fake_pipeline)
    # Neutralise la persistance complète de la pipeline
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)

    # Simule la CV pour éviter de lancer cross_val_score
    def fake_train_with_optional_cv(
        _request: train.TrainingRequest,
        _X: np.ndarray,
        _y: np.ndarray,
        _pipeline: train.Pipeline,
        _adapted_config: train.PipelineConfig,
    ):
        # Retourne la pipeline factice sans scores CV
        return np.array([]), fake_pipeline, None, None, None

    # Injecte le stub CV pour isoler run_training
    monkeypatch.setattr(train, "_train_with_optional_cv", fake_train_with_optional_cv)

    # Neutralise l'écriture de manifeste pour isoler run_training
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    # Prépare une requête d'entraînement minimale
    request = train.TrainingRequest(
        # Fixe un sujet minimal pour la structure d'artefacts
        subject="S01",
        # Fixe un run minimal pour la structure d'artefacts
        run="R01",
        # Configure la pipeline pour CSP afin d'activer spatial_filters
        pipeline_config=train.PipelineConfig(
            # Fixe la fréquence pour la cohérence des tests
            sfreq=50.0,
            # Sélectionne Welch pour reproduire un flux CSP réaliste
            feature_strategy="welch",
            # Active la normalisation pour refléter la config standard
            normalize_features=True,
            # Force la méthode CSP pour valider la branche spatial_filters
            dim_method="csp",
            # Utilise un nombre de composantes explicite
            n_components=2,
            # Conserve LDA comme classifieur stable
            classifier="lda",
            # Laisse le scaler désactivé pour isoler la persistance W
            scaler=None,
        ),
        # Fixe le répertoire data pour les numpy
        data_dir=tmp_path / "data",
        # Fixe le répertoire d'artefacts pour la sauvegarde
        artifacts_dir=tmp_path / "artifacts",
        # Fixe le répertoire raw pour le contrat de build
        raw_dir=tmp_path / "raw",
    )

    # Lance l'entraînement pour persister la matrice W
    report = train.run_training(request)

    # Construit le chemin attendu du fichier W sauvegardé
    expected_path = tmp_path / "artifacts" / "S01" / "R01" / "w_matrix.joblib"
    # Vérifie que le chemin retourné correspond à l'artefact attendu
    assert report["w_matrix_path"] == expected_path
    # Vérifie que la matrice W a bien été persistée sur disque
    assert expected_path.exists()
    # Recharge l'artefact pour valider le contenu W sauvegardé
    payload = train.joblib.load(expected_path)
    # Vérifie que la matrice W sauvegardée correspond au filtre factice
    assert np.allclose(payload["w_matrix"], spatial_filters.w_matrix)


def test_run_training_prints_exact_warning_when_cross_validation_is_disabled(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
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
    # Vérifie le message d'info lorsque la CV est désactivée
    assert (
        "INFO: validation croisée indisponible, " "entraînement direct sans cross-val"
    ) in stdout


# Vérifie que le splitter stratifié conserve une seed stable
def test_run_training_builds_stratified_shuffle_split_with_stable_random_state(
    # Reçoit le chemin temporaire pour simuler les artefacts
    tmp_path: Path,
    # Reçoit le monkeypatch pour injecter les doubles
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Définit un loader factice pour maîtriser les labels
    def fake_load_data(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
        # Prépare un tenseur minimal compatible pipeline
        X = np.zeros((6, 1, 4), dtype=float)
        # Déclare deux classes équilibrées pour la CV
        y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        # Retourne les données factices pour l'entraînement
        return X, y

    # Injecte le loader factice dans le module train
    monkeypatch.setattr(train, "_load_data", fake_load_data)

    # Déclare un réducteur factice pour respecter l'API save
    class FakeDimReducer:
        # Fournit un save no-op pour éviter l'I/O réelle
        def save(self, _path: Path) -> None:
            # Retourne None pour garder une API stable
            return None

    # Déclare une pipeline factice pour contrôler l'appel fit
    class FakePipeline:
        # Construit la pipeline factice avec l'étape requise
        def __init__(self) -> None:
            # Expose l'étape de réduction attendue par run_training
            self.named_steps = {"dimensionality": FakeDimReducer()}

        # Simule l'entraînement en renvoyant self
        def fit(self, _X: np.ndarray, _y: np.ndarray) -> "FakePipeline":
            # Retourne l'instance pour coller au contrat sklearn
            return self

    # Injecte la pipeline factice pour éviter un vrai fit
    monkeypatch.setattr(train, "build_pipeline", lambda _cfg: FakePipeline())
    # Neutralise la sauvegarde du modèle pour accélérer le test
    monkeypatch.setattr(train, "save_pipeline", lambda _p, _path: None)
    # Force un manifeste factice pour ne pas écrire sur disque
    monkeypatch.setattr(
        train,
        "_write_manifest",
        lambda *_args, **_kwargs: {
            "json": tmp_path / "m.json",
            "csv": tmp_path / "m.csv",
        },
    )

    # Prépare un dictionnaire pour capturer l'instanciation CV
    captured: dict[str, object] = {}

    # Déclare un spy pour capturer les arguments du splitter
    class SpyStratifiedShuffleSplit:
        # Capture args/kwargs pour assertions ultérieures
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Mémorise les args pour inspection
            captured["args"] = args
            # Mémorise les kwargs pour inspection
            captured["kwargs"] = kwargs

    # Déclare une version contrôlée de cross_val_score
    def fake_cross_val_score(
        _pipeline: object,
        _X: object,
        _y: object,
        cv: object,
        **_kwargs: object,
    ):
        # Capture l'objet CV construit par run_training
        captured["cv"] = cv
        # Retourne un score fixe pour stabiliser le test
        return np.array([0.5], dtype=float)

    # Injecte le spy sur le splitter stratifié shuffle
    monkeypatch.setattr(train, "StratifiedShuffleSplit", SpyStratifiedShuffleSplit)
    # Injecte la fonction cross_val_score factice
    monkeypatch.setattr(train, "cross_val_score", fake_cross_val_score)

    # Prépare une requête d'entraînement minimale
    request = train.TrainingRequest(
        # Définit un sujet fictif pour le test
        subject="S01",
        # Définit un run fictif pour le test
        run="R01",
        # Fournit une config pipeline valide
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=None,
            classifier="lda",
            scaler=None,
        ),
        # Injecte un data_dir factice
        data_dir=tmp_path / "data",
        # Injecte un artifacts_dir factice
        artifacts_dir=tmp_path / "artifacts",
        # Injecte un raw_dir factice
        raw_dir=tmp_path / "raw",
    )

    # Exécute l'entraînement pour déclencher la création du splitter
    train.run_training(request)

    # Vérifie que les kwargs du splitter sont présents
    assert "kwargs" in captured
    # Récupère les kwargs capturés pour inspection
    kwargs = cast(dict[str, object], captured["kwargs"])
    # Vérifie que le nombre de splits respecte la consigne 10
    assert kwargs.get("n_splits") == train.DEFAULT_CV_SPLITS
    # Vérifie que la taille de test suit la contrainte minimale
    assert kwargs.get("test_size") == pytest.approx(1 / 3)
    # Vérifie la présence de random_state pour la reproductibilité
    assert "random_state" in kwargs
    # Vérifie la valeur du random_state pour stabilité
    assert kwargs["random_state"] == train.DEFAULT_RANDOM_STATE


# Vérifie que le splitter retourne None quand une seule classe est présente
def test_build_cv_splitter_returns_none_with_single_class() -> None:
    # Prépare un vecteur de labels avec une seule classe
    y = np.array([0, 0, 0], dtype=int)
    # Construit le splitter avec les paramètres par défaut
    splitter = train._build_cv_splitter(y, train.DEFAULT_CV_SPLITS)
    # Vérifie que le splitter est None en cas de classe unique
    assert splitter is None


# Vérifie que le splitter refuse les effectifs trop faibles
def test_build_cv_splitter_returns_shuffle_for_low_class_counts() -> None:
    # Prépare un vecteur de labels avec une classe rare
    y = np.array([0, 0, 0, 1], dtype=int)
    # Construit le splitter avec les paramètres par défaut
    splitter = train._build_cv_splitter(y, train.DEFAULT_CV_SPLITS)
    # Vérifie que le splitter bascule sur un shuffle non stratifié
    assert isinstance(splitter, train.ShuffleSplit)


def test_filter_shuffle_splits_keeps_two_classes_in_train() -> None:
    # Prépare un vecteur de labels avec une minorité de classe
    y = np.array([0, 0, 0, 1], dtype=int)
    # Crée un splitter shuffle pour simuler le fallback low count
    splitter = train.ShuffleSplit(
        n_splits=5,
        test_size=0.25,
        random_state=0,
    )
    # Exécute le filtrage des splits pour garder deux classes en train
    filtered = train._filter_shuffle_splits_for_binary_train(y, splitter, 5)
    # Vérifie que la liste filtrée est non vide pour cette distribution
    assert filtered
    # Vérifie que chaque split conserve deux classes en train
    assert all(np.unique(y[train_idx]).size == 2 for train_idx, _ in filtered)


# Vérifie le stop sur limite max quand aucun split n'est valide
def test_filter_shuffle_splits_breaks_on_max_attempts_when_no_valid_splits() -> None:
    # Prépare un vecteur de labels mono-classe pour bloquer la validation
    y = np.zeros((6,), dtype=int)
    # Construit un splitter riche en splits pour dépasser la limite
    splitter = train.ShuffleSplit(n_splits=20, test_size=0.5, random_state=0)
    # Exécute le filtrage avec un nombre minimal de splits demandés
    filtered = train._filter_shuffle_splits_for_binary_train(y, splitter, 1)
    # Vérifie que la liste reste vide lorsque tous les splits sont rejetés
    assert filtered == []


# Vérifie l'arrêt immédiat quand le quota de splits valides est atteint
def test_filter_shuffle_splits_stops_after_requested_splits() -> None:
    # Prépare un vecteur de labels équilibré pour générer des splits valides
    y = np.array([0, 0, 1, 1], dtype=int)
    # Construit un splitter court pour limiter la boucle
    splitter = train.ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    # Exécute le filtrage avec un seul split attendu
    filtered = train._filter_shuffle_splits_for_binary_train(y, splitter, 1)
    # Vérifie que la liste est tronquée au premier split valide
    assert len(filtered) == 1


# Vérifie le fallback lorsque le filtrage shuffle ne trouve aucun split
def test_resolve_cv_splits_reports_empty_filtered_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prépare un vecteur de labels pour déclencher la branche shuffle
    y = np.array([0, 0, 1, 1], dtype=int)
    # Force un splitter ShuffleSplit pour la résolution
    monkeypatch.setattr(
        train,
        "_build_cv_splitter",
        lambda *_args, **_kwargs: train.ShuffleSplit(
            n_splits=3,
            test_size=0.5,
            random_state=0,
        ),
    )
    # Force un filtrage vide pour simuler l'absence de splits valides
    monkeypatch.setattr(
        train,
        "_filter_shuffle_splits_for_binary_train",
        lambda *_args, **_kwargs: [],
    )
    # Résout le splitter pour couvrir la branche de rejet
    cv, reason = train._resolve_cv_splits(y, train.DEFAULT_CV_SPLITS)
    # Vérifie que la CV est signalée indisponible
    assert cv is None
    # Vérifie que la raison est bien renseignée
    assert reason


def test_describe_cv_unavailability_reports_single_class() -> None:
    # Prépare un vecteur de labels à classe unique
    y = np.array([0, 0, 0], dtype=int)
    # Exécute le diagnostic pour la CV indisponible
    message = train._describe_cv_unavailability(y, train.DEFAULT_CV_SPLITS)
    # Vérifie que le message mentionne l'unicité de classe
    assert "une seule classe" in message


def test_describe_cv_unavailability_reports_empty_samples() -> None:
    # Prépare un vecteur vide pour simuler l'absence d'échantillons
    y = np.array([], dtype=int)
    # Exécute le diagnostic pour la CV indisponible
    message = train._describe_cv_unavailability(y, train.DEFAULT_CV_SPLITS)
    # Vérifie que le message mentionne l'absence d'échantillons
    assert "aucun échantillon" in message


def test_main_prints_cv_unavailability_reason(monkeypatch, capsys):
    # Définit un run_training factice avec une raison explicite
    def fake_run_training(_request):
        # Retourne un rapport minimal avec une raison d'indisponibilité
        return {
            "cv_scores": np.array([]),
            "cv_splits_requested": 5,
            "cv_unavailability_reason": "raison test",
        }

    # Remplace run_training pour isoler l'output CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec les arguments minimaux
    exit_code = train.main(["S001", "R01"])

    # Vérifie un code de sortie nominal
    assert exit_code == 0
    # Capture la sortie standard pour validation
    stdout_lines = capsys.readouterr().out.splitlines()
    # Définit la sortie attendue avec la raison CV
    expected = [
        # Vérifie l'affichage des splits demandés
        "CV_SPLITS: 5 (scores: 0)",
        # Vérifie l'affichage de la raison d'indisponibilité
        "INFO: CV indisponible (raison test)",
        # Vérifie le tableau vide attendu
        "[]",
        # Vérifie le score moyen nul affiché
        "cross_val_score: 0.0",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


def test_main_prints_cv_error_warning(monkeypatch, capsys):
    # Définit un run_training factice avec une erreur CV
    def fake_run_training(_request):
        # Retourne un rapport minimal avec un message d'erreur CV
        return {
            "cv_scores": np.array([]),
            "cv_splits_requested": 3,
            "cv_error": "erreur CV",
        }

    # Remplace run_training pour isoler l'output CLI
    monkeypatch.setattr(train, "run_training", fake_run_training)
    # Neutralise l'accès au dataset réel pour garder le test hermétique
    monkeypatch.setattr(train, "resolve_sampling_rate", lambda *_args, **_kwargs: 50.0)

    # Exécute la CLI avec les arguments minimaux
    exit_code = train.main(["S001", "R01"])

    # Vérifie un code de sortie nominal
    assert exit_code == 0
    # Capture la sortie standard pour validation
    stdout_lines = capsys.readouterr().out.splitlines()
    # Définit la sortie attendue avec l'alerte CV
    expected = [
        # Vérifie l'affichage des splits demandés
        "CV_SPLITS: 3 (scores: 0)",
        # Vérifie l'affichage de l'alerte d'erreur CV
        "AVERTISSEMENT: cross_val_score échoué (erreur CV)",
        # Vérifie le tableau vide attendu
        "[]",
        # Vérifie le score moyen nul affiché
        "cross_val_score: 0.0",
    ]
    # Compare la sortie observée avec l'attendu
    assert stdout_lines == expected


def test_run_training_uses_grid_search_and_captures_best_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La recherche grid search doit piloter le pipeline et exposer les scores."""

    X = np.random.default_rng(0).standard_normal((6, 2, 4))
    y = np.array([0, 0, 0, 1, 1, 1])

    monkeypatch.setattr(train, "_load_data", lambda *_args: (X, y))

    class FakeDimReducer:
        def save(self, _path: Path) -> None:
            return None

    class FakePipeline:
        def __init__(self) -> None:
            self.named_steps = {"dimensionality": FakeDimReducer(), "scaler": None}

        # Simule l'API fit pour la pipeline factice
        def fit(self, *_args, **_kwargs):
            # Retourne self pour émuler scikit-learn
            return self

    class FakeSearch:
        def __init__(self, estimator, param_grid, cv, scoring, refit) -> None:
            self.estimator = estimator
            self.param_grid = param_grid
            self.cv = cv
            self.scoring = scoring
            self.refit = refit
            self.best_estimator_ = FakePipeline()
            self.best_params_ = {"classifier": "lda"}
            self.best_score_ = 0.75
            self.best_index_ = 0
            self.cv_results_ = {
                "split0_test_score": [0.7],
                "split1_test_score": [0.8],
                "split2_test_score": [0.75],
            }

        def fit(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(train, "build_search_pipeline", lambda *_cfg: FakePipeline())
    monkeypatch.setattr(train, "GridSearchCV", FakeSearch)
    # Force cross_val_score pour isoler la validation finale
    monkeypatch.setattr(
        # Cible le module d'entraînement pour le patch
        train,
        # Cible explicitement cross_val_score
        "cross_val_score",
        # Retourne des scores déterministes pour le test
        lambda *_args, **_kwargs: np.array([0.62, 0.68]),
    )
    monkeypatch.setattr(train, "save_pipeline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train.joblib, "dump", lambda *_args, **_kwargs: None)

    captured: dict[str, object] = {}

    def fake_write_manifest(
        _request, _target_dir, cv_scores, _artifacts, search_summary=None
    ):
        captured["cv_scores"] = cv_scores
        captured["search_summary"] = search_summary
        return {"json": tmp_path / "m.json", "csv": tmp_path / "m.csv"}

    monkeypatch.setattr(train, "_write_manifest", fake_write_manifest)

    request = train.TrainingRequest(
        subject="S001",
        run="R01",
        pipeline_config=train.PipelineConfig(
            sfreq=50.0,
            feature_strategy="fft",
            normalize_features=True,
            dim_method="pca",
            n_components=2,
            classifier="lda",
            scaler=None,
        ),
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        raw_dir=tmp_path / "raw",
        enable_grid_search=True,
        grid_search_splits=3,
    )

    train.run_training(request)

    cv_scores = np.array(cast(np.ndarray, captured["cv_scores"]))
    assert cv_scores.shape == (2,)
    search_summary = cast(dict[str, object], captured["search_summary"])
    assert search_summary["best_score"] == pytest.approx(0.75)


def test_run_training_creates_target_dir_with_exist_ok_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_data(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
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
    def fake_load_data(
        _subject: str,
        _run: str,
        _build_context: train.NpyBuildContext,
    ):
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


# Valide la résolution de la fréquence via métadonnées EDF
def test_resolve_sampling_rate_prefers_metadata_when_default_requested(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Définit un identifiant de sujet pour préparer le chemin EDF
    subject = "S001"
    # Définit un run cohérent avec la logique de naming
    run = "R03"
    # Construit le répertoire racine simulant data/
    raw_dir = tmp_path
    # Construit le dossier du sujet pour y déposer un EDF fictif
    subject_dir = raw_dir / subject
    # Crée l'arborescence du sujet pour satisfaire .exists()
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Compose le chemin EDF attendu par la résolution automatique
    raw_path = subject_dir / f"{subject}{run}.edf"
    # Crée un fichier EDF vide pour activer la branche de lecture
    raw_path.write_text("", encoding="utf-8")

    # Prépare un faux objet Raw compatible avec raw.close()
    class DummyRaw:
        # Autorise un close() no-op pour simuler MNE Raw
        def close(self) -> None:
            # Simule la libération des ressources sans effet secondaire
            return None

    # Définit un loader factice renvoyant une fréquence de 160 Hz
    def fake_loader(
        path: Path, reference: str | None = "average"
    ) -> tuple[DummyRaw, dict[str, object]]:
        # Vérifie que le chemin transmis est celui attendu
        assert path == raw_path
        # Vérifie que la référence attendue est transmise
        assert reference == "average"
        # Retourne un Raw simulé et des métadonnées cohérentes
        return DummyRaw(), {"sampling_rate": 160.0}

    # Remplace load_physionet_raw pour éviter une lecture EDF réelle
    monkeypatch.setattr(train.preprocessing, "load_physionet_raw", fake_loader)

    # Demande la résolution avec la fréquence par défaut
    resolved = train.resolve_sampling_rate(
        subject,
        run,
        raw_dir,
        train.DEFAULT_SAMPLING_RATE,
        "average",
    )

    # Vérifie que la fréquence détectée est utilisée
    assert resolved == 160.0


# Valide le fallback lorsque la fréquence lue est invalide
def test_resolve_sampling_rate_falls_back_on_invalid_metadata(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Définit un identifiant de sujet pour préparer le chemin EDF
    subject = "S002"
    # Définit un run cohérent avec la logique de naming
    run = "R04"
    # Construit le répertoire racine simulant data/
    raw_dir = tmp_path
    # Construit le dossier du sujet pour y déposer un EDF fictif
    subject_dir = raw_dir / subject
    # Crée l'arborescence du sujet pour satisfaire .exists()
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Compose le chemin EDF attendu par la résolution automatique
    raw_path = subject_dir / f"{subject}{run}.edf"
    # Crée un fichier EDF vide pour activer la branche de lecture
    raw_path.write_text("", encoding="utf-8")

    # Prépare un faux objet Raw compatible avec raw.close()
    class DummyRaw:
        # Autorise un close() no-op pour simuler MNE Raw
        def close(self) -> None:
            # Simule la libération des ressources sans effet secondaire
            return None

    # Définit un loader factice renvoyant une fréquence invalide
    def fake_loader(
        path: Path, reference: str | None = "average"
    ) -> tuple[DummyRaw, dict[str, object]]:
        # Vérifie que le chemin transmis est celui attendu
        assert path == raw_path
        # Vérifie que la référence attendue est transmise
        assert reference == "average"
        # Retourne un Raw simulé et une valeur non convertible
        return DummyRaw(), {"sampling_rate": {"invalid": True}}

    # Remplace load_physionet_raw pour éviter une lecture EDF réelle
    monkeypatch.setattr(train.preprocessing, "load_physionet_raw", fake_loader)

    # Demande la résolution avec une fréquence explicite
    resolved = train.resolve_sampling_rate(
        subject,
        run,
        raw_dir,
        123.0,
        "average",
    )

    # Vérifie que la fréquence demandée est conservée
    assert resolved == 123.0
