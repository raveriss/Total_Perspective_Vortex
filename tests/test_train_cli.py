import argparse
import json
from typing import cast

from scripts import train


def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


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
    assert config.sfreq == 120.0
    assert config.normalize_features is True
    assert captured["dirs"] == (
        tmp_path / "data",
        tmp_path / "artifacts",
        tmp_path / "raw",
    )


def test_main_returns_error_code_when_training_files_missing(monkeypatch, capsys):
    def fake_run_training(_):
        raise FileNotFoundError("données manquantes pour S001 R01")

    monkeypatch.setattr(train, "run_training", fake_run_training)

    exit_code = train.main(["S001", "R01"])

    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "ERREUR: données manquantes pour S001 R01" in stdout
