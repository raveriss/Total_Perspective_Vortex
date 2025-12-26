import argparse
from pathlib import Path

from scripts import predict


def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


def test_build_parser_exposes_compatibility_defaults_and_paths() -> None:
    parser = predict.build_parser()

    classifier_action = _get_action(parser, "classifier")
    scaler_action = _get_action(parser, "scaler")
    feature_action = _get_action(parser, "feature_strategy")
    dim_action = _get_action(parser, "dim_method")
    n_components_action = _get_action(parser, "n_components")
    no_normalize_action = _get_action(parser, "no_normalize_features")
    sfreq_action = _get_action(parser, "sfreq")
    data_dir_action = _get_action(parser, "data_dir")
    artifacts_dir_action = _get_action(parser, "artifacts_dir")
    raw_dir_action = _get_action(parser, "raw_dir")

    assert classifier_action.choices is not None
    assert tuple(classifier_action.choices) == (
        "lda",
        "logistic",
        "svm",
        "centroid",
    )
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
    assert no_normalize_action.default is False
    assert sfreq_action.type is float
    assert sfreq_action.default == 50.0
    assert data_dir_action.type is Path
    assert data_dir_action.default == predict.DEFAULT_DATA_DIR
    assert artifacts_dir_action.type is Path
    assert artifacts_dir_action.default == predict.DEFAULT_ARTIFACTS_DIR
    assert raw_dir_action.type is Path
    assert raw_dir_action.default == predict.DEFAULT_RAW_DIR


def test_build_parser_parses_defaults_and_suppresses_n_components() -> None:
    parser = predict.build_parser()

    args = parser.parse_args(["S123", "R02"])

    assert args.classifier == "lda"
    assert args.scaler == "none"
    assert args.feature_strategy == "fft"
    assert args.dim_method == "pca"
    assert args.no_normalize_features is False
    assert args.sfreq == 50.0
    assert "n_components" not in vars(args)
    assert isinstance(args.data_dir, Path)
    assert args.data_dir == predict.DEFAULT_DATA_DIR
    assert isinstance(args.artifacts_dir, Path)
    assert args.artifacts_dir == predict.DEFAULT_ARTIFACTS_DIR
    assert isinstance(args.raw_dir, Path)
    assert args.raw_dir == predict.DEFAULT_RAW_DIR
