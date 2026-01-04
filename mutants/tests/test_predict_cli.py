# ruff: noqa: PLR0915
import argparse
import builtins
from pathlib import Path

from scripts import predict


def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


def test_build_parser_exposes_compatibility_defaults_and_paths() -> (
    None
):  # noqa: PLR0915
    parser = predict.build_parser()

    assert (
        parser.description == "Charge une pipeline TPV entraînée et produit un rapport"
    )
    subject_action = _get_action(parser, "subject")
    run_action = _get_action(parser, "run")
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

    assert subject_action.help == "Identifiant du sujet (ex: S001)"
    assert run_action.help == "Identifiant du run (ex: R01)"

    assert classifier_action.choices is not None
    assert tuple(classifier_action.choices) == (
        "lda",
        "logistic",
        "svm",
        "centroid",
    )
    assert classifier_action.default == "lda"
    assert (
        classifier_action.help
        == "Classifieur final (ignoré en prédiction, pour compatibilité CLI)"
    )

    assert scaler_action.choices is not None
    assert tuple(scaler_action.choices) == ("standard", "robust", "none")
    assert scaler_action.default == "none"
    assert (
        scaler_action.help == "Scaler appliqué en entraînement (ignoré en prédiction)"
    )

    assert feature_action.choices is not None
    assert tuple(feature_action.choices) == ("fft", "wavelet")
    assert feature_action.default == "fft"
    assert (
        feature_action.help
        == "Stratégie de features utilisée à l'entraînement (ignorée ici)"
    )

    assert dim_action.choices is not None
    assert tuple(dim_action.choices) == ("pca", "csp")
    assert dim_action.default == "pca"
    assert (
        dim_action.help == "Méthode de réduction de dimension (ignorée en prédiction)"
    )

    assert n_components_action.help == "Nombre de composantes (ignoré en prédiction)"
    assert no_normalize_action.help == "Flag de normalisation (ignoré en prédiction)"
    assert sfreq_action.help == "Fréquence utilisée en features (ignorée ici)"

    assert n_components_action.default is argparse.SUPPRESS
    assert n_components_action.type is int
    assert no_normalize_action.default is False
    assert sfreq_action.type is float
    assert sfreq_action.default == 50.0
    assert data_dir_action.type is Path
    assert data_dir_action.default == predict.DEFAULT_DATA_DIR
    assert data_dir_action.help == "Répertoire racine contenant les fichiers numpy"

    assert artifacts_dir_action.type is Path
    assert artifacts_dir_action.default == predict.DEFAULT_ARTIFACTS_DIR
    assert artifacts_dir_action.help == "Répertoire racine où lire le modèle"

    assert raw_dir_action.type is Path
    assert raw_dir_action.default == predict.DEFAULT_RAW_DIR
    assert raw_dir_action.help == "Répertoire racine contenant les fichiers EDF bruts"

    normalized_help = " ".join(parser.format_help().split())
    assert "Charge une pipeline TPV entraînée et produit un rapport" in normalized_help
    assert (
        "Méthode de réduction de dimension (ignorée en prédiction)" in normalized_help
    )
    assert "Nombre de composantes (ignoré en prédiction)" in normalized_help
    assert "Flag de normalisation (ignoré en prédiction)" in normalized_help
    assert "Fréquence utilisée en features (ignorée ici)" in normalized_help
    assert "Répertoire racine contenant les fichiers numpy" in normalized_help
    assert "Répertoire racine où lire le modèle" in normalized_help
    assert "Répertoire racine contenant les fichiers EDF bruts" in normalized_help


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


def test_main_renders_epoch_log_and_accuracy(monkeypatch, capsys, tmp_path):
    subject = "S10"
    run = "R20"
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    captured_result: dict[str, object] = {}
    y_pred = [1, 0]
    y_true = [1, 1]

    zip_strict_values: list[object] = []
    real_zip = builtins.zip

    def spy_zip(*iterables, **kwargs):
        if len(iterables) == 2 and iterables[0] is y_pred and iterables[1] is y_true:
            zip_strict_values.append(kwargs.get("strict", "__missing__"))
        return real_zip(*iterables, **kwargs)

    monkeypatch.setattr(builtins, "zip", spy_zip)

    def fake_evaluate_run(
        subject_arg: str,
        run_arg: str,
        data_dir_arg: Path,
        artifacts_dir_arg: Path,
        raw_dir_arg: Path,
    ) -> dict[str, object]:
        captured_result["subject"] = subject_arg
        captured_result["run"] = run_arg
        captured_result["data_dir"] = data_dir_arg
        captured_result["artifacts_dir"] = artifacts_dir_arg
        captured_result["raw_dir"] = raw_dir_arg
        return {
            "subject": subject_arg,
            "run": run_arg,
            "predictions": y_pred,
            "y_true": y_true,
            "accuracy": 0.5,
        }

    def fake_build_report(result: dict[str, object]) -> dict[str, object]:
        captured_result["report_input"] = result
        return {"global": result["accuracy"]}

    monkeypatch.setattr(predict, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(predict, "build_report", fake_build_report)

    exit_code = predict.main(
        [
            subject,
            run,
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
        ]
    )

    assert exit_code == 0
    assert zip_strict_values == [True]
    assert captured_result == {
        "subject": subject,
        "run": run,
        "data_dir": data_dir,
        "artifacts_dir": artifacts_dir,
        "raw_dir": predict.DEFAULT_RAW_DIR,
        "report_input": {
            "subject": subject,
            "run": run,
            "predictions": y_pred,
            "y_true": y_true,
            "accuracy": 0.5,
        },
    }

    stdout = capsys.readouterr().out.splitlines()
    assert stdout == [
        "epoch nb: [prediction] [truth] equal?",
        "epoch 00: [1] [1] True",
        "epoch 01: [0] [1] False",
        "Accuracy: 0.5000",
    ]
