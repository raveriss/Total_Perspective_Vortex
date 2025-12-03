# Préserve argparse pour comparer les valeurs par défaut
import argparse

# Préserve sys pour récupérer l'interpréteur courant dans les assertions
import sys

# Garantit l'accès aux types génériques pour annoter les fakes
from typing import Any

# Utilise pytest pour vérifier les erreurs de parsing
import pytest

import mybci

# Fixe le code d'échec simulé pour valider la propagation dans main
MODULE_FAILURE_CODE = 5

EXIT_USAGE = 2

# Fixe le nombre de composantes demandé lors du mode train
TRAIN_COMPONENTS = 7

# Fixe le nombre de composantes demandé lors du mode predict
PREDICT_COMPONENTS = 9


def test_parse_args_returns_expected_namespace():
    args = mybci.parse_args(["S01", "R01", "train"])

    assert args.subject == "S01"
    assert args.run == "R01"
    assert args.mode == "train"


def test_build_parser_metadata():
    parser = mybci.build_parser()

    assert (
        parser.description == "Pilote un workflow d'entraînement ou de prédiction TPV"
    )
    assert (
        parser.usage or ""
    ).strip() == "python mybci.py <subject> <run> {train,predict,realtime}"


def test_build_parser_defines_expected_arguments():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    subject_arg = get_action("subject")
    run_arg = get_action("run")
    mode_arg = get_action("mode")

    assert subject_arg.help == "Identifiant du sujet (ex: S01)"
    assert run_arg.help == "Identifiant du run (ex: R01)"
    assert mode_arg.help == "Choix du pipeline à lancer"
    assert tuple(mode_arg.choices) == ("train", "predict", "realtime")


def test_build_parser_argument_types_and_defaults():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    n_components_action = get_action("n_components")
    no_normalize_action = get_action("no_normalize_features")

    assert n_components_action.type is int
    assert n_components_action.default is argparse.SUPPRESS
    assert n_components_action.option_strings == ["--n-components"]
    assert no_normalize_action.option_strings == ["--no-normalize-features"]
    assert no_normalize_action.default is False


def test_parse_args_rejects_invalid_mode(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(["S01", "R01", "invalid"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr


def test_parse_args_requires_all_positional_arguments(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(["S01", "R01"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "usage: python mybci.py <subject> <run> {train,predict,realtime}" in stderr


def test_main_invokes_train_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S01", "R02", "train"])

    assert exit_code == 0
    assert called["args"][0] == "tpv.train"
    assert called["args"][1].subject == "S01"
    assert called["args"][1].run == "R02"


def test_main_invokes_predict_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S02", "R03", "predict"])

    assert exit_code == 0
    assert called["args"][0] == "tpv.predict"
    assert called["args"][1].subject == "S02"
    assert called["args"][1].run == "R03"


def test_call_module_executes_python_module(monkeypatch):
    recorded_command: list[str] = []

    class DummyCompletedProcess:
        returncode = 0

    def fake_run(command: list[str], check: bool) -> DummyCompletedProcess:
        recorded_command.extend(command)
        assert check is False
        return DummyCompletedProcess()

    monkeypatch.setattr(mybci.subprocess, "run", fake_run)

    exit_code = mybci._call_module(
        "tpv.train",
        mybci.ModuleCallConfig(
            subject="S03",
            run="R04",
            classifier="lda",
            scaler=None,
            feature_strategy="fft",
            dim_method="pca",
            n_components=None,
            normalize_features=True,
        ),
    )

    assert exit_code == 0
    assert recorded_command == [
        sys.executable,
        "-m",
        "tpv.train",
        "S03",
        "R04",
        "--classifier",
        "lda",
        "--feature-strategy",
        "fft",
        "--dim-method",
        "pca",
    ]


def test_call_module_appends_optional_arguments(monkeypatch):
    recorded_command: list[str] = []

    class DummyCompletedProcess:
        returncode = 0

    def fake_run(command: list[str], check: bool) -> DummyCompletedProcess:
        recorded_command.extend(command)
        assert check is False
        return DummyCompletedProcess()

    monkeypatch.setattr(mybci.subprocess, "run", fake_run)

    exit_code = mybci._call_module(
        "tpv.predict",
        mybci.ModuleCallConfig(
            subject="S05",
            run="R06",
            classifier="svm",
            scaler="standard",
            feature_strategy="wavelet",
            dim_method="csp",
            n_components=12,
            normalize_features=False,
        ),
    )

    assert exit_code == 0
    assert recorded_command == [
        sys.executable,
        "-m",
        "tpv.predict",
        "S05",
        "R06",
        "--classifier",
        "svm",
        "--scaler",
        "standard",
        "--feature-strategy",
        "wavelet",
        "--dim-method",
        "csp",
        "--n-components",
        "12",
        "--no-normalize-features",
    ]


# Vérifie que main remonte l'échec propagé par un module sous-jacent
def test_main_propagates_module_failure(monkeypatch):
    # Force un code retour non nul pour simuler un pipeline échoué
    monkeypatch.setattr(mybci, "_call_module", lambda *_: MODULE_FAILURE_CODE)
    # Capture le code de sortie afin d'observer la propagation d'erreur
    exit_code = mybci.main(["S01", "R01", "train"])
    # Valide que main renvoie exactement le code d'échec du module
    assert exit_code == MODULE_FAILURE_CODE


def test_build_parser_defines_optional_defaults_and_choices():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    classifier_action = get_action("classifier")
    scaler_action = get_action("scaler")
    feature_action = get_action("feature_strategy")
    dim_action = get_action("dim_method")
    n_components_action = get_action("n_components")
    normalize_action = get_action("no_normalize_features")

    assert tuple(classifier_action.choices) == ("lda", "logistic", "svm")
    assert classifier_action.default == "lda"
    assert tuple(scaler_action.choices) == ("standard", "robust", "none")
    assert scaler_action.default == "none"
    assert tuple(feature_action.choices) == ("fft", "wavelet")
    assert feature_action.default == "fft"
    assert tuple(dim_action.choices) == ("pca", "csp")
    assert dim_action.default == "pca"
    assert n_components_action.default is argparse.SUPPRESS
    assert normalize_action.default is False


def test_build_parser_help_messages():
    parser = mybci.build_parser()

    help_text = parser.format_help()

    assert "Choix du classifieur final" in help_text
    assert "Scaler optionnel appliqué après les features" in help_text
    assert "Méthode d'extraction des features" in help_text
    assert "Technique de réduction de dimension" in help_text
    assert "Nombre de composantes à conserver" in help_text
    assert "Désactive la normalisation des features" in help_text


def test_build_parser_help_fields_are_exact():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    classifier_action = get_action("classifier")
    scaler_action = get_action("scaler")
    feature_action = get_action("feature_strategy")
    dim_action = get_action("dim_method")
    n_components_action = get_action("n_components")
    normalize_action = get_action("no_normalize_features")

    assert classifier_action.help == "Choix du classifieur final"
    assert scaler_action.help == "Scaler optionnel appliqué après les features"
    assert feature_action.help == "Méthode d'extraction des features"
    assert dim_action.help == "Technique de réduction de dimension"
    assert n_components_action.help == "Nombre de composantes à conserver"
    assert normalize_action.help == "Désactive la normalisation des features"


def test_parse_args_defaults_match_parser_configuration():
    args = mybci.parse_args(["S07", "R08", "predict"])

    assert args.classifier == "lda"
    assert args.scaler == "none"
    assert args.feature_strategy == "fft"
    assert args.dim_method == "pca"
    assert "n_components" not in vars(args)
    assert args.no_normalize_features is False


def test_parse_args_rejects_unknown_optional_choices(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(
            [
                "S01",
                "R01",
                "train",
                "--classifier",
                "invalid",
            ]
        )

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr

    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(
            [
                "S01",
                "R01",
                "train",
                "--n-components",
                "invalid",
            ]
        )

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid int value" in stderr

    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(
            [
                "S01",
                "R01",
                "train",
                "--scaler",
                "invalid",
            ]
        )

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr


def test_main_builds_config_with_scaler_none_and_defaults(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        captured["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S09", "R10", "predict"])

    assert exit_code == 0
    called_module, config = captured["args"]
    assert called_module == "tpv.predict"
    assert config.scaler is None
    assert config.normalize_features is True
    assert config.n_components is None


def test_main_respects_no_normalize_flag(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(
        [
            "S11",
            "R12",
            "train",
            "--no-normalize-features",
            "--n-components",
            str(TRAIN_COMPONENTS),
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.normalize_features is False
    assert config.n_components == TRAIN_COMPONENTS


def test_main_propagates_all_cli_options(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        captured["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(
        [
            "S13",
            "R14",
            "predict",
            "--classifier",
            "svm",
            "--scaler",
            "robust",
            "--feature-strategy",
            "wavelet",
            "--dim-method",
            "csp",
            "--n-components",
            str(PREDICT_COMPONENTS),
        ]
    )

    assert exit_code == 0
    module_name, config = captured["args"]
    assert module_name == "tpv.predict"
    assert config.classifier == "svm"
    assert config.scaler == "robust"
    assert config.feature_strategy == "wavelet"
    assert config.dim_method == "csp"
    assert config.n_components == PREDICT_COMPONENTS
    assert config.normalize_features is True
