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

# Fixe le code de sortie du sous-processus temps réel pour simuler l'échec
REALTIME_SUBPROCESS_CODE = 7

# Fixe le code de sortie retourné lors du routage temps réel dans main
REALTIME_ROUTING_CODE = 12

# Fixe la taille de fenêtre par défaut utilisée en mode realtime
REALTIME_WINDOW_DEFAULT = 50

# Fixe le pas de fenêtre par défaut utilisé en mode realtime
REALTIME_STEP_DEFAULT = 25

# Fixe la taille de buffer par défaut utilisée en mode realtime
REALTIME_BUFFER_DEFAULT = 3

# Fixe la fréquence d'échantillonnage par défaut pour le streaming realtime
REALTIME_SFREQ_DEFAULT = 50.0

# Fixe la latence maximale par défaut pour le streaming realtime
REALTIME_LATENCY_DEFAULT = 2.0


def test_main_runs_global_evaluation_when_no_arguments(monkeypatch):
    # Prépare un registre pour vérifier l'appel du runner global
    called: dict[str, bool] = {}

    # Simule le runner global pour éviter une boucle lourde en test
    def fake_runner() -> int:
        # Marque l'activation du runner global
        called["triggered"] = True
        # Retourne un succès pour suivre le contrat du runner global
        return 0

    # Remplace le runner global par le double de test
    monkeypatch.setattr(mybci, "_run_global_evaluation", fake_runner)
    # Exécute main sans aucun argument pour couvrir la nouvelle branche
    exit_code = mybci.main([])

    # Vérifie que le runner global a bien été invoqué
    assert called.get("triggered") is True
    # Vérifie que main renvoie le code de succès du runner global
    assert exit_code == 0


def test_parse_args_returns_expected_namespace():
    args = mybci.parse_args(["S001", "R01", "train"])

    assert args.subject == "S001"
    assert args.run == "R01"
    assert args.mode == "train"


def test_call_realtime_executes_python_module(monkeypatch):
    captured: dict[str, Any] = {}

    class _Completed:
        def __init__(self, code: int):
            self.returncode = code

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return _Completed(code=7)

    monkeypatch.setattr(mybci.subprocess, "run", fake_run)

    exit_code = mybci._call_realtime(
        mybci.RealtimeCallConfig(
            subject="S20",
            run="R21",
            window_size=10,
            step_size=5,
            buffer_size=3,
            # Fige la latence max pour respecter la contrainte realtime
            max_latency=REALTIME_LATENCY_DEFAULT,
            sfreq=42.0,
            data_dir="data",
            artifacts_dir="artifacts",
        )
    )

    assert exit_code == REALTIME_SUBPROCESS_CODE
    assert captured["check"] is False
    assert captured["command"] == [
        sys.executable,
        "-m",
        "tpv.realtime",
        "S20",
        "R21",
        "--window-size",
        "10",
        "--step-size",
        "5",
        "--buffer-size",
        "3",
        "--max-latency",
        str(REALTIME_LATENCY_DEFAULT),
        "--sfreq",
        "42.0",
        "--data-dir",
        "data",
        "--artifacts-dir",
        "artifacts",
    ]


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

    assert subject_arg.help == "Identifiant du sujet (ex: S001)"
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
        mybci.parse_args(["S001", "R01", "invalid"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr


def test_parse_args_requires_all_positional_arguments(capsys):
    with pytest.raises(SystemExit) as excinfo:
        mybci.parse_args(["S001", "R01"])

    assert excinfo.value.code == EXIT_USAGE
    stderr = capsys.readouterr().err
    assert "usage: python mybci.py <subject> <run> {train,predict,realtime}" in stderr


def test_main_invokes_train_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S001", "R02", "train"])

    assert exit_code == 0
    assert called["args"][0] == "tpv.train"
    assert called["args"][1].subject == "S001"
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
    exit_code = mybci.main(["S001", "R01", "train"])
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

    assert tuple(classifier_action.choices) == ("lda", "logistic", "svm", "centroid")
    assert classifier_action.default == "lda"
    assert tuple(scaler_action.choices) == ("standard", "robust", "none")
    assert scaler_action.default == "none"
    assert tuple(feature_action.choices) == ("fft", "wavelet")
    assert feature_action.default == "fft"
    assert tuple(dim_action.choices) == ("pca", "csp")
    assert dim_action.default == "pca"
    assert n_components_action.default is argparse.SUPPRESS
    assert normalize_action.default is False


# Vérifie les options spécifiques au mode realtime pour éviter les mutations
def test_build_parser_defines_realtime_defaults_and_help():
    # Construit le parser afin d'inspecter les actions realtime
    parser = mybci.build_parser()

    # Fournit un accès direct aux actions via leur dest
    def get_action(dest: str):
        # Recherche l'action correspondant au dest ciblé
        return next(action for action in parser._actions if action.dest == dest)

    # Récupère l'action de taille de fenêtre pour contrôler son type
    window_action = get_action("window_size")
    # Récupère l'action de pas de fenêtre pour vérifier le défaut
    step_action = get_action("step_size")
    # Récupère l'action de buffer pour valider le paramétrage par défaut
    buffer_action = get_action("buffer_size")
    # Récupère l'action de latence pour contrôler le SLA
    latency_action = get_action("max_latency")
    # Récupère l'action de fréquence pour surveiller la valeur par défaut
    sfreq_action = get_action("sfreq")
    # Récupère l'action data-dir pour bloquer les mutations de défaut
    data_dir_action = get_action("data_dir")
    # Récupère l'action artifacts-dir pour fixer le chemin par défaut
    artifacts_action = get_action("artifacts_dir")

    # Vérifie que la conversion window-size reste un entier
    assert window_action.type is int
    # Vérifie que la valeur par défaut de window-size reste 50
    assert window_action.default == REALTIME_WINDOW_DEFAULT
    # Vérifie que l'aide décrit correctement window-size
    assert window_action.help == "Taille de fenêtre glissante pour le mode realtime"
    # Vérifie que la conversion step-size reste un entier
    assert step_action.type is int
    # Vérifie que la valeur par défaut de step-size reste 25
    assert step_action.default == REALTIME_STEP_DEFAULT
    # Vérifie que l'aide de step-size reste présente
    assert step_action.help == "Pas entre deux fenêtres en streaming realtime"
    # Vérifie que buffer-size reste typé en entier
    assert buffer_action.type is int
    # Vérifie que la valeur par défaut de buffer-size reste 3
    assert buffer_action.default == REALTIME_BUFFER_DEFAULT
    # Vérifie que l'aide de buffer-size reste intacte
    assert buffer_action.help == "Taille du buffer de lissage pour le mode realtime"
    # Vérifie que max-latency reste typé en float pour surveiller le SLA
    assert latency_action.type is float
    # Vérifie que la valeur par défaut de max-latency reste 2.0
    assert latency_action.default == REALTIME_LATENCY_DEFAULT
    # Vérifie que l'aide de max-latency reste explicite
    assert latency_action.help == "Latence maximale autorisée par fenêtre realtime"
    # Vérifie que sfreq reste typé en float pour protéger le parsing
    assert sfreq_action.type is float
    # Vérifie que la valeur par défaut de sfreq reste 50.0
    assert sfreq_action.default == REALTIME_SFREQ_DEFAULT
    # Vérifie que l'aide de sfreq reste descriptive
    assert sfreq_action.help == "Fréquence d'échantillonnage appliquée au flux realtime"
    # Vérifie que data-dir conserve le défaut attendu
    assert data_dir_action.default == "data"
    # Vérifie que l'aide de data-dir reste informative
    assert data_dir_action.help == "Répertoire racine contenant les fichiers numpy"
    # Vérifie qu'artifacts-dir conserve le défaut attendu
    assert artifacts_action.default == "artifacts"
    # Vérifie que l'aide d'artifacts-dir reste descriptive
    assert artifacts_action.help == "Répertoire racine où récupérer le modèle entraîné"


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
                "S001",
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
                "S001",
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
                "S001",
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


def test_main_routes_to_realtime(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_realtime(config: mybci.RealtimeCallConfig) -> int:
        captured["config"] = config
        return 12

    monkeypatch.setattr(mybci, "_call_realtime", fake_realtime)

    exit_code = mybci.main(
        [
            "S30",
            "R31",
            "realtime",
            "--window-size",
            "8",
            "--step-size",
            "4",
            "--buffer-size",
            "5",
            "--max-latency",
            "1.5",
            "--sfreq",
            "64.0",
            "--data-dir",
            "custom-data",
            "--artifacts-dir",
            "custom-artifacts",
        ]
    )

    assert exit_code == REALTIME_ROUTING_CODE
    assert captured["config"] == mybci.RealtimeCallConfig(
        subject="S30",
        run="R31",
        window_size=8,
        step_size=4,
        buffer_size=5,
        max_latency=1.5,
        sfreq=64.0,
        data_dir="custom-data",
        artifacts_dir="custom-artifacts",
    )
