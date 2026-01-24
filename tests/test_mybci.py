# Préserve argparse pour vérifier les valeurs par défaut des options
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


# Vérifie que l'absence de scikit-learn produit un message actionnable
def test_require_dependency_reports_missing_sklearn(monkeypatch):
    # Simule l'absence de scikit-learn dans l'environnement courant
    monkeypatch.setattr(mybci.importlib.util, "find_spec", lambda _: None)
    # Capture l'exception SystemExit pour inspecter le message utilisateur
    with pytest.raises(SystemExit) as exc_info:
        # Lance la vérification des dépendances ML
        mybci._ensure_ml_dependencies()
    # Convertit l'exception en chaîne pour analyser le message
    message = str(exc_info.value)
    # Vérifie que le module manquant est clairement cité
    assert "sklearn" in message
    # Vérifie que la commande d'installation Poetry est suggérée
    assert "poetry install" in message


# Vérifie que le rapport agrège correctement toutes les alertes attendues
def test_report_missing_artifacts_summarizes_all_alerts(capsys):
    # Rassemble douze couples manquants pour tester le découpage à dix éléments
    missing_entries = [f"S{idx:03d}/R{idx:02d}" for idx in range(1, 13)]
    # Prépare des modèles manquants pour couvrir les runs partiels et vides
    missing_models_by_run = {
        # Conserve un run partiel pour vérifier le résumé par run
        "R01": ["S001", "S002"],
        # Conserve un run complet pour activer la liste des runs vides
        "R02": ["S001", "S002", "S003"],
        # Ajoute un run volumineux pour valider la limitation des exemples affichés
        "R03": [f"S{idx:03d}" for idx in range(1, 8)],
    }
    # Construit une expérience ignorée pour déclencher l'avertissement final
    skipped_experiments = [mybci.ExperimentDefinition(index=1, run="R02")]
    # Exécute le rapport afin de capturer la totalité des messages émis
    mybci._report_missing_artifacts(
        missing_entries,
        missing_models_by_run,
        skipped_experiments,
        3,
    )
    # Capture la sortie standard pour inspecter les alertes imprimées
    stdout = capsys.readouterr().out
    # Vérifie que l'avertissement de données conserve sa casse officielle
    assert any(
        line.startswith("AVERTISSEMENT: certaines données EDF ou .npy sont manquantes.")
        for line in stdout.splitlines()
    )
    # Vérifie que le volume de couples manquants est correctement résumé
    assert "Couples sujet/run concernés: 12" in stdout
    # Vérifie que seuls les dix premiers couples apparaissent dans l'aperçu
    assert (
        "Premiers manquants: "
        "S001/R01, S002/R02, S003/R03, S004/R04, S005/R05, "
        "S006/R06, S007/R07, S008/R08, S009/R09, S010/R10"
    ) in stdout
    # Vérifie que les références au-delà de la dixième sont exclues
    assert "S011/R11" not in stdout
    # Vérifie que l'alerte sur les modèles absents reste inchangée
    assert any(
        line.startswith("AVERTISSEMENT: certains modèles entraînés sont absents.")
        for line in stdout.splitlines()
    )
    # Vérifie que l'avertissement sur les modèles absents est présent
    assert "modèles entraînés sont absents" in stdout
    # Vérifie que les runs totalement vides sont listés pour prioriser
    assert "Runs sans aucun modèle disponible: R02" in stdout
    # Vérifie que le résumé par run mentionne les sujets manquants avec exemples
    assert "Run R01: modèles manquants pour 2 sujets (exemples: S001, S002)" in stdout
    # Vérifie que les exemples sont limités à cinq sujets lorsqu'ils sont nombreux
    expected_r03 = (
        "Run R03: modèles manquants pour 7 sujets "
        "(exemples: S001, S002, S003, S004, S005)"
    )
    # Vérifie que l'aperçu du run volumineux reste borné à cinq sujets
    assert expected_r03 in stdout
    # Vérifie que la ligne dédiée au run volumineux n'affiche pas les sujets suivants
    run_r03_line = next(
        line for line in stdout.splitlines() if line.startswith("Run R03")
    )
    # Vérifie que les sujets au-delà du cinquième sont absents de l'aperçu ciblé
    assert "S006" not in run_r03_line
    # Vérifie que le rappel de commande de génération reste affiché
    # sans préfixe parasite
    assert any(
        line.startswith("Pour générer un modèle manquant, lancez par exemple :")
        for line in stdout.splitlines()
    )
    # Vérifie que les expériences ignorées conservent le préfixe d'avertissement
    assert "AVERTISSEMENT: les expériences suivantes ont été ignorées" in stdout
    # Vérifie que les expériences ignorées sont récapitulées
    assert "expériences suivantes ont été ignorées faute de modèles: 1 (R02)" in stdout


# Vérifie que le rapport reste silencieux lorsqu'aucun élément ne manque
def test_report_missing_artifacts_skips_empty_inputs(capsys):
    # Appelle le rapport avec des structures entièrement vides
    mybci._report_missing_artifacts([], {}, [], 0)
    # Capture la sortie pour vérifier l'absence totale de messages
    stdout = capsys.readouterr().out
    # Vérifie qu'aucune ligne n'est produite lorsqu'il n'y a aucun manque
    assert stdout == ""


# Vérifie que seuls les modèles manquants déclenchent un avertissement ciblé
def test_report_missing_artifacts_only_reports_models(capsys):
    # Simule un run partiel sans données ni expériences ignorées
    missing_models_by_run = {"R05": ["S010"]}
    # Appelle le rapport pour déclencher uniquement l'alerte modèles
    mybci._report_missing_artifacts([], missing_models_by_run, [], 2)
    # Capture la sortie pour inspecter les messages émis
    stdout = capsys.readouterr().out
    # Vérifie que le message sur les modèles absents est présent
    assert "modèles entraînés sont absents" in stdout
    # Vérifie que l'alerte données manquantes n'est pas imprimée
    assert "Couples sujet/run concernés" not in stdout
    # Vérifie que l'alerte sur les expériences ignorées reste muette
    assert "expériences suivantes ont été ignorées" not in stdout
    # Vérifie que la liste des runs entièrement vides n'est pas affichée
    assert "Runs sans aucun modèle disponible" not in stdout


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


def test_main_defaults_to_sys_argv_and_runs_global_evaluation(monkeypatch):
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
    # Simule une exécution CLI sans arguments utilisateur
    monkeypatch.setattr(mybci.sys, "argv", ["mybci.py"])
    # Exécute main sans argv pour couvrir la branche sys.argv[1:]
    exit_code = mybci.main()

    # Vérifie que le runner global a bien été invoqué
    assert called.get("triggered") is True
    # Vérifie que main renvoie le code de succès du runner global
    assert exit_code == 0


def test_main_defaults_to_sys_argv_and_routes_train(monkeypatch):
    # Prépare un registre pour capturer le module et la config transmis
    called: dict[str, Any] = {}

    # Simule l'appel de module pour éviter un sous-processus réel
    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        # Capture les arguments pour vérification fine
        called["args"] = (module_name, config)
        # Retourne un succès contrôlé
        return 0

    # Remplace l'exécution du module par le double de test
    monkeypatch.setattr(mybci, "_call_module", fake_call)
    # Simule une exécution CLI standard avec 3 positionnels
    monkeypatch.setattr(mybci.sys, "argv", ["mybci.py", "S001", "R02", "train"])
    # Exécute main sans argv pour valider sys.argv[1:] (et pas sys.argv[2:])
    exit_code = mybci.main()

    # Vérifie que main retourne le succès propagé par _call_module
    assert exit_code == 0
    # Vérifie que le routage train sélectionne bien le module attendu
    assert called["args"][0] == "tpv.train"
    # Vérifie que les 2 premiers args CLI sont bien conservés
    assert called["args"][1].subject == "S001"
    assert called["args"][1].run == "R02"


def test_parse_args_returns_expected_namespace():
    # Lance le parsing sur des identifiants numériques comme dans le sujet
    args = mybci.parse_args(["4", "14", "train"])

    # Vérifie que le sujet est normalisé au format Sxxx
    assert args.subject == "S004"
    # Vérifie que le run est normalisé au format Rxx
    assert args.run == "R14"
    assert args.mode == "train"
    # Vérifie que les options facultatives restent absentes sans override explicite
    assert not hasattr(args, "feature_strategy")
    assert not hasattr(args, "dim_method")
    assert not hasattr(args, "classifier")
    assert not hasattr(args, "scaler")


# Vérifie que la liste des expériences par défaut correspond aux 6 runs attendus
def test_build_default_experiments_matches_six_runs():
    # Construit la liste des expériences définies par défaut
    experiments = mybci._build_default_experiments()
    # Extrait les runs pour valider l'ordre et la complétude
    runs = [experiment.run for experiment in experiments]
    # Valide le nombre d'expériences attendu par la consigne
    assert len(experiments) == 6
    # Valide l'ordre strict des runs attendus pour l'évaluation globale
    assert runs == ["R03", "R04", "R05", "R06", "R07", "R08"]


# Vérifie que l'évaluation globale affiche une ligne par sujet évalué
def test_evaluate_experiments_reports_subject_accuracy(capsys, monkeypatch):
    # Prépare une expérience unique pour limiter la sortie à vérifier
    experiments = [mybci.ExperimentDefinition(index=0, run="R03")]
    # Prépare une table de sujets disponibles pour l'expérience ciblée
    available_subjects_by_run = {"R03": [1, 2]}

    # Simule le calcul d'accuracy pour éviter des dépendances lourdes
    def fake_evaluate(*_args, **_kwargs) -> float:
        # Retourne une accuracy fixe pour un affichage déterministe
        return 0.5

    # Remplace la fonction d'évaluation par un double contrôlé
    monkeypatch.setattr(mybci, "_evaluate_experiment_subject", fake_evaluate)

    # Prépare le répertoire data pour la construction des chemins
    data_root = mybci.Path("data")
    # Prépare le répertoire artifacts pour l'évaluation simulée
    artifacts_root = mybci.Path("artifacts")
    # Prépare le répertoire raw pour la construction des chemins
    raw_root = mybci.Path("data")
    # Construit l'objet de chemins attendu par l'évaluation
    paths = mybci.EvaluationPaths(
        # Associe le répertoire data pour la lecture des features
        data_root=data_root,
        # Associe le répertoire artifacts pour les modèles simulés
        artifacts_root=artifacts_root,
        # Associe le répertoire raw pour compléter les chemins EDF
        raw_root=raw_root,
    )

    # Lance l'évaluation sur les sujets factices sans barre de progression
    mybci._evaluate_experiments(experiments, available_subjects_by_run, paths, None)

    # Capture la sortie pour vérifier les lignes par sujet
    stdout = capsys.readouterr().out
    # Vérifie l'affichage de l'accuracy pour le sujet 1
    assert "experiment 0: subject 001: accuracy = 0.5000" in stdout
    # Vérifie l'affichage de l'accuracy pour le sujet 2
    assert "experiment 0: subject 002: accuracy = 0.5000" in stdout


def test_build_parser_metadata():
    parser = mybci.build_parser()

    assert (
        parser.description == "Pilote un workflow d'entraînement ou de prédiction TPV"
    )
    assert (
        parser.usage or ""
    ).strip() == "python mybci.py <subject> <run> {train,predict}"


def test_build_parser_defines_expected_arguments():
    parser = mybci.build_parser()

    def get_action(dest: str):
        return next(action for action in parser._actions if action.dest == dest)

    subject_arg = get_action("subject")
    run_arg = get_action("run")
    mode_arg = get_action("mode")
    feature_arg = get_action("feature_strategy")
    dim_arg = get_action("dim_method")

    # Vérifie que l'aide du sujet mentionne l'exemple numérique attendu
    assert subject_arg.help == "Identifiant du sujet (ex: 4)"
    # Vérifie que l'aide du run mentionne l'exemple numérique attendu
    assert run_arg.help == "Identifiant du run (ex: 14)"
    # Vérifie que l'aide du mode reste inchangée
    assert mode_arg.help == "Choix du pipeline à lancer"
    assert tuple(mode_arg.choices) == ("train", "predict")
    # Prépare la liste attendue des choix de features pour la CLI
    expected_feature_choices = ("fft", "welch", "wavelet", "pca", "csp", "svd")
    # Vérifie que la stratégie de features expose les choix attendus + alias
    assert tuple(feature_arg.choices) == expected_feature_choices
    assert feature_arg.default is argparse.SUPPRESS
    # Vérifie que la réduction de dimension reste optionnelle sans override
    assert tuple(dim_arg.choices) == ("pca", "csp", "svd")
    assert dim_arg.default is argparse.SUPPRESS


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
    assert "usage: python mybci.py <subject> <run> {train,predict}" in stderr


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
    assert called["args"][1].module_args == []


def test_main_relays_feature_strategy_override(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(
        [
            "S001",
            "R01",
            "train",
            "--feature-strategy",
            "wavelet",
            "--scaler",
            "standard",
        ]
    )

    assert exit_code == 0
    assert called["args"][0] == "tpv.train"
    assert called["args"][1].module_args == [
        "--scaler",
        "standard",
        "--feature-strategy",
        "wavelet",
    ]


def test_main_maps_csp_feature_alias_to_dim_method(monkeypatch, capsys):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(["S001", "R01", "train", "--feature-strategy", "CSP"])

    assert exit_code == 0
    assert called["args"][1].module_args == ["--dim-method", "csp"]
    stdout = capsys.readouterr().out
    assert "--feature-strategy csp est interprété comme --dim-method csp" in stdout


# Vérifie que l'alias PCA est redirigé vers --dim-method pca
def test_main_maps_pca_feature_alias_to_dim_method(monkeypatch, capsys):
    # Prépare le conteneur de capture pour l'appel de module
    called: dict[str, Any] = {}

    # Définit un faux appel module pour enregistrer la configuration
    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        # Enregistre les arguments transmis par mybci
        called["args"] = (module_name, config)
        # Retourne un succès pour simuler l'exécution
        return 0

    # Injecte le faux call_module pour isoler la logique CLI
    monkeypatch.setattr(mybci, "_call_module", fake_call)

    # Exécute mybci avec l'alias PCA en stratégie de features
    exit_code = mybci.main(["S001", "R01", "train", "--feature-strategy", "pca"])

    # Vérifie que le code de sortie remonte le succès simulé
    assert exit_code == 0
    # Vérifie que la CLI relaie la méthode de réduction PCA
    assert called["args"][1].module_args == ["--dim-method", "pca"]
    # Capture la sortie standard pour vérifier l'information CLI
    stdout = capsys.readouterr().out
    # Vérifie que le message d'information mentionne l'alias PCA
    assert "--feature-strategy pca est interprété comme --dim-method pca" in stdout


def test_main_warns_when_csp_alias_conflicts_with_dim_method(monkeypatch, capsys):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    exit_code = mybci.main(
        [
            "S001",
            "R01",
            "train",
            "--feature-strategy",
            "csp",
            "--dim-method",
            "pca",
        ]
    )

    assert exit_code == 0
    assert called["args"][1].module_args == ["--dim-method", "csp"]
    stdout = capsys.readouterr().out
    assert "AVERTISSEMENT: --feature-strategy csp force --dim-method csp" in stdout


def test_main_invokes_predict_pipeline(monkeypatch):
    called: dict[str, Any] = {}

    def fake_call(module_name: str, config: mybci.ModuleCallConfig) -> int:
        called["args"] = (module_name, config)
        return 0

    monkeypatch.setattr(mybci, "_call_module", fake_call)

    # Lance mybci avec des identifiants numériques pour valider la normalisation
    exit_code = mybci.main(["2", "3", "predict"])

    # Vérifie que l'exécution renvoie un succès
    assert exit_code == 0
    # Vérifie que le module de prédiction est sélectionné
    assert called["args"][0] == "tpv.predict"
    # Vérifie la normalisation du sujet au format Sxxx
    assert called["args"][1].subject == "S002"
    # Vérifie la normalisation du run au format Rxx
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
            module_args=["--feature-strategy", "welch", "--dim-method", "pca"],
        ),
    )

    assert exit_code == 0
    assert recorded_command == [
        sys.executable,
        "-m",
        "tpv.train",
        "S03",
        "R04",
        "--feature-strategy",
        "welch",
        "--dim-method",
        "pca",
    ]


# Vérifie que main remonte l'échec propagé par un module sous-jacent
def test_main_propagates_module_failure(monkeypatch):
    # Force un code retour non nul pour simuler un pipeline échoué
    monkeypatch.setattr(mybci, "_call_module", lambda *_: MODULE_FAILURE_CODE)
    # Capture le code de sortie afin d'observer la propagation d'erreur
    exit_code = mybci.main(["S001", "R01", "train"])
    # Valide que main renvoie exactement le code d'échec du module
    assert exit_code == MODULE_FAILURE_CODE


def test_report_missing_artifacts_prioritizes_empty_runs(capsys):
    missing_models_by_run = {
        "R04": ["S01", "S02"],
        "R05": ["S02"],
        "R06": ["S01", "S02"],
    }

    mybci._report_missing_artifacts(
        missing_entries=[],
        missing_models_by_run=missing_models_by_run,
        skipped_experiments=[
            mybci.ExperimentDefinition(index=1, run="R04"),
            mybci.ExperimentDefinition(index=3, run="R06"),
        ],
        expected_subject_count=2,
    )

    output = capsys.readouterr().out

    # Vérifie que les runs sans modèles sont listés en premier
    assert "Runs sans aucun modèle disponible: R04, R06" in output
    # Vérifie que les runs partiels conservent le comptage des sujets
    assert "Run R05: modèles manquants pour 1 sujets" in output
    # Vérifie que les expériences ignorées sont présentées avec la séparation attendue
    assert (
        "AVERTISSEMENT: les expériences suivantes ont été ignorées "
        "faute de modèles: 1 (R04), 3 (R06)"
    ) in output
