"""Tests du workflow d'entraînement et de prédiction sur données jouets."""

# Préserve numpy pour construire des données EEG synthétiques
# Offre la lecture des artefacts sauvegardés par joblib
# Permet de relire le CSV produit par l'agrégateur
import csv

# Permet de valider la sérialisation JSON des rapports
import json

# Charge les artefacts joblib générés durant les tests
import joblib
import numpy as np

# Garantit l'accès aux fixtures temporaires et assertions
import pytest

# Importe la logique de prédiction pour vérifier l'accuracy
# Importe la logique d'entraînement pour orchestrer la sauvegarde
# Importe l'agrégation pour résumer les accuracies multi-runs
from scripts import aggregate_accuracy as aggregate_cli
from scripts import aggregate_scores as aggregate_scores_cli
from scripts import predict as predict_cli
from scripts import train as train_cli

# Fige le seuil minimal d'accuracy exigé pour valider le run jouet
EXPECTED_MIN_ACCURACY = 0.9


# Construit un jeu de données EEG linéairement séparable
def _build_toy_dataset(sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Génère des signaux distincts par classe pour tester la pipeline."""

    # Crée l'axe temporel pour simuler une seconde d'enregistrement
    t = np.arange(0, 1, 1 / sfreq)
    # Génère un signal theta sur le premier canal pour la classe A
    theta = np.sin(2 * np.pi * 6 * t)
    # Génère un signal alpha sur le second canal pour la classe B
    alpha = np.sin(2 * np.pi * 10 * t)
    # Construit la matrice X en empilant des essais séparables
    class_a = np.stack([theta, np.zeros_like(theta)])
    # Construit la matrice X pour la classe B avec énergie sur le second canal
    class_b = np.stack([np.zeros_like(alpha), alpha])
    # Assemble plusieurs essais pour renforcer la séparation
    trials = [
        class_a,
        class_b,
        class_a * 1.1,
        class_b * 1.1,
        class_a * 0.9,
        class_b * 0.9,
    ]
    # Convertit la liste en tenseur (essai, canal, temps)
    X = np.stack(trials)
    # Construit les labels correspondants pour chaque essai
    y = np.array([0, 1, 0, 1, 0, 1])
    # Retourne les features et labels synthétiques
    return X, y


# Vérifie que l'entraînement produit des artefacts complets
@pytest.mark.parametrize("scaler_option", [None, "standard"])
def test_training_saves_artifacts(tmp_path, scaler_option):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet S001
    data_dir = tmp_path / "data" / "S001"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R01_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R01_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la configuration alignée sur la CLI pour l'entraînement
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=scaler_option,
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject="S001",
        run="R01",
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Exécute l'entraînement complet et récupère les chemins sauvegardés
    result = train_cli.run_training(request)
    # Vérifie que le modèle joblib a bien été sauvegardé
    assert result["model_path"].exists()
    # Vérifie que la matrice W est bien persistée pour le streaming
    assert result["w_matrix_path"].exists()
    # Charge le contenu de la matrice W pour s'assurer qu'il n'est pas vide
    matrix_payload = joblib.load(result["w_matrix_path"])
    # Vérifie que la matrice de projection possède des coefficients
    assert matrix_payload["w_matrix"].size > 0


# Vérifie que le classifieur centroid atteint une accuracy élevée sur le jeu jouet
def test_centroid_classifier_reaches_expected_accuracy(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet S99
    data_dir = tmp_path / "data" / "S99"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R99_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R99_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la configuration alignée sur la CLI pour tester le classifieur léger
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
        n_components=2,
        classifier="centroid",
        scaler="standard",
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject="S99",
        run="R99",
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Exécute l'entraînement complet et récupère les chemins sauvegardés
    result = train_cli.run_training(request)
    # Charge le modèle pour vérifier la performance sur le même jeu
    model = joblib.load(result["model_path"])
    # Prédit sur les données jouets pour évaluer l'accuracy
    predictions = model.predict(X)
    # Vérifie que le classifieur léger atteint une accuracy quasi parfaite
    assert (predictions == y).mean() >= EXPECTED_MIN_ACCURACY


# Vérifie que la prédiction restitue un rapport structuré cohérent
def test_prediction_report(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet S02
    data_dir = tmp_path / "data" / "S02"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R02_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R02_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la configuration alignée sur la CLI pour l'entraînement
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject="S02",
        run="R02",
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Entraîne une pipeline pour alimenter la prédiction
    train_cli.run_training(request)
    # Évalue le run entraîné pour produire un rapport
    result = predict_cli.evaluate_run("S02", "R02", tmp_path / "data", artifacts_dir)
    # Construit le rapport agrégé par run et sujet
    report = predict_cli.build_report(result)
    # Vérifie que l'accuracy par run est présente et dépasse le seuil cible
    assert report["by_run"]["R02"] > EXPECTED_MIN_ACCURACY
    # Vérifie que l'accuracy par sujet reflète la même valeur
    assert report["by_subject"]["S02"] == report["by_run"]["R02"]
    # Vérifie que l'accuracy globale correspond à la mesure du run
    assert pytest.approx(report["global"], rel=0.01) == report["by_run"]["R02"]


def test_build_report_tracks_confusion_and_reports_reference():
    # Prépare un résultat simulé minimal pour l'agrégateur
    reports = {
        "confusion": [[5, 1], [2, 6]],
        "details": {"note": "toy-evaluation"},
    }
    # Construit un dictionnaire identique à ce que renvoie evaluate_run
    result = {
        "accuracy": 0.625,
        "run": "R77",
        "subject": "S77",
        "reports": reports,
    }
    # Construit le rapport structuré pour la CLI
    report = predict_cli.build_report(result)
    # Vérifie la propagation des accuracies dans les trois agrégations
    assert report["by_run"] == {"R77": result["accuracy"]}
    assert report["by_subject"] == {"S77": result["accuracy"]}
    assert report["global"] == result["accuracy"]
    # Vérifie que la confusion est bien exposée et partagée sans copie
    assert report["confusion_matrix"] is reports["confusion"]
    assert report["reports"] is reports


# Vérifie que la CLI d'entraînement parsée atteint la sauvegarde attendue
def test_training_cli_main_covers_parser_and_paths(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet normalisé S003
    data_dir = tmp_path / "data" / "S003"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R03_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R03_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la liste d'arguments simulant un appel mybci
    argv = [
        "3",
        "3",
        "--classifier",
        "lda",
        "--feature-strategy",
        "fft",
        "--dim-method",
        "pca",
        "--n-components",
        "2",
        "--data-dir",
        str(tmp_path / "data"),
        "--artifacts-dir",
        str(artifacts_dir),
        "--sfreq",
        str(sfreq),
        "--scaler",
        "none",
    ]
    # Exécute la CLI d'entraînement et récupère le code de sortie
    exit_code = train_cli.main(argv)
    # Vérifie que la CLI retourne un succès standard
    assert exit_code == 0
    # Construit le chemin du modèle pour valider la création d'artefacts
    model_path = artifacts_dir / "S003" / "R03" / "model.joblib"
    # Confirme que le modèle joblib est bien présent après l'appel CLI
    assert model_path.exists()


# Vérifie que la CLI de prédiction couvre le parsing et l'exécution complète
def test_predict_cli_main_covers_parser_and_report(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le sujet numérique pour valider la normalisation CLI
    subject = "4"
    # Construit le run numérique pour valider la normalisation CLI
    run = "4"
    # Calcule l'identifiant normalisé du sujet pour les chemins disque
    normalized_subject = "S004"
    # Calcule l'identifiant normalisé du run pour les chemins disque
    normalized_run = "R04"
    # Construit le répertoire des données pour le sujet normalisé
    data_dir = tmp_path / "data" / normalized_subject
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / f"{normalized_run}_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / f"{normalized_run}_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la configuration alignée sur la CLI pour l'entraînement
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject=normalized_subject,
        run=normalized_run,
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Entraîne une pipeline pour alimenter la prédiction CLI
    train_cli.run_training(request)
    # Construit la liste d'arguments simulant un appel mybci
    argv = [
        subject,
        run,
        "--data-dir",
        str(tmp_path / "data"),
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    # Exécute la CLI de prédiction et récupère le code de sortie
    exit_code = predict_cli.main(argv)
    # Vérifie que la CLI retourne un succès standard
    assert exit_code == 0


# Vérifie que l'agrégation restitue les métriques multi-runs synthétiques
def test_accuracy_aggregation_across_runs(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire racine des données temporaires
    data_dir = tmp_path / "data"
    # Construit le répertoire racine des artefacts temporaires
    artifacts_dir = tmp_path / "artifacts"
    # Déclare la configuration de pipeline identique pour tous les runs
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Liste les couples (sujet, run) à entraîner pour l'agrégation
    runs = [("S10", "R01"), ("S10", "R02"), ("S11", "R01")]
    # Entraîne chaque run pour générer les artefacts attendus
    for subject, run in runs:
        # Construit le répertoire propre au sujet courant
        subject_dir = data_dir / subject
        # Assure la création du répertoire avant les sauvegardes numpy
        subject_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarde les features au format attendu par la CLI
        np.save(subject_dir / f"{run}_X.npy", X)
        # Sauvegarde les labels au format attendu par la CLI
        np.save(subject_dir / f"{run}_y.npy", y)
        # Regroupe les paramètres d'entraînement dans une requête dédiée
        request = train_cli.TrainingRequest(
            subject=subject,
            run=run,
            pipeline_config=config,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
        )
        # Exécute l'entraînement et la sauvegarde des artefacts
        train_cli.run_training(request)
    # Agrège les accuracies calculées à partir des artefacts générés
    report = aggregate_cli.aggregate_accuracies(data_dir, artifacts_dir)
    # Vérifie que tous les runs sont bien présents dans le rapport
    assert set(report["by_run"].keys()) == {"S10/R01", "S10/R02", "S11/R01"}
    # Vérifie que chaque run dépasse le seuil minimal d'accuracy
    for accuracy in report["by_run"].values():
        # Garantit une performance élevée sur le dataset synthétique
        assert accuracy > EXPECTED_MIN_ACCURACY
    # Vérifie que les agrégations par sujet couvrent les deux identifiants
    assert set(report["by_subject"].keys()) == {"S10", "S11"}
    # Calcule la moyenne attendue des accuracies individuelles
    expected_global = float(np.mean(list(report["by_run"].values())))
    # Vérifie que l'accuracy globale reflète la moyenne des runs
    assert pytest.approx(report["global"], rel=0.01) == expected_global


# Vérifie que l'agrégation gère l'absence d'artefacts sans planter
def test_aggregate_accuracy_handles_missing_artifacts(tmp_path):
    # Construit un chemin d'artefacts inexistant pour simuler l'absence de run
    missing_artifacts_dir = tmp_path / "artifacts_missing"
    # Appelle la découverte des runs pour vérifier le retour vide
    discovered = aggregate_cli._discover_runs(missing_artifacts_dir)
    # Vérifie qu'aucun run n'est détecté lorsque le dossier manque
    assert discovered == []
    # Calcule le rapport complet pour verrouiller le cas sans run
    report = aggregate_cli.aggregate_accuracies(
        tmp_path / "data", missing_artifacts_dir
    )
    # Vérifie que l'agrégation par run est vide sans artefact
    assert report["by_run"] == {}
    # Vérifie que l'agrégation par sujet est vide sans artefact
    assert report["by_subject"] == {}
    # Vérifie que l'accuracy globale retombe à zéro sans run
    assert report["global"] == 0.0
    # Crée un répertoire d'artefacts avec un fichier parasite
    artifacts_dir = tmp_path / "artifacts"
    # Assure la création du répertoire exploré par l'agrégateur
    artifacts_dir.mkdir()
    # Dépose un fichier pour couvrir la branche qui ignore les éléments non dossiers
    stray_file = artifacts_dir / "README.txt"
    # Écrit un contenu pour matérialiser le fichier à ignorer
    stray_file.write_text("placeholder")
    # Crée un sujet valide pour vérifier que la découverte ne s'arrête pas
    subject_dir = artifacts_dir / "S20"
    # Crée un run valide pour couvrir la détection des modèles persistés
    run_dir = subject_dir / "R01"
    # Assure la création du dossier de run attendu par l'agrégateur
    run_dir.mkdir(parents=True)
    # Crée un modèle factice pour activer l'ajout du couple (sujet, run)
    (run_dir / "model.joblib").write_text("stub", encoding="utf-8")
    # Relance la découverte pour couvrir le saut sur les fichiers
    discovered_with_file = aggregate_cli._discover_runs(artifacts_dir)
    # Vérifie que le fichier parasite est ignoré et que le run reste détecté
    assert discovered_with_file == [("S20", "R01")]


# Vérifie le parsing CLI, le formatage et l'affichage du tableau agrégé
def test_aggregate_accuracy_cli_parser_and_table(tmp_path, capsys):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire racine des données temporaires
    data_dir = tmp_path / "data"
    # Construit le répertoire racine des artefacts temporaires
    artifacts_dir = tmp_path / "artifacts"
    # Construit le répertoire dédié au sujet S20
    subject_dir = data_dir / "S20"
    # Assure la création du répertoire avant les sauvegardes numpy
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(subject_dir / "R01_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(subject_dir / "R01_y.npy", y)
    # Construit la configuration alignée sur la CLI pour l'entraînement
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject="S20",
        run="R01",
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    # Exécute l'entraînement et la sauvegarde des artefacts
    train_cli.run_training(request)
    # Construit un parser via la fonction dédiée pour couvrir les options
    parser = aggregate_cli.build_parser()
    # Parse les arguments fournis pour s'assurer de la configuration
    args = parser.parse_args(
        ["--data-dir", str(data_dir), "--artifacts-dir", str(artifacts_dir)]
    )
    # Vérifie que le parser restitue les chemins attendus
    assert args.data_dir == data_dir
    # Vérifie que le parser retourne le répertoire d'artefacts fourni
    assert args.artifacts_dir == artifacts_dir
    # Calcule un rapport complet en réutilisant les artefacts générés
    report = aggregate_cli.aggregate_accuracies(args.data_dir, args.artifacts_dir)
    # Formate le tableau pour vérifier l'inclusion du run et du sujet
    table = aggregate_cli.format_accuracy_table(report)
    # Découpe le tableau en lignes pour verrouiller le format tabulaire
    lines = table.splitlines()
    # Verrouille le nombre de lignes attendu pour 1 run et 1 sujet
    assert len(lines) == 5
    # Verrouille l'en-tête pour détecter toute mutation de libellé
    assert lines[0] == "Run\tSubject\tAccuracy"
    # Découpe la ligne de run pour verrouiller l'ordre et les séparateurs
    run_parts = lines[1].split("\t")
    # Verrouille le nombre de colonnes pour la ligne de run
    assert len(run_parts) == 3
    # Verrouille l'identifiant du run pour tuer les mutations de texte
    assert run_parts[0] == "R01"
    # Verrouille l'identifiant du sujet pour tuer les mutations de texte
    assert run_parts[1] == "S20"
    # Valide que l'accuracy est un float convertible et borné
    run_accuracy = float(run_parts[2])
    # Assure une accuracy normalisée entre 0 et 1
    assert 0.0 <= run_accuracy <= 1.0
    # Verrouille le séparateur de section pour détecter toute mutation
    assert lines[2] == "Subject\tMean Accuracy"
    # Découpe la ligne moyenne du sujet pour valider le format attendu
    subject_parts = lines[3].split("\t")
    # Verrouille le nombre de colonnes pour la ligne moyenne sujet
    assert len(subject_parts) == 2
    # Verrouille l'identifiant du sujet agrégé pour tuer les mutations de texte
    assert subject_parts[0] == "S20"
    # Valide que la moyenne sujet est un float convertible et borné
    subject_accuracy = float(subject_parts[1])
    # Assure une accuracy moyenne normalisée entre 0 et 1
    assert 0.0 <= subject_accuracy <= 1.0
    # Découpe la ligne globale pour verrouiller le libellé final
    global_parts = lines[4].split("\t")
    # Verrouille le nombre de colonnes pour la ligne globale
    assert len(global_parts) == 2
    # Verrouille le libellé Global pour tuer les mutations de texte
    assert global_parts[0] == "Global"
    # Valide que la moyenne globale est un float convertible et borné
    global_accuracy = float(global_parts[1])
    # Assure une accuracy globale normalisée entre 0 et 1
    assert 0.0 <= global_accuracy <= 1.0
    # Exécute la CLI principale pour couvrir l'impression du tableau
    exit_code = aggregate_cli.main(
        ["--data-dir", str(data_dir), "--artifacts-dir", str(artifacts_dir)]
    )
    # Vérifie que la CLI retourne un code de succès
    assert exit_code == 0
    # Capture la sortie standard pour contrôler la présence du tableau
    captured = capsys.readouterr()
    # Vérifie que la sortie inclut l'accuracy globale formatée
    assert "Global" in captured.out


def test_aggregate_accuracy_parser_description_defaults_and_help_contract() -> None:
    """Le parser doit exposer description, defaults et help stables pour la CLI."""

    # Construit le parser de la commande pour inspecter sa configuration
    parser = aggregate_cli.build_parser()
    # Verrouille la description pour tuer les mutations sur le texte ou None
    assert (
        parser.description
        == "Agrège les accuracies par run, sujet et global à partir des artefacts"
    )
    # Parse sans arguments pour valider les valeurs par défaut
    args = parser.parse_args([])
    # Verrouille le default data_dir pour tuer default=None ou default supprimé
    assert args.data_dir == aggregate_cli.DEFAULT_DATA_DIR
    # Verrouille le default artifacts_dir pour tuer default=None ou default supprimé
    assert args.artifacts_dir == aggregate_cli.DEFAULT_ARTIFACTS_DIR

    # Récupère l'action associée à --data-dir pour vérifier default/help
    data_action = next(
        action for action in parser._actions if action.dest == "data_dir"
    )
    # Verrouille le default de l'option pour tuer les mutations sur default
    assert data_action.default == aggregate_cli.DEFAULT_DATA_DIR
    # Verrouille le help de l'option pour tuer help=None ou help supprimé
    assert (
        data_action.help
        == "Répertoire racine contenant les matrices numpy utilisées pour le scoring"
    )

    # Récupère l'action associée à --artifacts-dir pour vérifier default/help
    artifacts_action = next(
        action for action in parser._actions if action.dest == "artifacts_dir"
    )
    # Verrouille le default de l'option pour tuer les mutations sur default
    assert artifacts_action.default == aggregate_cli.DEFAULT_ARTIFACTS_DIR
    # Verrouille le help de l'option pour tuer help=None ou help supprimé
    assert (
        artifacts_action.help
        == "Répertoire racine où sont stockés les modèles et matrices W"
    )


# Vérifie que l'agrégateur exporte CSV/JSON et valide les seuils
def test_aggregate_scores_exports_files_and_thresholds(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire racine des données temporaires
    data_dir = tmp_path / "data"
    # Construit le répertoire racine des artefacts temporaires
    artifacts_dir = tmp_path / "artifacts"
    # Déclare la configuration de pipeline identique pour tous les runs
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Liste les couples (sujet, run) à entraîner pour le reporting
    runs = [("S30", "R01"), ("S31", "R01")]
    # Entraîne chaque run pour générer les artefacts attendus
    for subject, run in runs:
        # Construit le répertoire propre au sujet courant
        subject_dir = data_dir / subject
        # Assure la création du répertoire avant les sauvegardes numpy
        subject_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarde les features au format attendu par la CLI
        np.save(subject_dir / f"{run}_X.npy", X)
        # Sauvegarde les labels au format attendu par la CLI
        np.save(subject_dir / f"{run}_y.npy", y)
        # Regroupe les paramètres d'entraînement dans une requête dédiée
        request = train_cli.TrainingRequest(
            subject=subject,
            run=run,
            pipeline_config=config,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
        )
        # Exécute l'entraînement et la sauvegarde des artefacts
        train_cli.run_training(request)
    # Agrège les accuracies calculées à partir des artefacts générés
    report = aggregate_scores_cli.aggregate_scores(data_dir, artifacts_dir)
    # Vérifie que les deux runs apparaissent dans le rapport agrégé
    assert {(entry["subject"], entry["run"]) for entry in report["runs"]} == set(runs)
    # Vérifie que tous les runs dépassent les seuils minimum et cible
    for entry in report["runs"]:
        # Garantit une performance élevée sur le dataset synthétique
        assert entry["accuracy"] > EXPECTED_MIN_ACCURACY
        # Confirme que le seuil minimal est franchi
        assert entry["meets_minimum"] is True
        # Confirme que la cible ambitieuse est atteinte
        assert entry["meets_target"] is True
    # Vérifie que chaque sujet possède une moyenne cohérente
    for subject_entry in report["subjects"]:
        # Confirme que la moyenne par sujet dépasse le seuil cible
        assert subject_entry["accuracy"] > EXPECTED_MIN_ACCURACY
        # Confirme que les drapeaux reflètent la performance élevée
        assert subject_entry["meets_target"] is True
    # Vérifie que l'accuracy globale respecte les objectifs
    assert report["global"]["accuracy"] > EXPECTED_MIN_ACCURACY
    # Confirme que les drapeaux globaux signalent la conformité
    assert report["global"]["meets_target"] is True
    # Prépare les chemins de sortie pour le CSV et le JSON
    csv_path = tmp_path / "reports" / "scores.csv"
    json_path = tmp_path / "reports" / "scores.json"
    # Écrit le CSV consolidé pour inspection manuelle
    aggregate_scores_cli.write_csv(report, csv_path)
    # Écrit le JSON consolidé pour réutilisation en CI
    aggregate_scores_cli.write_json(report, json_path)
    # Vérifie que les fichiers sont bien créés
    assert csv_path.exists()
    assert json_path.exists()
    # Relit le CSV pour valider le contenu et le type global
    with csv_path.open() as handle:
        # Charge les lignes du rapport CSV
        rows = list(csv.DictReader(handle))
    # Vérifie la présence d'une ligne globale avec drapeaux positifs
    assert any(
        row["type"] == "global" and row["meets_target"] == "True" for row in rows
    )
    # Relit le JSON pour valider la structure sérialisée
    with json_path.open() as handle:
        # Charge le contenu JSON écrit par l'agrégateur
        serialized = json.load(handle)
    # Vérifie que les drapeaux JSON reflètent la conformité aux seuils
    assert serialized["global"]["meets_minimum"] is True


# Vérifie que le parser fournit des valeurs par défaut exploitables
def test_aggregate_scores_parser_and_missing_artifacts(tmp_path):
    # Construit le parser pour couvrir la configuration des options
    parser = aggregate_scores_cli.build_parser()
    # Parse les arguments vides pour vérifier les valeurs par défaut
    args = parser.parse_args([])
    # Vérifie que le répertoire de données par défaut est bien exposé
    assert args.data_dir == aggregate_scores_cli.DEFAULT_DATA_DIR
    # Vérifie que le répertoire d'artefacts par défaut est bien exposé
    assert args.artifacts_dir == aggregate_scores_cli.DEFAULT_ARTIFACTS_DIR
    # Vérifie que l'export CSV est désactivé par défaut
    assert args.csv_output is None
    # Vérifie que l'export JSON est désactivé par défaut
    assert args.json_output is None
    # Construit un répertoire d'artefacts inexistant pour déclencher le cas absent
    artifacts_dir = tmp_path / "artifacts_missing"
    # Calcule un rapport en absence totale d'artefacts sauvegardés
    report = aggregate_scores_cli.aggregate_scores(tmp_path / "data", artifacts_dir)
    # Vérifie que la liste des runs est vide lorsque rien n'existe
    assert report["runs"] == []
    # Vérifie que la moyenne par sujet est vide sans run détecté
    assert report["subjects"] == []
    # Vérifie que l'accuracy globale retombe à zéro sans artefact
    assert report["global"]["accuracy"] == 0.0


def test_score_run_applies_thresholds(monkeypatch, tmp_path):
    """Valide les drapeaux meets_minimum/target sur accuracy contrôlée."""

    # Capture les appels evaluate_run pour valider la propagation des chemins
    calls: list[tuple[str, str, object, object]] = []
    # Prépare deux accuracies sous seuil et sous la cible
    accuracies = [0.5, 0.8]

    # Déclare un stub evaluate_run pour renvoyer des accuracies maîtrisées
    def fake_evaluate_run(subject: str, run: str, data_dir, artifacts_dir):
        # Enregistre les paramètres pour vérifier l'ordonnancement
        calls.append((subject, run, data_dir, artifacts_dir))
        # Dépile l'accuracy suivante pour contrôler les drapeaux
        return {"accuracy": accuracies.pop(0)}

    # Remplace evaluate_run par un stub pour isoler _score_run du disque
    monkeypatch.setattr(
        aggregate_scores_cli.predict_cli, "evaluate_run", fake_evaluate_run
    )
    # Construit un répertoire de données temporaire pour passer à _score_run
    data_dir = tmp_path / "data"
    # Construit un répertoire d'artefacts temporaire pour passer à _score_run
    artifacts_dir = tmp_path / "artifacts"
    # Évalue un run avec une accuracy sous le seuil minimal attendu
    below_entry = aggregate_scores_cli._score_run("S10", "R01", data_dir, artifacts_dir)
    # Vérifie que l'accuracy propagée correspond à la valeur injectée
    assert below_entry["accuracy"] == 0.5
    # Vérifie que le drapeau minimum est désactivé sous le seuil requis
    assert below_entry["meets_minimum"] is False
    # Vérifie que le drapeau cible est désactivé sous la cible ambitieuse
    assert below_entry["meets_target"] is False
    # Évalue un second run avec une accuracy au-dessus du minimum
    above_entry = aggregate_scores_cli._score_run("S10", "R02", data_dir, artifacts_dir)
    # Vérifie que l'accuracy reflète la deuxième valeur injectée
    assert above_entry["accuracy"] == 0.8
    # Vérifie que le drapeau minimum se déclenche dès le seuil atteint
    assert above_entry["meets_minimum"] is True
    # Vérifie que le drapeau cible reste faux sous le seuil cible
    assert above_entry["meets_target"] is False
    # Vérifie que evaluate_run reçoit les deux appels avec les bons chemins
    assert calls == [
        ("S10", "R01", data_dir, artifacts_dir),
        ("S10", "R02", data_dir, artifacts_dir),
    ]


def test_aggregate_scores_discover_runs_filters_invalid_entries(tmp_path):
    # Construit un chemin inexistant pour simuler l'absence d'artefacts
    missing_artifacts_dir = tmp_path / "artifacts_missing"
    # Vérifie que la découverte retourne une liste vide quand rien n'existe
    assert aggregate_scores_cli._discover_runs(missing_artifacts_dir) == []
    # Construit le répertoire racine des artefacts temporaires
    artifacts_dir = tmp_path / "artifacts"
    # Assure la création du répertoire pour déposer plusieurs cas
    artifacts_dir.mkdir()
    # Ajoute un fichier parasite pour vérifier que l'exploration continue
    stray_file = artifacts_dir / "README.txt"
    # Écrit un contenu factice pour matérialiser le fichier à ignorer
    stray_file.write_text("placeholder", encoding="utf-8")
    # Construit un sujet sans modèle pour vérifier l'exclusion
    subject_without_model = artifacts_dir / "S10"
    # Construit un run dépourvu de modèle entraîné
    run_without_model = subject_without_model / "R01"
    # Crée l'arborescence du run sans déposer de modèle
    run_without_model.mkdir(parents=True)
    # Construit un sujet valide pour vérifier la détection positive
    subject_with_model = artifacts_dir / "S11"
    # Construit le run contenant un modèle sérialisé
    run_with_model = subject_with_model / "R02"
    # Crée l'arborescence du run qui sera considéré éligible
    run_with_model.mkdir(parents=True)
    # Dépose un modèle factice pour activer la sélection du run
    (run_with_model / "model.joblib").write_text("stub", encoding="utf-8")
    # Découvre les runs après avoir mélangé fichiers et dossiers
    discovered_runs = aggregate_scores_cli._discover_runs(artifacts_dir)
    # Vérifie que seul le run muni d'un modèle est retenu
    assert discovered_runs == [("S11", "R02")]


def test_aggregate_scores_aggregates_stubbed_runs(monkeypatch, tmp_path):
    # Force la découverte de plusieurs runs répartis sur deux sujets
    stubbed_runs = [("S10", "R01"), ("S10", "R02"), ("S11", "R01")]
    monkeypatch.setattr(
        aggregate_scores_cli, "_discover_runs", lambda artifacts_dir: stubbed_runs
    )
    # Prépare des accuracies distinctes pour valider les moyennes
    accuracies = iter([0.4, 0.8, 0.6])

    # Stub _score_run pour retourner les accuracies contrôlées
    def fake_score_run(subject, run, data_dir, artifacts_dir):
        accuracy = next(accuracies)
        return {
            "subject": subject,
            "run": run,
            "accuracy": accuracy,
            "meets_minimum": accuracy >= aggregate_scores_cli.MINIMUM_ACCURACY,
            "meets_target": accuracy >= aggregate_scores_cli.TARGET_ACCURACY,
        }

    monkeypatch.setattr(aggregate_scores_cli, "_score_run", fake_score_run)
    # Exécute l'agrégation avec des chemins temporaires neutres
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    report = aggregate_scores_cli.aggregate_scores(data_dir, artifacts_dir)
    # Vérifie que tous les runs stubés apparaissent dans le rapport
    assert [(entry["subject"], entry["run"]) for entry in report["runs"]] == [
        ("S10", "R01"),
        ("S10", "R02"),
        ("S11", "R01"),
    ]
    # Vérifie que les accuracies sont conservées dans l'ordre des runs
    assert [entry["accuracy"] for entry in report["runs"]] == [
        pytest.approx(0.4),
        pytest.approx(0.8),
        pytest.approx(0.6),
    ]
    # Vérifie les moyennes par sujet (S10: 0.6, S11: 0.6)
    subject_means = {
        entry["subject"]: entry["accuracy"] for entry in report["subjects"]
    }
    assert subject_means == {
        "S10": pytest.approx(0.6),
        "S11": pytest.approx(0.6),
    }
    # Vérifie que les drapeaux reflètent les seuils du module agrégateur
    assert all(entry["meets_minimum"] is False for entry in report["subjects"])
    assert all(entry["meets_target"] is False for entry in report["subjects"])
    # Vérifie la moyenne globale sur l'ensemble des runs
    assert report["global"]["accuracy"] == pytest.approx(0.6)
    assert report["global"]["meets_minimum"] is False
    assert report["global"]["meets_target"] is False


def test_aggregate_scores_returns_zero_when_no_stubbed_runs(monkeypatch, tmp_path):
    # Force la découverte à renvoyer aucun run
    monkeypatch.setattr(aggregate_scores_cli, "_discover_runs", lambda _: [])
    # S'assure que _score_run n'est jamais invoqué dans ce scénario
    monkeypatch.setattr(
        aggregate_scores_cli,
        "_score_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_score_run should not be called")
        ),
    )
    # Exécute l'agrégation et vérifie le rapport vide
    report = aggregate_scores_cli.aggregate_scores(
        tmp_path / "data", tmp_path / "artifacts"
    )
    assert report["runs"] == []
    assert report["subjects"] == []
    assert report["global"]["accuracy"] == 0.0
    assert report["global"]["meets_minimum"] is False
    assert report["global"]["meets_target"] is False


# Vérifie que la CLI principale sérialise les rapports CSV et JSON
def test_aggregate_scores_main_writes_requested_outputs(tmp_path, monkeypatch):
    # Définit un rapport synthétique pour limiter les calculs en test
    stub_report = {
        "runs": [
            {
                "subject": "S50",
                "run": "R01",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "subjects": [
            {
                "subject": "S50",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "global": {"accuracy": 0.8, "meets_minimum": True, "meets_target": True},
    }
    # Force l'agrégateur à renvoyer le rapport synthétique préconstruit
    monkeypatch.setattr(
        aggregate_scores_cli, "aggregate_scores", lambda *_: stub_report
    )
    # Construit un répertoire d'artefacts contenant un fichier non dossier
    artifacts_dir = tmp_path / "artifacts"
    # Crée le répertoire d'artefacts pour tester l'exploration
    artifacts_dir.mkdir()
    # Ajoute un fichier factice pour couvrir le parcours qui ignore les fichiers
    (artifacts_dir / "README.txt").write_text("placeholder")
    # Vérifie que l'exploration ignore les chemins qui ne sont pas des dossiers
    assert aggregate_scores_cli._discover_runs(artifacts_dir) == []
    # Prépare les chemins de sortie demandés à la CLI principale
    csv_path = tmp_path / "reports" / "scores.csv"
    # Prépare le chemin JSON pour vérifier la sérialisation secondaire
    json_path = tmp_path / "reports" / "scores.json"
    # Construit les arguments CLI incluant les chemins de sortie
    argv = [
        "--data-dir",
        str(tmp_path / "data"),
        "--artifacts-dir",
        str(artifacts_dir),
        "--csv-output",
        str(csv_path),
        "--json-output",
        str(json_path),
    ]
    # Exécute la CLI principale pour générer les fichiers attendus
    exit_code = aggregate_scores_cli.main(argv)
    # Vérifie que la CLI signale un succès standard
    assert exit_code == 0
    # Vérifie que le fichier CSV a bien été écrit par la CLI
    assert csv_path.exists()
    # Vérifie que le fichier JSON a bien été écrit par la CLI
    assert json_path.exists()


# Vérifie que la CLI n'écrit rien lorsque aucun chemin de sortie n'est fourni
def test_aggregate_scores_main_skips_outputs_when_not_requested(tmp_path, monkeypatch):
    # Prépare un rapport synthétique pour limiter l'exécution en test
    stub_report = {
        "runs": [],
        "subjects": [],
        "global": {"accuracy": 0.0, "meets_minimum": False, "meets_target": False},
    }
    # Capture les appels d'écriture pour vérifier les garde-fous
    writes = {"csv": 0, "json": 0}
    # Force l'agrégateur à renvoyer le rapport synthétique préconstruit
    monkeypatch.setattr(
        aggregate_scores_cli, "aggregate_scores", lambda *_: stub_report
    )
    # Remplace l'écriture CSV pour compter les appels potentiels
    monkeypatch.setattr(
        aggregate_scores_cli,
        "write_csv",
        lambda report, path: writes.update({"csv": writes["csv"] + 1}),
    )
    # Remplace l'écriture JSON pour compter les appels potentiels
    monkeypatch.setattr(
        aggregate_scores_cli,
        "write_json",
        lambda report, path: writes.update({"json": writes["json"] + 1}),
    )
    # Construit les arguments sans options d'export pour tester les garde-fous
    argv = [
        "--data-dir",
        str(tmp_path / "data"),
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
    ]
    # Exécute la CLI pour vérifier qu'aucun export n'est déclenché
    exit_code = aggregate_scores_cli.main(argv)
    # Vérifie que la CLI signale un succès standard
    assert exit_code == 0
    # Vérifie que l'écriture CSV n'a pas été appelée
    assert writes["csv"] == 0
    # Vérifie que l'écriture JSON n'a pas été appelée
    assert writes["json"] == 0


def test_aggregate_scores_main_passes_cli_arguments(tmp_path, monkeypatch):
    # Prépare un rapport synthétique pour les écritures de sortie
    stub_report: dict[str, object] = {"runs": [], "subjects": [], "global": {}}
    # Capture les paramètres transmis à aggregate_scores et aux writers
    calls: dict[str, tuple[object, object] | tuple[object, object, object]] = {}

    # Injecte un agrégateur espion pour vérifier les chemins fournis
    def recording_aggregate(data_dir, artifacts_dir):
        calls["aggregate"] = (data_dir, artifacts_dir)
        return stub_report

    # Remplace l'agrégateur par la version espion
    monkeypatch.setattr(aggregate_scores_cli, "aggregate_scores", recording_aggregate)
    # Capture les écritures CSV en enregistrant le rapport et le chemin
    monkeypatch.setattr(
        aggregate_scores_cli,
        "write_csv",
        lambda report, path: calls.update({"csv": (report, path)}),
    )
    # Capture les écritures JSON en enregistrant le rapport et le chemin
    monkeypatch.setattr(
        aggregate_scores_cli,
        "write_json",
        lambda report, path: calls.update({"json": (report, path)}),
    )
    # Construit les chemins de données et d'artefacts attendus
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    # Prépare les destinations de sortie pour le CSV et le JSON
    csv_path = tmp_path / "scores.csv"
    json_path = tmp_path / "scores.json"
    # Exécute la CLI avec des arguments explicites
    exit_code = aggregate_scores_cli.main(
        [
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--csv-output",
            str(csv_path),
            "--json-output",
            str(json_path),
        ]
    )
    # Vérifie que la CLI termine avec succès
    assert exit_code == 0
    # Vérifie que aggregate_scores a reçu les deux chemins fournis
    assert calls["aggregate"] == (data_dir, artifacts_dir)
    # Vérifie que write_csv reçoit le rapport calculé et le chemin CSV
    assert calls["csv"] == (stub_report, csv_path)
    # Vérifie que write_json reçoit le rapport calculé et le chemin JSON
    assert calls["json"] == (stub_report, json_path)
