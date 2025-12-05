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
    # Construit le répertoire des données pour le sujet S01
    data_dir = tmp_path / "data" / "S01"
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
        subject="S01",
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


# Vérifie que la CLI d'entraînement parsée atteint la sauvegarde attendue
def test_training_cli_main_covers_parser_and_paths(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet S03
    data_dir = tmp_path / "data" / "S03"
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
        "S03",
        "R03",
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
    model_path = artifacts_dir / "S03" / "R03" / "model.joblib"
    # Confirme que le modèle joblib est bien présent après l'appel CLI
    assert model_path.exists()


# Vérifie que la CLI de prédiction couvre le parsing et l'exécution complète
def test_predict_cli_main_covers_parser_and_report(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère des données jouets linéairement séparables
    X, y = _build_toy_dataset(sfreq)
    # Construit le répertoire des données pour le sujet S04
    data_dir = tmp_path / "data" / "S04"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R04_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R04_y.npy", y)
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
        subject="S04",
        run="R04",
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Entraîne une pipeline pour alimenter la prédiction CLI
    train_cli.run_training(request)
    # Construit la liste d'arguments simulant un appel mybci
    argv = [
        "S04",
        "R04",
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
    # Crée un répertoire d'artefacts avec un fichier parasite
    artifacts_dir = tmp_path / "artifacts"
    # Assure la création du répertoire exploré par l'agrégateur
    artifacts_dir.mkdir()
    # Dépose un fichier pour couvrir la branche qui ignore les éléments non dossiers
    stray_file = artifacts_dir / "README.txt"
    # Écrit un contenu pour matérialiser le fichier à ignorer
    stray_file.write_text("placeholder")
    # Relance la découverte pour couvrir le saut sur les fichiers
    discovered_with_file = aggregate_cli._discover_runs(artifacts_dir)
    # Vérifie que le fichier parasite est ignoré et qu'aucun run n'est trouvé
    assert discovered_with_file == []


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
    # Vérifie que le tableau contient l'identifiant du run et du sujet
    assert "R01\tS20" in table
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
