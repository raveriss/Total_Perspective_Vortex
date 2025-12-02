"""Tests du workflow d'entraînement et de prédiction sur données jouets."""

# Préserve numpy pour construire des données EEG synthétiques
# Offre la lecture des artefacts sauvegardés par joblib
import joblib
import numpy as np

# Garantit l'accès aux fixtures temporaires et assertions
import pytest

# Importe la logique de prédiction pour vérifier l'accuracy
# Importe la logique d'entraînement pour orchestrer la sauvegarde
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
