"""Tests de cohérence des artefacts pour un usage temps-réel."""

# Préserve numpy pour construire des données EEG synthétiques
import numpy as np

# Offre la lecture des artefacts sauvegardés par joblib
import joblib

# Importe la logique d'entraînement pour orchestrer la sauvegarde
from scripts import train as train_cli

# Importe la logique de prédiction pour vérifier les matrices W
from scripts import predict as predict_cli


# Vérifie que la matrice W sauvegardée est cohérente avec la pipeline
def test_w_matrix_matches_pipeline(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère un signal simple pour deux classes opposées
    t = np.arange(0, 1, 1 / sfreq)
    # Crée un essai pour la classe 0 avec énergie sur le premier canal
    class_a = np.stack([np.sin(2 * np.pi * 6 * t), np.zeros_like(t)])
    # Crée un essai pour la classe 1 avec énergie sur le second canal
    class_b = np.stack([np.zeros_like(t), np.sin(2 * np.pi * 10 * t)])
    # Assemble les essais dans un tenseur (essai, canal, temps)
    X = np.stack([class_a, class_b, class_a * 1.1, class_b * 1.1])
    # Construit les labels associés pour chaque essai
    y = np.array([0, 1, 0, 1])
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
    # Entraîne une pipeline pour alimenter la prédiction
    train_cli.run_training(
        subject="S03",
        run="R03",
        classifier="lda",
        scaler=None,
        feature_strategy="fft",
        dim_method="pca",
        n_components=2,
        normalize_features=False,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
        sfreq=sfreq,
    )
    # Charge la pipeline complète pour comparer la matrice interne
    pipeline = joblib.load(artifacts_dir / "S03" / "R03" / "model.joblib")
    # Charge la matrice W sauvegardée par le réducteur
    stored_matrix = joblib.load(artifacts_dir / "S03" / "R03" / "w_matrix.joblib")
    # Recharge le réducteur pour simuler une utilisation temps-réel
    loaded_reducer = predict_cli._load_w_matrix(artifacts_dir / "S03" / "R03" / "w_matrix.joblib")
    # Vérifie que la matrice W rechargée correspond à celle de la pipeline
    assert np.allclose(loaded_reducer.w_matrix, pipeline.named_steps["dimensionality"].w_matrix)
    # Vérifie que la matrice W stockée contient la même structure
    assert np.allclose(stored_matrix["w_matrix"], loaded_reducer.w_matrix)
