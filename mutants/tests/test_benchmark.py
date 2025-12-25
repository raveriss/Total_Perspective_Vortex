# Importe Path pour vérifier les fichiers de sortie générés
from pathlib import Path

# Importe pandas pour inspecter les DataFrames produits
import pandas as pd

# Importe les fonctions de benchmark synthétique
from scripts import benchmark


def test_run_synthetic_benchmark_produces_expected_columns():
    # Exécute un benchmark réduit pour limiter le temps de test
    df = benchmark.run_synthetic_benchmark(n_samples=20, sfreq=64.0)
    # Vérifie que toutes les combinaisons attendues sont présentes
    assert set(df["feature_strategy"]) == {"fft", "wavelet"}
    # Vérifie que les classifieurs testés incluent le bonus centroid
    assert set(df["classifier"]) == {"lda", "logistic", "centroid"}
    # Vérifie que les accuracies sont bornées entre 0 et 1
    assert df["accuracy"].between(0.0, 1.0).all()


def test_save_reports_writes_markdown_and_json(tmp_path):
    # Construit un DataFrame minimal pour tester la sauvegarde
    df = pd.DataFrame(
        {
            "feature_strategy": ["fft"],
            "classifier": ["centroid"],
            "accuracy": [0.95],
            "train_seconds": [0.01],
            "predict_seconds": [0.001],
        }
    )
    # Exécute la sauvegarde vers un répertoire temporaire
    benchmark._save_reports(df, Path(tmp_path))
    # Vérifie que le fichier JSON est présent
    assert (tmp_path / "benchmark_results.json").exists()
    # Vérifie que le fichier Markdown est présent
    assert (tmp_path / "benchmark_results.md").exists()
