# Importe Path pour vérifier les fichiers de sortie générés
import importlib

# Importe inspect pour verrouiller les defaults via introspection
import inspect
import sys
from pathlib import Path

# Importe pandas pour inspecter les DataFrames produits
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype


def test_run_synthetic_benchmark_exposes_expected_defaults() -> None:
    # Force l'exécution de la ligne `def` *pendant* ce test (mutmut par-test)
    original_sys_path = list(sys.path)
    original_module = sys.modules.get("scripts.benchmark")
    try:
        # Force un import frais pour exécuter la définition dans ce test
        sys.modules.pop("scripts.benchmark", None)
        bench = importlib.import_module("scripts.benchmark")
        # Capture la signature pour détecter toute dérive des defaults
        signature = inspect.signature(bench.run_synthetic_benchmark)
    finally:
        # Restaure l'état d'import pour ne pas polluer les autres tests
        if original_module is not None:
            sys.modules["scripts.benchmark"] = original_module
        else:
            sys.modules.pop("scripts.benchmark", None)
        # Restaure sys.path car benchmark.py le modifie à l'import
        sys.path[:] = original_sys_path
    # Verrouille n_samples=120 pour garantir la reproductibilité du benchmark
    assert signature.parameters["n_samples"].default == 120
    # Verrouille sfreq=128.0 pour stabiliser la taille des epochs
    assert signature.parameters["sfreq"].default == 128.0


def test_run_synthetic_benchmark_produces_expected_columns(monkeypatch):
    bench = importlib.import_module("scripts.benchmark")
    # Prépare un dataset minimal stable pour contrôler les timings
    sfreq = 128.0
    times = np.arange(0.0, 1.0, 1.0 / sfreq)
    base_epoch = np.stack(
        [
            np.sin(2 * np.pi * 8.0 * times),
            0.2 * np.sin(2 * np.pi * 12.0 * times),
        ],
        axis=0,
    )
    alt_epoch = np.stack(
        [
            0.1 * np.sin(2 * np.pi * 20.0 * times),
            np.sin(2 * np.pi * 15.0 * times),
        ],
        axis=0,
    )
    samples = []
    labels = []
    for idx in range(12):
        jitter = 0.01 * (idx + 1)
        if idx % 2 == 0:
            samples.append(base_epoch + jitter)
            labels.append(0)
        else:
            samples.append(alt_epoch - jitter)
            labels.append(1)
    X = np.stack(samples)
    y = np.array(labels)

    # Monkeypatch pour forcer l'usage du dataset stable et vérifier les valeurs par défaut
    def tiny_dataset(n_samples: int, sfreq: float):
        assert n_samples == 120
        assert sfreq == 128.0
        return X, y

    monkeypatch.setattr(bench, "_generate_dataset", tiny_dataset)

    # Exécute le benchmark avec les valeurs par défaut contrôlées par le patch
    df = bench.run_synthetic_benchmark()
    expected_columns = {
        "feature_strategy",
        "classifier",
        "accuracy",
        "train_seconds",
        "predict_seconds",
    }
    assert set(df.columns) == expected_columns
    # Vérifie que les six combinaisons feature/classifier sont présentes
    expected_pairs = {
        ("fft", "lda"),
        ("fft", "logistic"),
        ("fft", "centroid"),
        ("wavelet", "lda"),
        ("wavelet", "logistic"),
        ("wavelet", "centroid"),
    }
    assert len(df) == 6
    assert set(zip(df["feature_strategy"], df["classifier"])) == expected_pairs
    # Vérifie les types et bornes attendus
    assert is_float_dtype(df["accuracy"])
    assert is_float_dtype(df["train_seconds"])
    assert is_float_dtype(df["predict_seconds"])
    assert df["accuracy"].between(0.0, 1.0).all()
    assert (df["train_seconds"] >= 0).all()
    assert (df["predict_seconds"] >= 0).all()


def test_save_reports_writes_markdown_and_json(tmp_path):
    bench = importlib.import_module("scripts.benchmark")
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
    bench._save_reports(df, Path(tmp_path))
    # Vérifie que le fichier JSON est présent
    assert (tmp_path / "benchmark_results.json").exists()
    # Vérifie que le fichier Markdown est présent
    assert (tmp_path / "benchmark_results.md").exists()


# Valide la génération synthétique pour couvrir le helper interne
def test_generate_dataset_shapes_and_labels_are_balanced() -> None:
    # Importe le module benchmark pour accéder à l'helper interne
    bench = importlib.import_module("scripts.benchmark")
    # Demande un petit dataset synthétique pour vérifier les formes
    X, y = bench._generate_dataset(n_samples=4, sfreq=64.0)
    # Vérifie la forme attendue (n_samples, n_channels, n_times)
    assert X.shape == (4, 2, 64)
    # Vérifie que les labels couvrent bien deux classes
    assert set(y.tolist()) == {0, 1}
    # Vérifie l'équilibre attendu sur un échantillonnage pair
    assert int((y == 0).sum()) == int((y == 1).sum())


# Valide le parser et le flux main pour couvrir la CLI benchmark
def test_build_parser_defaults_and_main_flow(monkeypatch, tmp_path) -> None:
    # Importe le module benchmark pour tester le parser et main
    bench = importlib.import_module("scripts.benchmark")
    # Construit le parser pour inspecter les defaults
    parser = bench.build_parser()
    # Parse sans arguments pour vérifier les valeurs par défaut
    args = parser.parse_args([])
    # Vérifie la valeur par défaut de n_samples
    assert args.n_samples == 120
    # Vérifie la valeur par défaut de sfreq
    assert args.sfreq == 128.0
    # Vérifie que le output_dir pointe vers docs/project par défaut
    assert args.output_dir == Path("docs/project")

    # Prépare un DataFrame vide pour contrôler les colonnes explicitement
    dummy_df = pd.DataFrame()
    # Remplit la colonne feature_strategy pour simuler un résultat minimal
    dummy_df["feature_strategy"] = ["fft"]
    # Remplit la colonne classifier pour simuler un résultat minimal
    dummy_df["classifier"] = ["lda"]
    # Remplit la colonne accuracy avec une valeur cohérente
    dummy_df["accuracy"] = [1.0]
    # Remplit la colonne train_seconds avec un temps d'entraînement factice
    dummy_df["train_seconds"] = [0.01]
    # Remplit la colonne predict_seconds avec une latence factice
    dummy_df["predict_seconds"] = [0.001]

    # Remplace le benchmark synthétique pour éviter le coût réel
    monkeypatch.setattr(bench, "run_synthetic_benchmark", lambda **_: dummy_df)
    # Remplace la sauvegarde pour éviter d'écrire hors du tmp_path
    monkeypatch.setattr(bench, "_save_reports", lambda *_args, **_kwargs: None)

    # Prépare les arguments CLI pour la fonction main
    main_args = ["--n-samples", "2", "--sfreq", "64", "--output-dir", str(tmp_path)]
    # Exécute main avec un output_dir temporaire
    exit_code = bench.main(main_args)
    # Vérifie que le code de sortie signale le succès
    assert exit_code == 0
