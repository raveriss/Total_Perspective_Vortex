# ruff: noqa: PLR0915
from pathlib import Path

# Importe cast pour typer explicitement la requête capturée
from typing import cast

import numpy as np

from scripts import predict as predict_cli


def test_evaluate_run_trains_and_loads_missing_artifacts(
    tmp_path, monkeypatch, capsys
):  # noqa: PLR0915
    """Couvre la régression d'evaluate_run lorsque les artefacts sont absents."""

    subject = "S77"
    run = "R11"
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    target_dir = artifacts_dir / subject / run

    X = np.ones((2, 1, 4), dtype=float)
    y = np.array([0, 1], dtype=int)
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    np.save(subject_dir / f"{run}_X.npy", X)
    np.save(subject_dir / f"{run}_y.npy", y)

    assert not target_dir.exists()

    train_calls: list[tuple[str, str, Path, Path, Path]] = []

    def fake_train_missing(
        subject_arg: str,
        run_arg: str,
        data_dir_arg: Path,
        artifacts_dir_arg: Path,
        options_arg=None,
    ) -> None:
        raw_dir = (
            options_arg.raw_dir
            if options_arg is not None
            else predict_cli.DEFAULT_RAW_DIR
        )
        train_calls.append(
            (subject_arg, run_arg, data_dir_arg, artifacts_dir_arg, raw_dir)
        )
        ensured_target = artifacts_dir_arg / subject_arg / run_arg
        ensured_target.mkdir(parents=True, exist_ok=True)
        (ensured_target / "model.joblib").write_bytes(b"model-bytes")
        (ensured_target / "w_matrix.joblib").write_bytes(b"w-bytes")

    monkeypatch.setattr(predict_cli, "_train_missing_pipeline", fake_train_missing)

    loaded_models: list[Path] = []

    class DummyPipeline:
        def __init__(self, expected_shape: tuple[int, ...]):
            self.expected_shape = expected_shape

        def predict(self, X_input: np.ndarray) -> np.ndarray:
            assert X_input.shape == self.expected_shape
            return np.zeros(X_input.shape[0], dtype=int)

        def score(self, X_input: np.ndarray, y_input: np.ndarray) -> float:
            predictions = self.predict(X_input)
            return float((predictions == y_input).mean())

    def fake_load_pipeline(path: str) -> DummyPipeline:
        loaded_models.append(Path(path))
        return DummyPipeline(X.shape)

    monkeypatch.setattr(predict_cli, "load_pipeline", fake_load_pipeline)

    loaded_w_paths: list[Path] = []

    class DummyReducer:
        def load(self, path: Path) -> None:
            loaded_w_paths.append(Path(path))

    monkeypatch.setattr(predict_cli, "TPVDimReducer", DummyReducer)

    report_calls: list[float | None] = []

    def fake_write_reports(
        target_dir_arg: Path,
        identifiers_arg: dict[str, str],
        y_true_arg: np.ndarray,
        y_pred_arg: np.ndarray,
        accuracy_arg: float,
    ) -> dict:
        assert target_dir_arg == target_dir
        assert identifiers_arg == {"subject": subject, "run": run}
        assert np.array_equal(y_true_arg, y)
        assert y_pred_arg.shape == y.shape
        report_calls.append(accuracy_arg)
        return {"confusion": [[0, 0], [0, 0]]}

    monkeypatch.setattr(predict_cli, "_write_reports", fake_write_reports)

    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)

    stdout = capsys.readouterr().out.splitlines()
    assert stdout == [
        f"INFO: modèle absent pour {subject} {run}, entraînement automatique en cours..."
    ]

    assert train_calls == [
        (subject, run, data_dir, artifacts_dir, predict_cli.DEFAULT_RAW_DIR)
    ]
    assert target_dir.exists()
    assert loaded_models == [target_dir / "model.joblib"]
    assert loaded_w_paths == [target_dir / "w_matrix.joblib"]
    assert result["predictions"].shape == y.shape
    assert isinstance(result["w_matrix"], DummyReducer)
    assert "truth" in result
    assert np.array_equal(result["truth"], y)
    assert "y_true" in result
    assert np.array_equal(result["y_true"], y)
    assert report_calls == [result["accuracy"]]


def test_evaluate_run_triggers_training_when_only_w_matrix_missing(
    tmp_path, monkeypatch, capsys
):
    """Verrouille le OR: un seul artefact manquant doit déclencher l'auto-train."""

    subject = "S78"
    run = "R12"
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    target_dir = artifacts_dir / subject / run

    X = np.ones((2, 1, 4), dtype=float)
    y = np.array([0, 1], dtype=int)
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    np.save(subject_dir / f"{run}_X.npy", X)
    np.save(subject_dir / f"{run}_y.npy", y)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "model.joblib").write_bytes(b"existing-model")
    assert not (target_dir / "w_matrix.joblib").exists()

    train_calls: list[tuple[str, str, Path, Path, Path]] = []

    def fake_train_missing(
        subject_arg: str,
        run_arg: str,
        data_dir_arg: Path,
        artifacts_dir_arg: Path,
        options_arg=None,
    ) -> None:
        raw_dir = (
            options_arg.raw_dir
            if options_arg is not None
            else predict_cli.DEFAULT_RAW_DIR
        )
        train_calls.append(
            (subject_arg, run_arg, data_dir_arg, artifacts_dir_arg, raw_dir)
        )
        ensured_target = artifacts_dir_arg / subject_arg / run_arg
        ensured_target.mkdir(parents=True, exist_ok=True)
        (ensured_target / "model.joblib").write_bytes(b"model-bytes")
        (ensured_target / "w_matrix.joblib").write_bytes(b"w-bytes")

    monkeypatch.setattr(predict_cli, "_train_missing_pipeline", fake_train_missing)

    class DummyPipeline:
        def predict(self, X_input: np.ndarray) -> np.ndarray:
            return np.zeros(X_input.shape[0], dtype=int)

        def score(self, X_input: np.ndarray, y_input: np.ndarray) -> float:
            predictions = self.predict(X_input)
            return float((predictions == y_input).mean())

    monkeypatch.setattr(predict_cli, "load_pipeline", lambda _: DummyPipeline())

    loaded_w_paths: list[Path] = []

    class DummyReducer:
        def load(self, path: Path) -> None:
            loaded_w_paths.append(Path(path))

    monkeypatch.setattr(predict_cli, "TPVDimReducer", DummyReducer)

    report_calls: list[float | None] = []

    def fake_write_reports(
        target_dir_arg: Path,
        identifiers_arg: dict[str, str],
        y_true_arg: np.ndarray,
        y_pred_arg: np.ndarray,
        accuracy_arg: float,
    ) -> dict:
        report_calls.append(accuracy_arg)
        return {"confusion": [[0, 0], [0, 0]]}

    monkeypatch.setattr(predict_cli, "_write_reports", fake_write_reports)

    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)

    stdout = capsys.readouterr().out.splitlines()
    assert stdout == [
        f"INFO: modèle absent pour {subject} {run}, entraînement automatique en cours..."
    ]
    assert train_calls == [
        (subject, run, data_dir, artifacts_dir, predict_cli.DEFAULT_RAW_DIR)
    ]
    assert loaded_w_paths == [target_dir / "w_matrix.joblib"]
    assert "truth" in result
    assert np.array_equal(result["truth"], y)
    assert report_calls == [result["accuracy"]]


def test_evaluate_run_skips_training_when_artifacts_present_and_forwards_raw_dir(
    tmp_path, monkeypatch, capsys
):
    """Verrouille l'absence d'auto-train + la propagation raw_dir vers _load_data."""

    subject = "S79"
    run = "R13"
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir = tmp_path / "raw"
    target_dir = artifacts_dir / subject / run

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "model.joblib").write_bytes(b"model-bytes")
    (target_dir / "w_matrix.joblib").write_bytes(b"w-bytes")

    X = np.ones((3, 1, 2), dtype=float)
    y = np.array([1, 0, 1], dtype=int)
    load_calls: list[tuple[Path | None, str | None]] = []

    def fake_load_data(
        subject_arg: str,
        run_arg: str,
        data_dir_arg: Path,
        raw_dir_arg: Path,
        eeg_reference: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert subject_arg == subject
        assert run_arg == run
        assert data_dir_arg == data_dir
        load_calls.append((raw_dir_arg, eeg_reference))
        return X, y

    monkeypatch.setattr(predict_cli, "_load_data", fake_load_data)

    def fail_train_missing(*_args, **_kwargs) -> None:
        raise AssertionError("_train_missing_pipeline ne doit pas être invoqué")

    monkeypatch.setattr(predict_cli, "_train_missing_pipeline", fail_train_missing)

    class DummyPipeline:
        def predict(self, X_input: np.ndarray) -> np.ndarray:
            return np.zeros(X_input.shape[0], dtype=int)

        def score(self, X_input: np.ndarray, y_input: np.ndarray) -> float:
            predictions = self.predict(X_input)
            return float((predictions == y_input).mean())

    monkeypatch.setattr(predict_cli, "load_pipeline", lambda _: DummyPipeline())

    loaded_w_paths: list[Path] = []

    class DummyReducer:
        def load(self, path: Path) -> None:
            loaded_w_paths.append(Path(path))

    monkeypatch.setattr(predict_cli, "TPVDimReducer", DummyReducer)

    report_calls: list[float | None] = []

    def fake_write_reports(
        target_dir_arg: Path,
        identifiers_arg: dict[str, str],
        y_true_arg: np.ndarray,
        y_pred_arg: np.ndarray,
        accuracy_arg: float,
    ) -> dict:
        report_calls.append(accuracy_arg)
        return {"confusion": [[0, 0], [0, 0]]}

    monkeypatch.setattr(predict_cli, "_write_reports", fake_write_reports)

    options = predict_cli.PredictionOptions(raw_dir=raw_dir)
    result = predict_cli.evaluate_run(
        subject,
        run,
        data_dir,
        artifacts_dir,
        options,
    )

    stdout = capsys.readouterr().out
    assert stdout == ""
    assert load_calls == [(raw_dir, "average")]
    assert loaded_w_paths == [target_dir / "w_matrix.joblib"]
    assert "truth" in result
    assert np.array_equal(result["truth"], y)
    assert report_calls == [result["accuracy"]]


def test_train_missing_pipeline_builds_expected_request(tmp_path, monkeypatch) -> None:
    """Valide la configuration construite lors de l'auto-entraînement."""

    # Définit un sujet/run pour l'auto-entraînement
    subject = "S90"
    # Définit un run pour vérifier la propagation des identifiants
    run = "R01"
    # Prépare le répertoire data requis par la requête d'entraînement
    data_dir = tmp_path / "data"
    # Prépare le répertoire d'artefacts pour stocker le modèle
    artifacts_dir = tmp_path / "artifacts"
    # Prépare un répertoire raw dédié pour la fréquence d'échantillonnage
    raw_dir = tmp_path / "raw"

    # Capture la requête d'entraînement générée par _train_missing_pipeline
    captured: dict[str, object] = {}

    # Force une fréquence d'échantillonnage connue pour la configuration
    monkeypatch.setattr(
        predict_cli.train_module,
        "resolve_sampling_rate",
        lambda *_args, **_kwargs: 128.0,
    )

    # Capture la requête transmise à run_training sans exécuter de fit réel
    def fake_run_training(request) -> None:
        captured["request"] = request

    monkeypatch.setattr(predict_cli.train_module, "run_training", fake_run_training)

    # Lance l'auto-entraînement pour produire la requête
    options = predict_cli.PredictionOptions(raw_dir=raw_dir)
    predict_cli._train_missing_pipeline(
        subject,
        run,
        data_dir,
        artifacts_dir,
        options,
    )

    # Vérifie que la requête a bien été capturée
    assert "request" in captured
    # Convertit la requête capturée pour aligner le typage mypy
    request = cast(predict_cli.train_module.TrainingRequest, captured["request"])
    # Vérifie que les identifiants sont propagés
    assert request.subject == subject
    # Vérifie que le run est propagé
    assert request.run == run
    # Vérifie que la fréquence d'échantillonnage est propagée
    assert request.pipeline_config.sfreq == 128.0
    # Vérifie la stratégie de features par défaut
    assert request.pipeline_config.feature_strategy == "fft"
    # Vérifie la normalisation par défaut
    assert request.pipeline_config.normalize_features is True
    # Vérifie la méthode de réduction par défaut
    assert request.pipeline_config.dim_method == "csp"
    # Vérifie la valeur par défaut des composantes CSP
    assert (
        request.pipeline_config.n_components
        == predict_cli.train_module.DEFAULT_CSP_COMPONENTS
    )
    # Vérifie le classifieur par défaut
    assert request.pipeline_config.classifier == "lda"
    # Vérifie l'absence de scaler par défaut
    assert request.pipeline_config.scaler is None
    # Vérifie le répertoire raw propagé
    assert request.raw_dir == raw_dir
    # Vérifie la désactivation de la recherche exhaustive
    assert request.enable_grid_search is False
    # Vérifie le nombre de splits par défaut
    assert request.grid_search_splits == 5


# Vérifie la résolution d'un alias de feature_strategy vers dim_method
def test_resolve_pipeline_overrides_alias_defaults_to_fft(capsys) -> None:
    # Prépare des overrides d'alias sans dim_method explicite
    overrides = {"feature_strategy": "pca"}
    # Résout les overrides via l'helper interne
    resolved = predict_cli._resolve_pipeline_overrides(overrides)
    # Capture les messages d'information produits
    stdout = capsys.readouterr().out

    # Vérifie que l'information sur l'alias est bien affichée
    assert "feature_strategy='fft' appliquée" in stdout
    # Vérifie que la stratégie est ramenée à FFT
    assert resolved.feature_strategy == "fft"
    # Vérifie que le dim_method suit l'alias fourni
    assert resolved.dim_method == "pca"
    # Vérifie que le classifieur par défaut est conservé
    assert resolved.classifier == "lda"
    # Vérifie que le scaler par défaut reste None
    assert resolved.scaler is None


# Vérifie le warning lorsque wavelet est combiné à CSP explicite
def test_resolve_pipeline_overrides_wavelet_warns_on_csp(capsys) -> None:
    # Prépare des overrides wavelet avec dim_method explicite
    overrides = {"feature_strategy": "wavelet", "dim_method": "csp"}
    # Résout les overrides via l'helper interne
    resolved = predict_cli._resolve_pipeline_overrides(overrides)
    # Capture les messages d'avertissement produits
    stdout = capsys.readouterr().out

    # Vérifie qu'un avertissement est bien affiché
    assert "AVERTISSEMENT: dim_method='csp' ignore feature_strategy" in stdout
    # Vérifie que la stratégie wavelet est conservée
    assert resolved.feature_strategy == "wavelet"
    # Vérifie que la méthode CSP reste en place
    assert resolved.dim_method == "csp"
