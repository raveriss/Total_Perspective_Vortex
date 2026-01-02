from pathlib import Path

import numpy as np

from scripts import predict as predict_cli


def test_evaluate_run_trains_and_loads_missing_artifacts(tmp_path, monkeypatch):
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
        raw_dir_arg: Path,
    ) -> None:
        train_calls.append(
            (subject_arg, run_arg, data_dir_arg, artifacts_dir_arg, raw_dir_arg)
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

    result = predict_cli.evaluate_run(subject, run, data_dir, artifacts_dir)

    assert train_calls == [
        (subject, run, data_dir, artifacts_dir, predict_cli.DEFAULT_RAW_DIR)
    ]
    assert target_dir.exists()
    assert loaded_models == [target_dir / "model.joblib"]
    assert loaded_w_paths == [target_dir / "w_matrix.joblib"]
    assert result["predictions"].shape == y.shape
    assert isinstance(result["w_matrix"], DummyReducer)
