"""Tests du protocole de score groupé par type d'expérience."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tpv import evaluation
from tpv.dimensionality import FilterBankCSP
from tpv.features import MIBIFSelector
from tpv.preprocessing import SpatialReference


def _motor_trials() -> tuple[np.ndarray, np.ndarray, float]:
    sfreq = 64.0
    times = np.arange(256) / sfreq
    labels = np.repeat([0, 1], 10)
    rng = np.random.default_rng(12)
    trials = rng.standard_normal((20, 4, times.size + 64)) * 0.05
    full_times = np.arange(trials.shape[2]) / sfreq
    trials[labels == 0, 0, :] += np.sin(2 * np.pi * 10.0 * full_times)
    trials[labels == 1, 1, :] += np.sin(2 * np.pi * 10.0 * full_times)
    return trials, labels, sfreq


def test_build_experiment_pipeline_contains_full_fold_local_processing() -> None:
    pipeline = evaluation.build_experiment_pipeline(160.0, "T1")

    assert list(pipeline.named_steps) == [
        "spatial_reference",
        "filter_bank_csp",
        "mibif",
        "classifier",
    ]
    assert isinstance(pipeline.named_steps["spatial_reference"], SpatialReference)
    assert isinstance(pipeline.named_steps["filter_bank_csp"], FilterBankCSP)
    assert isinstance(pipeline.named_steps["mibif"], MIBIFSelector)
    assert pipeline.named_steps["filter_bank_csp"].run_adaptive is True
    assert pipeline.named_steps["filter_bank_csp"].baseline_window is None
    assert pipeline.named_steps["classifier"].solver == "lsqr"
    assert pipeline.named_steps["classifier"].shrinkage == "auto"


def test_task_configs_distinguish_lateral_and_bilateral_motor_rois() -> None:
    lateral = evaluation.get_task_pipeline_config("T1")
    bilateral = evaluation.get_task_pipeline_config("T3")

    assert lateral.motor_roi != bilateral.motor_roi
    assert "C3" in lateral.motor_roi and "C4" in lateral.motor_roi
    assert "Cz" in bilateral.motor_roi
    assert 4 <= lateral.selected_features <= 16
    assert 4 <= bilateral.selected_features <= 16


def test_evaluate_subject_experiences_returns_fold_evidence(monkeypatch) -> None:
    trials, labels, sfreq = _motor_trials()
    monkeypatch.setattr(
        evaluation,
        "load_experience_epochs_with_runs",
        lambda *_args, **_kwargs: (
            trials,
            labels,
            np.tile([0, 1], 10),
            sfreq,
        ),
    )
    config = evaluation.ExperimentEvaluationConfig(
        cv_splits=2,
        inner_cv_splits=2,
        nested_cv=False,
    )

    results = evaluation.evaluate_subject_experiences(
        raw_dir=Path("data"),
        subject="S001",
        config=config,
    )

    assert set(results) == {"T1", "T2", "T3", "T4"}
    for result in results.values():
        assert len(cast(list[float], result["cv_scores"])) == 2
        assert len(cast(list[float], result["strict_run_scores"])) == 2
        assert result["n_trials"] == 20
        assert result["class_counts"] == {"0": 10, "1": 10}
        mean_score = cast(float, result["cv_mean"])
        assert 0.0 <= mean_score <= 1.0


def test_evaluate_all_subjects_keeps_one_mean_per_experience(monkeypatch) -> None:
    fake_results = {
        experience: {
            "runs": [],
            "n_trials": 45,
            "class_counts": {"0": 21, "1": 24},
            "cv_scores": [0.7, 0.8],
            "cv_mean": 0.75,
        }
        for experience in ("T1", "T2", "T3", "T4")
    }
    monkeypatch.setattr(
        evaluation,
        "evaluate_subject_experiences",
        lambda *_args, **_kwargs: fake_results,
    )
    progress: list[str] = []

    scores, evidence = evaluation.evaluate_all_subjects(
        Path("data"),
        ["S001", "S002"],
        evaluation.ExperimentEvaluationConfig(),
        progress=lambda subject, _results: progress.append(subject),
    )

    assert scores["S001"] == {experience: [0.75] for experience in fake_results}
    assert [entry["subject"] for entry in evidence] == ["S001", "S002"]
    assert progress == ["S001", "S002"]


def test_load_experience_epochs_concatenates_the_three_protocol_runs(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_load(_raw_dir, _subject, run, _config, _motor_roi):
        calls.append(run)
        return np.ones((15, 3, 256)), np.tile([0, 1, 0], 5), 64.0

    monkeypatch.setattr(evaluation, "_load_run_epochs", fake_load)

    trials, labels, sampling_rate = evaluation.load_experience_epochs(
        Path("data"),
        "S001",
        "T1",
        evaluation.ExperimentEvaluationConfig(),
    )

    assert calls == ["R03", "R07", "R11"]
    assert trials.shape == (45, 3, 256)
    assert labels.shape == (45,)
    assert sampling_rate == 64.0


def test_load_experience_epochs_rejects_unknown_experience() -> None:
    with pytest.raises(ValueError, match="unknown experience"):
        evaluation.load_experience_epochs(
            Path("data"),
            "S001",
            "T9",
            evaluation.ExperimentEvaluationConfig(),
        )


def test_evaluate_all_subjects_reuses_matching_cache(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    fake_results = {
        experience: {
            "runs": [],
            "n_trials": 45,
            "class_counts": {"0": 21, "1": 24},
            "cv_scores": [0.7, 0.8],
            "cv_mean": 0.75,
        }
        for experience in ("T1", "T2", "T3", "T4")
    }

    def fake_evaluate(_raw_dir, subject, _config):
        calls.append(subject)
        return fake_results

    monkeypatch.setattr(evaluation, "evaluate_subject_experiences", fake_evaluate)
    config = evaluation.ExperimentEvaluationConfig()
    arguments = (Path("data"), ["S001"], config)

    evaluation.evaluate_all_subjects(*arguments, cache_dir=tmp_path)
    evaluation.evaluate_all_subjects(*arguments, cache_dir=tmp_path)

    assert calls == ["S001"]
    assert (tmp_path / "S001.json").exists()


def test_score_cache_validation_rejects_nan_and_missing_experience() -> None:
    complete = {
        experience: {"cv_mean": 0.75} for experience in ("T1", "T2", "T3", "T4")
    }
    with_nan = {key: dict(value) for key, value in complete.items()}
    with_nan["T2"]["cv_mean"] = float("nan")

    assert evaluation._has_complete_finite_scores(complete) is True
    assert evaluation._has_complete_finite_scores(with_nan) is False
    assert evaluation._has_complete_finite_scores({"T1": {"cv_mean": 0.8}}) is False
    assert evaluation._has_complete_finite_scores(None) is False


def test_campaign_subjects_are_progressive_and_final_set_is_immutable() -> None:
    assert evaluation.campaign_subjects(10) == [f"S{i:03d}" for i in range(1, 11)]
    assert len(evaluation.campaign_subjects(30)) == 30
    final_subjects = evaluation.campaign_subjects(109)
    assert len(final_subjects) == 109
    assert final_subjects[-1] == "S109"


def test_task_pipeline_and_campaign_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown experience"):
        evaluation.get_task_pipeline_config("T9")
    with pytest.raises(ValueError, match="10, 30, or 109"):
        evaluation.campaign_subjects(11)
    with pytest.raises(ValueError, match="model_family"):
        evaluation.build_experiment_pipeline(160.0, "T1", "unknown")
    with pytest.raises(ValueError, match="too low"):
        evaluation._supported_bands(((8.0, 12.0),), 20.0)


def test_build_riemannian_pipeline_and_search_grids_are_explicit() -> None:
    pipeline = evaluation.build_experiment_pipeline(160.0, "T3", "riemannian")
    assert list(pipeline.named_steps) == [
        "spatial_reference",
        "covariance_tangent",
        "mibif",
        "classifier",
    ]
    riemannian = evaluation.ExperimentEvaluationConfig(model_family="riemannian")
    assert set(evaluation._search_parameter_grid(riemannian)) == {
        "mibif__k",
        "covariance_tangent__regularization",
    }
    ablation = evaluation.ExperimentEvaluationConfig(enable_preprocessing_ablation=True)
    grid = evaluation._search_parameter_grid(ablation)
    assert isinstance(grid, list)
    assert len(grid) == 6
    assert {variant["spatial_reference__method"][0] for variant in grid} == {
        "none",
        "car",
        "laplacian",
    }
    assert any(
        variant["filter_bank_csp__baseline_window"] == [(-1.0, 0.0)] for variant in grid
    )
    assert any(
        variant["filter_bank_csp__filter_mode"] == ["causal"] for variant in grid
    )


def test_load_run_epochs_maps_labels_and_checks_roi(monkeypatch) -> None:
    class FakeEpochs:
        ch_names = list(evaluation.MOTOR_ROI)
        events = np.array([[0, 0, 7], [1, 0, 8]])

        def get_data(self, picks, copy):
            assert picks == list(evaluation.MOTOR_ROI)
            assert copy is True
            return np.ones((2, len(picks), 16))

    monkeypatch.setattr(
        evaluation, "resolve_recording_paths", lambda *_: (Path("x"), Path("y"))
    )
    monkeypatch.setattr(
        evaluation,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (object(), {"sampling_rate": 160.0}),
    )
    monkeypatch.setattr(
        evaluation,
        "map_events_to_motor_labels",
        lambda _raw: (np.ones((2, 3), dtype=int), {"T1": 7, "T2": 8}, []),
    )
    monkeypatch.setattr(
        evaluation,
        "create_epochs_from_raw",
        lambda *_args, **_kwargs: FakeEpochs(),
    )

    trials, labels, sampling_rate = evaluation._load_run_epochs(
        Path("data"), "S001", "R03", evaluation.ExperimentEvaluationConfig()
    )

    assert trials.shape == (2, 15, 16)
    assert labels.tolist() == [0, 1]
    assert sampling_rate == 160.0


def test_load_run_epochs_rejects_missing_roi_and_motor_class(monkeypatch) -> None:
    class FakeEpochs:
        ch_names = ["C3"]
        events = np.array([[0, 0, 7]])

    monkeypatch.setattr(
        evaluation, "resolve_recording_paths", lambda *_: (Path("x"), Path("y"))
    )
    monkeypatch.setattr(
        evaluation,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (object(), {"sampling_rate": 160.0}),
    )
    monkeypatch.setattr(
        evaluation,
        "map_events_to_motor_labels",
        lambda _raw: (np.ones((1, 3), dtype=int), {"T1": 7}, []),
    )
    monkeypatch.setattr(
        evaluation,
        "create_epochs_from_raw",
        lambda *_args, **_kwargs: FakeEpochs(),
    )
    config = evaluation.ExperimentEvaluationConfig()

    with pytest.raises(ValueError, match="missing motor ROI"):
        evaluation._load_run_epochs(Path("data"), "S001", "R03", config)

    FakeEpochs.ch_names = list(evaluation.MOTOR_ROI)
    with pytest.raises(ValueError, match="both T1 and T2"):
        evaluation._load_run_epochs(Path("data"), "S001", "R03", config)


def test_load_experience_rejects_inconsistent_sampling_rates(monkeypatch) -> None:
    rates = iter((160.0, 128.0, 160.0))
    monkeypatch.setattr(
        evaluation,
        "_load_run_epochs",
        lambda *_args: (np.ones((2, 12, 16)), np.array([0, 1]), next(rates)),
    )
    with pytest.raises(ValueError, match="inconsistent sampling rates"):
        evaluation.load_experience_epochs_with_runs(
            Path("data"), "S001", "T1", evaluation.ExperimentEvaluationConfig()
        )


def test_evaluate_folds_builds_nested_grid_inside_external_cv(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_cross_val_score(estimator, X, y, cv, **kwargs):
        observed["estimator"] = estimator
        observed["splits"] = cv
        observed["kwargs"] = kwargs
        assert X.shape[0] == y.shape[0]
        return np.array([0.5, 0.75])

    monkeypatch.setattr(evaluation, "cross_val_score", fake_cross_val_score)
    trials = np.ones((8, len(evaluation.LATERAL_MOTOR_ROI), 320))
    labels = np.tile([0, 1], 4)
    groups = np.repeat([0, 1], 4)
    config = evaluation.ExperimentEvaluationConfig(
        cv_splits=2,
        inner_cv_splits=2,
        nested_cv=True,
    )

    scores, folds = evaluation._evaluate_folds(
        trials, labels, groups, 64.0, "T1", config, strict_runs=False
    )

    assert isinstance(observed["estimator"], evaluation.GridSearchCV)
    assert len(cast(list[tuple[np.ndarray, np.ndarray]], observed["splits"])) == 2
    assert scores == [0.5, 0.75]
    assert folds[0]["selection"] == "nested_grid_search"
    kwargs = cast(dict[str, object], observed["kwargs"])
    parameters = cast(dict[str, np.ndarray], kwargs["params"])
    assert parameters["filter_bank_csp__run_groups"] is groups
