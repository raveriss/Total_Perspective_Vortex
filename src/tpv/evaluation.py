"""Évaluation reproductible des quatre expériences EEGMMIDB."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, cast

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from tpv.dimensionality import CovarianceTangentSpace, FilterBankCSP
from tpv.features import MIBIFSelector
from tpv.preprocessing import (
    SpatialReference,
    create_epochs_from_raw,
    load_physionet_raw,
    map_events_to_motor_labels,
)
from tpv.protocol import EXPERIENCE_ORDER, EXPERIENCE_RUNS
from tpv.utils import resolve_recording_paths

MOTOR_ROI = (
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
)

LATERAL_MOTOR_ROI = (
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CP2",
    "CP4",
)

LAPLACIAN_NEIGHBOURS = {
    "FC3": ("FC1", "C3"),
    "FC1": ("FC3", "FCz", "C1"),
    "FCz": ("FC1", "FC2", "Cz"),
    "FC2": ("FCz", "FC4", "C2"),
    "FC4": ("FC2", "C4"),
    "C3": ("FC3", "C1", "CP3"),
    "C1": ("FC1", "C3", "Cz", "CP1"),
    "Cz": ("FCz", "C1", "C2", "CPz"),
    "C2": ("FC2", "Cz", "C4", "CP2"),
    "C4": ("FC4", "C2", "CP4"),
    "CP3": ("C3", "CP1"),
    "CP1": ("C1", "CP3", "CPz"),
    "CPz": ("Cz", "CP1", "CP2"),
    "CP2": ("C2", "CPz", "CP4"),
    "CP4": ("C4", "CP2"),
}

CANONICAL_FBCSP_BANDS = tuple((float(low), float(low + 4)) for low in range(4, 40, 4))
REFERENCE_FBCSP_BANDS = (
    (7.0, 12.0),
    (10.0, 15.0),
    (12.0, 18.0),
    (15.0, 22.0),
    (18.0, 26.0),
    (22.0, 30.0),
)
REFERENCE_FBCSP_WINDOWS = ((0.0, 2.0), (0.75, 2.75), (1.5, 3.5))


@dataclass(frozen=True)
class TaskPipelineConfig:
    """Configuration EEG spécifique à une famille de tâches motrices."""

    motor_roi: tuple[str, ...]
    bands: tuple[tuple[float, float], ...]
    windows: tuple[tuple[float, float], ...]
    selected_features: int = 12
    spatial_reference: str = "none"


TASK_PIPELINE_CONFIGS = {
    "T1": TaskPipelineConfig(
        LATERAL_MOTOR_ROI,
        REFERENCE_FBCSP_BANDS,
        REFERENCE_FBCSP_WINDOWS,
    ),
    "T2": TaskPipelineConfig(
        LATERAL_MOTOR_ROI,
        REFERENCE_FBCSP_BANDS,
        REFERENCE_FBCSP_WINDOWS,
    ),
    "T3": TaskPipelineConfig(
        MOTOR_ROI,
        REFERENCE_FBCSP_BANDS,
        REFERENCE_FBCSP_WINDOWS,
    ),
    "T4": TaskPipelineConfig(
        MOTOR_ROI,
        REFERENCE_FBCSP_BANDS,
        REFERENCE_FBCSP_WINDOWS,
    ),
}


def get_task_pipeline_config(experience: str) -> TaskPipelineConfig:
    """Retourne la configuration figée d'un type d'expérience EEGMMIDB."""

    try:
        return TASK_PIPELINE_CONFIGS[experience]
    except KeyError as error:
        raise ValueError(f"unknown experience: {experience}") from error


@dataclass(frozen=True)
class ExperimentEvaluationConfig:
    """Paramètres figés de la preuve de score FBCSP."""

    cv_splits: int = 5
    inner_cv_splits: int = 3
    random_state: int = 42
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0
    eeg_reference: str | None = "average"
    nested_cv: bool = True
    strict_run_report: bool = True
    enable_preprocessing_ablation: bool = False
    model_family: str = "fbcsp"
    selected_feature_counts: tuple[int, ...] = (4, 8, 12, 16)
    csp_component_counts: tuple[int, ...] = (2,)
    covariance_regularizations: tuple[float, ...] = (0.1,)
    filter_mode: str = "zero_phase"
    baseline_window: tuple[float, float] | None = None
    run_adaptive_filters: bool = True
    algorithm_version: str = "nested-mibif-fbcsp-v3.1"


def _load_run_epochs(
    raw_dir: Path,
    subject: str,
    run: str,
    config: ExperimentEvaluationConfig,
    motor_roi: tuple[str, ...] = MOTOR_ROI,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Charge un run en conservant les amplitudes utiles au CSP."""

    raw_path, _event_path = resolve_recording_paths(raw_dir, subject, run)
    raw, metadata = load_physionet_raw(
        raw_path,
        reference=config.eeg_reference,
    )
    events, event_id, _motor_labels = map_events_to_motor_labels(raw)
    epochs = create_epochs_from_raw(
        raw,
        events,
        event_id,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
    )
    missing_channels = sorted(set(motor_roi) - set(epochs.ch_names))
    if missing_channels:
        raise ValueError(
            f"{subject} {run}: missing motor ROI channels {missing_channels}"
        )
    class_a_code = event_id.get("T1")
    class_b_code = event_id.get("T2")
    if class_a_code is None or class_b_code is None:
        raise ValueError(f"{subject} {run}: both T1 and T2 events are required")
    labels = np.where(epochs.events[:, 2] == class_a_code, 0, 1)
    trials = epochs.get_data(picks=list(motor_roi), copy=True)
    sampling_rate = cast(float, metadata["sampling_rate"])
    return np.asarray(trials), np.asarray(labels), float(sampling_rate)


def load_experience_epochs(
    raw_dir: Path,
    subject: str,
    experience: str,
    config: ExperimentEvaluationConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Regroupe les trois runs d'un même type avant validation croisée."""

    trials, labels, _run_groups, sampling_rate = load_experience_epochs_with_runs(
        raw_dir,
        subject,
        experience,
        config,
    )
    return trials, labels, sampling_rate


def load_experience_epochs_with_runs(
    raw_dir: Path,
    subject: str,
    experience: str,
    config: ExperimentEvaluationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Regroupe les essais tout en conservant leur run d'origine."""

    if experience not in EXPERIENCE_RUNS:
        raise ValueError(f"unknown experience: {experience}")
    task_config = get_task_pipeline_config(experience)
    trial_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    run_batches: list[np.ndarray] = []
    sampling_rates: set[float] = set()
    for run_index, run in enumerate(EXPERIENCE_RUNS[experience]):
        trials, labels, sampling_rate = _load_run_epochs(
            raw_dir,
            subject,
            run,
            config,
            task_config.motor_roi,
        )
        trial_batches.append(trials)
        label_batches.append(labels)
        run_batches.append(np.full(labels.shape, run_index, dtype=int))
        sampling_rates.add(sampling_rate)
    if len(sampling_rates) != 1:
        raise ValueError(f"{subject} {experience}: inconsistent sampling rates")
    return (
        np.concatenate(trial_batches, axis=0),
        np.concatenate(label_batches, axis=0),
        np.concatenate(run_batches, axis=0),
        sampling_rates.pop(),
    )


def _supported_bands(
    bands: tuple[tuple[float, float], ...], sampling_rate: float
) -> tuple[tuple[float, float], ...]:
    """Écarte uniquement les bandes incompatibles avec le Nyquist d'un test."""

    supported = tuple(band for band in bands if band[1] < sampling_rate / 2.0)
    if not supported:
        raise ValueError("sampling rate is too low for the configured filter bank")
    return supported


def build_experiment_pipeline(
    sampling_rate: float,
    experience: str = "T1",
    model_family: str = "fbcsp",
    evaluation_config: ExperimentEvaluationConfig | None = None,
) -> Pipeline:
    """Construit le pipeline complet appris à l'intérieur de chaque pli."""

    task = get_task_pipeline_config(experience)
    config = evaluation_config or ExperimentEvaluationConfig(model_family=model_family)
    reference = SpatialReference(
        method=task.spatial_reference,
        channel_names=task.motor_roi,
        neighbours=LAPLACIAN_NEIGHBOURS,
    )
    if model_family == "riemannian":
        return Pipeline(
            [
                ("spatial_reference", reference),
                ("covariance_tangent", CovarianceTangentSpace()),
                (
                    "mibif",
                    MIBIFSelector(k=task.selected_features, random_state=42),
                ),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        )
    if model_family != "fbcsp":
        raise ValueError("model_family must be 'fbcsp' or 'riemannian'")
    return Pipeline(
        [
            ("spatial_reference", reference),
            (
                "filter_bank_csp",
                FilterBankCSP(
                    sfreq=sampling_rate,
                    bands=_supported_bands(task.bands, sampling_rate),
                    windows=task.windows,
                    n_components=2,
                    regularization=0.1,
                    filter_mode=config.filter_mode,
                    time_origin=config.epoch_tmin,
                    baseline_window=config.baseline_window,
                    run_adaptive=config.run_adaptive_filters,
                ),
            ),
            ("mibif", MIBIFSelector(k=task.selected_features, random_state=42)),
            (
                "classifier",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )


def _search_parameter_grid(
    config: ExperimentEvaluationConfig,
) -> dict[str, list[object]] | list[dict[str, list[object]]]:
    """Définit la recherche interne sans jamais toucher au test externe."""

    grid: dict[str, list[object]] = {
        "mibif__k": list(config.selected_feature_counts),
    }
    if config.model_family == "riemannian":
        grid["covariance_tangent__regularization"] = list(
            config.covariance_regularizations
        )
        return grid
    grid["filter_bank_csp__n_components"] = list(config.csp_component_counts)
    grid["filter_bank_csp__regularization"] = list(config.covariance_regularizations)
    if not config.enable_preprocessing_ablation:
        return grid
    common: dict[str, list[object]] = {
        "mibif__k": list(config.selected_feature_counts),
        "filter_bank_csp__n_components": list(config.csp_component_counts),
        "filter_bank_csp__regularization": list(config.covariance_regularizations),
    }
    variants = (
        ("none", REFERENCE_FBCSP_BANDS, None, "zero_phase"),
        ("none", REFERENCE_FBCSP_BANDS, (-1.0, 0.0), "zero_phase"),
        ("car", REFERENCE_FBCSP_BANDS, None, "zero_phase"),
        ("laplacian", REFERENCE_FBCSP_BANDS, None, "zero_phase"),
        ("none", REFERENCE_FBCSP_BANDS, None, "causal"),
        ("none", CANONICAL_FBCSP_BANDS, None, "zero_phase"),
    )
    return [
        {
            **common,
            "spatial_reference__method": [spatial_reference],
            "filter_bank_csp__bands": [bands],
            "filter_bank_csp__baseline_window": [baseline],
            "filter_bank_csp__filter_mode": [filter_mode],
        }
        for spatial_reference, bands, baseline, filter_mode in variants
    ]


def _outer_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentEvaluationConfig,
    strict_runs: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if strict_runs:
        yield from LeaveOneGroupOut().split(np.zeros(labels.shape), labels, groups)
        return
    splitter = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    yield from splitter.split(np.zeros(labels.shape), labels)


def _evaluate_folds(  # noqa: PLR0913 - frontière explicite de l'évaluation
    trials: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sampling_rate: float,
    experience: str,
    config: ExperimentEvaluationConfig,
    strict_runs: bool,
) -> tuple[list[float], list[dict[str, object]]]:
    """Exécute les plis externes et garde les meilleurs paramètres internes."""

    base_pipeline = build_experiment_pipeline(
        sampling_rate,
        experience,
        config.model_family,
        config,
    )
    estimator: Pipeline | GridSearchCV
    if config.nested_cv:
        inner_splitter = StratifiedKFold(
            n_splits=config.inner_cv_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        estimator = GridSearchCV(
            clone(base_pipeline),
            _search_parameter_grid(config),
            cv=inner_splitter,
            scoring="accuracy",
            n_jobs=1,
            error_score="raise",
            refit=True,
        )
    else:
        estimator = clone(base_pipeline)
    splits = list(_outer_splits(labels, groups, config, strict_runs))
    fit_parameters = (
        {"filter_bank_csp__run_groups": groups}
        if config.model_family == "fbcsp"
        else None
    )
    score_array = cross_val_score(
        estimator,
        trials,
        labels,
        cv=splits,
        scoring="accuracy",
        n_jobs=1,
        params=fit_parameters,
        error_score="raise",
    )
    scores = [float(score) for score in score_array]
    fold_evidence: list[dict[str, object]] = []
    for fold_index, ((train_indices, test_indices), score) in enumerate(
        zip(splits, scores, strict=True), start=1
    ):
        fold_evidence.append(
            {
                "fold": fold_index,
                "train_size": int(train_indices.size),
                "test_size": int(test_indices.size),
                "held_out_runs": sorted({int(value) for value in groups[test_indices]}),
                "score": score,
                "selection": "nested_grid_search" if config.nested_cv else "fixed",
            }
        )
    return scores, fold_evidence


def evaluate_subject_experiences(
    raw_dir: Path,
    subject: str,
    config: ExperimentEvaluationConfig,
) -> dict[str, dict[str, object]]:
    """Valide les quatre types d'expérience d'un sujet sur des essais non appris."""

    results: dict[str, dict[str, object]] = {}
    for experience in EXPERIENCE_ORDER:
        trials, labels, run_groups, sampling_rate = load_experience_epochs_with_runs(
            raw_dir, subject, experience, config
        )
        scores, fold_evidence = _evaluate_folds(
            trials,
            labels,
            run_groups,
            sampling_rate,
            experience,
            config,
            strict_runs=False,
        )
        strict_scores: list[float] = []
        strict_evidence: list[dict[str, object]] = []
        if config.strict_run_report:
            strict_scores, strict_evidence = _evaluate_folds(
                trials,
                labels,
                run_groups,
                sampling_rate,
                experience,
                config,
                strict_runs=True,
            )
        results[experience] = {
            "runs": list(EXPERIENCE_RUNS[experience]),
            "n_trials": int(trials.shape[0]),
            "class_counts": {
                str(label): int(np.count_nonzero(labels == label))
                for label in np.unique(labels)
            },
            "cv_scores": scores,
            "cv_mean": float(np.mean(scores)),
            "cv_folds": fold_evidence,
            "strict_run_scores": strict_scores,
            "strict_run_mean": (
                float(np.mean(strict_scores)) if strict_scores else None
            ),
            "strict_run_folds": strict_evidence,
        }
    return results


def _has_complete_finite_scores(results: object) -> bool:
    """Refuse un cache partiel ou contenant un score NaN/infini."""

    if not isinstance(results, dict):
        return False
    means = [
        result.get("cv_mean") for result in results.values() if isinstance(result, dict)
    ]
    return len(means) == len(EXPERIENCE_ORDER) and all(
        isinstance(mean, (float, int)) and np.isfinite(mean) for mean in means
    )


def evaluate_all_subjects(
    raw_dir: Path,
    subjects: list[str],
    config: ExperimentEvaluationConfig,
    progress: Callable[[str, dict[str, dict[str, object]]], None] | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, object]]]:
    """Produit les scores agrégeables et les folds auditables de tous les sujets."""

    subject_scores: dict[str, dict[str, list[float]]] = {}
    evidence: list[dict[str, object]] = []
    configuration = json.loads(json.dumps(asdict(config)))
    for subject in subjects:
        cache_path = cache_dir / f"{subject}.json" if cache_dir is not None else None
        results: dict[str, dict[str, object]]
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get(
                "configuration"
            ) == configuration and _has_complete_finite_scores(
                cached.get("experiences")
            ):
                results = cast(dict[str, dict[str, object]], cached["experiences"])
            else:
                results = evaluate_subject_experiences(raw_dir, subject, config)
        else:
            results = evaluate_subject_experiences(raw_dir, subject, config)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "subject": subject,
                        "configuration": configuration,
                        "experiences": results,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        subject_scores[subject] = {
            experience: [float(cast(float, results[experience]["cv_mean"]))]
            for experience in EXPERIENCE_ORDER
        }
        evidence.append({"subject": subject, "experiences": results})
        if progress is not None:
            progress(subject, results)
    return subject_scores, evidence


def campaign_subjects(size: int) -> list[str]:
    """Construit les cohortes figées 10 → 30 → 109 sans modifier le test final."""

    if size not in {10, 30, 109}:
        raise ValueError("campaign size must be 10, 30, or 109")
    return [f"S{subject_index:03d}" for subject_index in range(1, size + 1)]
