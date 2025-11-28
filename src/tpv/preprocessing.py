"""Utilities for loading and validating Physionet EEG datasets."""

# Explain why future annotations import is required for typing modernity
from __future__ import annotations

# Preserve hashing utilities to verify dataset integrity without extra deps
import hashlib

# Keep json to format error payloads for debugging run counts
import json

# Use pathlib to guarantee portable filesystem interactions
from pathlib import Path

# Type hints clarify expectations for callers and simplify tests
from typing import Any, Dict, List, Mapping, Tuple

# MNE is mandatory for EDF/BDF parsing and epoch management
import mne

# Numpy offers vectorized masks for fast event filtering
import numpy as np

# Provide a default mapping consistent with Physionet motor imagery labels
PHYSIONET_LABEL_MAP: Dict[str, int] = {"T0": 0, "T1": 1, "T2": 2}


def load_physionet_raw(
    file_path: Path, montage: str = "standard_1020"
) -> Tuple[mne.io.BaseRaw, Dict[str, object]]:
    """Load an EDF/BDF Physionet file with metadata."""

    # Resolve the input path to avoid surprises with relative locations
    normalized_path = Path(file_path).expanduser().resolve()
    # Load the recording with preload to enable immediate validation steps
    raw = mne.io.read_raw_edf(normalized_path, preload=True, verbose=False)
    # Attach the montage so downstream spatial filters assume 10-20 layout
    raw.set_montage(montage, on_missing="warn")
    # Extract sampling rate to guide later filtering and epoch durations
    sampling_rate = float(raw.info["sfreq"])
    # Capture channel names to expose them to downstream feature builders
    channel_names = list(raw.ch_names)
    # Record the montage name for traceability in integrity reports
    montage_name = montage
    # Bundle metadata for callers that need reproducible preprocessing config
    metadata = {
        "sampling_rate": sampling_rate,
        "channel_names": channel_names,
        "montage": montage_name,
        "path": str(normalized_path),
    }
    # Return both signal and metadata to keep the loader side-effect free
    return raw, metadata


def _extract_bad_intervals(raw: mne.io.BaseRaw) -> List[Tuple[float, float]]:
    """Return BAD annotation windows as (start, end) times in seconds."""

    # Start with an empty list to accumulate invalid windows
    bad_intervals: List[Tuple[float, float]] = []
    # Iterate annotations to translate BAD markers into explicit intervals
    for onset, duration, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        # Skip annotations not flagged BAD to avoid overzealous filtering
        if not desc.upper().startswith("BAD"):
            # Continue looping when the annotation is not an invalid segment
            continue
        # Append the interval boundaries to help later event rejection
        bad_intervals.append((float(onset), float(onset + duration)))
    # Return all invalid windows for consumers that exclude unsafe events
    return bad_intervals


def map_events_and_validate(
    raw: mne.io.BaseRaw, label_map: Mapping[str, int] | None = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map annotations to events while checking label consistency."""

    # Use caller-provided label map or default Physionet mapping for labels
    effective_label_map = (
        dict(label_map) if label_map is not None else dict(PHYSIONET_LABEL_MAP)
    )
    # Inspect annotations to ensure only expected labels remain
    present_labels = set(raw.annotations.description)
    # Identify labels that would break the supervised mapping stage
    unknown_labels = {
        lab
        for lab in present_labels
        if lab not in effective_label_map and not lab.upper().startswith("BAD")
    }
    # Stop early when unknown labels are detected to prevent silent errors
    if unknown_labels:
        # Raise a descriptive error to support dataset hygiene during setup
        raise ValueError(f"Unknown labels in annotations: {sorted(unknown_labels)}")
    # Extract invalid windows to support removal of corrupted epochs
    bad_intervals = _extract_bad_intervals(raw)
    # Convert annotations into events that MNE Epochs can consume
    events, _ = mne.events_from_annotations(
        raw, event_id=effective_label_map, verbose=False
    )
    # Preserve the full label map even if some labels are absent in a run
    event_id = dict(effective_label_map)
    # Initialize a boolean mask list to track valid events explicitly
    keep_mask: List[bool] = [True] * len(events)
    # Iterate over events to check whether they overlap a BAD interval
    for idx, (sample, _, _) in enumerate(events):
        # Convert sample index to seconds to compare against annotation times
        event_time = sample / raw.info["sfreq"]
        # Mark the event for removal when it lies within a BAD interval
        if any(start <= event_time <= end for start, end in bad_intervals):
            # Update the mask to drop contaminated events from the dataset
            keep_mask[idx] = False
    # Enforce boolean typing on the mask to avoid silent drops from bad values
    if not all(isinstance(flag, bool) for flag in keep_mask):
        # Raise when the mask contains non-boolean entries to surface errors early
        raise TypeError("Event mask contained non-boolean values")
    # Gather events whose mask entries remain explicitly True
    filtered_events_list = [
        event for flag, event in zip(keep_mask, events) if flag is True
    ]
    # Convert the preserved events back to a NumPy array for downstream consumers
    filtered_events = np.array(filtered_events_list)
    # Return clean events and the mapping for downstream epoch creation
    return filtered_events, event_id


def create_epochs_from_raw(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Mapping[str, int],
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> mne.Epochs:
    """Construct epochs with annotation-aware rejection."""

    # Build epochs while honoring BAD annotations to avoid contaminating data
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        reject_by_annotation=True,
        baseline=None,
        # Ignore labels missing from specific runs to keep epoching robust
        on_missing="ignore",
        verbose=False,
    )
    # Return epochs ready for feature extraction and model training
    return epochs


def _build_file_entry(
    data_root: Path,
    file_path: Path,
    expected_hashes: Mapping[str, str] | None,
) -> Dict[str, object]:
    """Compose a report entry for a single EDF file."""

    # Record the file size to detect incomplete downloads
    size_bytes = file_path.stat().st_size
    # Compute SHA256 only when reference hashes are provided for comparison
    file_hash = (
        hashlib.sha256(file_path.read_bytes()).hexdigest() if expected_hashes else None
    )
    # Build a stable relative key to align with expected hashes mapping
    rel_key = str(file_path.relative_to(data_root))
    # Evaluate hash parity when expectations exist to surface corruption
    hash_match = expected_hashes is None or expected_hashes.get(rel_key) == file_hash
    # Return a structured entry consumable by integrity reports
    return {
        "path": rel_key,
        "size": size_bytes,
        "sha256": file_hash,
        "hash_ok": hash_match,
    }


def _collect_run_counts(data_root: Path) -> Dict[str, int]:
    """Count EDF runs per subject directory."""

    # Initialize dictionary to aggregate run totals by subject
    subject_counts: Dict[str, int] = {}
    # Iterate over immediate child directories representing subjects
    for subject_dir in data_root.iterdir():
        # Ignore non-directories to focus exclusively on subject folders
        if not subject_dir.is_dir():
            # Continue scanning when encountering stray files at the root
            continue
        # Count EDF files within the subject directory to quantify runs
        run_count = len(list(subject_dir.glob("*.edf")))
        # Persist the count for downstream comparison against expectations
        subject_counts[subject_dir.name] = run_count
    # Return all computed run counts for further validation steps
    return subject_counts


def verify_dataset_integrity(
    base_path: Path,
    expected_hashes: Mapping[str, str] | None = None,
    expected_runs_per_subject: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Verify presence, size, and optional hashes for Physionet data."""

    # Resolve dataset root to ensure comparisons use absolute locations
    data_root = Path(base_path).expanduser().resolve()
    # Prepare a container for per-file reports to keep typing explicit
    file_entries: List[Dict[str, object]] = []
    # Prepare a report structure to feed monitoring or logging systems
    report: Dict[str, Any] = {"root": str(data_root), "files": file_entries}
    # Fail fast if the dataset directory is missing to avoid silent skips
    if not data_root.exists():
        # Raise an explicit error when the dataset root cannot be found
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")
    # Walk through EDF files to build a detailed integrity report
    for file_path in data_root.rglob("*.edf"):
        # Append structured entry for each discovered EDF recording
        file_entries.append(_build_file_entry(data_root, file_path, expected_hashes))
    # Validate expected run counts when provided by the caller
    if expected_runs_per_subject:
        # Collect run totals per subject to compare against expectations
        subject_counts = _collect_run_counts(data_root)
        # Attach run counts to the report for external visibility
        report["subject_run_counts"] = subject_counts
        # Identify subjects whose run counts deviate from expectations
        missing_runs = {
            subject: count
            for subject, count in subject_counts.items()
            if expected_runs_per_subject.get(subject) not in (None, count)
        }
        # Raise when any subject is incomplete to protect model validity
        if missing_runs:
            # Raise a clear error so dataset preparation can be fixed early
            raise ValueError(f"Run count mismatch: {json.dumps(missing_runs)}")
    # Return the report to enable higher-level monitoring or logging
    return report
