"""Unit tests for Physionet preprocessing helpers."""

# Import pathlib to construct temporary dataset layouts
# Import pathlib to type temporary paths for datasets
from pathlib import Path

# Import SimpleNamespace to build lightweight annotation holders
from types import SimpleNamespace

# Import mne to build synthetic Raw objects and annotations
import mne

# Import numpy to craft deterministic dummy EEG data
import numpy as np

# Import pytest to manage temporary directories and assertions
import pytest

# Import the preprocessing helpers under test
from tpv.preprocessing import (
    PHYSIONET_LABEL_MAP,
    _build_file_entry,
    _collect_run_counts,
    _extract_bad_intervals,
    create_epochs_from_raw,
    load_physionet_raw,
    map_events_and_validate,
    verify_dataset_integrity,
)


def _build_dummy_raw(sfreq: float = 128.0) -> mne.io.Raw:
    """Create a synthetic RawArray with Physionet-like annotations."""

    # Set two channels to keep tests lightweight while representative
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    # Generate deterministic data to guarantee reproducible hashes
    rng = np.random.default_rng(seed=42)
    # Create one second of data for each channel to minimize test duration
    data = rng.standard_normal((2, int(sfreq)))
    # Assemble the RawArray from the synthetic data
    raw = mne.io.RawArray(data, info)
    # Annotate two events representing motor imagery tasks
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.6],
            duration=[0.1, 0.1],
            description=["T1", "T2"],
        )
    )
    # Return the constructed Raw object for downstream export
    return raw


def test_load_physionet_raw_reads_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure loader returns Raw and metadata from EDF files."""

    # Build a dummy raw instance with consistent annotations
    raw = _build_dummy_raw()
    # Define EDF output path within the temporary directory
    edf_path = tmp_path / "subject01" / "run01.edf"
    # Ensure the directory exists to allow file export
    edf_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a small placeholder file to satisfy filesystem expectations
    edf_path.write_bytes(b"dummy")
    # Stub the MNE loader to return the synthetic Raw without EDF export
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    # Load the placeholder file using the preprocessing helper
    loaded_raw, metadata = load_physionet_raw(edf_path)
    # Assert sampling rate is preserved during loader execution
    assert metadata["sampling_rate"] == pytest.approx(128.0)
    # Assert montage is set to the default 10-20 scheme
    assert metadata["montage"] == "standard_1020"
    # Verify that channels are present in the returned metadata
    assert metadata["channel_names"] == ["C3", "C4"]
    # Confirm the loader returns an MNE Raw instance
    assert isinstance(loaded_raw, mne.io.BaseRaw)


def test_load_physionet_raw_applies_montage_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure loader applies montage and normalizes the path."""

    # Prepare a dummy raw with a patched montage setter to track invocation
    raw = _build_dummy_raw()
    # Initialize a list to capture montage calls for assertion
    montage_calls: list[tuple[str, dict[str, object]]] = []
    # Patch set_montage to record keyword arguments explicitly
    monkeypatch.setattr(
        raw,
        "set_montage",
        lambda name, **kwargs: montage_calls.append((name, kwargs)),
    )
    # Build a relative EDF path to verify normalization to absolute form
    edf_path = Path("./relative/path/to_run.edf")
    # Stub the reader to return the instrumented raw instance
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    # Load using the helper to trigger montage assignment and path normalization
    _, metadata = load_physionet_raw(edf_path, montage="custom_montage")
    # Confirm the montage setter was called exactly once with the provided name
    assert montage_calls == [("custom_montage", {"on_missing": "warn"})]
    # Ensure the metadata path is resolved to an absolute representation
    assert metadata["path"] == str(edf_path.resolve())


def test_load_physionet_raw_uses_resolved_path_and_reader_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the loader forwards the resolved path and preload flags."""

    # Prepare a raw object for the stubbed reader to return
    raw = _build_dummy_raw()
    # Capture the arguments received by the reader stub
    captured_args: list[tuple[tuple[object, ...], dict[str, object]]] = []

    # Define a stub reader that records calls and returns the dummy raw
    def reader_stub(*args: object, **kwargs: object) -> mne.io.BaseRaw:
        # Persist the invocation arguments for later assertions
        captured_args.append((args, kwargs))
        # Return the prepared Raw instance to mimic EDF loading
        return raw

    # Patch the reader to store invocation details before returning the raw
    monkeypatch.setattr(mne.io, "read_raw_edf", reader_stub)
    # Build a path with a user component to confirm resolution
    edf_path = Path("~") / "dataset" / "file.edf"
    # Execute the loader to trigger the patched reader
    _, metadata = load_physionet_raw(edf_path)
    # Confirm the reader received the fully resolved absolute path
    assert captured_args[0][0][0] == edf_path.expanduser().resolve()
    # Validate that preload and verbosity flags are forwarded as expected
    assert captured_args[0][1] == {"preload": True, "verbose": False}
    # Ensure the metadata path matches the resolved reader input
    assert metadata["path"] == str(edf_path.expanduser().resolve())


def test_map_events_filters_bad_segments(tmp_path: Path) -> None:
    """Validate event mapping drops events overlapping BAD intervals."""

    # Create a synthetic raw recording with annotations
    raw = _build_dummy_raw()
    # Extend annotations with a BAD interval covering the second event
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(onset=[0.55], duration=[0.2], description=["BAD_segment"])
    )
    # Derive events and event_id using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Ensure the label map matches the Physionet default even when labels are missing
    assert event_id == PHYSIONET_LABEL_MAP
    # Confirm only the first event remains after BAD filtering
    assert events.shape[0] == 1
    # Verify the remaining event corresponds to the first annotation
    assert events[0, 0] == pytest.approx(round(0.1 * raw.info["sfreq"]))


def test_map_events_respects_custom_label_map_and_bad_boundaries() -> None:
    """Ensure custom label maps apply and BAD windows remove boundary events."""

    # Build a raw instance with two annotations for mapping
    raw = _build_dummy_raw()
    # Override annotations to place one event on a BAD boundary
    raw.set_annotations(
        mne.Annotations(
            onset=[0.2, 0.5, 0.5],
            duration=[0.05, 0.05, 0.1],
            description=["X1", "X2", "BAD_artifact"],
        )
    )
    # Define a custom label map covering the custom annotations
    label_map = {"X1": 7, "X2": 9}
    # Map events using the helper with the custom label map
    events, event_id = map_events_and_validate(raw, label_map=label_map)
    # Confirm the returned mapping matches the provided mapping
    assert event_id == label_map
    # Ensure the event overlapping the BAD onset is removed
    assert events.shape[0] == 1
    # Validate the surviving event corresponds to the first annotation
    assert events[0, 2] == label_map["X1"]


def test_map_events_does_not_mutate_input_label_map() -> None:
    """Ensure the helper copies the provided label map instead of mutating it."""

    # Build a raw instance with a single annotation
    raw = _build_dummy_raw()
    # Define a complete label map that should remain unchanged
    label_map = {"T0": 0, "T1": 5, "T2": 6}
    # Map events using the helper with the custom label map
    events, event_id = map_events_and_validate(raw, label_map=label_map)
    # Confirm the effective mapping matches the expected value
    assert event_id == label_map
    # Verify the original dictionary remains unchanged after the call
    assert label_map == {"T0": 0, "T1": 5, "T2": 6}
    # Define the expected count of preserved events when no BAD labels exist
    expected_event_count = 2
    # Ensure all events are retained when no BAD intervals exist
    assert events.shape[0] == expected_event_count


def test_extract_bad_intervals_handles_non_bad_and_bad_segments() -> None:
    """Ensure BAD extraction returns only flagged intervals."""

    # Build a raw instance with a mix of BAD and non-BAD annotations
    raw = _build_dummy_raw()
    # Set annotations combining valid labels and a BAD marker
    raw.set_annotations(
        mne.Annotations(
            onset=[0.2, 0.4, 0.8],
            duration=[0.1, 0.05, 0.05],
            description=["T1", "BAD_noise", "T2"],
        )
    )
    # Extract invalid intervals to validate filtering
    bad_intervals = _extract_bad_intervals(raw)
    # Confirm only the BAD window is returned with correct boundaries
    assert bad_intervals == [(0.4, 0.45)]


def test_extract_bad_intervals_returns_empty_when_no_bad() -> None:
    """Ensure no intervals are returned when annotations are clean."""

    # Build a raw instance with only valid annotations
    raw = _build_dummy_raw()
    # Extract intervals when no BAD tags are present
    bad_intervals = _extract_bad_intervals(raw)
    # Confirm the extraction yields an empty list
    assert bad_intervals == []


def test_extract_bad_intervals_raises_on_mismatched_lengths() -> None:
    """Ensure mismatched annotation fields raise under strict zip enforcement."""

    # Build an annotation container with inconsistent lengths
    annotations = SimpleNamespace(
        onset=[0.1],
        duration=[],
        description=["BAD_segment"],
    )
    # Build a lightweight raw-like object exposing annotations
    raw = SimpleNamespace(annotations=annotations)
    # Expect a ValueError due to zip strictness enforcing equal lengths
    with pytest.raises(ValueError):
        _extract_bad_intervals(raw)


def test_create_epochs_builds_clean_epochs(tmp_path: Path) -> None:
    """Check epochs creation integrates event validation."""

    # Build a raw recording with valid annotations
    raw = _build_dummy_raw()
    # Convert annotations to events using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Construct epochs and ensure no epoch is dropped unnecessarily
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.2)
    # Define the expected number of epochs for clarity in assertions
    expected_epoch_count = 2
    # Expect two epochs corresponding to the two annotations
    assert len(epochs) == expected_epoch_count
    # Ensure epoch labels reflect the annotation mapping
    assert set(epochs.events[:, 2]) == {1, 2}


def test_create_epochs_respects_custom_window(tmp_path: Path) -> None:
    """Ensure epoch timing arguments propagate to the Epochs object."""

    # Build a raw instance with valid annotations
    raw = _build_dummy_raw()
    # Map events using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Construct epochs with a custom window to detect mutation of defaults
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=-0.1, tmax=0.1)
    # Validate that the time window matches the provided bounds within tolerance
    assert epochs.tmin == pytest.approx(-0.1, abs=2e-3)
    # Confirm the maximum time reflects the requested limit within tolerance
    assert epochs.tmax == pytest.approx(0.1, abs=2e-3)


def test_create_epochs_uses_default_window() -> None:
    """Ensure the default epoch window remains unchanged."""

    # Build a raw instance with valid annotations
    raw = _build_dummy_raw()
    # Map events using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Construct epochs with default timing arguments
    epochs = create_epochs_from_raw(raw, events, event_id)
    # Confirm the default maximum time reflects the documented value
    assert epochs.tmax == pytest.approx(0.8, abs=1e-2)


def test_create_epochs_preserves_configuration_flags() -> None:
    """Ensure baseline and annotation rejection flags remain set."""

    # Build a raw instance with valid annotations
    raw = _build_dummy_raw()
    # Map events using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Construct epochs with default parameters
    epochs = create_epochs_from_raw(raw, events, event_id)
    # Confirm annotation rejection is enabled to drop BAD-labeled segments
    assert epochs.reject_by_annotation is True
    # Verify baseline remains disabled as configured
    assert epochs.baseline is None
    # Ensure the data are preloaded to allow immediate downstream processing
    assert epochs.preload is True


def test_create_epochs_rejects_events_over_bad_annotation() -> None:
    """Ensure epochs drop events overlapping BAD annotations."""

    # Build a raw instance with overlapping BAD annotation
    raw = _build_dummy_raw()
    # Attach a BAD annotation covering the second event
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(onset=[0.6], duration=[0.2], description=["BAD_segment"])
    )
    # Build events directly to include the contaminated event
    events, event_id = mne.events_from_annotations(raw, event_id=PHYSIONET_LABEL_MAP)
    # Create epochs while letting MNE reject events inside BAD spans
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Confirm the contaminated event has been rejected
    assert len(epochs) == 1


def test_verify_dataset_integrity_checks_hash_and_runs(tmp_path: Path) -> None:
    """Ensure integrity verification validates hashes and run counts."""

    # Create subject directory to mimic expected dataset structure
    subject_dir = tmp_path / "subject01"
    # Ensure the directory exists before placing the EDF placeholder
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Define path for the placeholder EDF recording
    edf_path = subject_dir / "run01.edf"
    # Write deterministic bytes to emulate a downloaded EDF file
    edf_path.write_bytes(b"edf-bytes")
    # Compute expected hash for the placeholder file
    expected_hash = mne.utils.hashfunc(edf_path, hash_type="sha256")
    # Verify integrity with expected hash and run count
    report = verify_dataset_integrity(
        tmp_path,
        expected_hashes={"subject01/run01.edf": expected_hash},
        expected_runs_per_subject={"subject01": 1},
    )
    # Confirm the report captures the exported file
    assert report["files"][0]["path"] == "subject01/run01.edf"
    # Ensure the hash check passed successfully
    assert report["files"][0]["hash_ok"] is True
    # Validate the recorded run count for the subject
    assert report["subject_run_counts"]["subject01"] == 1


def test_map_events_rejects_unknown_labels() -> None:
    """Ensure unknown labels raise a clear validation error."""

    # Create a raw recording with an unsupported annotation label
    raw = _build_dummy_raw()
    # Replace annotations to include an unknown class name
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1],
            duration=[0.1],
            description=["UNKNOWN"],
        )
    )
    # Expect the mapping helper to reject unexpected labels explicitly
    with pytest.raises(ValueError) as exc:
        map_events_and_validate(raw)
    # Confirm the error message originates from the explicit unknown label guard
    assert "Unknown labels" in str(exc.value)


def test_verify_dataset_integrity_missing_root_raises(tmp_path: Path) -> None:
    """Ensure missing dataset roots trigger an explicit error."""

    # Define a path that does not exist under the temporary directory
    missing_root = tmp_path / "absent_dataset"
    # Expect integrity verification to fail fast when the directory is absent
    with pytest.raises(FileNotFoundError):
        verify_dataset_integrity(missing_root)


def test_verify_dataset_integrity_run_mismatch_and_skip_files(tmp_path: Path) -> None:
    """Validate run count mismatches raise while non-directories are ignored."""

    # Create a stray file at the dataset root to trigger the skip branch
    stray_file = tmp_path / "unexpected.txt"
    # Write placeholder content to persist the stray file
    stray_file.write_text("extra")
    # Create a subject directory with fewer runs than expected
    subject_dir = tmp_path / "subject02"
    # Ensure the subject directory exists for run counting
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Write one EDF file while expecting two runs
    (subject_dir / "run01.edf").write_bytes(b"edf-run")
    # Expect a ValueError when run counts do not match expectations
    with pytest.raises(ValueError):
        verify_dataset_integrity(tmp_path, expected_runs_per_subject={"subject02": 2})


def test_build_file_entry_without_expected_hashes(tmp_path: Path) -> None:
    """Ensure file entry omits hashes when no expectations are provided."""

    # Create a directory to host a dummy EDF file
    data_root = tmp_path / "root"
    # Ensure the root exists prior to writing
    data_root.mkdir(parents=True, exist_ok=True)
    # Write a minimal EDF placeholder file
    file_path = data_root / "sample.edf"
    # Write content to permit size measurement
    file_path.write_bytes(b"edf")
    # Build the file entry without expected hashes
    entry = _build_file_entry(data_root, file_path, expected_hashes=None)
    # Confirm the SHA field remains unset when no reference hash exists
    assert entry["sha256"] is None
    # Validate that hash_ok is considered true in the absence of expectations
    assert entry["hash_ok"] is True


def test_build_file_entry_with_expected_hashes(tmp_path: Path) -> None:
    """Ensure hash computation and relative paths are reported."""

    # Create a dataset root directory
    data_root = tmp_path / "physionet"
    # Ensure the directory hierarchy exists
    data_root.mkdir(parents=True, exist_ok=True)
    # Create an EDF placeholder file with deterministic content
    file_path = data_root / "subjectA" / "run01.edf"
    # Ensure parent directories exist before writing
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes to compute a reproducible hash
    file_path.write_bytes(b"edf-hash")
    # Compute the expected hash for comparison
    expected_hash = mne.utils.hashfunc(file_path, hash_type="sha256")
    # Build the file entry with the expected hash mapping
    entry = _build_file_entry(
        data_root, file_path, expected_hashes={"subjectA/run01.edf": expected_hash}
    )
    # Confirm the hash matches the computed value
    assert entry["sha256"] == expected_hash
    # Ensure the relative path matches the dataset layout
    assert entry["path"] == "subjectA/run01.edf"
    # Validate the hash check outcome is positive
    assert entry["hash_ok"] is True
    # Confirm the recorded size matches the file contents
    assert entry["size"] == len(b"edf-hash")


def test_verify_dataset_integrity_flags_hash_mismatch(tmp_path: Path) -> None:
    """Ensure hash mismatches are reported without raising."""

    # Create subject directory for the sample dataset
    subject_dir = tmp_path / "subject03"
    # Ensure the directory hierarchy exists
    subject_dir.mkdir(parents=True, exist_ok=True)
    # Create an EDF placeholder file with deterministic bytes
    edf_path = subject_dir / "run01.edf"
    # Write content to compute an actual hash
    edf_path.write_bytes(b"edf-run")
    # Provide an incorrect expected hash to trigger mismatch tracking
    wrong_hash = "0" * 64
    # Run integrity verification with the incorrect hash
    report = verify_dataset_integrity(
        tmp_path,
        expected_hashes={"subject03/run01.edf": wrong_hash},
        expected_runs_per_subject={"subject03": 1},
    )
    # Confirm the hash mismatch is recorded in the report
    assert report["files"][0]["hash_ok"] is False


def test_collect_run_counts_ignores_files(tmp_path: Path) -> None:
    """Ensure run counting ignores non-directory entries."""

    # Create a dataset root containing a file and a subject directory
    root = tmp_path / "dataset"
    # Ensure the root exists before writing content
    root.mkdir(parents=True, exist_ok=True)
    # Write a stray file that should be ignored
    (root / "note.txt").write_text("note")
    # Create a subject directory with multiple EDF files
    subj = root / "subjectX"
    # Ensure the subject directory exists
    subj.mkdir(parents=True, exist_ok=True)
    # Write two EDF placeholders to represent runs
    (subj / "run1.edf").write_bytes(b"r1")
    # Write a second EDF file to ensure counting reflects both runs
    (subj / "run2.edf").write_bytes(b"r2")
    # Count runs using the helper
    counts = _collect_run_counts(root)
    # Confirm only the subject directory contributes to the counts
    assert counts == {"subjectX": 2}


def test_verify_dataset_integrity_reports_root_and_file_count(tmp_path: Path) -> None:
    """Ensure integrity reports include root path and discovered files."""

    # Create dataset structure with a single EDF file
    data_root = tmp_path / "dataset"
    # Ensure the dataset root exists
    data_root.mkdir(parents=True, exist_ok=True)
    # Write an EDF placeholder to be discovered
    edf_path = data_root / "record.edf"
    # Persist bytes to register file size
    edf_path.write_bytes(b"edf")
    # Run integrity verification without expected metadata
    report = verify_dataset_integrity(data_root)
    # Confirm the report exposes the root path key
    assert report["root"] == str(data_root.resolve())
    # Ensure exactly one file entry was recorded
    assert len(report["files"]) == 1
