"""Unit tests for Physionet preprocessing helpers."""

# Import pathlib to construct temporary dataset layouts
# Import pathlib to type temporary paths for datasets
from pathlib import Path

# Import mne to build synthetic Raw objects and annotations
import mne

# Import numpy to craft deterministic dummy EEG data
import numpy as np

# Import pytest to manage temporary directories and assertions
import pytest

# Import the preprocessing helpers under test
from tpv.preprocessing import (
    PHYSIONET_LABEL_MAP,
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
