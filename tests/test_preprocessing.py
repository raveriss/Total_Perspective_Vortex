"""Unit tests for Physionet preprocessing helpers."""

# Import pathlib to construct temporary dataset layouts
# Import pathlib to type temporary paths for datasets
# Import json to inspect structured error payloads
import json
from pathlib import Path

# Import SimpleNamespace to build lightweight annotation holders
from types import SimpleNamespace

# Import Mapping to annotate captured motor mapping structures
from typing import Mapping

# Import mne to build synthetic Raw objects and annotations
import mne

# Import numpy to craft deterministic dummy EEG data
import numpy as np

# Import pytest to manage temporary directories and assertions
import pytest

# Import the preprocessing helpers under test
from tpv.preprocessing import (
    MOTOR_EVENT_LABELS,
    PHYSIONET_LABEL_MAP,
    _build_file_entry,
    _build_keep_mask,
    _collect_run_counts,
    _extract_bad_intervals,
    _is_bad_description,
    create_epochs_from_raw,
    generate_epoch_report,
    load_mne_raw_checked,
    load_physionet_raw,
    map_events_and_validate,
    quality_control_epochs,
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
    # Annotate three events representing motor imagery tasks
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.3, 0.6],
            duration=[0.1, 0.1, 0.1],
            description=["T0", "T1", "T2"],
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
    raw = _build_dummy_raw(sfreq=100.0)
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
    raw = _build_dummy_raw(sfreq=100.0)
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


def test_load_mne_raw_checked_validates_sampling_and_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the MNE loader enforces montage, sampling, and channels."""

    # Build a deterministic raw array matching expected montage and channels
    raw = _build_dummy_raw(sfreq=256.0)
    # Capture montage applications to ensure configuration is enforced
    montage_calls: list[tuple[str, dict[str, object]]] = []

    # Record montage invocations while preserving actual behavior
    def montage_stub(name: str, **kwargs: object) -> None:
        # Track the call to validate montage propagation
        montage_calls.append((name, kwargs))
        # Delegate to the true method to keep montage attachment intact
        mne.io.Raw.set_montage(raw, name, **kwargs)

    # Patch the EDF reader to return the synthetic recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Patch set_montage to observe invocation parameters
    monkeypatch.setattr(raw, "set_montage", montage_stub)
    # Execute the validated loader with matching expectations
    loaded_raw = load_mne_raw_checked(
        Path("record.edf"),
        expected_montage="standard_1020",
        expected_sampling_rate=256.0,
        expected_channels=["C3", "C4"],
    )
    # Confirm the returned object is the original Raw instance
    assert loaded_raw is raw
    # Ensure the montage setter was called with the expected configuration
    assert montage_calls == [("standard_1020", {"on_missing": "warn"})]


def test_load_mne_raw_checked_raises_on_sampling_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure sampling rate mismatches raise clear errors."""

    # Build a raw object with a known sampling frequency
    raw = _build_dummy_raw(sfreq=128.0)
    # Stub the EDF reader to return the synthetic recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Expect a ValueError when the sampling rate differs from the expectation
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage="standard_1020",
            expected_sampling_rate=512.0,
            expected_channels=["C3", "C4"],
        )
    # Confirm the message explains the sampling rate discrepancy
    assert "Expected sampling rate" in str(exc.value)


def test_map_events_and_validate_rejects_unknown_labels() -> None:
    """Ensure event mapping fails when annotations contain unknown labels."""

    # Build a dummy raw instance with an unsupported label to trigger validation
    raw = _build_dummy_raw()
    # Replace annotations with an invalid label to exercise the error path
    raw.set_annotations(
        mne.Annotations(onset=[0.1], duration=[0.1], description=["TX"])
    )
    # Expect a ValueError when the annotation label is absent from the mapping
    with pytest.raises(ValueError) as exc:
        map_events_and_validate(raw, label_map=PHYSIONET_LABEL_MAP)
    # Confirm the message surfaces the unknown label for debugging
    assert "Unknown labels" in str(exc.value)


def test_quality_control_and_reporting(tmp_path: Path) -> None:
    """Ensure artifact epochs are removed and reports capture remaining counts."""

    # Build a dummy raw instance to generate events and epochs
    raw = _build_dummy_raw()
    # Map annotations to events with validation to obtain event identifiers
    events, event_id = map_events_and_validate(raw)
    # Create epochs around the events with a non-negative start to keep them valid
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.3)
    # Amplify the first epoch to simulate an artifact exceeding the threshold
    epoch_data = epochs.get_data(copy=False)
    # Inject a large positive swing on the first channel to inflate peak-to-peak
    epoch_data[0, 0, 0] = 50.0
    # Inject a large negative swing on the second channel to widen amplitude
    epoch_data[0, 1, 0] = -50.0
    # Apply quality control with rejection to drop the corrupted epoch
    filtered_epochs, flagged = quality_control_epochs(epochs, max_peak_to_peak=10.0)
    # Confirm the first epoch was flagged as an artifact for removal
    assert flagged["artifact"] == [0]
    # Confirm no incomplete epochs were detected in the synthetic data
    assert flagged["incomplete"] == []
    # Generate a JSON report summarizing the surviving epoch counts
    report_path = tmp_path / "reports" / "summary.json"
    # Persist the report to disk for later inspection
    generate_epoch_report(
        filtered_epochs,
        event_id,
        {"subject": "S01", "run": "R01"},
        report_path,
    )
    # Load the report to verify the counts reflect the filtered epochs
    report_content = json.loads(report_path.read_text(encoding="utf-8"))
    # Confirm subject and run identifiers are preserved in the report
    assert report_content["subject"] == "S01"
    # Confirm run identifier matches the provided input
    assert report_content["run"] == "R01"
    # Confirm only two epochs remain after dropping the artifact
    remaining_epochs = 2
    assert report_content["total_epochs"] == remaining_epochs
    # Confirm each remaining label appears exactly once in the counts
    assert report_content["counts"] == {"T0": 0, "T1": 1, "T2": 1}


def test_quality_control_marking_handles_incomplete_epochs() -> None:
    """Confirm marking mode annotates incomplete epochs without dropping them."""

    # Build a dummy raw instance to generate events and epochs
    raw = _build_dummy_raw()
    # Map annotations to events with validation to obtain event identifiers
    events, event_id = map_events_and_validate(raw)
    # Create epochs around the events with a non-negative start to keep them valid
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.3)
    # Inject a NaN to trigger the incomplete flag while preserving shape
    epoch_data = epochs.get_data(copy=False)
    epoch_data[0, 0, 0] = np.nan
    # Apply quality control in marking mode to retain epochs
    marked_epochs, flagged = quality_control_epochs(
        epochs, max_peak_to_peak=10.0, mode="mark"
    )
    # Confirm the first epoch was flagged as incomplete instead of dropped
    assert flagged["incomplete"] == [0]
    # Confirm metadata was created and records the incomplete status
    assert marked_epochs.metadata is not None
    assert marked_epochs.metadata.loc[0, "quality_flag"] == "incomplete"


def test_quality_control_rejects_invalid_mode() -> None:
    """Ensure unsupported mode strings raise a clear ValueError."""

    # Build a dummy raw instance to generate events and epochs
    raw = _build_dummy_raw()
    # Map annotations to events with validation to obtain event identifiers
    events, event_id = map_events_and_validate(raw)
    # Create epochs around the events with a non-negative start to keep them valid
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.3)
    # Expect a ValueError when an unknown mode is supplied
    with pytest.raises(ValueError) as exc:
        quality_control_epochs(epochs, max_peak_to_peak=10.0, mode="unknown")
    assert "mode must be either" in str(exc.value)


def test_generate_epoch_report_outputs_csv_and_validates_format(tmp_path: Path) -> None:
    """Validate CSV export and the error path for unsupported formats."""

    # Build a dummy raw instance to generate events and epochs
    raw = _build_dummy_raw()
    # Map annotations to events with validation to obtain event identifiers
    events, event_id = map_events_and_validate(raw)
    # Create epochs around the events with a non-negative start to keep them valid
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.3)
    # Generate a CSV report to exercise the CSV serialization path
    csv_path = tmp_path / "reports" / "summary.csv"
    output_path = generate_epoch_report(
        epochs,
        event_id,
        {"subject": "S02", "run": "R02"},
        csv_path,
        fmt="csv",
    )
    # Confirm the CSV file exists and contains per-label counts
    csv_content = output_path.read_text(encoding="utf-8").splitlines()
    assert csv_content[0] == "subject,run,label,count"
    assert "S02,R02,T0," in csv_content[1]
    # Expect a ValueError when an unsupported format is requested
    with pytest.raises(ValueError) as exc:
        generate_epoch_report(
            epochs, event_id, {"subject": "S02", "run": "R02"}, csv_path, fmt="xml"
        )
    assert "fmt must be either" in str(exc.value)


def test_load_mne_raw_checked_raises_when_montage_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure a missing montage triggers a descriptive error."""

    # Build a deterministic raw array to patch montage behavior
    raw = _build_dummy_raw(sfreq=128.0)
    # Stub the EDF reader to return the synthetic recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Patch set_montage to bypass montage attachment silently
    monkeypatch.setattr(raw, "set_montage", lambda *_args, **_kwargs: None)
    # Patch get_montage to mimic a failed attachment
    monkeypatch.setattr(raw, "get_montage", lambda: None)
    # Expect the loader to raise when montage application fails
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4"],
        )
    # Confirm the error message cites the missing montage
    assert "could not be applied" in str(exc.value)


def test_load_mne_raw_checked_reports_channel_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure channel inconsistencies raise structured errors."""

    # Build a raw object whose channels differ from the expected layout
    raw = _build_dummy_raw()
    # Extend the channels to trigger the mismatch handler
    raw.rename_channels({"C4": "C4-extra"})
    # Stub the EDF reader to return the modified recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Expect the loader to raise because channels do not match expectations
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4"],
        )
    # Confirm the structured error highlights missing channel names
    assert "missing" in str(exc.value)


def test_load_mne_raw_checked_resolves_path_and_reader_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the loader normalizes paths and forwards reader kwargs."""

    # Build a deterministic raw object to reuse in the reader stub
    raw = _build_dummy_raw()
    # Capture reader arguments to verify path normalization and flags
    captured_args: list[tuple[tuple[object, ...], dict[str, object]]] = []

    # Record reader invocations while returning the prepared raw object
    def reader_stub(*args: object, **kwargs: object) -> mne.io.Raw:
        # Store positional and keyword arguments for later assertions
        captured_args.append((args, kwargs))
        # Provide the synthetic raw to satisfy loader expectations
        return raw

    # Patch the EDF reader with the recording stub
    monkeypatch.setattr(mne.io, "read_raw_edf", reader_stub)
    # Use a path containing a tilde to force expansion and resolution
    edf_path = Path("~") / "relative" / "record.edf"
    # Invoke the loader with matching expectations to avoid validation errors
    load_mne_raw_checked(
        edf_path,
        expected_montage="standard_1020",
        expected_sampling_rate=128.0,
        expected_channels=["C3", "C4"],
    )
    # Ensure the reader received the expanded and resolved path
    assert captured_args[0][0][0] == edf_path.expanduser().resolve()
    # Confirm preload and verbosity flags are forwarded unchanged
    assert captured_args[0][1] == {"preload": True, "verbose": False}


def test_load_mne_raw_checked_flags_extra_only_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure extra channels alone trigger the mismatch error."""

    # Build a raw object containing both expected channels
    raw = _build_dummy_raw()
    # Stub the EDF reader to return the prepared recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Expect a ValueError because an extra channel is present without missing ones
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3"],
        )
    # Confirm the error payload documents the unexpected channel
    assert "extra" in str(exc.value)


def test_load_mne_raw_checked_flags_missing_only_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure missing channels without extras raise a structured error."""

    # Build a raw object missing a channel expected by the caller
    raw = _build_dummy_raw()
    # Stub the EDF reader to return the prepared recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Expect a ValueError because one expected channel is absent
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4", "Cz"],
        )
    # Parse the JSON payload to inspect the structured error contents
    payload = json.loads(str(exc.value))
    # Confirm the top-level error message remains stable for CI assertions
    assert payload["error"] == "Channel mismatch"
    # Ensure no extra channels are reported when only missing channels occur
    assert payload["extra"] == []
    # Ensure the missing list highlights the absent channel explicitly
    assert payload["missing"] == ["Cz"]


def test_map_events_validates_motor_label_mapping() -> None:
    """Ensure motor label mapping enforces A/B coverage and known keys."""

    # Build a raw instance with default annotations for motor imagery
    raw = _build_dummy_raw()
    # Expect a ValueError when motor mapping omits required labels
    with pytest.raises(ValueError) as exc_missing:
        map_events_and_validate(raw, motor_label_map={"T1": "A"})
    # Confirm the message lists the missing event label
    assert "missing labels for events" in str(exc_missing.value)
    # Expect a ValueError when motor mapping includes invalid targets
    with pytest.raises(ValueError) as exc_invalid:
        map_events_and_validate(raw, motor_label_map={"T1": "Left", "T2": "B"})
    # Confirm the message identifies the unsupported motor label value
    assert "Motor labels must be within" in str(exc_invalid.value)
    # Expect a ValueError when motor mapping omits the B target
    with pytest.raises(ValueError) as exc_missing_target:
        map_events_and_validate(raw, motor_label_map={"T1": "A", "T2": "A"})
    # Confirm the error reports the missing motor imagery target
    assert "missing ['B']" in str(exc_missing_target.value)
    # Expect a ValueError when motor mapping references unknown event keys
    with pytest.raises(ValueError) as exc_unknown_key:
        map_events_and_validate(raw, motor_label_map={"T1": "A", "T2": "B", "X3": "A"})
    # Confirm the error highlights the extraneous motor mapping key
    assert "unknown events" in str(exc_unknown_key.value)


def test_map_events_invokes_motor_validation_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure motor validation runs when motor labels appear in annotations."""

    # Build a raw recording that contains default motor imagery labels
    raw = _build_dummy_raw()
    # Track motor validation invocations to assert that validation is enforced
    motor_calls: list[Mapping[str, str]] = []

    # Record the motor mapping provided to the validation helper
    def motor_stub(
        raw_arg: mne.io.Raw,
        effective_label_map: Mapping[str, int],
        motor_label_map: Mapping[str, str],
    ) -> Mapping[str, str]:
        # Store the received motor mapping for later assertions
        motor_calls.append(dict(motor_label_map))
        # Return the mapping unchanged to preserve downstream behavior
        return motor_label_map

    # Patch the validation helper to record invocations without altering logic
    monkeypatch.setattr("tpv.preprocessing._validate_motor_mapping", motor_stub)
    # Map events using the default annotation set
    events, event_id = map_events_and_validate(raw)
    # Confirm the events remain intact when validation succeeds
    assert events.shape[0] == len(PHYSIONET_LABEL_MAP)
    # Ensure the event mapping preserves the Physionet defaults
    assert event_id == PHYSIONET_LABEL_MAP
    # Verify the motor validation helper was invoked once with default mapping
    assert motor_calls == [MOTOR_EVENT_LABELS]


def test_map_events_skips_motor_validation_when_no_motor_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure motor validation is bypassed when only BAD labels exist."""

    # Build a raw recording to host custom BAD-only annotations
    raw = _build_dummy_raw()
    # Override annotations so no motor labels are present
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1], duration=[0.1], description=["BAD"], orig_time=None
        )
    )
    # Raise if motor validation is invoked when it should be skipped
    monkeypatch.setattr(
        "tpv.preprocessing._validate_motor_mapping",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("motor validation should not run")
        ),
    )
    # Stub event extraction to avoid relying on unknown labels
    monkeypatch.setattr(
        mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (np.empty((0, 3), int), {"BAD": 0}),
    )
    # Map events using the BAD-only annotations
    events, event_id = map_events_and_validate(raw, label_map={"BAD": 0})
    # Confirm no events remain because only BAD annotations were present
    assert events.shape[0] == 0
    # Ensure the returned mapping matches the provided label map
    assert event_id == {"BAD": 0}


def test_map_events_handles_mixed_case_bad_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure mixed-case BAD annotations are ignored during validation."""

    # Build a raw recording with a mixed-case BAD annotation only
    raw = _build_dummy_raw()
    # Override annotations to include solely a mixed-case BAD marker
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1], duration=[0.1], description=["Bad_artifact"], orig_time=None
        )
    )
    # Stub event extraction to avoid mismatches for the non-labeled annotation
    monkeypatch.setattr(
        mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (np.empty((0, 3), int), {}),
    )
    # Execute mapping with an empty label map to focus on annotation filtering
    events, event_id = map_events_and_validate(raw, label_map={})
    # Confirm no events are produced because only BAD annotations were present
    assert events.shape[0] == 0
    # Ensure the returned mapping remains empty as provided
    assert event_id == {}


def test_is_bad_description_is_case_insensitive() -> None:
    """Ensure BAD detection tolerates mixed casing."""

    # Confirm mixed-case BAD markers are detected correctly
    assert _is_bad_description("Bad_marker") is True
    # Ensure regular event labels are not mislabeled as BAD
    assert _is_bad_description("T1") is False


def test_map_events_filters_bad_segments(tmp_path: Path) -> None:
    """Validate event mapping drops events overlapping BAD intervals."""

    # Create a synthetic raw recording with annotations
    raw = _build_dummy_raw(sfreq=100.0)
    # Extend annotations with a BAD interval covering the second event
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(onset=[0.25], duration=[0.2], description=["BAD_segment"])
    )
    # Derive events and event_id using the validation helper
    events, event_id = map_events_and_validate(raw)
    # Ensure the label map matches the Physionet default even when labels are missing
    assert event_id == PHYSIONET_LABEL_MAP
    # Confirm the first and last events remain after BAD filtering
    expected_event_count = 2
    assert events.shape[0] == expected_event_count
    # Verify the remaining events correspond to the first annotation
    assert events[0, 0] == pytest.approx(round(0.1 * raw.info["sfreq"]))
    # Verify the remaining events include the third annotation
    assert events[1, 0] == pytest.approx(round(0.6 * raw.info["sfreq"]))


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


def test_map_events_includes_bad_interval_end_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure events ending exactly on a BAD boundary are removed."""

    # Build a raw instance with deterministic sampling for boundary alignment
    raw = _build_dummy_raw(sfreq=100.0)
    # Define a boundary-matching event time at exactly one second
    boundary_sample = int(raw.info["sfreq"] * 1.0)
    # Compute the exact event time produced by the boundary sample
    boundary_time = boundary_sample / raw.info["sfreq"]
    # Prepare a single event that lands precisely on the BAD interval end
    stub_events = np.array([[boundary_sample, 0, PHYSIONET_LABEL_MAP["T1"]]])
    # Stub BAD intervals to end exactly at the event time
    stub_bad_intervals = [(boundary_time, boundary_time)]

    # Patch annotation extraction to return the boundary interval
    monkeypatch.setattr(
        "tpv.preprocessing._extract_bad_intervals", lambda _raw: stub_bad_intervals
    )
    # Patch events_from_annotations to return the boundary event deterministically
    monkeypatch.setattr(
        mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (stub_events, PHYSIONET_LABEL_MAP),
    )

    # Map events using the patched helpers to enforce boundary overlap
    events, _ = map_events_and_validate(raw)
    # Confirm the boundary event is removed due to end-inclusive filtering
    assert events.shape[0] == 0


def test_map_events_passes_verbose_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure events_from_annotations receives verbose=False."""

    # Build a raw instance with standard annotations
    raw = _build_dummy_raw()
    # Prepare a placeholder event array to return from the stub
    stub_events = np.array([[1, 0, PHYSIONET_LABEL_MAP["T1"]]])
    # Capture keyword arguments passed to events_from_annotations
    captured_kwargs: dict[str, object] = {}

    # Define a stub that records its keyword arguments
    def events_stub(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, dict[str, int]]:
        # Store the keyword arguments for later assertion
        captured_kwargs.update(kwargs)
        # Return the prepared events and label map to mimic MNE behavior
        return stub_events, dict(PHYSIONET_LABEL_MAP)

    # Patch the MNE helper with the recording stub
    monkeypatch.setattr(mne, "events_from_annotations", events_stub)
    # Map events using the preprocessing helper to trigger the stub
    events, event_id = map_events_and_validate(raw)
    # Confirm the stub returned events are propagated to the caller
    assert np.array_equal(events, stub_events)
    # Ensure the effective label map is preserved from the stub
    assert event_id == PHYSIONET_LABEL_MAP
    # Verify the verbose flag is explicitly set to False
    assert captured_kwargs["verbose"] is False


def test_map_events_raises_when_keep_mask_length_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure strict zip raises when keep_mask length diverges from events."""

    # Build a raw instance with standard annotations for mapping
    raw = _build_dummy_raw()
    # Patch the mask builder to return a shorter mask than the events length
    monkeypatch.setattr(
        "tpv.preprocessing._build_keep_mask", lambda *_args, **_kwargs: [True]
    )
    # Expect a ValueError due to strict zip enforcing equal lengths
    with pytest.raises(ValueError):
        map_events_and_validate(raw)


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
    expected_event_count = 3
    # Ensure all events are retained when no BAD intervals exist
    assert events.shape[0] == expected_event_count


def test_build_keep_mask_rejects_non_boolean_forced_masks() -> None:
    """Ensure forced masks containing non-bools raise an explicit error."""

    # Build a minimal event matrix with a single sample for masking
    events = np.array([[100, 0, 1]])
    # Provide a mask containing an invalid non-boolean entry
    forced_mask = ["invalid"]
    # Expect a TypeError because the forced mask violates boolean typing
    with pytest.raises(TypeError) as exc:
        _build_keep_mask(
            events, sampling_rate=100.0, bad_intervals=[], forced_mask=forced_mask
        )
    # Confirm the error message matches the strict validation text
    assert str(exc.value) == "Event mask contained non-boolean values"


def test_build_keep_mask_copies_valid_forced_mask() -> None:
    """Ensure valid forced masks are copied and preserved as booleans."""

    # Build a minimal event matrix for masking
    events = np.array([[100, 0, 1], [200, 0, 2]])
    # Provide a valid boolean mask to force deterministic filtering
    forced_mask = [True, False]
    # Invoke the mask builder to apply the forced mask
    keep_mask = _build_keep_mask(
        events, sampling_rate=100.0, bad_intervals=[], forced_mask=forced_mask
    )
    # Confirm the returned mask matches the forced mask values
    assert keep_mask == forced_mask
    # Ensure the returned mask is a distinct copy from the caller-provided list
    assert keep_mask is not forced_mask


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
    expected_epoch_count = 3
    # Expect three epochs corresponding to the annotations
    assert len(epochs) == expected_epoch_count
    # Ensure epoch labels reflect the annotation mapping
    assert set(epochs.events[:, 2]) == {0, 1, 2}


def test_create_epochs_passes_configuration_to_mne(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure create_epochs_from_raw forwards critical arguments."""

    # Build a raw instance with valid annotations for epoching
    raw = _build_dummy_raw()
    # Convert annotations to events for the stubbed Epochs call
    events, event_id = map_events_and_validate(raw)
    # Capture the time window to avoid magic-number assertions
    custom_window = (-0.2, 0.4)
    # Capture the arguments passed to the Epochs constructor
    captured_args: tuple[tuple[object, ...], dict[str, object]] = ((), {})
    # Prepare a sentinel object to return from the stubbed Epochs
    sentinel = object()

    # Define a stub for mne.Epochs that records invocation details
    def epochs_stub(*args: object, **kwargs: object) -> object:
        # Persist the positional and keyword arguments for assertions
        nonlocal captured_args
        captured_args = (args, kwargs)
        # Return a sentinel value to confirm stub propagation
        return sentinel

    # Patch mne.Epochs with the recording stub
    monkeypatch.setattr(mne, "Epochs", epochs_stub)
    # Invoke epoch creation to trigger the stubbed constructor
    result = create_epochs_from_raw(
        raw, events, event_id, tmin=custom_window[0], tmax=custom_window[1]
    )
    # Confirm the stub result is returned unchanged
    assert result is sentinel
    # Extract positional and keyword arguments from the captured call
    args, kwargs = captured_args
    # Verify the raw object is the first positional argument
    assert args[0] is raw
    # Confirm the events array is forwarded without mutation after coercion
    assert np.array_equal(np.asarray(kwargs["events"]), events)
    # Ensure the event_id mapping is provided to the constructor
    assert kwargs["event_id"] == event_id
    # Confirm the time window honors the values provided by the caller
    assert kwargs["tmin"] == custom_window[0]
    # Validate that the maximum window aligns with the requested end time
    assert kwargs["tmax"] == custom_window[1]
    # Validate that annotation-based rejection remains enabled
    assert kwargs["reject_by_annotation"] is True
    # Confirm missing labels are ignored to avoid errors on incomplete runs
    assert kwargs["on_missing"] == "ignore"
    # Confirm verbosity is explicitly silenced
    assert kwargs["verbose"] is False
    # Ensure epochs are preloaded to support downstream processing
    assert kwargs["preload"] is True
    # Verify baseline remains disabled to prevent unintended correction
    assert kwargs["baseline"] is None


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
    # Confirm only the unaffected events remain after rejection
    expected_epoch_count = 2
    assert len(epochs) == expected_epoch_count


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
    with pytest.raises(FileNotFoundError) as exc:
        verify_dataset_integrity(missing_root)
    # Confirm the error message includes the missing root path
    assert str(exc.value) == f"Dataset directory not found: {missing_root.resolve()}"


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
    with pytest.raises(ValueError) as exc:
        verify_dataset_integrity(tmp_path, expected_runs_per_subject={"subject02": 2})
    # Confirm the mismatch report highlights the affected subject
    assert "subject02" in str(exc.value)


def test_collect_run_counts_continues_after_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure stray files do not halt run counting for later subjects."""

    # Create fake path objects to control iteration order explicitly
    file_path = Path("/data/note.txt")
    # Build a subject directory placeholder with EDF files
    subject_dir = Path("/data/subjectY")
    # Prepare a subject directory structure in a temporary folder
    tmp_root = Path("/data")

    # Define a stub for iterdir returning a file before the subject
    def iterdir_stub() -> list[Path]:
        # Return a list where the file appears before the directory
        return [file_path, subject_dir]

    # Define a stub for is_dir to mark only the subject path as a directory
    def is_dir_stub(self: Path) -> bool:
        # Treat the subject directory as a directory and the note as a file
        return self == subject_dir

    # Define a stub for glob that yields two EDF files for the subject
    def glob_stub(self: Path, pattern: str):
        # Yield EDF files only when invoked on the subject directory
        if self != subject_dir:
            # Return an empty iterator for non-subject paths
            return iter([])
        # Return two EDF paths to represent available runs
        return iter([self / "run1.edf", self / "run2.edf"])

    # Confirm the stub returns no files when applied to non-subject paths
    assert list(glob_stub(file_path, "*.edf")) == []

    # Patch Path.iterdir, Path.is_dir, and Path.glob to use the stubs
    monkeypatch.setattr(Path, "iterdir", lambda self: iterdir_stub())
    monkeypatch.setattr(Path, "is_dir", is_dir_stub)
    monkeypatch.setattr(Path, "glob", glob_stub)
    # Invoke run counting on the stubbed dataset root
    counts = _collect_run_counts(tmp_root)
    # Confirm run counting continues past the stray file
    assert counts == {"subjectY": 2}


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
