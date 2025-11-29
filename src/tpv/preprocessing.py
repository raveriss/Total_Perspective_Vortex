"""Utilities for loading and validating Physionet EEG datasets.

Filtrage EEG 8–40 Hz (FIR/IIR) avec padding (FIR auto ~2–4*fs, IIR ordre 4).

Le filtre par défaut suit la contrainte WBS 3.1.1 : bande 8–40 Hz avec FIR
zero-phase (longueur auto MNE équivalente à un ordre ~401 sur des segments
>1s) ou IIR Butterworth (ordre 4) et un padding réfléchissant de 0.5 seconde
pour limiter les effets de bord sur les segments fenêtrés.
"""

# Garantit la compatibilité des annotations de type pour l’ensemble du module
from __future__ import annotations

# Préserve les fonctions de hachage pour valider l’intégrité des datasets
import hashlib

# Conserve json pour formater les rapports d’erreurs de comptage de runs
import json

# Utilise pathlib pour assurer la portabilité des interactions fichiers
from pathlib import Path

# Centralise les hints pour clarifier les attentes des appels et des tests
from typing import Any, Dict, List, Mapping, Tuple

# MNE est obligatoire pour le parsing EDF/BDF et la gestion des epochs
import mne

# Numpy offre des masques vectorisés pour filtrer rapidement les événements
import numpy as np

# Pandas gère les métadonnées annotées pour le contrôle qualité
import pandas as pd

# Provide a default mapping consistent with Physionet motor imagery labels
PHYSIONET_LABEL_MAP: Dict[str, int] = {"T0": 0, "T1": 1, "T2": 2}
# Provide an explicit mapping from Physionet events to motor imagery labels
MOTOR_EVENT_LABELS: Dict[str, str] = {"T1": "A", "T2": "B"}


def apply_bandpass_filter(
    raw: mne.io.BaseRaw,
    method: str = "fir",
    freq_band: Tuple[float, float] = (8.0, 40.0),
    order: int | str | None = None,
    pad_duration: float = 0.5,
) -> mne.io.BaseRaw:
    """Apply a padded 8–40 Hz band-pass filter using FIR or IIR designs."""

    # Clone the raw object to avoid mutating caller buffers during filtering
    filtered_raw = raw.copy().load_data()
    # Normalize the method string to simplify downstream comparisons
    normalized_method = method.lower()
    # Enforce supported methods to avoid silent fallbacks inside MNE
    if normalized_method not in {"fir", "iir"}:
        # Raise early to force callers to pick an explicit filter family
        raise ValueError("method must be 'fir' or 'iir'")
    # Extract the sampling frequency to derive padding and design parameters
    sampling_rate = float(filtered_raw.info["sfreq"])
    # Décompose la bande en fréquences basse et haute pour le filtrage
    l_freq, h_freq = freq_band
    # Translate the padding duration into sample counts for symmetrical padding
    pad_samples = max(int(round(pad_duration * sampling_rate)), 0)
    # Fetch the data array once to avoid repeated MNE access overhead
    data = filtered_raw.get_data()
    # Build a reflect-padded buffer to minimize edge artifacts during filtering
    if pad_samples > 0:
        # Use symmetric reflection to keep boundary continuity without phase jumps
        padded_data = np.pad(data, ((0, 0), (pad_samples, pad_samples)), mode="reflect")
    else:
        # Skip padding when the caller explicitly disables it via pad_duration=0.0
        padded_data = data
    # Prepare FIR-specific arguments when a linear-phase design is required
    if normalized_method == "fir":
        # Sélectionne l’ordre FIR explicite ou l’auto-calcul par défaut
        fir_length = order if order is not None else "auto"
        # Configure FIR parameters to balance roll-off and latency for MI bands
        filter_kwargs: Dict[str, Any] = {
            "method": "fir",
            "fir_design": "firwin",
            "fir_window": "hamming",
            "filter_length": fir_length,
            "phase": "zero-double",
        }
    else:
        # Définit l’ordre IIR à 4 par défaut pour limiter la latence
        iir_order = int(order) if order is not None else 4
        # Configure a Butterworth IIR design to minimize latency during streaming
        filter_kwargs = {
            "method": "iir",
            "iir_params": {"order": iir_order, "ftype": "butter"},
            "phase": "zero-double",
        }
    # Apply the selected filter to the padded buffer to obtain band-limited data
    filtered_data = mne.filter.filter_data(
        padded_data,
        sfreq=sampling_rate,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False,
        **filter_kwargs,
    )
    # Remove artificial padding to restore the original signal duration
    if pad_samples > 0:
        # Slice out the central segment corresponding to the unpadded recording
        filtered_data = filtered_data[:, pad_samples:-pad_samples]
    # Assign the filtered samples back into the cloned Raw object for return
    filtered_raw._data = filtered_data
    # Return the filtered recording to feed downstream epoching and features
    return filtered_raw


def _is_bad_description(description: str) -> bool:
    """Return True when an annotation description denotes a BAD marker."""

    # Normalize the description to uppercase for case-insensitive detection
    normalized_description = description.upper()
    # Identify MNE BAD-prefixed annotations regardless of original casing
    return normalized_description.startswith("BAD")


def load_mne_raw_checked(
    file_path: Path,
    expected_montage: str,
    expected_sampling_rate: float,
    expected_channels: List[str],
) -> mne.io.BaseRaw:
    """Load a raw MNE file and validate montage, sampling rate, and channels."""

    # Normalize the file path to avoid surprises from relative inputs
    normalized_path = Path(file_path).expanduser().resolve()
    # Load the raw file with preload enabled for immediate validation
    raw = mne.io.read_raw_edf(normalized_path, preload=True, verbose=False)
    # Apply the montage to ensure spatial layout matches expectations
    raw.set_montage(expected_montage, on_missing="warn")
    # Retrieve the effective montage to confirm it has been attached
    montage = raw.get_montage()
    # Fail loudly when the montage could not be established on the recording
    if montage is None:
        # Raise a clear error describing the missing montage configuration
        raise ValueError(f"Montage '{expected_montage}' could not be applied")
    # Extract the sampling frequency reported by the recording
    sampling_rate = float(raw.info["sfreq"])
    # Validate the sampling frequency against the expected configuration
    if not np.isclose(sampling_rate, expected_sampling_rate):
        # Raise a descriptive error when the sampling rate deviates
        raise ValueError(
            f"Expected sampling rate {expected_sampling_rate}Hz "
            f"but got {sampling_rate}Hz"
        )
    # Gather channel names from the recording for consistency checks
    channel_names = list(raw.ch_names)
    # Identify unexpected channels that would break downstream spatial filters
    extra_channels = sorted(set(channel_names) - set(expected_channels))
    # Identify missing channels that would prevent feature extraction
    missing_channels = sorted(set(expected_channels) - set(channel_names))
    # Raise an explicit error when the channel layout is inconsistent
    if extra_channels or missing_channels:
        # Compose a readable error describing both types of discrepancies
        raise ValueError(
            json.dumps(
                {
                    "error": "Channel mismatch",
                    "extra": extra_channels,
                    "missing": missing_channels,
                }
            )
        )
    # Return the validated raw object for downstream preprocessing steps
    return raw


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
        if not _is_bad_description(desc):
            # Continue looping when the annotation is not an invalid segment
            continue
        # Append the interval boundaries to help later event rejection
        bad_intervals.append((float(onset), float(onset + duration)))
    # Return all invalid windows for consumers that exclude unsafe events
    return bad_intervals


def map_events_and_validate(
    raw: mne.io.BaseRaw,
    label_map: Mapping[str, int] | None = None,
    motor_label_map: Mapping[str, str] | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map annotations to events while checking label consistency."""

    # Use caller-provided label map or default Physionet mapping for labels
    effective_label_map = (
        dict(label_map) if label_map is not None else dict(PHYSIONET_LABEL_MAP)
    )
    # Confirm annotations contain only labels that the mapping can handle
    _validate_annotation_labels(raw, effective_label_map)
    # Detect whether motor mapping validation should be applied
    motor_labels_present = any(
        desc in MOTOR_EVENT_LABELS
        for desc in raw.annotations.description
        if not _is_bad_description(desc)
    )
    # Build a motor mapping to make motor imagery labels explicit when needed
    if motor_labels_present or motor_label_map is not None:
        # Validate the mapping either provided by the caller or defaulted
        _validate_motor_mapping(
            raw,
            effective_label_map,
            motor_label_map if motor_label_map is not None else MOTOR_EVENT_LABELS,
        )
    # Extract invalid windows to support removal of corrupted epochs
    bad_intervals = _extract_bad_intervals(raw)
    # Convert annotations into events that MNE Epochs can consume
    events, _ = mne.events_from_annotations(
        raw, event_id=effective_label_map, verbose=False
    )
    # Preserve the full label map even if some labels are absent in a run
    event_id = dict(effective_label_map)
    # Build a boolean mask describing which events survive BAD intervals
    keep_mask = _build_keep_mask(events, raw.info["sfreq"], bad_intervals)
    # Gather events whose mask entries remain explicitly True
    filtered_events_list = [
        event for flag, event in zip(keep_mask, events, strict=True) if flag is True
    ]
    # Convert the preserved events back to a NumPy array for downstream consumers
    filtered_events = np.array(filtered_events_list)
    # Return clean events and the mapping for downstream epoch creation
    return filtered_events, event_id


def _validate_annotation_labels(
    raw: mne.io.BaseRaw, effective_label_map: Mapping[str, int]
) -> None:
    """Ensure annotations only include labels present in the mapping."""

    # Inspect annotations to ensure only expected labels remain
    present_labels = set(raw.annotations.description)
    # Identify labels that would break the supervised mapping stage
    unknown_labels = {
        lab
        for lab in present_labels
        if lab not in effective_label_map and not _is_bad_description(lab)
    }
    # Stop early when unknown labels are detected to prevent silent errors
    if unknown_labels:
        # Raise a descriptive error to support dataset hygiene during setup
        raise ValueError(f"Unknown labels in annotations: {sorted(unknown_labels)}")


def _validate_motor_mapping(
    raw: mne.io.BaseRaw,
    effective_label_map: Mapping[str, int],
    motor_label_map: Mapping[str, str],
) -> Dict[str, str]:
    """Validate motor mapping covers all events with A/B labels."""

    # Copy the mapping to avoid mutating caller dictionaries during validation
    effective_motor_map = dict(motor_label_map)
    # Restrict allowed motor labels to the binary A/B tasks defined by the project
    allowed_motor_labels = {"A", "B"}
    # Detect invalid motor labels that would break downstream training splits
    invalid_motor_labels = set(effective_motor_map.values()) - allowed_motor_labels
    # Raise a clear error when unsupported motor labels are provided
    if invalid_motor_labels:
        # Surface which labels are invalid to guide mapping corrections
        raise ValueError(
            f"Motor labels must be within {sorted(allowed_motor_labels)}: "
            f"found {sorted(invalid_motor_labels)}"
        )
    # Collect all annotation labels excluding BAD markers for completeness checks
    observed_labels = {
        desc for desc in raw.annotations.description if not _is_bad_description(desc)
    }
    # Restrict completeness checks to motor-related annotation labels
    motor_label_candidates = {
        desc for desc in observed_labels if desc in MOTOR_EVENT_LABELS
    }
    # Identify observed motor labels not covered by the motor mapping
    missing_motor_keys = motor_label_candidates - set(effective_motor_map.keys())
    # Raise a descriptive error when observed labels lack motor interpretations
    if missing_motor_keys:
        # Include unknown label names to speed up dataset adjustments
        raise ValueError(
            f"Motor mapping missing labels for events: {sorted(missing_motor_keys)}"
        )
    # Identify motor labels that are expected but absent from the mapping outputs
    missing_targets = allowed_motor_labels - set(effective_motor_map.values())
    # Raise when A or B is not reachable from the mapping configuration
    if missing_targets:
        # Provide actionable feedback by listing missing motor targets explicitly
        raise ValueError(
            f"Motor mapping must include targets {sorted(allowed_motor_labels)}: "
            f"missing {sorted(missing_targets)}"
        )
    # Identify motor keys that are not part of the annotation label map
    unknown_keys = set(effective_motor_map.keys()) - set(effective_label_map.keys())
    # Stop when the motor mapping references labels outside the event ID map
    if unknown_keys:
        # Include stray keys in the error to steer label alignment quickly
        raise ValueError(
            f"Motor mapping references unknown events: {sorted(unknown_keys)}"
        )
    # Return the validated motor mapping for optional downstream logging
    return effective_motor_map


def _build_keep_mask(
    events: np.ndarray,
    sampling_rate: float,
    bad_intervals: List[Tuple[float, float]],
    forced_mask: List[Any] | None = None,
) -> List[bool]:
    """Return a boolean mask that excludes events overlapping BAD spans."""

    # Declare the mask variable once to maintain consistent typing across branches
    keep_mask: List[Any]
    # Accept an externally supplied mask to validate defensive branches explicitly
    if forced_mask is not None:
        # Copy the forced mask to avoid caller-side mutations affecting validation
        keep_mask = list(forced_mask)
    else:
        # Initialize a boolean mask list to track valid events explicitly
        keep_mask = [True] * len(events)
        # Iterate over events to check whether they overlap a BAD interval
        for idx, (sample, _, _) in enumerate(events):
            # Convert sample index to seconds to compare against annotation times
            event_time = sample / sampling_rate
            # Mark the event for removal when it lies within a BAD interval
            if any(start <= event_time <= end for start, end in bad_intervals):
                # Update the mask to drop contaminated events from the dataset
                keep_mask[idx] = False
    # Enforce boolean typing on the mask to avoid silent drops from bad values
    if not all(isinstance(flag, bool) for flag in keep_mask):
        # Raise when the mask contains non-boolean entries to surface errors early
        raise TypeError("Event mask contained non-boolean values")
    # Return the vetted mask for downstream event filtering
    return keep_mask


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


def _expected_epoch_samples(epochs: mne.Epochs) -> int:
    """Compute the expected number of samples per epoch."""

    # Derive duration-based sample count to catch truncated segments
    return int(round((epochs.tmax - epochs.tmin) * epochs.info["sfreq"])) + 1


def _flag_epoch_quality(
    epoch: np.ndarray, max_peak_to_peak: float, expected_samples: int
) -> List[str]:
    """Identify quality issues for a single epoch."""

    # Track reasons to support later reporting or masking
    reasons: List[str] = []
    # Measure peak-to-peak amplitude to reveal sharp artifacts
    ptp_value = float(np.ptp(epoch))
    # Record amplitude excursions beyond the threshold to protect models
    if ptp_value > max_peak_to_peak:
        reasons.append("artifact")
    # Detect incomplete epochs via shape or NaN inspection
    if epoch.shape[1] < expected_samples or np.isnan(epoch).any():
        reasons.append("incomplete")
    # Return accumulated flags for the caller to aggregate
    return reasons


def _apply_marking(
    safe_epochs: mne.Epochs, flagged: Dict[str, List[int]]
) -> Tuple[mne.Epochs, Dict[str, List[int]]]:
    """Mark flagged epochs in metadata and return the updated set."""

    # Initialize metadata so downstream code can track quality per epoch
    if safe_epochs.metadata is None:
        # Build a DataFrame with a single column matching epoch count
        safe_epochs.metadata = pd.DataFrame({"quality_flag": ["ok"] * len(safe_epochs)})
    # Iterate to update quality flags for each identified issue
    for reason, indices in flagged.items():
        # Apply the reason to all recorded indices for transparency
        for idx in indices:
            # Overwrite the quality flag to reflect the detected issue
            safe_epochs.metadata.at[idx, "quality_flag"] = reason
    # Return the annotated epochs along with the indexed reasons
    return safe_epochs, flagged


def _apply_rejection(
    safe_epochs: mne.Epochs, flagged: Dict[str, List[int]]
) -> Tuple[mne.Epochs, Dict[str, List[int]]]:
    """Drop flagged epochs and return the pruned set."""

    # Build a set of indices to remove for efficient membership tests
    removed_indices = set(flagged["artifact"]) | set(flagged["incomplete"])
    # Construct a boolean mask that preserves only unflagged epochs
    keep_mask = [idx not in removed_indices for idx in range(len(safe_epochs))]
    # Apply the mask to drop contaminated epochs before returning
    return safe_epochs[keep_mask], flagged


def quality_control_epochs(
    epochs: mne.Epochs,
    max_peak_to_peak: float,
    mode: str = "reject",
) -> Tuple[mne.Epochs, Dict[str, List[int]]]:
    """Screen epochs for artifacts or incompleteness and flag or drop them."""

    # Copy the epochs to avoid mutating caller data during quality enforcement
    safe_epochs = epochs.copy()
    # Retrieve the data to inspect amplitude and completeness metrics
    data = safe_epochs.get_data(copy=True)
    # Compute expected sample count to detect truncated epoch shapes
    expected_samples = _expected_epoch_samples(safe_epochs)
    # Prepare buckets to track artifact and incomplete epoch indices
    flagged: Dict[str, List[int]] = {"artifact": [], "incomplete": []}
    # Iterate over epochs to check amplitude and completeness constraints
    for idx, epoch in enumerate(data):
        # Aggregate all reasons identified for the current epoch
        for reason in _flag_epoch_quality(epoch, max_peak_to_peak, expected_samples):
            # Record the reason to enable downstream removal or marking
            flagged[reason].append(idx)
    # When the mode is reject, remove flagged epochs to stabilize training
    if mode == "reject":
        # Delegate rejection to simplify complexity for linting and tests
        return _apply_rejection(safe_epochs, flagged)
    # When the mode is mark, annotate metadata instead of dropping epochs
    if mode == "mark":
        # Delegate marking to reuse the metadata update logic
        return _apply_marking(safe_epochs, flagged)
    # Raise when the mode is unsupported to avoid silent misuse
    raise ValueError("mode must be either 'reject' or 'mark'")


def detect_artifacts(
    signal: np.ndarray,
    amplitude_threshold: float,
    variance_threshold: float,
    mode: str = "reject",
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect amplitude or variance artifacts and reject or interpolate."""

    # Copie le signal pour éviter de modifier les buffers d'appelant
    safe_signal = np.array(signal, copy=True)
    # Calcule l'amplitude absolue pour repérer les excursions extrêmes
    amplitude_mask = np.abs(safe_signal) > amplitude_threshold
    # Calcule la variance par échantillon pour capturer les déviations croisées
    variance_per_sample = np.var(safe_signal, axis=0)
    # Étend le masque de variance à toutes les voies pour uniformiser le traitement
    variance_mask = variance_per_sample > variance_threshold
    # Combine les critères pour identifier chaque échantillon contaminé
    combined_mask = amplitude_mask | variance_mask
    # Déduit un masque global indiquant les colonnes à exclure ou corriger
    sample_mask = combined_mask.any(axis=0)
    # Traite la branche de rejet pour supprimer les échantillons fautifs
    if mode == "reject":
        # Construit un masque de conservation pour filtrer les colonnes sûres
        keep_mask = ~sample_mask
        # Supprime les colonnes contaminées afin de stabiliser l'apprentissage
        return safe_signal[:, keep_mask], sample_mask
    # Traite la branche d'interpolation pour conserver la structure temporelle
    if mode == "interpolate":
        # Localise les indices sûrs pour guider l'interpolation linéaire
        valid_indices = np.flatnonzero(~sample_mask)
        # Gère le cas où tous les échantillons sont invalides en conservant le signal brut
        if len(valid_indices) == 0:
            # Retourne le signal initial lorsque l'interpolation est impossible
            return safe_signal, sample_mask
        # Prépare un vecteur d'indices cible pour reconstituer chaque colonne
        target_indices = np.arange(safe_signal.shape[1])
        # Itère sur chaque canal pour appliquer une interpolation indépendante
        for channel in range(safe_signal.shape[0]):
            # Extrait les valeurs sûres du canal courant pour alimenter l'interpolation
            valid_values = safe_signal[channel, valid_indices]
            # Remplace les échantillons fautifs par l'interpolation linéaire
            safe_signal[channel] = np.interp(
                target_indices, valid_indices, valid_values
            )
        # Retourne le signal interpolé pour préserver la longueur temporelle
        return safe_signal, sample_mask
    # Lève une erreur explicite pour les modes non supportés
    raise ValueError("mode must be either 'reject' or 'interpolate'")


def normalize_channels(
    signal: np.ndarray, method: str = "zscore", epsilon: float = 1e-8
) -> np.ndarray:
    """Normalize each channel using z-score or robust statistics."""

    # Copie le signal pour préserver l'entrée originale lors de la normalisation
    safe_signal = np.array(signal, copy=True)
    # Uniformise le nom de méthode pour éviter les confusions de casse
    normalized_method = method.lower()
    # Applique une normalisation z-score basée sur moyenne et écart-type
    if normalized_method == "zscore":
        # Calcule la moyenne par canal pour centrer la distribution
        mean_per_channel = np.mean(safe_signal, axis=1, keepdims=True)
        # Calcule l'écart-type par canal et ajoute epsilon pour la stabilité
        std_per_channel = np.std(safe_signal, axis=1, keepdims=True) + epsilon
        # Centre et réduit chaque canal pour homogénéiser les amplitudes
        return (safe_signal - mean_per_channel) / std_per_channel
    # Applique une normalisation robuste basée sur médiane et IQR
    if normalized_method == "robust":
        # Calcule la médiane par canal pour neutraliser les valeurs extrêmes
        median_per_channel = np.median(safe_signal, axis=1, keepdims=True)
        # Calcule l'IQR par canal et ajoute epsilon pour éviter les divisions nulles
        iqr_per_channel = (
            np.percentile(safe_signal, 75, axis=1, keepdims=True)
            - np.percentile(safe_signal, 25, axis=1, keepdims=True)
            + epsilon
        )
        # Centre et met à l'échelle chaque canal selon les statistiques robustes
        return (safe_signal - median_per_channel) / iqr_per_channel
    # Lève une erreur explicite pour les méthodes non supportées
    raise ValueError("method must be either 'zscore' or 'robust'")


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


def generate_epoch_report(
    epochs: mne.Epochs,
    event_id: Mapping[str, int],
    run_metadata: Mapping[str, str],
    output_path: Path,
    fmt: str = "json",
) -> Path:
    """Persist epoch counts per class, subject, and run in JSON or CSV."""

    # Convertit le format en minuscules pour uniformiser les comparaisons
    fmt_normalized = fmt.lower()
    # Refuse les formats non minuscules pour éviter les ambiguïtés silencieuses
    if fmt != fmt_normalized:
        # Arrête l'exécution pour imposer une convention de nommage explicite
        raise ValueError("fmt must be lowercase")
    # Vérifie que le format fourni est limité aux options minuscules supportées
    if fmt_normalized not in {"json", "csv"}:
        # Interrompt tôt pour éviter d'écrire un rapport avec un format ambigu
        raise ValueError("fmt must be either 'json' or 'csv'")
    # Normalise le chemin pour garantir des écritures cohérentes sur disque
    # Ensure the parent directory exists to make the report path valid
    output_path = Path(output_path)
    # Create parent directories so the report can be written without errors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Build a reverse lookup to translate event codes into label names
    reverse_map = {code: label for label, code in event_id.items()}
    # Count epochs for every label present in the event mapping
    label_counts = {
        label: int(np.sum(epochs.events[:, 2] == code))
        for code, label in reverse_map.items()
    }
    # Compose a structured payload describing the run content
    payload: Dict[str, Any] = {
        "subject": run_metadata["subject"],
        "run": run_metadata["run"],
        "total_epochs": int(len(epochs)),
        "counts": label_counts,
    }
    # Serialize the payload to JSON when requested by the caller
    if fmt_normalized == "json":
        # Write the JSON content with indentation for human readability
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Return the path so callers can locate the generated report
        return output_path
    # Serialize the payload to CSV rows when CSV is requested
    else:
        # Build header and rows capturing each class count explicitly
        lines = ["subject,run,label,count"]
        # Iterate over label counts to materialize per-class entries
        for label, count in label_counts.items():
            # Append a CSV line detailing subject, run, label, and count
            lines.append(
                f"{run_metadata['subject']},{run_metadata['run']},{label},{count}"
            )
        # Write all lines with newline separation to the output path
        output_path.write_text("\n".join(lines), encoding="utf-8")
        # Return the path so downstream processes can load the CSV
        return output_path
