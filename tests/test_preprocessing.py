"""Unit tests for Physionet preprocessing helpers."""

# Import pathlib to construct temporary dataset layouts
# Import pathlib to type temporary paths for datasets
# Import json to inspect structured error payloads
# Import inspect to introspect function signatures when required
import inspect
import json
import time

# Imports warnings to verify the warning filters applied by loaders
import warnings
from pathlib import Path

# Import SimpleNamespace to build lightweight annotation holders
from types import SimpleNamespace

# Import Mapping to annotate captured motor mapping structures
# Import cast to préciser le type des dictionnaires capturés
from typing import Any, Callable, Dict, List, Literal, Mapping, Tuple, cast

# Import mne to build synthetic Raw objects and annotations
import mne

# Import numpy to craft deterministic dummy EEG data
import numpy as np

# Import pytest to manage temporary directories and assertions
import pytest

# Import the preprocessing module to instrument dtype enforcement
from tpv import preprocessing

# Import the preprocessing helpers under test
# Importe les utilitaires de contrôle d'échantillons et de qualité
# Importe le marqueur pour vérifier la structure des métadonnées
# Importe l'utilitaire de conversion d'unités pour le contrôle qualité
from tpv.preprocessing import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_NORMALIZE_EPSILON,
    DEFAULT_NORMALIZE_METHOD,
    MOTOR_EVENT_LABELS,
    PHYSIONET_LABEL_MAP,
    ReportConfig,
    _apply_marking,
    _assert_expected_labels_present,
    _build_file_entry,
    _build_keep_mask,
    _collect_run_counts,
    _ensure_label_alignment,
    _expected_epoch_samples,
    _extract_bad_intervals,
    _flag_epoch_quality,
    _is_bad_description,
    _normalize_report_config,
    _rename_channels_for_montage,
    _validate_motor_mapping,
    apply_bandpass_filter,
    create_epochs_from_raw,
    detect_artifacts,
    drop_non_eeg_channels,
    ensure_volts_units,
    generate_epoch_report,
    load_mne_motor_run,
    load_mne_raw_checked,
    load_physionet_raw,
    map_events_and_validate,
    map_events_to_motor_labels,
    normalize_channels,
    quality_control_epochs,
    report_epoch_anomalies,
    summarize_epoch_quality,
    verify_dataset_integrity,
)

# Fixe une amplitude maximale attendue pour repérer une dérive de filtrage
MAX_FILTER_AMPLITUDE = 10.0
# Fixe l'ordre IIR personnalisé pour verrouiller le paramètre transmis
CUSTOM_IIR_ORDER = 6
# Fixe l'ordre IIR par défaut pour surveiller la configuration interne
EXPECTED_IIR_DEFAULT_ORDER = 4
# Fixe un identifiant de sujet de test pour les rapports synthétiques
TEST_SUBJECT = "S-test"
# Fixe un identifiant de run de test pour homogénéiser les métadonnées
TEST_RUN = "R-test"


def _build_dummy_raw(sfreq: float = 128.0, duration: float = 1.0) -> mne.io.Raw:
    """Create a synthetic RawArray with Physionet-like annotations."""

    # Set two channels to keep tests lightweight while representative
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    # Generate deterministic data to guarantee reproducible hashes
    rng = np.random.default_rng(seed=42)
    # Create a configurable duration to stress filtering when needed
    data = rng.standard_normal((2, int(sfreq * duration)))
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


def _build_epoch_array(
    epoch_count: int = 4, sfreq: float = 128.0
) -> tuple[mne.Epochs, list[str]]:
    """Create deterministic epochs with balanced motor labels."""

    # Construit les métadonnées des canaux pour aligner MNE et les tests
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    # Génère des données à faible amplitude pour éviter les rejets inattendus
    rng = np.random.default_rng(seed=123)
    # Calcule le nombre d'échantillons attendu pour un segment de 1 seconde
    samples = int(round(sfreq)) + 1
    # Crée un cube de données stable pour maintenir les tests déterministes
    data = rng.normal(scale=0.01, size=(epoch_count, 2, samples))
    # Bâtit une liste d'événements espacés pour satisfaire la construction MNE
    events = np.column_stack(
        [
            np.arange(epoch_count),
            np.zeros(epoch_count, dtype=int),
            np.ones(epoch_count, dtype=int),
        ]
    )
    # Génère des labels équilibrés pour tester le comptage par classe
    labels = ["A" if idx % 2 == 0 else "B" for idx in range(epoch_count)]
    # Assemble des epochs à partir des données synthétiques
    epochs = mne.EpochsArray(data=data, info=info, events=events, tmin=0.0)
    # Retourne les epochs et les labels alignés pour alimenter les tests
    return epochs, labels


def test_drop_non_eeg_channels_removes_eog_emg() -> None:
    """Ensure EOG/EMG channels are removed from raw recordings."""

    # Construit des métadonnées incluant des canaux EEG et non-EEG
    info = mne.create_info(
        ch_names=["C3", "C4", "EOG1", "EMG1"],
        sfreq=128.0,
        ch_types=["eeg", "eeg", "eog", "emg"],
    )
    # Génère des données déterministes pour ces canaux
    rng = np.random.default_rng(seed=7)
    data = rng.standard_normal((4, 128))
    # Assemble un RawArray pour tester le filtrage des canaux
    raw = mne.io.RawArray(data, info)
    # Applique la suppression des canaux non EEG
    cleaned = drop_non_eeg_channels(raw)
    # Vérifie que seuls les canaux EEG restent
    assert cleaned.ch_names == ["C3", "C4"]


def test_apply_bandpass_filter_preserves_shape_and_stability() -> None:
    """Ensure FIR and IIR filtering keep shapes and finite amplitudes."""

    # Build a short raw recording to constrain test runtime
    raw = _build_dummy_raw(sfreq=256.0, duration=2.0)
    # Apply a FIR filter with padding to limit boundary distortions
    fir_filtered = apply_bandpass_filter(raw, method="fir", pad_duration=0.25)
    # Apply an IIR filter to compare stability against the FIR baseline
    iir_filtered = apply_bandpass_filter(raw, method="iir", pad_duration=0.25)
    # Confirm the filtered data preserves the original sampling shape
    assert fir_filtered.get_data().shape == raw.get_data().shape
    # Check the IIR path also returns data with unchanged shape
    assert iir_filtered.get_data().shape == raw.get_data().shape
    # Verify that FIR filtering produces finite outputs across channels
    assert np.isfinite(fir_filtered.get_data()).all()
    # Verify that IIR filtering remains stable without NaN or inf values
    assert np.isfinite(iir_filtered.get_data()).all()
    # Ensure both filters stay within a reasonable amplitude envelope
    assert np.max(np.abs(fir_filtered.get_data())) < MAX_FILTER_AMPLITUDE
    # Ensure IIR filtering also remains within the expected amplitude range
    assert np.max(np.abs(iir_filtered.get_data())) < MAX_FILTER_AMPLITUDE


def test_apply_bandpass_filter_latency_benchmark() -> None:
    """Benchmark FIR vs IIR latency while checking reproducibility."""

    # Generate a longer dummy recording to stress filtering performance
    raw = _build_dummy_raw(sfreq=512.0, duration=2.0)
    # Measure FIR latency using a monotonic clock for robustness
    fir_start = time.perf_counter()
    apply_bandpass_filter(raw, method="fir", pad_duration=0.5)
    # Capture FIR duration to enable relative latency comparisons
    fir_latency = time.perf_counter() - fir_start
    # Measure IIR latency on the same input to compare efficiency
    iir_start = time.perf_counter()
    apply_bandpass_filter(raw, method="iir", pad_duration=0.5)
    # Capture IIR duration to validate latency improvements
    iir_latency = time.perf_counter() - iir_start
    # Ensure both measurements are positive to confirm timer correctness
    assert fir_latency > 0
    # Confirm IIR filtering completes with a measurable positive duration
    assert iir_latency > 0
    # Expect IIR latency to stay below a generous multiple of FIR time
    assert iir_latency < fir_latency * 2.5


def test_apply_bandpass_filter_rejects_unknown_method() -> None:
    """Validate that the filter refuses unsupported design names."""

    # Construit un enregistrement court pour atteindre la validation rapide
    raw = _build_dummy_raw(sfreq=128.0, duration=0.5)
    # Vérifie qu'une méthode inconnue déclenche une erreur explicite
    with pytest.raises(ValueError):
        # Force un nom de méthode non pris en charge pour tester la garde
        apply_bandpass_filter(raw, method="fft", pad_duration=0.25)


def test_apply_bandpass_filter_uses_expected_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate that default padding length is applied and removed."""

    # Construit un enregistrement très court pour forcer un seul échantillon de pad
    raw = _build_dummy_raw(sfreq=2.0, duration=1.0)
    # Capture la forme des données transmises à la fonction de filtrage
    captured: dict[str, object] = {}

    # Remplace la fonction de filtrage MNE pour observer le tampon fourni
    def _fake_filter(data: np.ndarray, **kwargs: object) -> np.ndarray:
        # Stocke la forme pour vérifier la présence du padding symétrique
        captured["shape"] = data.shape
        # Conserve les données pour analyser le contenu du padding
        captured["data"] = data.copy()
        # Retourne le tampon inchangé pour que la découpe retire le pad
        return data

    # Patch la fonction de filtrage afin d'éviter le coût du vrai filtrage
    monkeypatch.setattr("mne.filter.filter_data", _fake_filter)
    # Applique le filtre afin de déclencher l'ajout puis le retrait du pad
    filtered = apply_bandpass_filter(raw, method="fir")
    # Vérifie que la fonction de filtrage a bien reçu un pad symétrique
    assert captured["shape"] == (2, raw.get_data().shape[1] + 2)
    # Contrôle que le padding reflète correctement les bords du signal source
    padded = np.asarray(captured["data"])
    assert np.allclose(padded[:, 0], raw.get_data()[:, 1])
    assert np.allclose(padded[:, -1], raw.get_data()[:, -2])
    # Vérifie que les données finales ont retrouvé leur longueur initiale
    np.testing.assert_array_equal(filtered.get_data(), raw.get_data())


def test_apply_bandpass_filter_invalid_method_message() -> None:
    """Check that the guardrails keep the explicit validation message."""

    # Prépare un enregistrement minimal pour déclencher la validation tôt
    raw = _build_dummy_raw(sfreq=64.0, duration=0.25)
    # Vérifie que le message d'erreur reste précis sur les méthodes autorisées
    with pytest.raises(ValueError, match=r"^method must be 'fir' or 'iir'$"):
        # Passe une méthode erronée pour couvrir les variantes mutées du message
        apply_bandpass_filter(raw, method="bad")


def test_report_epoch_anomalies_reports_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure reports stay empty of anomalies for clean epochs."""

    # Construit des epochs synthétiques sans artefacts pour le rapport
    epochs, labels = _build_epoch_array()
    # Prépare les métadonnées de run attendues par la routine de rapport
    run_metadata = {"subject": TEST_SUBJECT, "run": TEST_RUN}
    # Force un chemin imbriqué pour exiger la création des parents
    output_path = tmp_path / "reports" / "nested" / "report.json"
    # Construit la configuration de rapport en imposant le format JSON
    report_config = preprocessing.ReportConfig(path=output_path, fmt="json")

    # Trace l'encodage et le contenu pour verrouiller l'écriture JSON
    write_calls: list[tuple[str | None, str]] = []
    # Conserve l'implémentation originale pour déléguer l'écriture réelle
    original_write_text = Path.write_text

    # Intercepte l'écriture afin d'imposer un encodage explicite et stable
    def spy_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        # Archive l'appel pour détecter les mutations d'arguments critiques
        write_calls.append((encoding, data))
        # Délègue pour conserver la lecture disque déjà testée plus bas
        return original_write_text(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    # Patch Path.write_text pour tracer les paramètres effectifs
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    # Génère le rapport tout en récupérant les epochs nettoyées
    cleaned, report, path = report_epoch_anomalies(
        epochs,
        labels,
        run_metadata,
        max_peak_to_peak=1.0,
        report_config=report_config,
    )
    # Vérifie que le chemin correspond au fichier produit
    assert path == output_path
    # Vérifie que les dossiers parents ont été créés automatiquement
    assert output_path.parent.is_dir()
    # S'assure que le fichier de rapport a bien été écrit sur disque
    assert output_path.exists()
    # Contrôle que les epochs conservées respectent le comptage attendu
    assert len(cleaned) == len(epochs)
    # Valide l'absence totale d'anomalies dans le rapport synthétique
    assert report["anomalies"] == {"artifact": [], "incomplete": []}
    # Vérifie que chaque classe conserve deux occurrences après filtrage nul
    assert report["counts"] == {"A": 2, "B": 2}
    # Charge le fichier pour garantir la cohérence du contenu sérialisé
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    # Confirme que la charge reflète les mêmes données que le rapport en mémoire
    assert loaded == report

    # Verrouille l'unicité de l'écriture JSON pour éviter des doubles writes
    assert len(write_calls) == 1
    # Verrouille l'encodage explicite pour éviter la dépendance au locale système
    assert write_calls[0][0] == "utf-8"
    # Verrouille l'indentation stable pour faciliter le diff humain des rapports
    assert write_calls[0][1] == json.dumps(report, indent=2)


def test_report_epoch_anomalies_forwards_report_to_label_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure label validation receives the report payload (not None)."""

    # Construit des epochs synthétiques sans artefacts pour le rapport
    epochs, labels = _build_epoch_array()
    # Prépare les métadonnées de run attendues par la routine de rapport
    run_metadata = {"subject": TEST_SUBJECT, "run": TEST_RUN}
    # Fixe le chemin de sortie pour valider l'écriture JSON
    output_path = tmp_path / "report.json"
    # Construit la configuration de rapport en imposant le format JSON
    report_config = preprocessing.ReportConfig(path=output_path, fmt="json")
    # Conserve l'implémentation réelle pour déléguer le comportement nominal
    original_validator = preprocessing._assert_expected_labels_present
    # Initialise les captures à None pour confirmer l'exécution du spy
    captured_report: dict[str, Any] | None = None
    # Initialise les captures à None pour vérifier la cohérence des comptes
    captured_counts: dict[str, int] | None = None

    def spy_validator(report: dict[str, Any], counts: dict[str, int]) -> None:
        """Capture les arguments pour verrouiller la propagation du rapport."""

        # Expose les variables externes pour stocker l'appel intercepté
        nonlocal captured_report, captured_counts
        # Archive le rapport reçu afin de détecter une valeur None mutée
        captured_report = report
        # Archive les comptes reçus pour comparer au rapport final
        captured_counts = counts
        # Délègue au validateur réel pour conserver la sémantique nominale
        original_validator(report, counts)

    # Patch le validateur afin d'inspecter les arguments transmis par la fonction
    monkeypatch.setattr(preprocessing, "_assert_expected_labels_present", spy_validator)
    # Lance la génération du rapport pour déclencher l'appel au validateur
    _cleaned, report, _path = report_epoch_anomalies(
        epochs,
        labels,
        run_metadata,
        max_peak_to_peak=1.0,
        report_config=report_config,
    )
    # Confirme que le spy a bien capturé un appel effectif
    assert captured_report is not None
    # Confirme que le spy a bien capturé un dictionnaire de comptes
    assert captured_counts is not None
    # Vérifie que le validateur a bien reçu l'objet rapport construit
    assert captured_report is report
    # Vérifie que les comptes transmis au validateur reflètent le rapport final
    assert captured_counts == report["counts"]


def test_apply_bandpass_filter_accepts_uppercase_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure normalization preserves the FIR branch with uppercase input."""

    # Construit un signal court pour passer rapidement le filtrage patché
    raw = _build_dummy_raw(sfreq=64.0, duration=0.5)
    # Capture les arguments transmis à la fonction de filtrage MNE
    captured: dict[str, dict[str, object]] = {}

    # Remplace la fonction de filtrage pour inspecter les paramètres reçus
    def _fake_filter(data: np.ndarray, **kwargs: object) -> np.ndarray:
        # Stocke le dictionnaire pour vérifier la méthode sélectionnée
        captured["kwargs"] = kwargs
        # Retourne le tampon inchangé afin de conserver la longueur originale
        return data

    # Patch la fonction de filtrage afin d'éviter une exécution coûteuse
    monkeypatch.setattr("mne.filter.filter_data", _fake_filter)
    # Applique le filtre avec un nom de méthode en majuscules
    apply_bandpass_filter(raw, method="FIR", pad_duration=0.0)
    # Extrait les paramètres capturés avec un typage explicite pour mypy
    kwargs: dict[str, object] = captured["kwargs"]
    # Vérifie que la méthode transmise reste bien le chemin FIR attendu
    assert kwargs.get("method") == "fir"


def test_report_epoch_anomalies_flags_corrupted_segments(tmp_path: Path) -> None:
    """Validate that corrupted epochs are removed and reported in CSV."""

    # Construit un lot d'epochs avec labels équilibrés pour la détection
    epochs, labels = _build_epoch_array()
    # Extrait les données brutes pour injecter des anomalies contrôlées
    corrupted_data = epochs.get_data(copy=True)
    # Insère un pic artificiel pour dépasser le seuil d'amplitude
    corrupted_data[0, 0, 0] = 50.0
    # Ajoute un NaN pour marquer une epoch incomplète détectable
    corrupted_data[1] = np.nan
    # Remplace les données internes afin que le contrôle qualité les inspecte
    epochs._data = corrupted_data
    # Prépare les métadonnées nécessaires au rapport CSV
    run_metadata = {"subject": TEST_SUBJECT, "run": TEST_RUN}
    # Positionne le chemin de sortie en CSV pour valider le format tabulaire
    output_path = tmp_path / "report.csv"
    # Construit la configuration de rapport pour la sortie CSV
    report_config = preprocessing.ReportConfig(path=output_path, fmt="csv")
    # Lance la génération du rapport avec un seuil strict pour capturer le pic
    cleaned, report, path = report_epoch_anomalies(
        epochs,
        labels,
        run_metadata,
        max_peak_to_peak=1.0,
        report_config=report_config,
    )
    # Vérifie que le chemin retourné pointe vers le CSV attendu
    assert path == output_path
    # Fixe le nombre d'epochs attendues après suppression des anomalies
    expected_kept_epochs = 2
    # Verrouille la propagation du sujet dans le rapport JSON retourné
    assert report["subject"] == TEST_SUBJECT
    # Verrouille la propagation du run dans le rapport JSON retourné
    assert report["run"] == TEST_RUN
    # Verrouille le nombre d'epochs avant filtrage dans le rapport JSON retourné
    assert report["total_epochs_before"] == len(epochs)
    # Verrouille le nombre d'epochs conservées dans le rapport JSON retourné
    assert report["kept_epochs"] == expected_kept_epochs
    # Contrôle que deux epochs seulement restent après suppression des anomalies
    assert len(cleaned) == expected_kept_epochs
    # Vérifie que les indices d'artefact et d'incomplétude sont bien signalés
    assert report["anomalies"] == {"artifact": [0], "incomplete": [1]}
    # Confirme que le comptage par classe reflète la suppression de deux epochs
    assert report["counts"] == {"A": 1, "B": 1}
    # Lit le contenu du CSV pour contrôler la présence des anomalies
    rows = output_path.read_text(encoding="utf-8").splitlines()
    # Vérifie que le CSV contient un en-tête et deux lignes de données
    assert rows[0].startswith("subject,run,total_epochs_before")
    # Inspecte la ligne de la classe A pour valider les indices d'artefacts
    assert f"{TEST_SUBJECT},{TEST_RUN},4,2,0,1,A,1" in rows
    # Inspecte la ligne de la classe B pour valider le comptage restant
    assert f"{TEST_SUBJECT},{TEST_RUN},4,2,0,1,B,1" in rows


def test_build_epoch_report_exposes_total_epochs_before_key() -> None:
    """Lock report keys so downstream QA exports stay stable."""

    # Prépare des métadonnées synthétiques pour construire un rapport stable
    run_metadata = {"subject": TEST_SUBJECT, "run": TEST_RUN}
    # Fixe le nombre total d'epochs observées avant nettoyage pour le contrôle
    total_epochs = 4
    # Fixe le nombre d'epochs conservées après nettoyage pour le suivi
    kept_epochs = 2
    # Définit un comptage minimal pour remplir la structure du rapport
    counts = {"A": 1}
    # Définit des indices signalés pour verrouiller la structure anomalies
    flagged = {"artifact": [0], "incomplete": [1]}
    # Construit le rapport interne afin de tester les clés exposées
    report = preprocessing._build_epoch_report(
        run_metadata,
        total_epochs,
        kept_epochs,
        counts,
        flagged,
    )
    # Verrouille les clés attendues pour éviter une dérive silencieuse de l'API
    assert set(report.keys()) == {
        "subject",
        "run",
        "total_epochs_before",
        "kept_epochs",
        "counts",
        "anomalies",
    }
    # Vérifie que le total initial est publié sous la clé documentée
    assert report["total_epochs_before"] == total_epochs


def test_write_csv_report_creates_parents_and_uses_utf8_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure CSV writer creates parents and enforces UTF-8 encoding."""

    # Construit un chemin imbriqué pour exiger la création des dossiers parents
    target = tmp_path / "reports" / "nested" / "report.csv"
    # Prépare un rapport minimal pour produire une ligne de données CSV
    report = {"subject": TEST_SUBJECT, "run": TEST_RUN, "kept_epochs": 1}
    # Définit des indices signalés pour alimenter les colonnes anomalies
    flagged = {"artifact": [0], "incomplete": [1]}
    # Définit un comptage déterministe pour générer une ligne par classe
    counts = {"A": 1}
    # Fixe le total initial pour remplir la colonne total_epochs_before
    total_epochs = 2
    # Prépare une liste pour enregistrer les encodages utilisés à l'écriture
    write_encodings: list[str | None] = []
    # Capture l'implémentation originale pour conserver l'écriture réelle
    original_write_text = Path.write_text

    # Définit un espion afin de capturer l'encodage passé à write_text
    def spy_write_text(self, data, encoding=None, errors=None):
        # Mémorise l'encodage pour valider le contrat de sérialisation
        write_encodings.append(encoding)
        # Délègue à Path.write_text pour conserver le comportement nominal
        return original_write_text(self, data, encoding=encoding, errors=errors)

    # Remplace Path.write_text pour tracer l'encodage utilisé par l'écrivain CSV
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    # Écrit le rapport CSV avec une arborescence à créer
    preprocessing._write_csv_report(report, flagged, counts, total_epochs, target)
    # Confirme que l'arborescence a bien été créée et que le fichier existe
    assert target.exists()
    # Vérifie que l'encodage transmis est exactement utf-8
    assert write_encodings == ["utf-8"]


def test_write_csv_report_uses_semicolon_delimiter_for_indices(tmp_path: Path) -> None:
    """Lock the delimiter for anomaly index columns in the CSV output."""

    # Fixe un chemin simple pour concentrer le test sur le contenu écrit
    target = tmp_path / "report.csv"
    # Prépare un rapport minimal afin de remplir les colonnes sujet/run
    report = {"subject": TEST_SUBJECT, "run": TEST_RUN, "kept_epochs": 1}
    # Injecte plusieurs indices pour forcer l'utilisation d'un séparateur
    flagged = {"artifact": [0, 2], "incomplete": [1, 3]}
    # Définit un comptage minimal pour générer exactement une ligne de données
    counts = {"A": 1}
    # Fixe le total initial pour alimenter la colonne dédiée du CSV
    total_epochs = 4
    # Écrit le CSV via l'utilitaire interne afin de verrouiller le format
    preprocessing._write_csv_report(report, flagged, counts, total_epochs, target)
    # Lit les lignes pour isoler la ligne de données produite
    rows = target.read_text(encoding="utf-8").splitlines()
    # Vérifie la présence de l'en-tête et d'une ligne de données
    assert len(rows) == 2
    # Découpe la ligne de données pour inspecter précisément les colonnes
    columns = rows[1].split(",")
    # Confirme l'usage du séparateur ';' pour les indices artefacts
    assert columns[4] == "0;2"
    # Confirme l'usage du séparateur ';' pour les indices incomplets
    assert columns[5] == "1;3"


def test_apply_quality_control_passes_reject_mode_to_quality_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure reject mode is forwarded explicitly to quality control."""

    # Construit des epochs propres pour isoler la vérification d'appel
    epochs, labels = _build_epoch_array()
    # Conserve l'implémentation réelle pour déléguer le comportement nominal
    original_quality_control_epochs = preprocessing.quality_control_epochs
    # Prépare une liste pour confirmer que le spy a bien intercepté l'appel
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy_quality_control_epochs(*args: Any, **kwargs: Any):
        """Capture les arguments pour imposer un mode explicite stable."""

        # Archive l'appel pour valider qu'il a bien eu lieu
        calls.append((args, dict(kwargs)))
        # Récupère le mode depuis kwargs ou depuis l'argument positionnel
        mode = kwargs.get("mode")
        # Gère la variante où le mode serait passé en positionnel
        if mode is None and len(args) >= 3:
            # Importe le mode pour éviter de dépendre de la valeur par défaut
            mode = args[2]
        # Force la spécification explicite de la stratégie de rejet
        assert mode == "reject"
        # Délègue au comportement réel pour conserver la sémantique testée
        return cast(Any, original_quality_control_epochs)(*args, **kwargs)

    # Remplace la fonction pour observer précisément les arguments fournis
    monkeypatch.setattr(
        preprocessing, "quality_control_epochs", spy_quality_control_epochs
    )
    # Lance le contrôle qualité via la fonction interne ciblée par le mutant
    cleaned_epochs, flagged, cleaned_labels = preprocessing._apply_quality_control(
        epochs,
        labels,
        max_peak_to_peak=1.0,
    )
    # Vérifie que le contrôle qualité a bien été appelé via le spy
    assert calls
    # Vérifie que les labels restent alignés quand aucune epoch n'est rejetée
    assert cleaned_labels == labels
    # Vérifie que le rapport d'anomalies reste vide sur des données propres
    assert flagged == {"artifact": [], "incomplete": []}
    # Vérifie que le nombre d'epochs reste identique en absence d'anomalies
    assert len(cleaned_epochs) == len(epochs)


def test_summarize_epoch_quality_forwards_reject_mode_and_builds_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure summarize_epoch_quality locks reject mode and report keys."""

    # Construit des epochs équilibrés pour exercer la routine de synthèse
    epochs, labels = _build_epoch_array()
    # Fixe la session attendue pour verrouiller les champs subject/run
    session = (TEST_SUBJECT, TEST_RUN)
    # Prépare une liste d'appels pour confirmer l'interception du spy
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy_quality_control_epochs(*args: Any, **kwargs: Any):
        """Force un mode explicite pour éviter une dérive silencieuse."""

        # Archive l'appel pour vérifier la présence de paramètres critiques
        calls.append((args, dict(kwargs)))
        # Récupère le mode depuis kwargs ou depuis un argument positionnel
        mode = kwargs.get("mode")
        # Gère la variante où le mode serait passé en positionnel
        if mode is None and len(args) >= 3:
            # Capture la valeur pour éviter de dépendre de la valeur par défaut
            mode = args[2]
        # Exige un mode explicitement fourni pour verrouiller l'API
        assert mode is not None
        # Exige la stratégie de rejet pour conserver la sémantique attendue
        assert mode == "reject"
        # Retourne les epochs inchangés pour isoler la construction du rapport
        return args[0], {"artifact": [], "incomplete": []}

    # Remplace la fonction réelle afin de tracer les arguments transmis
    monkeypatch.setattr(
        preprocessing,
        "quality_control_epochs",
        spy_quality_control_epochs,
    )
    # Lance la synthèse pour vérifier les clés et valeurs du rapport
    cleaned_epochs, report, cleaned_labels = summarize_epoch_quality(
        epochs,
        labels,
        session,
        max_peak_to_peak=1.0,
    )
    # Vérifie que le contrôle qualité a bien été invoqué via le spy
    assert calls
    # Vérifie que les epochs restent inchangées en absence d'anomalies
    assert len(cleaned_epochs) == len(epochs)
    # Vérifie que les labels restent alignés quand rien n'est rejeté
    assert cleaned_labels == labels
    # Verrouille la présence et la valeur du sujet dans le rapport synthétique
    assert report["subject"] == TEST_SUBJECT
    # Verrouille la présence et la valeur du run dans le rapport synthétique
    assert report["run"] == TEST_RUN
    # Verrouille la structure du comptage des rejets calculés depuis flagged
    assert report["dropped"] == {"artifact": 0, "incomplete": 0}
    # Verrouille le comptage des classes pour détecter toute régression
    assert report["counts"] == {"A": 2, "B": 2}


def test_build_epoch_report_exposes_total_epochs_before_key_when_clean() -> None:
    """Ensure build report exposes stable keys when no epochs are dropped."""

    # Prépare des métadonnées minimales pour exercer la construction du rapport
    run_metadata = {"subject": TEST_SUBJECT, "run": TEST_RUN}
    # Prépare un comptage minimal pour verrouiller la structure retournée
    counts = {"A": 1, "B": 1}
    # Prépare un dictionnaire d'anomalies pour verrouiller la clé imbriquée
    flagged = {"artifact": [0], "incomplete": [1]}
    # Construit le rapport pour valider les clés indispensables du dictionnaire
    report = preprocessing._build_epoch_report(
        run_metadata,
        total_epochs=4,
        kept_epochs=2,
        counts=counts,
        flagged=flagged,
    )
    # Verrouille les clés d'identité afin d'éviter des ruptures silencieuses
    assert report["subject"] == TEST_SUBJECT
    # Verrouille la clé de run afin d'éviter des ruptures silencieuses
    assert report["run"] == TEST_RUN
    # Verrouille le nom de clé pour le total d'epochs avant filtrage
    assert report["total_epochs_before"] == 4
    # Verrouille la structure des anomalies pour éviter les régressions de clé
    assert report["anomalies"] == {"artifact": [0], "incomplete": [1]}


def test_ensure_label_alignment_reports_mismatch() -> None:
    """Ensure misaligned label counts raise a structured error."""

    # Construit des epochs équilibrés pour détecter le désalignement
    epochs, labels = _build_epoch_array()
    # Retire un label pour provoquer un écart entre événements et labels
    mismatched_labels = labels[:-1]
    # Vérifie que la validation remonte une erreur de désalignement
    with pytest.raises(ValueError) as excinfo:
        _ensure_label_alignment(epochs, mismatched_labels)
    # Convertit le message en dictionnaire pour faciliter les assertions
    payload = json.loads(str(excinfo.value))
    # Contrôle que le rapport explicite bien les longueurs attendues
    assert payload == {
        "error": "Label/event mismatch",
        "expected_events": len(epochs),
        "labels": len(mismatched_labels),
    }


def test_assert_expected_labels_present_reports_missing() -> None:
    """Confirm missing labels trigger a detailed diagnostic payload."""

    # Prépare un rapport minimal pour contextualiser l'erreur attendue
    report = {"subject": TEST_SUBJECT, "run": TEST_RUN, "counts": {}}
    # Construit un comptage lacunaire pour simuler une classe absente
    counts = {"A": 0, "B": 1}
    # Vérifie que l'absence de label est signalée via une erreur structurée
    with pytest.raises(ValueError) as excinfo:
        _assert_expected_labels_present(report, counts)
    # Convertit la charge utile JSON pour vérifier le contenu détaillé
    payload = json.loads(str(excinfo.value))
    # Contrôle que l'erreur inclut le libellé et la classe manquante
    assert payload == {
        "subject": TEST_SUBJECT,
        "run": TEST_RUN,
        "counts": {},
        "error": "Missing labels",
        "missing_labels": ["A"],
    }


def test_normalize_report_config_rejects_uppercase_format(tmp_path: Path) -> None:
    """Reject uppercase format names to keep serialization predictable."""

    # Fixe un chemin valide pour initialiser la configuration de rapport
    report_config = ReportConfig(path=tmp_path / "report.json", fmt="JSON")
    # Vérifie que la normalisation refuse la casse incorrecte
    with pytest.raises(ValueError, match=r"^fmt must be lowercase$"):
        _normalize_report_config(report_config)


def test_normalize_report_config_rejects_unknown_format(tmp_path: Path) -> None:
    """Reject unsupported formats to avoid ambiguous file outputs."""

    # Fixe un chemin valide pour initialiser la configuration de rapport
    report_config = ReportConfig(path=tmp_path / "report.txt", fmt="txt")
    # Vérifie que la validation refuse tout format hors JSON ou CSV
    with pytest.raises(ValueError, match=r"^fmt must be either 'json' or 'csv'$"):
        _normalize_report_config(report_config)


def test_apply_bandpass_filter_defaults_to_fir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure default arguments stay valid and keep the FIR branch active."""

    # Construit un enregistrement minimal pour exercer la valeur par défaut
    raw = _build_dummy_raw(sfreq=64.0, duration=0.25)
    # Capture les paramètres transmis à la fonction de filtrage
    captured: dict[str, dict[str, object]] = {}

    # Remplace la fonction de filtrage pour analyser les paramètres effectifs
    def _fake_filter(data: np.ndarray, **kwargs: object) -> np.ndarray:
        # Stocke les arguments pour vérifier la sélection du chemin FIR
        captured["kwargs"] = kwargs
        # Retourne le tampon inchangé pour simplifier la vérification
        return data

    # Patch la fonction de filtrage afin d'éviter le calcul réel
    monkeypatch.setattr("mne.filter.filter_data", _fake_filter)
    # Applique le filtre sans préciser de méthode pour déclencher la valeur par défaut
    apply_bandpass_filter(raw)
    # Extrait les paramètres capturés avec un typage clair pour les assertions
    kwargs: dict[str, object] = captured["kwargs"]
    # Vérifie que la méthode reste fermement définie sur le filtre FIR
    assert kwargs.get("method") == "fir"
    # Confirme que la fenêtre FIR par défaut reste sur le mode auto attendu
    assert kwargs.get("filter_length") == "auto"
    # Vérifie que les bornes de bande sont correctement transmises
    assert kwargs.get("l_freq") == pytest.approx(8.0)
    assert kwargs.get("h_freq") == pytest.approx(30.0)
    # Contrôle que le filtrage se fait en mode silencieux pour les tests
    assert kwargs.get("verbose") is False


def test_apply_bandpass_filter_skips_padding_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure padding is bypassed when the duration is zero."""

    # Construit un enregistrement pour activer la branche sans padding
    raw = _build_dummy_raw(sfreq=256.0, duration=1.0)
    # Remplace np.pad par un échec direct pour détecter toute invocation
    monkeypatch.setattr(
        "tpv.preprocessing.np.pad",
        lambda *args, **kwargs: pytest.fail("np.pad must stay unused"),
    )
    # Applique le filtre avec pad à zéro pour traverser la branche else
    filtered = apply_bandpass_filter(raw, method="fir", pad_duration=0.0)
    # Confirme que la forme reste identique sans utilisation de padding
    assert filtered.get_data().shape == raw.get_data().shape


def test_apply_bandpass_filter_accepts_custom_iir_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate that custom IIR order is forwarded without alteration."""

    # Construit un enregistrement court pour limiter le volume de calcul
    raw = _build_dummy_raw(sfreq=128.0, duration=0.25)
    # Capture les paramètres transmis à la fonction de filtrage simulée
    captured: dict[str, dict[str, object]] = {}

    # Remplace la fonction de filtrage pour inspecter les paramètres IIR
    def _fake_filter(data: np.ndarray, **kwargs: object) -> np.ndarray:
        # Stocke les paramètres afin de valider l'ordre IIR transmis
        captured["kwargs"] = kwargs
        # Retourne les données pour permettre la découpe sans calcul
        return data

    # Patch la fonction de filtrage pour éviter l'exécution réelle
    monkeypatch.setattr("mne.filter.filter_data", _fake_filter)
    # Applique le filtre IIR avec un ordre personnalisé pour verrouiller le comportement
    apply_bandpass_filter(raw, method="iir", order=CUSTOM_IIR_ORDER, pad_duration=0.0)
    # Extrait les paramètres capturés pour analyser l'ordre IIR appliqué
    kwargs: dict[str, object] = captured["kwargs"]
    # Extrait le dictionnaire IIR avec un typage sûr pour mypy
    iir_params = cast(dict[str, object], kwargs.get("iir_params", {}))
    # Vérifie que l'ordre IIR transmis correspond exactement à la valeur demandée
    assert iir_params.get("order") == CUSTOM_IIR_ORDER


def test_apply_bandpass_filter_defaults_to_expected_iir_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the default IIR configuration remains stable."""

    # Prépare un signal court pour tester la branche IIR par défaut
    raw = _build_dummy_raw(sfreq=128.0, duration=0.25)
    # Capture les paramètres transmis à la fonction de filtrage simulée
    captured: dict[str, dict[str, object]] = {}

    # Remplace la fonction de filtrage pour inspecter les paramètres IIR
    def _fake_filter(data: np.ndarray, **kwargs: object) -> np.ndarray:
        # Stocke les paramètres pour valider l'ordre et la verbosité
        captured["kwargs"] = kwargs
        # Retourne les données sans modification pour accélérer le test
        return data

    # Patch la fonction de filtrage pour éviter le calcul réel
    monkeypatch.setattr("mne.filter.filter_data", _fake_filter)
    # Applique le filtre avec la configuration IIR par défaut
    apply_bandpass_filter(raw, method="iir", pad_duration=0.0)
    # Extrait les paramètres capturés avec un typage explicite pour mypy
    kwargs: dict[str, object] = captured["kwargs"]
    # Extrait les paramètres IIR afin de valider l'ordre par défaut
    iir_params = cast(dict[str, object], kwargs.get("iir_params", {}))
    # Vérifie que l'ordre IIR par défaut reste fixé à quatre
    assert iir_params.get("order") == EXPECTED_IIR_DEFAULT_ORDER
    # Contrôle que le paramètre verbose reste désactivé
    assert kwargs.get("verbose") is False


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
    # Stores warning filter calls to ensure the loader silences known MNE noise
    filter_calls: List[Tuple[Tuple[str, ...], Dict[str, object]]] = []
    # Keeps a reference to the original filterwarnings for safe delegation
    original_filterwarnings = warnings.filterwarnings

    # Captures and forwards filterwarnings calls for strict signature validation
    def filterwarnings_spy(
        action: Literal[
            "default",
            "error",
            "ignore",
            "always",
            "all",
            "module",
            "once",
        ],
        message: str = "",
        category: type[Warning] = Warning,
        module: str = "",
        lineno: int = 0,
        append: bool = False,
    ) -> None:
        # Records calls so mutants changing arguments are detected by tests
        filter_calls.append(
            (
                (action,),
                {
                    "message": message,
                    "category": category,
                    "module": module,
                    "lineno": lineno,
                    "append": append,
                },
            )
        )
        # Delegates to preserve the underlying warnings machinery
        original_filterwarnings(
            action,
            message=message,
            category=category,
            module=module,
            lineno=lineno,
            append=append,
        )

    # Monkeypatches warnings.filterwarnings to observe the configured filter
    monkeypatch.setattr(warnings, "filterwarnings", filterwarnings_spy)
    loaded_raw, metadata = load_physionet_raw(edf_path)
    # Ensures exactly one warning filter is installed for this known MNE warning
    assert len(filter_calls) == 1
    # Extracts the captured signature for strict comparison
    args, kwargs = filter_calls[0]
    # Validates the ignore action is passed positionally as intended
    assert args == ("ignore",)
    # Validates all keyword arguments match the intended warning filter
    assert kwargs == {
        "message": "Limited .*annotation.*outside the data range",
        "category": RuntimeWarning,
        "module": "mne",
        "lineno": 0,
        "append": False,
    }
    # Assert sampling rate is preserved during loader execution
    assert metadata["sampling_rate"] == pytest.approx(128.0)
    # Assert montage is set to the default 10-20 scheme
    assert metadata["montage"] == "standard_1020"
    # Verify that channels are present in the returned metadata
    assert metadata["channel_names"] == ["C3", "C4"]
    # Confirm the loader returns an MNE Raw instance
    assert isinstance(loaded_raw, mne.io.BaseRaw)


def test_load_physionet_raw_applies_average_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the loader applies the requested EEG reference."""

    # Prépare des métadonnées simples pour un RawArray contrôlé
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=128.0, ch_types="eeg")
    # Définit des données non centrées pour détecter le re-référencement
    data = np.array([[1e-4, 2e-4, 3e-4], [4e-4, 5e-4, 6e-4]])
    # Construit un RawArray avec des valeurs déterministes
    raw = mne.io.RawArray(data, info)
    # Calcule la moyenne inter-canaux avant re-référencement
    baseline_mean = raw.get_data().mean(axis=0)
    # Vérifie que la moyenne initiale n'est pas déjà nulle
    assert not np.allclose(baseline_mean, 0.0)
    # Prépare un chemin EDF factice pour le loader
    edf_path = tmp_path / "subject02" / "run02.edf"
    # Crée le répertoire parent pour le chemin factice
    edf_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un fichier factice pour satisfaire les accès disque
    edf_path.write_bytes(b"dummy")
    # Stubbe le lecteur MNE pour renvoyer le Raw contrôlé
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw)
    # Charge le fichier via le loader avec re-référencement moyen
    loaded_raw, metadata = load_physionet_raw(edf_path, reference="average")
    # Calcule la moyenne inter-canaux après re-référencement
    referenced_mean = loaded_raw.get_data().mean(axis=0)
    # Vérifie que la moyenne est proche de zéro après re-référencement
    assert np.allclose(referenced_mean, 0.0, atol=1e-10)
    # Confirme que la référence est exposée dans les métadonnées
    assert metadata["reference"] == "average"


def test_ensure_volts_units_converts_microvolts_for_rejection_thresholds() -> None:
    """Ensure microvolt inputs are converted to volts for rejection checks."""

    # Prépare une configuration MNE minimale pour deux canaux EEG
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=100.0, ch_types="eeg")
    # Définit un signal microvolté avec un pic-à-pic de 100 µV
    data_uv = np.array(
        [
            [50.0, -50.0, 50.0, -50.0],
            [25.0, -25.0, 25.0, -25.0],
        ]
    )
    # Construit un RawArray simulant un fichier encodé en microvolts
    raw_uv = mne.io.RawArray(data_uv, info)
    # Convertit les données en volts via l'utilitaire dédié
    converted_raw = ensure_volts_units(raw_uv)
    # Prépare un epoch unique pour exécuter le contrôle qualité
    epochs = mne.EpochsArray(
        data=converted_raw.get_data()[np.newaxis, ...],
        info=converted_raw.info,
        tmin=0.0,
    )
    # Applique un seuil en volts pour vérifier l'absence de rejet après conversion
    _, flagged_converted = quality_control_epochs(
        epochs, max_peak_to_peak=200e-6, mode="mark"
    )
    # Vérifie que l'epoch converti n'est pas marqué comme artefact
    assert flagged_converted["artifact"] == []
    # Reconstruit un RawArray sans conversion pour comparer le même seuil
    raw_unscaled = mne.io.RawArray(data_uv, info)
    # Crée un epoch non converti pour vérifier le comportement attendu
    epochs_unscaled = mne.EpochsArray(
        data=raw_unscaled.get_data()[np.newaxis, ...],
        info=raw_unscaled.info,
        tmin=0.0,
    )
    # Applique le seuil en volts sur les données non converties
    _, flagged_unscaled = quality_control_epochs(
        epochs_unscaled, max_peak_to_peak=200e-6, mode="mark"
    )
    # Confirme que l'absence de conversion déclenche un rejet d'artefact
    assert flagged_unscaled["artifact"] == [0]


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
    assert "Unknown annotation labels" in str(exc.value)


def test_map_events_to_motor_labels_reports_unknown_event_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure a structured error surfaces when events reference unknown codes."""

    # Build a raw instance to satisfy the function signature
    raw = _build_dummy_raw()
    # Craft an event array with a numeric code absent from the event_id map
    fake_events = np.array([[0, 0, 999]])
    # Provide a minimal event_id map that omits the fabricated code
    fake_event_id = {"T1": 1}
    # Stub the mapping helper to return the crafted events and identifiers
    monkeypatch.setattr(
        "tpv.preprocessing.map_events_and_validate",
        lambda *_args, **_kwargs: (fake_events, fake_event_id),
    )
    # Force motor mapping validation to succeed with a minimal mapping
    monkeypatch.setattr(
        "tpv.preprocessing._validate_motor_mapping",
        lambda *_args, **_kwargs: {"T1": "A"},
    )
    # Expect a ValueError exposing the unknown event code in JSON form
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw, label_map=fake_event_id)
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "Unknown event codes"
    assert payload["unknown_codes"] == [999]


def test_validate_motor_mapping_rejects_duplicate_targets() -> None:
    """Ensure motor mappings cannot collapse distinct labels to one target."""

    # Construit un Raw synthétique contenant des annotations moteur
    raw = _build_dummy_raw()
    # Définit un mapping qui duplique la cible A et omet la cible B requise
    motor_map = {"T1": "A", "T2": "A"}
    # Vérifie que la validation refuse l'absence de cible B dans le mapping
    with pytest.raises(ValueError, match=r"missing \['B'\]"):
        _validate_motor_mapping(raw, PHYSIONET_LABEL_MAP, motor_map)


def test_rename_channels_for_montage_ignores_missing_entries() -> None:
    """Ensure absent mapping keys leave channel names untouched."""

    # Construit un Raw avec un canal sans correspondance dans le mapping par défaut
    info = mne.create_info(
        ch_names=["C3..", "Unknown", "T9.."], sfreq=100.0, ch_types="eeg"
    )
    # Génère des données nulles pour stabiliser le Raw de test
    data = np.zeros((3, 10))
    # Assemble le RawArray pour appliquer le renommage partiel
    raw = mne.io.RawArray(data, info)
    # Applique le renommage afin de transformer uniquement les canaux connus
    renamed = _rename_channels_for_montage(raw)
    # Vérifie que les canaux connus sont renommés tandis que les autres restent intacts
    assert renamed.ch_names == ["C3", "Unknown", "T9"]


def test_rename_channels_for_montage_preserves_channel_order() -> None:
    """Ensure remapping keeps channel order stable after rename."""

    # Construit un Raw avec des canaux devant être renommés mais ordonnés spécifiquement
    info = mne.create_info(
        ch_names=["T9..", "C4..", "C3.."], sfreq=200.0, ch_types="eeg"
    )
    # Génère des données nulles pour stabiliser l'instance Raw
    data = np.zeros((3, 20))
    # Assemble le RawArray afin de vérifier l'ordre après renommage
    raw = mne.io.RawArray(data, info)
    # Applique le mapping de renommage pour aligner les noms sur le montage 10-20
    renamed = _rename_channels_for_montage(raw)
    # Contrôle que l'ordre d'origine est respecté malgré le renommage
    assert renamed.ch_names == ["T9", "C4", "C3"]


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
        {"subject": "S001", "run": "R01"},
        report_path,
    )
    # Load the report to verify the counts reflect the filtered epochs
    report_content = json.loads(report_path.read_text(encoding="utf-8"))
    # Confirm subject and run identifiers are preserved in the report
    assert report_content["subject"] == "S001"
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


def test_expected_epoch_samples_uses_duration_and_rate() -> None:
    """Assure le calcul d'échantillons attendus depuis la durée des epochs."""

    # Construit un Raw synthétique avec fréquence stable pour contrôler la durée
    raw = _build_dummy_raw(sfreq=50.0)
    # Convertit les annotations en événements pour paramétrer l'epoching
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs couvrant une fenêtre asymétrique pour stresser la formule
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=-0.1, tmax=0.2)
    # Évalue la fonction utilitaire pour obtenir le nombre d'échantillons
    expected_samples = _expected_epoch_samples(epochs)
    # Calcule l'attendu pour éviter une valeur constante dans l'assertion
    manual_expected = int(round((epochs.tmax - epochs.tmin) * epochs.info["sfreq"])) + 1
    # Vérifie que le calcul suit bien durée × fréquence + 1 échantillon
    assert expected_samples == manual_expected


def test_flag_epoch_quality_excludes_equal_threshold_artifacts() -> None:
    """Garantit qu'un pic égal au seuil ne déclenche pas de drapeau."""

    # Crée un epoch constant pour isoler la logique de seuil d'amplitude
    epoch = np.zeros((2, 4))
    # Injecte un pic symétrique pour obtenir une amplitude pic-à-pic de 2
    epoch[0, 0] = 1.0
    # Finalise le pic négatif pour atteindre exactement le seuil cible
    epoch[1, 0] = -1.0
    # Calcule les raisons de rejet avec un seuil strictement supérieur à 2
    reasons = _flag_epoch_quality(epoch, max_peak_to_peak=2.0, expected_samples=4)
    # Valide qu'aucun artefact n'est marqué quand l'amplitude égale le seuil
    assert "artifact" not in reasons
    # Valide qu'aucune incomplétude n'est signalée sur des formes correctes
    assert "incomplete" not in reasons


def test_apply_marking_preserves_default_quality_metadata() -> None:
    """Vérifie que les métadonnées par défaut restent cohérentes."""

    # Construit un Raw synthétique pour générer plusieurs epochs
    raw = _build_dummy_raw()
    # Mappe les annotations en événements pour alimenter l'epoching
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs pour disposer de drapeaux applicables
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Prépare un marquage ciblant uniquement le deuxième epoch
    flagged = {"artifact": [1], "incomplete": []}
    # Applique le marquage pour générer ou mettre à jour les métadonnées
    marked, _ = _apply_marking(epochs, flagged)
    # Confirme que la colonne attendue est bien utilisée pour la qualité
    assert list(marked.metadata.columns) == ["quality_flag"]
    # Confirme que les valeurs par défaut sont "ok" hors des indices marqués
    assert list(marked.metadata["quality_flag"]) == ["ok", "artifact", "ok"]


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
    # Confirme que le message respecte le format attendu sans altération
    assert str(exc.value) == "mode must be either 'reject' or 'mark'"


def test_quality_control_epochs_requests_data_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assure que la qualité ne modifie pas l'array original des epochs."""

    # Crée un Raw synthétique pour disposer d'epochs reproductibles
    raw = _build_dummy_raw()
    # Mappe les annotations vers des événements pour l'epoching
    events, event_id = map_events_and_validate(raw)
    # Construit des epochs pour vérifier les appels à get_data
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Conserve l'implémentation native pour exécuter la logique MNE réelle
    original_get_data = mne.Epochs.get_data
    # Prépare une liste pour tracer le paramètre copy demandé
    copy_calls: list[bool] = []

    # Définit un espion qui capture la valeur du paramètre copy
    def spy_get_data(self, copy: bool = True):
        # Enregistre la valeur passée pour valider l'appel attendu
        copy_calls.append(copy)
        # Délègue à l'implémentation originale pour conserver le comportement
        return original_get_data(self, copy=copy)

    # Remplace get_data sur la classe Epochs pour observer l'appel interne
    monkeypatch.setattr(mne.Epochs, "get_data", spy_get_data)
    # Exécute le contrôle qualité pour déclencher l'appel espion
    quality_control_epochs(epochs, max_peak_to_peak=5.0)
    # Vérifie que l'appel a explicitement demandé une copie des données
    assert copy_calls == [True]


def test_generate_epoch_report_creates_nested_directories(tmp_path: Path) -> None:
    """Vérifie que la génération de rapport crée les répertoires profonds."""

    # Construit un Raw synthétique pour disposer d'events prévisibles
    raw = _build_dummy_raw()
    # Mappe les annotations vers des événements pour l'epoching
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs avec une fenêtre courte pour accélérer le test
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Déclare un chemin imbriqué nécessitant la création récursive des dossiers
    nested_path = tmp_path / "deep" / "nested" / "reports" / "summary.json"
    # Génère le rapport JSON pour tester la création récursive
    output_path = generate_epoch_report(
        epochs,
        event_id,
        {"subject": "S03", "run": "R03"},
        nested_path,
    )
    # Charge le rapport pour vérifier l'exactitude du contenu
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # Valide que le sujet et le run correspondent aux métadonnées fournies
    assert payload["subject"] == "S03"
    # Valide que le nombre total d'epochs correspond à la taille générée
    assert payload["total_epochs"] == len(epochs)
    # Valide que chaque label issu de la carte event_id apparaît dans le rapport
    assert set(payload["counts"].keys()) == set(event_id.keys())


def test_generate_epoch_report_default_format_is_json() -> None:
    """Assure que le format par défaut reste "json" en minuscules."""

    # Récupère la signature de la fonction pour lire la valeur par défaut
    default_fmt = inspect.signature(generate_epoch_report).parameters["fmt"].default
    # Vérifie que le format par défaut est exactement la chaîne "json"
    assert default_fmt == "json"


def test_generate_epoch_report_rejects_uppercase_format(tmp_path: Path) -> None:
    """Interdit les formats en majuscules pour éviter des corrections implicites."""

    # Construit un Raw synthétique pour obtenir des événements reproductibles
    raw = _build_dummy_raw()
    # Mappe les annotations vers des événements pour préparer les epochs
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs minimales pour alimenter la génération de rapport
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Définit un chemin de sortie factice pour vérifier l'échec précoce
    output_path = tmp_path / "upper.json"
    # Vérifie qu'un format non minuscule provoque une erreur explicite
    with pytest.raises(ValueError, match=r"^fmt must be lowercase$"):
        generate_epoch_report(
            epochs,
            event_id,
            {"subject": "S05", "run": "R05"},
            output_path,
            fmt="JSON",
        )


def test_generate_epoch_report_formats_json_with_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assure une sortie JSON indentée et encodée explicitement en UTF-8."""

    # Construit un Raw synthétique pour préparer un petit ensemble d'epochs
    raw = _build_dummy_raw()
    # Mappe les annotations en événements pour alimenter la génération du rapport
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs courts pour limiter la taille du rapport
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Trace les encodages utilisés lors des écritures sur disque
    write_encodings: list[str | None] = []
    # Capture l'implémentation originale pour conserver le comportement
    original_write_text = Path.write_text

    # Définit un espion pour enregistrer l'encodage avant l'écriture
    def spy_write_text(self, data, encoding=None, errors=None):
        # Enregistre l'encodage demandé pour la persistance
        write_encodings.append(encoding)
        # Délègue à l'implémentation de Path pour effectuer l'écriture réelle
        return original_write_text(self, data, encoding=encoding, errors=errors)

    # Remplace Path.write_text afin de capturer l'encodage effectif
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    # Génère le rapport JSON avec le format par défaut
    output_path = generate_epoch_report(
        epochs,
        event_id,
        {"subject": "S04", "run": "R04"},
        tmp_path / "report.json",
    )
    # Lit le contenu pour vérifier la présence d'une indentation multi-ligne
    content = output_path.read_text(encoding="utf-8")
    # Vérifie que la sortie contient des retours à la ligne attendus
    assert "\n" in content
    # Récupère la ligne du sujet pour mesurer l'indentation effective
    subject_line = content.splitlines()[1]
    # Vérifie que la ligne commence par exactement deux espaces
    assert subject_line.startswith('  "subject"')
    # Vérifie que l'encodage utilisé est explicitement UTF-8
    assert write_encodings == ["utf-8"]


def test_generate_epoch_report_csv_uses_utf8_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garantit l'encodage UTF-8 lors de l'export CSV."""

    # Construit un Raw synthétique pour préparer des epochs simples
    raw = _build_dummy_raw()
    # Mappe les annotations en événements pour alimenter l'epoching
    events, event_id = map_events_and_validate(raw)
    # Crée des epochs minimalistes pour produire un petit CSV
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Prépare une liste pour enregistrer les encodages utilisés
    write_encodings: list[str | None] = []
    # Capture l'implémentation originale pour conserver l'écriture réelle
    original_write_text = Path.write_text

    # Définit un espion pour contrôler l'encodage passé à write_text
    def spy_write_text(self, data, encoding=None, errors=None):
        # Mémorise l'encodage pour validation après l'écriture
        write_encodings.append(encoding)
        # Délègue à l'implémentation Path pour conserver le comportement
        return original_write_text(self, data, encoding=encoding, errors=errors)

    # Remplace Path.write_text pour tracer l'encodage
    monkeypatch.setattr(Path, "write_text", spy_write_text)
    # Génère un rapport CSV pour déclencher l'écriture surveillée
    generate_epoch_report(
        epochs,
        event_id,
        {"subject": "S05", "run": "R05"},
        tmp_path / "report.csv",
        fmt="csv",
    )
    # Vérifie que l'encodage appliqué est bien UTF-8
    assert write_encodings == ["utf-8"]


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
    # Vérifie que le message d'erreur suit exactement la formulation attendue
    assert str(exc.value) == "fmt must be either 'json' or 'csv'"


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
    # Parse the JSON payload to assert stable keys and values
    payload = json.loads(str(exc.value))
    # Define the exact structured payload expected for CI stability
    expected = {
        "error": "Channel mismatch",
        "extra": ["C4-extra"],
        "missing": ["C4"],
    }
    # Ensure the structured payload remains identical for debugging tools
    assert payload == expected


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
    # Parse the JSON payload to assert stable keys and values
    payload = json.loads(str(exc.value))
    # Define the exact structured payload expected for extra-only mismatches
    expected = {
        "error": "Channel mismatch",
        "extra": ["C4"],
        "missing": [],
    }
    # Ensure the structured payload remains identical for debugging tools
    assert payload == expected


def test_load_mne_raw_checked_rejects_unknown_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject non-EDF/BDF files with a structured JSON payload."""

    # Ensure the EDF reader is never called for unsupported formats
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: None)
    # Expect a ValueError when the extension is neither EDF nor BDF
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.txt"),
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4", "Cz"],
        )
    # Parse the payload to validate its JSON structure
    payload = json.loads(str(exc.value))
    # Capture the normalized path to assert the resolved context is surfaced
    expected_path = str(Path("record.txt").expanduser().resolve())
    # Define the exact structured payload expected for unsupported suffixes
    expected = {
        "error": "Unsupported file format",
        "path": expected_path,
        "suffix": ".txt",
    }
    # Ensure the structured payload remains identical for debugging tools
    assert payload == expected


def test_load_mne_raw_checked_surfaces_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure missing recordings raise FileNotFoundError with path context."""

    # Construit un chemin vers un enregistrement inexistant pour simuler une absence
    missing_file = tmp_path / "S404" / "run99.edf"
    # Capture le chemin résolu pour vérifier sa présence dans le message
    resolved_missing = missing_file.resolve()
    # Simule l'échec de lecture en répliquant l'erreur d'absence de fichier
    monkeypatch.setattr(
        mne.io,
        "read_raw_edf",
        lambda path, *_args, **_kwargs: (
            (_ for _ in ()).throw(FileNotFoundError(f"No recording at {path}"))
        ),
    )
    # Vérifie que l'erreur inclut bien le chemin normalisé
    with pytest.raises(FileNotFoundError) as exc:
        load_mne_raw_checked(
            missing_file,
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4"],
        )
    # Parse le payload JSON pour inspecter les champs contextualisés
    payload = json.loads(str(exc.value))
    # Valide que le chemin résolu est bien exposé dans l'erreur
    assert payload["path"] == str(resolved_missing)
    # Vérifie que l'erreur est bien catégorisée comme fichier absent
    assert payload["error"] == "Missing recording file"


def test_load_mne_raw_checked_reports_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure corrupted recordings raise a structured parsing error."""

    # Prépare un chemin factice pour simuler un fichier corrompu
    corrupted_file = tmp_path / "bad.edf"
    # Simule un échec MNE typique lors du parsing
    monkeypatch.setattr(
        mne.io,
        "read_raw_edf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("EOF")),
    )
    # Vérifie que l'erreur remonte un diagnostic JSON structuré
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            corrupted_file,
            expected_montage="standard_1020",
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4"],
        )
    # Parse le payload JSON pour inspecter le contexte fourni
    payload = json.loads(str(exc.value))
    # Vérifie que le chemin corrompu est reporté
    assert payload["path"] == str(corrupted_file.resolve())
    # Vérifie que la catégorie d'erreur est explicite
    assert payload["error"] == "MNE parse failure"
    # Vérifie que l'exception d'origine est mentionnée
    assert payload["exception"] == "RuntimeError"


def test_load_physionet_raw_surfaces_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure Physionet loader reports missing files with context."""

    # Construit un chemin inexistant pour simuler un manque de fichier
    missing_file = tmp_path / "missing.edf"
    # Simule l'erreur MNE d'absence de fichier
    monkeypatch.setattr(
        mne.io,
        "read_raw_edf",
        lambda path, *_args, **_kwargs: (
            (_ for _ in ()).throw(FileNotFoundError(f"No file at {path}"))
        ),
    )
    # Vérifie que l'erreur est bien remontée sous forme structurée
    with pytest.raises(FileNotFoundError) as exc:
        load_physionet_raw(missing_file)
    # Parse le payload JSON pour extraire les champs
    payload = json.loads(str(exc.value))
    # Vérifie que le chemin résolu est présent dans le message
    assert payload["path"] == str(missing_file.resolve())
    # Vérifie que la catégorie d'erreur est explicite
    assert payload["error"] == "Missing recording file"


def test_load_mne_raw_checked_flags_missing_only_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure missing channels without extras raise a structured error."""

    # Build a raw object missing a channel expected by the caller
    raw = _build_dummy_raw()
    # Stub the EDF reader to return the prepared recording
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_args, **_kwargs: raw)
    # Reuse the montage name across call + assertions to lock the payload key
    expected_montage = "standard_1020"
    # Expect a ValueError because one expected channel is absent
    with pytest.raises(ValueError) as exc:
        load_mne_raw_checked(
            Path("record.edf"),
            expected_montage=expected_montage,
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4", "Cz"],
        )
    # Parse the JSON payload to inspect the structured error contents
    payload = json.loads(str(exc.value))
    # Confirm the top-level error message remains stable for CI assertions
    assert payload["error"] == "Montage missing expected channels"
    # Ensure no extra channels are reported when only missing channels occur
    assert payload["extra"] == []
    # Ensure the missing list highlights the absent channel explicitly
    assert payload["missing_channels"] == ["Cz"]
    # Ensure the payload preserves the montage context under a stable key
    assert payload["montage"] == expected_montage


def test_load_mne_raw_checked_supports_bdf_and_montage_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validate BDF loading, montage 10-20 coverage, and sampling checks."""

    # Construit un enregistrement 10-20 minimal pour simuler un fichier BDF
    raw = _build_dummy_raw(sfreq=128.0, duration=0.5)
    # Déclare la fréquence attendue pour éviter une valeur magique dans les tests
    expected_sampling_rate = 128.0
    # Applique le montage pour fournir des positions attendues aux validations
    raw.set_montage("standard_1020")
    # Capture les arguments transmis à la lecture EDF/BDF pour vérification
    captured_args: list[tuple[tuple[object, ...], dict[str, object]]] = []

    # Remplace le lecteur EDF/BDF pour tracer les appels sans accès disque
    def reader_stub(*args: object, **kwargs: object) -> mne.io.BaseRaw:
        # Enregistre les arguments afin de valider l'appel au chargeur
        captured_args.append((args, kwargs))
        # Retourne une copie pour éviter de modifier l'instance d'origine
        return raw.copy()

    # Patch la fonction de lecture pour intercepter les requêtes de chargement
    monkeypatch.setattr(mne.io, "read_raw_edf", reader_stub)
    # Crée un chemin BDF factice pour valider la prise en charge de l'extension
    bdf_path = tmp_path / "sample.bdf"
    # Charge l'enregistrement en imposant les paramètres attendus
    loaded_raw = load_mne_raw_checked(
        bdf_path,
        expected_montage="standard_1020",
        expected_sampling_rate=expected_sampling_rate,
        expected_channels=["C3", "C4"],
    )
    # Vérifie que la fonction de lecture a bien reçu le chemin normalisé
    assert captured_args[0][0][0] == bdf_path.resolve()
    # Contrôle que le taux d'échantillonnage est bien respecté
    assert loaded_raw.info["sfreq"] == expected_sampling_rate
    # S'assure que le montage reste attaché après la validation des canaux
    assert loaded_raw.get_montage() is not None


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


def test_create_epochs_rejects_non_numeric_events() -> None:
    """Ensure event validation rejects non-integer indices."""

    # Construit un enregistrement brut compatible avec la validation des événements
    raw = _build_dummy_raw()
    # Injecte un code d'événement non entier pour provoquer une erreur ciblée
    invalid_events = np.array([[0, 0, "a"]], dtype=object)
    # Associe le label non entier à un identifiant pour simuler un usage incorrect
    event_id = {"a": 1}
    # Vérifie que le message d'erreur reste strict pour détecter les mutations
    with pytest.raises(
        ValueError, match=r"^events must contain integer-coded sample indices$"
    ):
        # Appelle le helper afin de déclencher l'exception attendue
        create_epochs_from_raw(raw, invalid_events, event_id)


def test_create_epochs_preserves_integer_events(monkeypatch) -> None:
    """Ensure integer-coded events are reused without unnecessary copies."""

    # Construit un enregistrement brut afin de fournir les métadonnées nécessaires
    raw = _build_dummy_raw()
    # Prépare un tableau d'événements déjà typé pour détecter un recopiage inutile
    events = np.array([[0, 0, 1]], dtype=int)
    # Associe le label attendu à l'identifiant numérique pour l'appel Epochs
    event_id = {"a": 1}
    # Capture la référence des événements transmis à mne.Epochs pour vérifier l'identité
    captured: dict[str, np.ndarray] = {}

    # Déclare une fabrique d'Epochs qui enregistre l'argument events reçu
    def fake_epochs(raw_input: mne.io.BaseRaw, **kwargs: object) -> str:
        # Confirme que la construction reçoit bien l'objet Raw fourni par le test
        assert raw_input is raw
        # Extrait le tableau d'événements transmis via kwargs pour analyse
        # Confirme que la référence capturée représente l'objet original
        captured_events = cast(np.ndarray, kwargs["events"])
        # Enregistre la référence exacte des événements pour détecter une copie
        captured["events"] = captured_events
        # Retourne un marqueur simple pour éviter l'instanciation réelle d'Epochs
        return "sentinel"

    # Remplace mne.Epochs pour observer l'argument events sans effets de bord
    monkeypatch.setattr(mne, "Epochs", fake_epochs)
    # Crée les epochs via le helper en conservant la référence events originale
    result = create_epochs_from_raw(raw, events, event_id)
    # Vérifie que le résultat correspond au marqueur renvoyé par la fabrique factice
    assert result == "sentinel"
    # Contrôle que le tableau d'événements n'a pas été copié lors de l'appel
    assert captured["events"] is events


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
    assert "Unknown annotation labels" in str(exc.value)


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


def test_detect_artifacts_rejects_amplitude_and_variance() -> None:
    """Ensure artifact detection removes samples exceeding thresholds."""

    # Construit un signal synthétique avec des excursions élevées ciblées
    signal = np.array(
        [
            [0.1, 0.2, 5.0, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.1, 0.1, 0.2, 0.1, 0.1, 3.0, 0.1, 0.1],
        ]
    )
    # Applique le détecteur en mode rejet pour retirer les échantillons fautifs
    cleaned, mask = detect_artifacts(
        signal, amplitude_threshold=1.0, variance_threshold=0.5, mode="reject"
    )
    # Vérifie que les indices contaminés ont été correctement marqués
    assert mask.tolist() == [False, False, True, False, False, True, False, False]
    # Contrôle que les colonnes marquées ont été supprimées du signal nettoyé
    assert cleaned.shape[1] == signal.shape[1] - 2


def test_detect_artifacts_interpolates_flagged_samples() -> None:
    """Ensure interpolation preserves length while smoothing spikes."""

    # Construit un signal avec un pic isolé sur un canal
    signal = np.array([[0.0, 0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    # Applique le détecteur en mode interpolation pour combler le pic
    interpolated, mask = detect_artifacts(
        signal, amplitude_threshold=1.0, variance_threshold=0.2, mode="interpolate"
    )
    # Vérifie que le masque identifie uniquement le pic
    assert mask.tolist() == [False, False, True, False, False]
    # Confirme que la longueur reste identique après interpolation
    assert interpolated.shape == signal.shape
    # Contrôle que le pic a été remplacé par une valeur interpolée nulle
    assert pytest.approx(interpolated[0, 2]) == 0.0


def test_detect_artifacts_interpolation_with_no_valid_points() -> None:
    """Keep raw signal when interpolation cannot find valid anchors."""

    # Construit un signal où chaque colonne dépasse les seuils définis
    signal = np.array([[5.0, 5.0], [5.0, 5.0]])
    # Applique l'interpolation sans aucun point fiable pour guider le calcul
    interpolated, mask = detect_artifacts(
        signal, amplitude_threshold=1.0, variance_threshold=0.1, mode="interpolate"
    )
    # Vérifie que toutes les colonnes sont marquées comme artefacts
    assert mask.tolist() == [True, True]
    # Confirme que le signal d'origine est renvoyé intact
    assert np.allclose(interpolated, signal)


def test_detect_artifacts_defaults_to_reject_mode() -> None:
    """Ensure default behavior removes flagged samples without interpolation."""

    # Construit un signal où une seule colonne dépasse le seuil d'amplitude
    signal = np.array([[0.1, 2.5, 0.1]])
    # Applique le détecteur sans préciser le mode pour tester la valeur par défaut
    cleaned, mask = detect_artifacts(
        signal, amplitude_threshold=1.0, variance_threshold=1.0
    )
    # Vérifie que le masque identifie uniquement la colonne hors tolérance
    assert mask.tolist() == [False, True, False]
    # Confirme que la colonne fautive a été retirée du signal nettoyé
    np.testing.assert_array_equal(cleaned, np.array([[0.1, 0.1]]))


def test_detect_artifacts_rejects_unknown_mode() -> None:
    """Raise an explicit error for unsupported artifact modes."""

    # Construit un signal minimal pour vérifier la validation du mode
    signal = np.zeros((1, 2))
    # Vérifie que l'appel avec un mode inconnu lève une erreur descriptive
    with pytest.raises(
        ValueError, match=r"^mode must be either 'reject' or 'interpolate'$"
    ):
        detect_artifacts(signal, 1.0, 0.1, mode="invalid")


def test_detect_artifacts_thresholds_and_dtypes() -> None:
    """Validate strict comparisons and float casting for integer inputs."""

    # Construit un signal entier dont une valeur égale au seuil ne doit pas être rejetée
    signal = np.array([[2, 2, 2]], dtype=int)
    # Applique la détection avec un seuil égal à l'échantillon central
    cleaned, mask = detect_artifacts(
        signal, amplitude_threshold=2, variance_threshold=0.0
    )
    # Vérifie qu'une valeur exactement au seuil n'est pas considérée comme artefact
    assert mask.tolist() == [False, False, False]
    # Confirme que la sortie est bien promue en flottants pour les traitements suivants
    assert cleaned.dtype == float


def test_normalize_channels_supports_zscore_and_robust() -> None:
    """Ensure both normalization modes return centered channels."""

    # Construit un signal simple pour vérifier les deux normalisations
    signal = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    # Applique la normalisation z-score pour obtenir un écart-type unitaire
    zscore_signal = normalize_channels(signal, method="zscore")
    # Vérifie que chaque canal z-score est centré autour de zéro
    assert np.allclose(np.mean(zscore_signal, axis=1), 0.0)
    # Applique la normalisation robuste pour neutraliser l'influence des extrêmes
    robust_signal = normalize_channels(signal, method="robust")
    # Contrôle que la médiane robuste est centrée à zéro pour chaque canal
    assert np.allclose(np.median(robust_signal, axis=1), 0.0)


def test_normalize_channels_robust_respects_epsilon() -> None:
    """Ensure robust scaling keeps a positive IQR with epsilon added."""

    # Construit un signal avec un IQR non nul pour observer l'impact d'epsilon
    signal = np.array([[0.0, 1.0, 2.0]])
    # Calcule l'attendu avec un epsilon large pour différencier les mutations
    expected_iqr = (
        np.percentile(signal, 75, axis=1, keepdims=True)
        - np.percentile(signal, 25, axis=1, keepdims=True)
        + 1.0
    )
    expected = (signal - np.median(signal, axis=1, keepdims=True)) / expected_iqr
    # Vérifie que la normalisation robuste respecte cette construction
    normalized = normalize_channels(signal, method="robust", epsilon=1.0)
    # Confirme que le résultat suit l'échelle robuste attendue
    np.testing.assert_allclose(normalized, expected)


def test_normalize_channels_converts_non_float_inputs() -> None:
    """Ensure inputs convertible to float are coerced before scaling."""

    # Construit un signal au format chaîne qui nécessite une conversion explicite
    signal = np.array([["1.0", "2.0"], ["3.0", "4.0"]], dtype=object)
    # Applique la normalisation pour vérifier la conversion en flottants
    normalized = normalize_channels(signal, method="zscore")
    # Vérifie que le résultat est bien de type flottant numpy
    assert normalized.dtype.kind == "f"
    # Contrôle que la moyenne de chaque canal normalisé est centrée
    assert np.allclose(np.mean(normalized, axis=1), 0.0)


def test_normalize_channels_robust_computes_per_channel() -> None:
    """Ensure percentiles are computed channel-wise, not globally."""

    # Construit un signal où les canaux possèdent des répartitions contrastées
    signal = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
    # Calcule les statistiques attendues canal par canal pour chaque ligne
    medians = np.median(signal, axis=1, keepdims=True)
    iqrs = (
        np.percentile(signal, 75, axis=1, keepdims=True)
        - np.percentile(signal, 25, axis=1, keepdims=True)
        + 1e-3
    )
    expected = (signal - medians) / iqrs
    # Calcule la normalisation robuste réelle pour comparaison
    normalized = normalize_channels(signal, method="robust", epsilon=1e-3)
    # Vérifie que chaque canal reste centré indépendamment
    assert np.allclose(np.median(normalized, axis=1), 0.0)
    # Confirme que les échelles distinctes des canaux sont bien conservées
    assert not np.allclose(normalized[0], normalized[1])
    # Compare la sortie à l'attendu construit canal par canal
    np.testing.assert_allclose(normalized, expected)


def test_normalize_channels_defaults_apply_epsilon() -> None:
    """Validate the default z-score path and epsilon addition."""

    # Construit un signal non centré pour observer l'effet de l'epsilon explicite
    signal = np.array([[1.0, 2.0], [2.0, 4.0]])
    # Calcule l'attendu avec epsilon important pour discriminer les mutations
    expected = (signal - np.mean(signal, axis=1, keepdims=True)) / (
        np.std(signal, axis=1, keepdims=True) + 1.0
    )
    # Vérifie que la fonction applique la même formule avec les paramètres par défaut
    normalized = normalize_channels(signal, epsilon=1.0)
    # Contrôle que la sortie correspond strictement à l'attendu numérique
    np.testing.assert_allclose(normalized, expected)


def test_normalize_channels_rejects_unknown_method() -> None:
    """Raise an explicit error when normalization method is invalid."""

    # Construit un signal simple pour valider la gestion des méthodes inconnues
    signal = np.array([[1.0, 2.0]])
    # Vérifie que l'appel avec un nom de méthode non supporté échoue clairement
    with pytest.raises(
        ValueError, match=r"^method must be either 'zscore' or 'robust'$"
    ):
        normalize_channels(signal, method="invalid")


# Garantit deux appels dtype explicites pour bloquer les mutations silencieuses
EXPECTED_NORMALIZE_DTYPE_CALLS = 2


def test_normalize_channels_calls_asarray_with_dtype_for_zscore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce dtype hints on z-score output to kill silent mutations."""

    # Construit une liste pour suivre les dtype passés aux conversions numpy
    recorded_dtypes: list[object] = []
    # Capture l'implémentation réelle pour déléguer la conversion
    original_asarray = preprocessing.np.asarray

    # Enveloppe np.asarray pour enregistrer les dtype demandés
    def recording_asarray(*args: Any, **kwargs: Any) -> np.ndarray:
        recorded_dtypes.append(kwargs.get("dtype"))
        return original_asarray(*args, **kwargs)

    # Injecte l'enveloppe dans le module de prétraitement
    monkeypatch.setattr(preprocessing.np, "asarray", recording_asarray)
    # Construit un signal entier pour différencier les conversions explicites
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    # Lance la normalisation z-score pour déclencher les conversions internes
    normalize_channels(signal, method="zscore", epsilon=0.5)
    # Vérifie que les conversions incluent deux requêtes dtype=float
    assert recorded_dtypes.count(float) == EXPECTED_NORMALIZE_DTYPE_CALLS


def test_normalize_channels_calls_asarray_with_dtype_for_robust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce dtype hints on robust output to kill silent mutations."""

    # Construit un journal des dtype transmis aux conversions robustes
    recorded_dtypes: list[object] = []
    # Capture np.asarray pour déléguer tout en observant les appels
    original_asarray = preprocessing.np.asarray

    # Enveloppe np.asarray pour contrôler les paramètres dtype
    def recording_asarray(*args: Any, **kwargs: Any) -> np.ndarray:
        recorded_dtypes.append(kwargs.get("dtype"))
        return original_asarray(*args, **kwargs)

    # Remplace np.asarray dans le module pour enregistrer les appels ciblés
    monkeypatch.setattr(preprocessing.np, "asarray", recording_asarray)
    # Construit un signal asymétrique pour exercer la branche robuste
    signal = np.array([[0, 1, 2], [5, 5, 5]], dtype=int)
    # Applique la normalisation robuste pour forcer les conversions finales
    normalize_channels(signal, method="robust", epsilon=0.25)
    # Contrôle que deux conversions dtype=float ont été requises
    assert recorded_dtypes.count(float) == EXPECTED_NORMALIZE_DTYPE_CALLS


def test_load_mne_motor_run_reports_channel_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure explicit JSON reports surface when channels are missing."""

    # Construit un enregistrement minimal conforme au montage 10-20
    raw = _build_dummy_raw(sfreq=128.0, duration=0.5)
    raw.set_montage("standard_1020")
    file_path = tmp_path / "sample.edf"
    monkeypatch.setattr("mne.io.read_raw_edf", lambda *_args, **_kwargs: raw)
    # Vérifie qu'un canal absent déclenche un rapport structuré
    with pytest.raises(ValueError) as excinfo:
        load_mne_motor_run(
            file_path,
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4", "Cz"],
        )
    message = str(excinfo.value)
    payload = json.loads(message)
    assert payload["error"] == "Montage missing expected channels"
    assert payload["missing_channels"] == ["Cz"]


def test_load_mne_motor_run_maps_motor_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load a run end-to-end and keep only mapped motor events."""

    # Construit un enregistrement motorisé avec une annotation de repos T0
    raw = _build_dummy_raw(sfreq=128.0, duration=0.5)
    # Recalibre les annotations pour rester dans la fenêtre temporelle utile
    raw.set_annotations(
        mne.Annotations(
            onset=[0.05, 0.15, 0.25],
            duration=[0.05, 0.05, 0.05],
            description=["T0", "T1", "T2"],
        )
    )
    # Applique le montage attendu pour aligner la validation des canaux
    raw.set_montage("standard_1020")
    # Fixe un chemin factice pour simuler la lecture disque
    file_path = tmp_path / "motor.edf"
    # Remplace la lecture EDF pour renvoyer l'enregistrement synthétique
    monkeypatch.setattr("mne.io.read_raw_edf", lambda *_args, **_kwargs: raw)
    # Charge le run en exigeant uniquement les canaux présents
    loaded_raw, events, event_id, motor_labels = load_mne_motor_run(
        file_path,
        expected_sampling_rate=128.0,
        expected_channels=["C3", "C4"],
    )
    # Vérifie que l'enregistrement retourné correspond à l'objet source
    assert loaded_raw is raw
    # Contrôle que seul le couple T1/T2 est conservé malgré la présence de T0
    assert motor_labels == ["A", "B"]
    # S'assure que les événements filtrés excluent l'annotation de repos
    assert np.array_equal(events[:, 2], np.array([event_id["T1"], event_id["T2"]]))


def test_load_mne_motor_run_reports_unknown_subject_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure an explicit FileNotFoundError is propagated for wrong IDs."""

    # Déclare des identifiants invalides pour simuler un chemin absent
    missing_subject = "S404"
    missing_run = "R99"
    missing_file = tmp_path / missing_subject / f"{missing_run}.edf"

    # Simule la lecture EDF qui échoue immédiatement
    def raising_reader(*_args: object, **_kwargs: object) -> mne.io.BaseRaw:
        raise FileNotFoundError("missing file")

    monkeypatch.setattr("mne.io.read_raw_edf", raising_reader)
    # Vérifie que le message explicite est conservé jusqu'à l'appelant
    with pytest.raises(FileNotFoundError) as exc:
        load_mne_motor_run(
            missing_file,
            expected_sampling_rate=128.0,
            expected_channels=["C3", "C4"],
        )
    # Parse le payload JSON pour vérifier le contexte remonté
    payload = json.loads(str(exc.value))
    # Vérifie que le chemin manquant est bien exposé
    assert payload["path"] == str(missing_file.resolve())
    # Vérifie que l'erreur est catégorisée comme fichier absent
    assert payload["error"] == "Missing recording file"


def test_map_events_to_motor_labels_rejects_runs_without_motor_activity() -> None:
    """Expose une erreur structurée lorsqu'aucun événement moteur n'existe."""

    # Crée un enregistrement ne contenant que des marqueurs de repos T0
    raw = _build_dummy_raw(sfreq=128.0, duration=0.25)
    # Force des annotations neutres pour éliminer toute cible A/B
    raw.set_annotations(
        mne.Annotations(
            onset=[0.05, 0.15],
            duration=[0.05, 0.05],
            description=["T0", "T0"],
        )
    )
    # Fixe le montage pour rester cohérent avec le pipeline standard
    raw.set_montage("standard_1020")
    # Vérifie qu'une erreur JSON est levée faute d'événements moteurs
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw)
    payload = json.loads(str(excinfo.value))
    # Confirme la nature de l'erreur et les étiquettes disponibles observées
    assert payload["error"] == "No motor events present"
    assert payload["available_labels"] == ["T0", "T1", "T2"]


def test_map_events_to_motor_labels_reports_unknown_codes() -> None:
    """Expose un rapport JSON lorsqu'un code d'événement est inconnu."""

    # Construit un enregistrement avec un code d'annotation hors mapping
    raw = _build_dummy_raw(sfreq=128.0, duration=0.25)
    # Remplace les annotations pour ne contenir que le code non référencé
    raw.set_annotations(
        mne.Annotations(
            onset=[0.05], duration=[0.05], description=["T9"], orig_time=None
        )
    )
    # Applique le montage standard pour rester cohérent avec la validation
    raw.set_montage("standard_1020")
    # Vérifie qu'une erreur structurée recense le code inconnu rencontré
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw)
    # Décode la charge utile JSON afin de vérifier le contenu détaillé
    payload = json.loads(str(excinfo.value))
    # Confirme la nature explicite de l'erreur et le code détecté
    assert payload["error"] == "Unknown annotation labels"
    assert payload["unknown_labels"] == ["T9"]


def test_map_events_to_motor_labels_flags_unknown_numeric_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose un rapport clair lorsque des codes numériques sont inconnus."""

    # Construit un enregistrement synthétique avec un montage standard
    raw = _build_dummy_raw(sfreq=256.0, duration=0.25)
    raw.set_montage("standard_1020")
    # Forge une séquence d'événements mélangeant codes valides et invalide
    event_id = dict(PHYSIONET_LABEL_MAP)
    mapped_events = np.array(
        [
            [0, 0, event_id["T1"]],
            [32, 0, 999],
        ]
    )
    # Force la validation à retourner le tableau artificiel pour isoler la branche
    monkeypatch.setattr(
        "tpv.preprocessing.map_events_and_validate",
        lambda *_args, **_kwargs: (mapped_events, event_id),
    )
    # Vérifie que l'erreur JSON répertorie précisément le code inconnu
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw)
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "Unknown event codes"
    assert payload["unknown_codes"] == [999]


def test_map_events_to_motor_labels_forwards_mapping_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure mapping helpers receive the explicit label and motor maps."""

    # Construit un Raw minimal pour satisfaire la signature sans dépendances I/O
    raw = _build_dummy_raw(sfreq=256.0, duration=0.25)
    # Applique un montage standard pour rester cohérent avec le pipeline
    raw.set_montage("standard_1020")
    # Définit un mapping moteur custom pour verrouiller la copie via dict(...)
    motor_map = {"T1": "A", "T2": "B"}
    # Définit un mapping de labels custom pour vérifier le forwarding inchangé
    label_map = {"T0": 0, "T1": 1, "T2": 2}
    # Prépare des événements factices stables pour alimenter le chemin nominal
    expected_events = np.array([[0, 0, 1], [32, 0, 2]])
    # Prépare un event_id minimal pour construire inv_event_id sans ambiguïté
    expected_event_id = {"T1": 1, "T2": 2}
    # Capture les arguments transmis au helper pour détecter les mutants
    captured: dict[str, object] = {}

    # Espionne l'appel afin de verrouiller les kwargs transmis au helper
    def spy_map_events_and_validate(
        raw_arg: mne.io.BaseRaw,
        *,
        label_map: Mapping[str, int],
        motor_label_map: Mapping[str, str],
    ) -> tuple[np.ndarray, dict[str, int]]:
        # Capture le Raw pour vérifier la propagation
        captured["raw"] = raw_arg
        # Capture le label_map pour refuser un None injecté par mutation
        captured["label_map"] = label_map
        # Capture le motor_label_map pour refuser les omissions ou None mutés
        captured["motor_label_map"] = motor_label_map
        # Retourne la paire events/event_id attendue par l'appelant
        return expected_events, expected_event_id

    # Remplace le helper par l'espion afin d'observer les arguments transmis
    monkeypatch.setattr(
        "tpv.preprocessing.map_events_and_validate",
        spy_map_events_and_validate,
    )

    # Appelle la fonction sous test avec des mappings explicites
    events, event_id, motor_labels = map_events_to_motor_labels(
        raw,
        label_map=label_map,
        motor_label_map=motor_map,
    )

    # Confirme que l'instance Raw transmise au helper est identique
    assert captured["raw"] is raw
    # Confirme que le label_map est propagé sans transformation ni suppression
    assert captured["label_map"] is label_map
    # Confirme que le mapping moteur est bien celui attendu en contenu
    assert captured["motor_label_map"] == motor_map
    # Confirme que dict(...) copie la structure pour éviter les aliasing
    assert captured["motor_label_map"] is not motor_map
    # Confirme que la sortie conserve les événements synthétiques renvoyés
    assert np.array_equal(events, expected_events)
    # Confirme que le mapping retourné est celui produit par le helper
    assert event_id == expected_event_id
    # Confirme que la convention T1/T2 -> A/B est respectée
    assert motor_labels == ["A", "B"]


def test_map_events_to_motor_labels_reports_all_unknown_numeric_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the report accumulates every unknown numeric event code."""

    # Construit un Raw minimal pour isoler la logique de mapping d'événements
    raw = _build_dummy_raw(sfreq=256.0, duration=0.25)
    # Applique un montage standard pour rester cohérent avec la validation
    raw.set_montage("standard_1020")
    # Fixe deux codes inconnus pour vérifier l'accumulation au lieu d'un arrêt
    first_unknown = 999
    # Fixe un second code inconnu pour détecter un break muté en boucle
    second_unknown = 888
    # Prépare un event_id minimal ne contenant qu'un label connu
    event_id = {"T1": 1}
    # Forge des événements où un inconnu précède et suit un code valide
    mapped_events = np.array(
        [
            [0, 0, first_unknown],
            [16, 0, event_id["T1"]],
            [32, 0, second_unknown],
        ]
    )
    # Force la validation à renvoyer les événements artificiels
    monkeypatch.setattr(
        "tpv.preprocessing.map_events_and_validate",
        lambda *_args, **_kwargs: (mapped_events, event_id),
    )

    # Vérifie que l'erreur JSON répertorie tous les codes inconnus rencontrés
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw)
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "Unknown event codes"
    assert payload["unknown_codes"] == [first_unknown, second_unknown]


def test_map_events_to_motor_labels_rejects_empty_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assure une erreur explicite lorsque aucun événement n'est disponible."""

    # Construit un enregistrement minimal avec montage appliqué
    raw = _build_dummy_raw(sfreq=128.0, duration=0.1)
    raw.set_montage("standard_1020")
    # Simule une extraction d'événements vide après filtrage des segments BAD
    empty_events = np.empty((0, 3), dtype=int)
    event_id = dict(PHYSIONET_LABEL_MAP)
    monkeypatch.setattr(
        "tpv.preprocessing.map_events_and_validate",
        lambda *_args, **_kwargs: (empty_events, event_id),
    )
    # Vérifie que l'erreur inclut les labels disponibles pour diagnostic
    with pytest.raises(ValueError) as excinfo:
        map_events_to_motor_labels(raw)
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "No motor events present"
    assert payload["available_labels"] == ["T0", "T1", "T2"]


def test_map_events_to_motor_labels_maps_left_and_right_cues() -> None:
    """Vérifie le mapping A/B des essais gauche (T1) et droite (T2)."""

    # Construit un enregistrement alternant les annotations main gauche/droite
    raw = _build_dummy_raw(sfreq=256.0, duration=0.5)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.05, 0.15, 0.25],
            duration=[0.05, 0.05, 0.05],
            description=["T1", "T2", "T1"],
        )
    )
    raw.set_montage("standard_1020")
    # Mappe les événements pour s'assurer que les labels moteurs suivent la convention
    events, event_id, motor_labels = map_events_to_motor_labels(raw)
    assert motor_labels == ["A", "B", "A"]
    assert event_id == {"T1": 1, "T2": 2}
    assert list(events[:, 2]) == [event_id["T1"], event_id["T2"], event_id["T1"]]


def test_summarize_epoch_quality_passes_reject_mode_and_reports_session_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the QC mode and report keys used by monitoring tooling."""

    # Construit des epochs équilibrés pour éviter les cas d'erreur sur les labels
    epochs, labels = _build_epoch_array()
    # Fixe une session stable pour verrouiller les valeurs reportées
    session = (TEST_SUBJECT, TEST_RUN)
    # Prépare un dictionnaire pour capturer le mode réellement transmis
    captured: dict[str, str] = {}

    # Définit un espion pour rendre le paramètre mode obligatoire côté appelant
    def spy_quality_control_epochs(epochs, *, max_peak_to_peak, mode):
        # Mémorise le mode afin de vérifier qu'il est transmis explicitement
        captured["mode"] = mode
        # Retourne les epochs inchangées pour isoler la logique de reporting
        return epochs, {"artifact": [], "incomplete": []}

    # Patch la fonction de contrôle qualité pour observer les arguments transmis
    monkeypatch.setattr(
        preprocessing, "quality_control_epochs", spy_quality_control_epochs
    )
    # Lance le résumé pour déclencher l'appel surveillé à quality_control_epochs
    cleaned_epochs, report, cleaned_labels = summarize_epoch_quality(
        epochs,
        labels,
        session=session,
        max_peak_to_peak=0.5,
    )
    # Vérifie que le mode est explicitement défini sur reject pour verrouiller l'API
    assert captured["mode"] == "reject"
    # Confirme que les clés de session restent stables pour les exports
    assert report["subject"] == session[0]
    assert report["run"] == session[1]
    # Confirme que les labels retournés restent alignés avec les epochs conservées
    assert len(cleaned_epochs) == len(cleaned_labels)


def test_summarize_epoch_quality_counts_and_rejects_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop incomplete epochs then count A/B labels per run."""

    # Fixe le nombre d'epochs attendues pour clarifier la validation
    expected_valid_epochs = 2
    # Conserve l'implémentation native pour valider le mode demandé
    original_quality_control_epochs = preprocessing.quality_control_epochs
    # Prépare un journal d'appels pour vérifier le mode transmis
    qc_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    # Définit un espion pour capturer les arguments du contrôle qualité
    def spy_quality_control_epochs(*args: object, **kwargs: object):
        # Enregistre les arguments afin de verrouiller le mode utilisé
        qc_calls.append((args, dict(kwargs)))
        # Délègue à l'implémentation réelle pour préserver le comportement
        delegate = cast(
            Callable[..., tuple[mne.Epochs, dict[str, list[int]]]],
            original_quality_control_epochs,
        )
        return delegate(*args, **kwargs)

    # Patch la fonction pour observer le mode sans changer la production
    monkeypatch.setattr(
        preprocessing, "quality_control_epochs", spy_quality_control_epochs
    )
    # Construit un enregistrement synthétique avec trois essais moteurs
    info = mne.create_info(["C3", "C4"], sfreq=64.0, ch_types="eeg")
    data = np.ones((2, 64))
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.3, 0.5],
            duration=[0.1, 0.1, 0.1],
            description=["T1", "T2", "T1"],
        )
    )
    raw.set_montage("standard_1020")
    events, event_id, motor_labels = map_events_to_motor_labels(raw)
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    epochs._data[0, 0, 0] = np.nan
    # Fixe les identifiants de session pour vérifier leur propagation
    session = ("S001", "R01")
    cleaned_epochs, report, cleaned_labels = summarize_epoch_quality(
        epochs,
        motor_labels,
        session=session,
        max_peak_to_peak=0.5,
    )
    # Vérifie que le contrôle qualité a bien été invoqué une fois
    assert len(qc_calls) == 1
    # Extrait les arguments capturés pour valider le mode effectif
    args, kwargs = qc_calls[0]
    # Force un passage explicite du mode pour éviter un changement silencieux
    assert ("mode" in kwargs) or (len(args) >= 3)
    # Verrouille le mode attendu pour imposer le rejet des epochs invalides
    effective_mode = kwargs["mode"] if "mode" in kwargs else args[2]
    assert effective_mode == "reject"
    assert len(cleaned_epochs) == expected_valid_epochs
    assert cleaned_labels == ["B", "A"]
    assert set(report.keys()) == {"subject", "run", "dropped", "counts"}
    assert report["subject"] == session[0]
    assert report["run"] == session[1]
    assert report["counts"] == {"A": 1, "B": 1}
    assert report["dropped"]["incomplete"] == 1


def test_summarize_epoch_quality_reports_missing_labels() -> None:
    """Raise a structured error when a motor class is absent."""

    # Fixe le nombre d'essais pour la classe A afin d'éviter les valeurs magiques
    expected_trials_for_a = 2
    # Construit un enregistrement ne contenant qu'une seule classe motrice
    info = mne.create_info(["C3", "C4"], sfreq=64.0, ch_types="eeg")
    data = np.ones((2, 64))
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.4],
            duration=[0.1, 0.1],
            description=["T1", "T1"],
        )
    )
    raw.set_montage("standard_1020")
    events, event_id, motor_labels = map_events_to_motor_labels(raw)
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    with pytest.raises(ValueError) as excinfo:
        summarize_epoch_quality(
            epochs,
            motor_labels,
            session=("S99", "R02"),
            max_peak_to_peak=0.5,
        )
    payload = json.loads(str(excinfo.value))
    assert payload["error"] == "Missing labels"
    assert payload["missing_labels"] == ["B"]
    assert payload["counts"]["A"] == expected_trials_for_a


def test_summarize_epoch_quality_reports_label_mismatch() -> None:
    """Détecte un décalage entre événements et étiquettes fournies."""

    # Construit un enregistrement avec deux essais moteurs équilibrés
    raw = _build_dummy_raw(sfreq=64.0, duration=0.5)
    # Applique un montage standard pour rester aligné avec la validation
    raw.set_montage("standard_1020")
    # Convertit les annotations en événements et identifiants numériques
    events, event_id, motor_labels = map_events_to_motor_labels(raw)
    # Crée des epochs courtes autour des événements détectés
    epochs = create_epochs_from_raw(raw, events, event_id, tmin=0.0, tmax=0.1)
    # Supprime un label pour provoquer un décalage volontaire
    truncated_labels = motor_labels[:-1]
    # Vérifie qu'un rapport clair est produit pour ce décalage détecté
    with pytest.raises(ValueError) as excinfo:
        summarize_epoch_quality(
            epochs,
            truncated_labels,
            session=("S05", "R03"),
            max_peak_to_peak=0.5,
        )
    payload = json.loads(str(excinfo.value))
    # Confirme la présence du message d'erreur attendu dans le rapport JSON
    assert payload["error"] == "Label/event mismatch"
    assert payload["expected_events"] == len(epochs)
    assert payload["labels"] == len(truncated_labels)


def test_preprocessing_signatures_preserve_documented_defaults() -> None:
    """Protect the public API defaults used in documentation and examples."""

    # Capture la signature du filtre pour verrouiller la valeur par défaut
    filter_signature = inspect.signature(apply_bandpass_filter)
    # Vérifie que la méthode par défaut reste bien en minuscules pour la doc
    assert filter_signature.parameters["method"].default == DEFAULT_FILTER_METHOD
    # Capture la signature de la normalisation pour sécuriser le paramètre
    normalize_signature = inspect.signature(normalize_channels)
    # Confirme que le mode par défaut reste conforme aux guides utilisateur
    assert normalize_signature.parameters["method"].default == DEFAULT_NORMALIZE_METHOD
    # Vérifie que l'epsilon par défaut reste fixé à 1e-8
    assert normalize_signature.parameters["epsilon"].default == pytest.approx(
        DEFAULT_NORMALIZE_EPSILON
    )
