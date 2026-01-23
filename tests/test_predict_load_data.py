"""Tests ciblés sur scripts.predict._load_data pour sécuriser la reconstruction."""

import json
import sys
from pathlib import Path

import numpy as np

# Importe pytest pour capturer les exceptions attendues
import pytest

from scripts import predict


# Verrouille l'initialisation booléenne de needs_rebuild (mutant équivalent sinon)
def test_predict_load_data_initializes_needs_rebuild_as_false(
    tmp_path, monkeypatch
) -> None:
    """Verrouille needs_rebuild: bool False dès l'entrée dans l'implémentation."""

    subject = "S011"
    run = "R03"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)

    # Prépare des fichiers valides pour forcer le chemin "pas de rebuild".
    expected_X = np.arange(12).reshape(3, 2, 2)
    expected_y = np.array([0, 1, 0])
    np.save(subject_dir / f"{run}_X.npy", expected_X)
    np.save(subject_dir / f"{run}_y.npy", expected_y)

    # Verrouille l'absence de rebuild dans ce scénario.
    def _forbid_rebuild(*args, **kwargs):
        raise AssertionError("_build_npy_from_edf ne doit pas être appelé ici")

    monkeypatch.setattr(predict, "_build_npy_from_edf", _forbid_rebuild)

    captured: dict[str, object] = {}

    # Capture la première valeur observée de needs_rebuild dans une frame load_data.
    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        if not filename.endswith("scripts/predict.py"):
            return tracer
        if "load_data" not in name:
            return tracer
        if "needs_rebuild" not in frame.f_locals:
            return tracer
        if "value" not in captured:
            captured["value"] = frame.f_locals["needs_rebuild"]
        return tracer

    previous_tracer = sys.gettrace()
    sys.settrace(tracer)
    try:
        predict._load_data(subject, run, data_dir, raw_dir)
    finally:
        sys.settrace(previous_tracer)

    assert "value" in captured
    assert captured["value"] is False
    assert type(captured["value"]) is bool


# Vérifie que _build_npy_from_edf est invoqué dès que les .npy sont invalides
def test_predict_load_data_rebuilds_invalid_numpy_payloads(tmp_path, monkeypatch):
    """Force la reconstruction sur X 2D et sur un y désaligné."""

    # Prépare le sujet/run pour isoler les fichiers temporaires
    subject = "S010"
    run = "R02"
    # Construit les répertoires attendus par la signature
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    # Crée l'arborescence pour simuler des .npy déjà présents
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    # Prépare les tenseurs régénérés par la reconstruction simulée
    rebuilt_X = np.full((2, 3, 4), fill_value=5)
    rebuilt_y = np.array([0, 1])
    # Trace les appels de reconstruction pour vérifier la propagation des arguments
    calls: list[tuple[str, str, Path, Path]] = []

    # Stub de reconstruction qui remplace l'EDF pendant le test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        calls.append((subject_arg, run_arg, data_arg, raw_arg))
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        np.save(features_path, rebuilt_X)
        np.save(labels_path, rebuilt_y)
        return features_path, labels_path

    monkeypatch.setattr(predict, "_build_npy_from_edf", fake_build_npy)

    # 1) X 2D pour déclencher la reconstruction liée à la mauvaise dimension
    np.save(subject_dir / f"{run}_X.npy", np.ones((2, 4)))
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, data_dir, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

    # 2) y de longueur différente pour forcer la régénération
    calls.clear()
    np.save(subject_dir / f"{run}_X.npy", np.ones((3, 2, 2)))
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1, 1, 0]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, data_dir, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)


# Vérifie que la reconstruction n'est pas déclenchée si les .npy sont valides
def test_predict_load_data_skips_rebuild_for_valid_files(tmp_path, monkeypatch):
    """Couvre le chemin nominal lorsque X et y sont déjà conformes."""

    subject = "S011"
    run = "R03"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    # Génère des tensors valides pour éviter toute reconstruction
    expected_X = np.arange(12).reshape(3, 2, 2)
    expected_y = np.array([0, 1, 0])
    np.save(subject_dir / f"{run}_X.npy", expected_X)
    np.save(subject_dir / f"{run}_y.npy", expected_y)

    # Espionne np.load pour imposer mmap_mode="r" sur les chargements "candidate"
    real_load = predict.np.load
    load_calls: list[tuple[Path, str]] = []

    def spy_load(path, *args, **kwargs):
        mode = kwargs.get("mmap_mode", "__missing__")
        load_calls.append((Path(path), str(mode)))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(predict.np, "load", spy_load)

    # Stub qui provoquerait un échec si _build_npy_from_edf était appelé
    def fail_build_npy(*args, **kwargs):
        raise AssertionError("_build_npy_from_edf ne doit pas être invoqué")

    monkeypatch.setattr(predict, "_build_npy_from_edf", fail_build_npy)

    X, y = predict._load_data(subject, run, data_dir, raw_dir)

    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)

    features_path = subject_dir / f"{run}_X.npy"
    labels_path = subject_dir / f"{run}_y.npy"
    features_modes = [mode for path, mode in load_calls if path == features_path]
    labels_modes = [mode for path, mode in load_calls if path == labels_path]
    assert sorted(features_modes) == ["__missing__", "r"]
    assert sorted(labels_modes) == ["__missing__", "r"]


# Vérifie que la reconstruction est déclenchée si un seul des deux fichiers manque
def test_predict_load_data_rebuilds_when_one_numpy_file_missing(tmp_path, monkeypatch):
    """Verrouille le OR de présence (un seul manquant doit reconstruire)."""

    subject = "S012"
    run = "R04"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)

    rebuilt_X = np.zeros((2, 3, 4))
    rebuilt_y = np.array([0, 1])
    calls: list[tuple[str, str, Path]] = []

    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        calls.append((subject_arg, run_arg, raw_arg))
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        np.save(features_path, rebuilt_X)
        np.save(labels_path, rebuilt_y)
        return features_path, labels_path

    monkeypatch.setattr(predict, "_build_npy_from_edf", fake_build_npy)

    # Cas 1: X présent, y absent => rebuild obligatoire
    np.save(subject_dir / f"{run}_X.npy", np.ones((2, 3, 4)))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)

    # Cas 2: y présent, X absent => rebuild obligatoire
    calls.clear()
    (subject_dir / f"{run}_X.npy").unlink()
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    X, y = predict._load_data(subject, run, data_dir, raw_dir)
    assert calls == [(subject, run, raw_dir)]
    assert np.array_equal(X, rebuilt_X)
    assert np.array_equal(y, rebuilt_y)


# Vérifie la reconstruction EDF en respectant la fenêtre persistée
def test_build_npy_from_edf_uses_epoch_window_metadata(tmp_path, monkeypatch) -> None:
    """Valide l'usage de la fenêtre persistée pour l'epoching."""

    subject = "S999"
    run = "R03"
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("stub")

    # Persiste une fenêtre custom pour ce run
    window_path = data_dir / subject / f"{run}_epoch_window.json"
    window_path.parent.mkdir(parents=True, exist_ok=True)
    window_path.write_text(json.dumps({"tmin": 0.5, "tmax": 2.5}))

    # Définit un Raw factice sans logique MNE
    class DummyRaw:
        info = {"sfreq": 100.0}

    dummy_raw = DummyRaw()

    # Stub l'entrée EDF
    monkeypatch.setattr(
        predict.preprocessing,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (dummy_raw, {}),
    )
    # Stub le notch et le bandpass
    monkeypatch.setattr(
        predict.preprocessing, "apply_notch_filter", lambda raw, **_kwargs: raw
    )
    monkeypatch.setattr(
        predict.preprocessing, "apply_bandpass_filter", lambda raw, **_kwargs: raw
    )

    # Capture la fenêtre utilisée pour l'epoching
    captured: dict[str, float] = {}

    # Définit une classe d'epochs factice pour le test
    class DummyEpochs:
        def get_data(self, copy=True):
            return np.zeros((2, 2, 2))

    dummy_epochs = DummyEpochs()

    # Stub l'epoching pour valider la fenêtre reçue
    def _fake_create_epochs(*_args, **kwargs):
        captured["tmin"] = kwargs["tmin"]
        captured["tmax"] = kwargs["tmax"]
        return dummy_epochs

    monkeypatch.setattr(
        predict.preprocessing, "create_epochs_from_raw", _fake_create_epochs
    )
    # Stub le mapping d'événements
    monkeypatch.setattr(
        predict.preprocessing,
        "map_events_to_motor_labels",
        lambda *_args, **_kwargs: (np.array([]), {"A": 1}, ["A", "B"]),
    )
    # Stub le QC pour éviter les dépendances MNE
    monkeypatch.setattr(
        predict.preprocessing,
        "summarize_epoch_quality",
        lambda *_args, **_kwargs: (dummy_epochs, {}, ["A", "B"]),
    )

    # Exécute la reconstruction depuis l'EDF
    features_path, labels_path = predict._build_npy_from_edf(
        subject, run, data_dir, raw_dir
    )

    # Vérifie que la fenêtre persistée est bien utilisée
    assert captured["tmin"] == 0.5
    assert captured["tmax"] == 2.5
    # Vérifie que les fichiers sont générés
    assert features_path.exists()
    assert labels_path.exists()


# Vérifie le fallback vers la fenêtre par défaut lorsqu'aucun JSON n'existe
def test_read_epoch_window_metadata_defaults_when_missing(tmp_path) -> None:
    """Confirme que la fenêtre par défaut est utilisée sans fichier JSON."""

    # Définit un sujet/run fictif pour la lecture de fenêtre
    subject = "S404"
    # Définit un run fictif pour isoler le chemin du JSON
    run = "R04"
    # Définit un répertoire data vide pour simuler l'absence de JSON
    data_dir = tmp_path / "data"
    # Crée le répertoire racine de données sans fichier de fenêtre
    data_dir.mkdir(parents=True, exist_ok=True)

    # Exécute la lecture de fenêtre sans JSON présent
    window = predict._read_epoch_window_metadata(subject, run, data_dir)

    # Vérifie que la fenêtre par défaut est renvoyée
    assert window == predict.DEFAULT_EPOCH_WINDOW


# Vérifie l'erreur explicite lorsque l'EDF est introuvable
def test_build_npy_from_edf_raises_when_edf_missing(tmp_path) -> None:
    """Valide l'exception FileNotFoundError pour un EDF absent."""

    # Définit un sujet/run factice pour construire le chemin EDF
    subject = "S500"
    # Définit un run factice pour la reconstruction EDF
    run = "R05"
    # Prépare le répertoire data requis par l'API
    data_dir = tmp_path / "data"
    # Prépare le répertoire raw vide pour simuler l'absence d'EDF
    raw_dir = tmp_path / "raw"
    # Calcule le chemin EDF attendu pour validation du message
    expected_path = raw_dir / subject / f"{subject}{run}.edf"

    # Vérifie que l'appel échoue proprement quand l'EDF manque
    with pytest.raises(FileNotFoundError) as exc_info:
        # Tente de construire les .npy depuis un EDF inexistant
        predict._build_npy_from_edf(subject, run, data_dir, raw_dir)

    # Vérifie que le message contient le chemin attendu
    assert str(expected_path) in str(exc_info.value)


# Vérifie le fallback lorsque summarize_epoch_quality remonte un Missing labels
def test_build_npy_from_edf_handles_missing_labels(tmp_path, monkeypatch) -> None:
    """Valide le fallback quand un ValueError 'Missing labels' survient."""

    # Définit un sujet/run pour isoler les fichiers générés
    subject = "S600"
    # Définit un run pour construire le chemin EDF attendu
    run = "R06"
    # Prépare les répertoires data/raw nécessaires
    data_dir = tmp_path / "data"
    # Prépare le répertoire raw pour déposer un EDF factice
    raw_dir = tmp_path / "raw"
    # Construit le chemin EDF attendu par la reconstruction
    raw_path = raw_dir / subject / f"{subject}{run}.edf"
    # Crée l'arborescence du fichier EDF factice
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Écrit un contenu factice pour matérialiser l'EDF
    raw_path.write_text("stub")

    # Définit des données d'epochs factices alignées sur deux labels
    epochs_data = np.ones((2, 1, 4), dtype=float)

    # Déclare des epochs factices avec un get_data compatible
    class DummyEpochs:
        # Fournit la signature attendue par _build_npy_from_edf
        def get_data(self, copy: bool = True) -> np.ndarray:
            # Retourne les données factices pour l'écriture des .npy
            return epochs_data

    # Prépare un Raw factice sans logique interne
    dummy_raw = object()

    # Force la lecture EDF à retourner un Raw factice
    monkeypatch.setattr(
        predict.preprocessing,
        "load_physionet_raw",
        lambda *_args, **_kwargs: (dummy_raw, {}),
    )
    # Neutralise le notch en renvoyant le Raw inchangé
    monkeypatch.setattr(
        predict.preprocessing, "apply_notch_filter", lambda raw, **_kwargs: raw
    )
    # Neutralise le filtrage bande-passante pour rester minimal
    monkeypatch.setattr(
        predict.preprocessing, "apply_bandpass_filter", lambda raw, **_kwargs: raw
    )
    # Fournit un mapping d'événements minimal avec deux labels
    monkeypatch.setattr(
        predict.preprocessing,
        "map_events_to_motor_labels",
        lambda *_args, **_kwargs: (
            np.zeros((2, 3), dtype=int),
            {"left": 1, "right": 2},
            ["left", "right"],
        ),
    )
    # Retourne des epochs factices pour éviter les dépendances MNE
    monkeypatch.setattr(
        predict.preprocessing,
        "create_epochs_from_raw",
        lambda *_args, **_kwargs: DummyEpochs(),
    )
    # Force un ValueError Missing labels pour couvrir le fallback
    monkeypatch.setattr(
        predict.preprocessing,
        "summarize_epoch_quality",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Missing labels")),
    )
    # Neutralise la lecture de fenêtre pour éviter un fichier JSON
    monkeypatch.setattr(
        predict, "_read_epoch_window_metadata", lambda *_args, **_kwargs: (0.0, 1.0)
    )

    # Construit les .npy en déclenchant le fallback Missing labels
    features_path, labels_path = predict._build_npy_from_edf(
        subject, run, data_dir, raw_dir
    )

    # Vérifie que les fichiers générés existent bien
    assert features_path.exists()
    # Vérifie que le fichier des labels a été écrit
    assert labels_path.exists()
    # Vérifie que les données d'epochs sont bien persistées
    assert np.array_equal(np.load(features_path), epochs_data)
    # Vérifie que les labels sont correctement encodés
    assert np.array_equal(np.load(labels_path), np.array([0, 1]))
