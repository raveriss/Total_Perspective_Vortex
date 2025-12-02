"""Tests de fumée pour scripts/visualize_raw_filtered.py sans dataset."""

# Importe pathlib pour gérer les répertoires temporaires de sortie
from pathlib import Path

# Importe mne pour construire un Raw synthétique simulant Physionet
import mne

# Importe numpy pour générer des données EEG déterministes
import numpy as np

# Importe pytest pour bénéficier de monkeypatch et tmp_path
import pytest

# Importe le module à tester pour cibler directement visualize_run
import scripts.visualize_raw_filtered as viz


# Construit un Raw synthétique minimal pour les tests rapides
def _build_dummy_raw(sfreq: float = 128.0, duration: float = 1.0) -> mne.io.Raw:
    """Génère un RawArray avec deux canaux EEG C3/C4."""

    # Crée un objet info pour typer les canaux EEG
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    # Génère des données déterministes pour stabiliser les assertions
    rng = np.random.default_rng(seed=0)
    # Calcule le nombre d'échantillons en fonction de la durée souhaitée
    data = rng.standard_normal((2, int(sfreq * duration)))
    # Assemble le RawArray à partir des données synthétiques
    return mne.io.RawArray(data, info)


# Crée une version allégée du filtre pour éviter les dépendances dataset
def _mock_filter(raw: mne.io.Raw, **_: object) -> mne.io.Raw:
    """Retourne une copie atténuée du Raw pour la comparaison."""

    # Copie le Raw pour éviter de muter l'entrée
    filtered = raw.copy()
    # Atténue légèrement l'amplitude pour différencier visuellement
    filtered._data = raw.get_data() * 0.5
    # Retourne le Raw filtré synthétique
    return filtered


# Vérifie que visualize_run produit bien un PNG et un JSON sans dataset
# WBS 3.3.1-3.3.3 : garantir la génération des plots de référence
@pytest.mark.filterwarnings("ignore:MNE has detected*")
def test_visualize_run_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Assure qu'un run se visualise avec des mocks de données."""

    # Construit un Raw factice pour simuler une acquisition Physionet
    raw = _build_dummy_raw(sfreq=64.0, duration=0.5)
    # Prépare des métadonnées minimales pour accompagner la figure
    metadata = {
        "sampling_rate": float(raw.info["sfreq"]),
        "channel_names": raw.ch_names,
        "montage": "mock",
        "path": "ignored",
        "subject": "S01",
        "run": "R01",
    }
    # Remplace le loader pour éviter toute dépendance au filesystem
    monkeypatch.setattr(viz, "load_recording", lambda _: (raw, dict(metadata)))
    # Remplace le filtre pour éviter l'appel MNE coûteux en test
    monkeypatch.setattr(viz, "apply_bandpass_filter", _mock_filter)
    # Prépare la configuration pour limiter le nombre d'arguments transmis
    config = viz.VisualizationConfig(
        channels=["C3"],
        output_dir=tmp_path,
        filter_method="fir",
        freq_band=(8.0, 40.0),
        pad_duration=0.1,
        title="Smoke Test",
    )
    # Exécute la visualisation avec une sélection de canal unique
    output_path = viz.visualize_run(
        data_root=Path("data/raw"),
        subject="S01",
        run="R01",
        config=config,
    )
    # Vérifie que le PNG a bien été écrit
    assert output_path.exists()
    # Vérifie que la configuration associée a été sérialisée
    assert output_path.with_suffix(".json").exists()
