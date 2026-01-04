"""Tests de fumée pour scripts/visualize_raw_filtered.py sans dataset."""

# Importe runpy pour exécuter le module en mode script
import runpy

# Importe sys pour manipuler argv lors des tests CLI
import sys

# Importe types pour créer un module factice tpv.preprocessing
import types

# Importe pathlib pour gérer les répertoires temporaires de sortie
from pathlib import Path

# Importe typing pour typer explicitement les modules patchés
from typing import Any, cast

# Importe mne pour construire un Raw synthétique simulant Physionet
import mne

# Importe numpy pour générer des données EEG déterministes
import numpy as np

# Importe pytest pour bénéficier de monkeypatch et tmp_path
import pytest

# Importe le module à tester pour cibler directement visualize_run
import scripts.visualize_raw_filtered as viz

import argparse

from scripts import visualize_raw_filtered

# Centralise la valeur de padding par défaut pour éviter une constante magique
PAD_DURATION_DEFAULT = 0.5


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


# Vérifie que le parseur propose bien les valeurs par défaut attendues
def test_build_parser_defaults() -> None:
    """Valide la configuration CLI minimale sans overrides."""

    # Construit le parseur dédié à la visualisation
    parser = viz.build_parser()
    # Parse uniquement sujet/run pour activer les valeurs par défaut
    args = parser.parse_args(["S001", "R02"])
    # Vérifie la racine dataset par défaut
    assert args.data_root == "data"
    # Vérifie le répertoire de sortie par défaut
    assert args.output_dir == "docs/viz"
    # Vérifie la méthode de filtrage par défaut
    assert args.filter_method == "fir"
    # Vérifie la bande de fréquence par défaut
    assert tuple(args.freq_band) == (8.0, 40.0)
    # Vérifie la durée de padding par défaut
    assert args.pad_duration == PAD_DURATION_DEFAULT


# Vérifie que load_recording signale l'absence du fichier demandé
def test_load_recording_missing_file(tmp_path: Path) -> None:
    """Assure une erreur explicite lorsque le run n'existe pas."""

    # Construit un chemin inexistant pour déclencher l'erreur
    missing_path = tmp_path / "absent" / "R01.edf"
    # Vérifie que FileNotFoundError est bien levée
    with pytest.raises(FileNotFoundError) as excinfo:
        viz.load_recording(missing_path)
    # Vérifie que le message indique clairement le chemin manquant
    assert str(excinfo.value) == f"Recording not found: {missing_path}"


# Vérifie que load_recording enrichit les métadonnées lorsque le fichier existe
def test_load_recording_hydrates_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirme que sujet et run sont ajoutés aux métadonnées."""

    # Construit un fichier EDF factice pour passer le check d'existence
    recording_path = tmp_path / "S99" / "R05.edf"
    # Crée les répertoires nécessaires
    recording_path.parent.mkdir(parents=True)
    # Écrit un contenu vide pour matérialiser le fichier
    recording_path.write_bytes(b"")
    # Prépare un Raw et des métadonnées simulées
    raw = _build_dummy_raw()
    metadata = {"foo": "bar"}
    # Remplace le loader Physionet par un mock contrôlé
    monkeypatch.setattr(viz, "load_physionet_raw", lambda _: (raw, dict(metadata)))
    # Charge le fichier avec le mock
    loaded_raw, loaded_metadata = viz.load_recording(recording_path)
    # Vérifie que le Raw retourné correspond au mock
    assert loaded_raw is raw
    # Vérifie l'enrichissement des métadonnées avec sujet et run
    assert loaded_metadata["subject"] == "S99"
    # Vérifie le nom de run inséré
    assert loaded_metadata["run"] == "R05"


# Vérifie que pick_channels retourne le Raw original lorsque aucun filtre n'est demandé
def test_pick_channels_pass_through() -> None:
    """Garantit l'absence de copie quand channels est None."""

    # Construit un Raw synthétique pour l'appel
    raw = _build_dummy_raw()
    # Appelle pick_channels sans sélection
    picked = viz.pick_channels(raw, None)
    # Vérifie que l'objet retourné est identique
    assert picked is raw


# Vérifie que pick_channels signale les canaux inconnus
def test_pick_channels_unknown_channel() -> None:
    """S'assure qu'une erreur claire est levée sur un canal manquant."""

    # Construit un Raw synthétique pour l'appel
    raw = _build_dummy_raw()
    # Vérifie que la demande d'un canal absent lève une ValueError
    with pytest.raises(ValueError):
        viz.pick_channels(raw, ["CZ"])


# Vérifie que main construit bien la configuration et invoque visualize_run
def test_main_invokes_visualize_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valide le parcours CLI nominal sans échec."""

    # Prépare une fonction sentinelle pour capturer les arguments
    called = {}

    # Définit un stub qui mémorise les paramètres reçus
    def _spy_visualize_run(**kwargs: object) -> None:
        called.update(kwargs)

    # Injecte le stub à la place de visualize_run
    monkeypatch.setattr(viz, "visualize_run", _spy_visualize_run)
    # Configure argv pour simuler une invocation CLI avec options personnalisées
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "S10",
            "R03",
            "--data-root",
            "/custom/data",
            "--output-dir",
            "/tmp/figures",
            "--channels",
            "C3",
            "C4",
            "--filter-method",
            "iir",
            "--freq-band",
            "5",
            "55",
            "--pad-duration",
            "1.5",
            "--title",
            "Demo",
        ],
    )
    # Exécute main pour déclencher la construction de config
    viz.main()
    # Vérifie que visualize_run a bien été invoqué
    assert called["subject"] == "S10"
    # Vérifie que le run transmis correspond à argv
    assert called["run"] == "R03"
    # Vérifie que la configuration passée est une VisualizationConfig
    assert isinstance(called["config"], viz.VisualizationConfig)
    # Vérifie que la racine data est transmise à visualize_run
    assert called["data_root"] == Path("/custom/data")
    # Vérifie que les canaux fournis sont conservés dans la configuration
    assert called["config"].channels == ["C3", "C4"]
    # Vérifie que la destination des figures correspond à l'argument CLI
    assert called["config"].output_dir == Path("/tmp/figures")
    # Vérifie que la méthode de filtrage respecte l'argument fourni
    assert called["config"].filter_method == "iir"
    # Vérifie que la bande de fréquences est convertie depuis la CLI
    assert called["config"].freq_band == (5.0, 55.0)
    # Vérifie que la durée de padding CLI est transmise en float
    custom_pad_duration = 1.5
    # Vérifie que la durée de padding CLI est transmise en float
    assert called["config"].pad_duration == custom_pad_duration
    # Vérifie que le titre de la figure reprend la valeur fournie
    assert called["config"].title == "Demo"


# Vérifie que main convertit une erreur en code de sortie explicite
def test_main_exits_on_visualize_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force visualize_run à échouer pour tester la conversion en SystemExit
    monkeypatch.setattr(
        viz, "visualize_run", lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # Configure argv pour simuler une invocation CLI minimale
    monkeypatch.setattr(sys, "argv", ["prog", "S10", "R03"])
    # Vérifie que main termine avec un code de sortie non nul
    with pytest.raises(SystemExit) as exit_info:
        viz.main()
    # Vérifie que le code de sortie reflète l'échec de visualisation
    assert exit_info.value.code == 1


# Vérifie que le guard __main__ s'exécute sans lancer la CLI réelle
def test_main_guard_covered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exécute le module en mode script en isolant les dépendances."""

    # Crée un module preprocessing factice pour intercepter les imports
    fake_preprocessing = types.ModuleType("tpv.preprocessing")
    # Construit un Raw synthétique à renvoyer par le loader mocké
    dummy_raw = _build_dummy_raw()
    # Bascule le module factice en Any pour déclarer les attributs attendus
    typed_preprocessing = cast(Any, fake_preprocessing)
    # Fournit un filtre no-op pour éviter tout calcul lourd
    typed_preprocessing.apply_bandpass_filter = lambda raw, **_: raw
    # Fournit un loader factice qui renvoie le Raw synthétique
    typed_preprocessing.load_physionet_raw = lambda path: (
        dummy_raw.copy(),
        {"path": str(path)},
    )
    # Injecte le module factice dans sys.modules pour l'exécution runpy
    monkeypatch.setitem(sys.modules, "tpv.preprocessing", fake_preprocessing)
    # Prépare un répertoire data/root minimal pour satisfaire load_recording
    data_root = tmp_path / "data"
    # Crée le dossier sujet pour respecter la hiérarchie attendue
    (data_root / "S20").mkdir(parents=True)
    # Crée un fichier EDF vide pour passer le check d'existence
    (data_root / "S20" / "R09.edf").write_bytes(b"")
    # Configure argv pour pointer vers le dataset factice et un output temporaire
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "S20",
            "R09",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "viz"),
        ],
    )
    # Exécute le module comme un script pour couvrir le guard
    runpy.run_module(
        "scripts.visualize_raw_filtered", run_name="__main__", alter_sys=True
    )


# Vérifie que main relaie correctement les erreurs via SystemExit
def test_main_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assure la conversion des erreurs runtime en code de sortie non nul."""

    # Force visualize_run à lever une exception contrôlée
    monkeypatch.setattr(
        viz, "visualize_run", lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # Configure argv pour fournir les arguments requis
    monkeypatch.setattr(sys, "argv", ["prog", "S11", "R07"])
    # Vérifie que main convertit l'erreur en SystemExit
    with pytest.raises(SystemExit) as excinfo:
        viz.main()
    # Vérifie que le code de sortie est non nul
    assert excinfo.value.code == 1


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
        "subject": "S001",
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
        data_root=Path("data"),
        subject="S001",
        run="R01",
        config=config,
    )
    # Vérifie que le PNG a bien été écrit
    assert output_path.exists()
    # Vérifie que la configuration associée a été sérialisée
    assert output_path.with_suffix(".json").exists()


def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    # Parcourt toutes les actions déclarées dans argparse
    for action in parser._actions:  # pylint: disable=protected-access
        # Sélectionne l'action correspondant au dest demandé
        if action.dest == dest:
            # Retourne l'action trouvée pour inspection
            return action
    # Signale clairement un dest manquant
    raise AssertionError(f"Action argparse introuvable: dest={dest!r}")


def test_build_parser_description_and_help_texts_are_stable() -> None:
    # Construit le parser depuis la fonction de prod
    parser = visualize_raw_filtered.build_parser()

    # Verrouille la description pour tuer description=None / variantes
    assert (
        parser.description
        == "Charge un run Physionet, applique le filtre 8-40 Hz, "
        "et enregistre un plot brut vs filtré."
    )

    # Récupère les actions positionnelles attendues
    subject_action = _get_action(parser, "subject")
    run_action = _get_action(parser, "run")

    # Verrouille l'aide exacte de l'argument subject
    assert subject_action.help == "Identifiant du sujet ex: S001"
    # Verrouille l'aide exacte de l'argument run
    assert run_action.help == "Identifiant du run ex: R01"

    # Récupère les options dont l'aide est mutée par mutmut
    data_root_action = _get_action(parser, "data_root")
    output_dir_action = _get_action(parser, "output_dir")
    channels_action = _get_action(parser, "channels")
    filter_method_action = _get_action(parser, "filter_method")

    # Verrouille default + help pour tuer help=None / parenthèses supprimées
    assert data_root_action.default == "data"
    assert data_root_action.help == "Racine locale des données Physionet"

    # Verrouille default + help pour tuer help=None / variantes de texte
    assert output_dir_action.default == "docs/viz"
    assert output_dir_action.help == "Répertoire de sauvegarde PNG"

    # Verrouille nargs + help pour tuer help=None / variantes
    assert channels_action.nargs == "*"
    assert channels_action.help == "Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)"

    # Verrouille choices + default + help pour tuer choices=None / suppression
    assert filter_method_action.choices == ["fir", "iir"]
    assert filter_method_action.default == "fir"
    assert filter_method_action.help == "Famille de filtre à appliquer"
