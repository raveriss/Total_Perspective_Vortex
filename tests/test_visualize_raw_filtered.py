"""Tests de fumée pour scripts/visualize_raw_filtered.py sans dataset."""

# Importe runpy pour exécuter le module en mode script
import argparse

# Importe json pour verrouiller la sérialisation du sidecar
import json
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


# Vérifie que build_recording_path respecte la convention data/<subject>/<run>.edf
def test_build_recording_path_composes_expected_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verrouille la construction du chemin EDF pour tuer recording_path=None."""

    # Force un cwd déterministe pour tester les chemins relatifs
    monkeypatch.chdir(tmp_path)
    # Utilise une racine relative pour valider expanduser/resolve
    data_root = Path("data")
    # Construit le chemin attendu via la racine résolue
    expected_root = (tmp_path / "data").resolve()
    # Construit le chemin via la fonction de prod
    recording_path = viz.build_recording_path(data_root, "S01", "R02")
    # Verrouille le type retourné pour éviter None
    assert isinstance(recording_path, Path)
    # Verrouille l'extension EDF pour respecter la convention README
    assert recording_path.suffix == ".edf"
    # Verrouille le chemin complet pour détecter toute dérive
    assert recording_path == expected_root / "S01" / "S01R02.edf"


# Vérifie que build_recording_path sélectionne automatiquement la variante préfixée
def test_build_recording_path_prefers_subject_prefixed_file(tmp_path: Path) -> None:
    """Garantit la prise en charge des fichiers Physionet SxxxRyy.edf."""

    # Prépare une arborescence mimant data/S01/S01R02.edf
    data_root = tmp_path / "data"
    prefixed = data_root / "S01" / "S01R02.edf"
    prefixed.parent.mkdir(parents=True)
    prefixed.write_bytes(b"")
    # Doit retourner le chemin existant même si run n'est pas préfixé
    recording_path = viz.build_recording_path(data_root, "S01", "R02")
    assert recording_path == prefixed


# Vérifie que load_recording signale l'absence du fichier demandé
def test_load_recording_missing_file(tmp_path: Path) -> None:
    """Assure une erreur explicite lorsque le run n'existe pas."""

    # Construit un chemin inexistant pour déclencher l'erreur
    missing_path = tmp_path / "absent" / "R01.edf"
    # Vérifie que FileNotFoundError est bien levée
    with pytest.raises(FileNotFoundError) as excinfo:
        viz.load_recording(missing_path)
    # Vérifie que le message indique clairement le chemin manquant
    message_lines = str(excinfo.value).splitlines()
    assert message_lines[0] == f"Recording not found: {missing_path}"
    assert any("Structure attendue" in line for line in message_lines)
    assert any("Physionet EEG Motor Movement/Imagery" in line for line in message_lines)
    assert any("Répertoire sujet absent" in line for line in message_lines)


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
    # Capture l'argument passé au loader pour détecter load_physionet_raw(None)
    seen: dict[str, object] = {}

    # Définit un spy qui vérifie le chemin fourni au loader
    def _spy_load_physionet_raw(
        path: Path, reference: str | None = "average"
    ) -> tuple[mne.io.Raw, dict]:
        # Mémorise le chemin reçu pour assertion post-call
        seen["path"] = path
        # Mémorise la référence pour vérifier le passage des arguments
        seen["reference"] = reference
        # Retourne les objets mockés pour ne pas dépendre du dataset
        return raw, dict(metadata)

    # Remplace le loader Physionet par un spy strict
    monkeypatch.setattr(viz, "load_physionet_raw", _spy_load_physionet_raw)
    # Charge le fichier avec le mock
    loaded_raw, loaded_metadata = viz.load_recording(recording_path)
    # Vérifie que le Raw retourné correspond au mock
    assert loaded_raw is raw
    # Verrouille le chemin transmis au loader pour tuer load_physionet_raw(None)
    assert seen["path"] == recording_path
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
def test_pick_channels_unknown_channel_raises_explicit_message() -> None:
    """Tue missing=None et verrouille le message Unknown channels exact."""

    # Définit un Raw minimal pour isoler l'erreur de la logique MNE
    class _StubRaw:
        # Expose les noms de canaux attendus par pick_channels
        def __init__(self, ch_names: list[str]) -> None:
            # Stocke les canaux disponibles pour le check missing
            self.ch_names = ch_names
            # Marque si copy a été appelé pour détecter un mauvais chemin
            self.copy_called = False

        # Fournit une copie neutre pour détecter les appels inattendus
        def copy(self) -> "_StubRaw":
            # Marque l'appel copy pour diagnostics
            self.copy_called = True
            # Retourne une nouvelle instance pour simuler une copie
            return _StubRaw(list(self.ch_names))

        # Implémente pick mais ne valide rien pour isoler la logique manquante
        def pick(self, _: object) -> "_StubRaw":
            # Retourne self sans lever pour éviter les erreurs MNE
            return self

    # Construit un Raw stub avec deux canaux valides
    raw = _StubRaw(["C3", "C4"])
    # Vérifie que la demande de canaux absents lève une ValueError explicite
    with pytest.raises(ValueError) as excinfo:
        viz.pick_channels(raw, ["CZ", "PZ"])
    # Verrouille le message pour tuer ValueError(None) et join altéré
    assert str(excinfo.value) == "Unknown channels: CZ, PZ"
    # Verrouille l'absence d'appel copy en cas d'erreur précoce
    assert raw.copy_called is False


# Vérifie que pick_channels copie puis restreint aux canaux demandés
def test_pick_channels_returns_copied_and_picked_raw() -> None:
    """Tue picked=raw.copy().pick(None) et verrouille l'argument pick."""

    # Définit un Raw minimal pour tracer les appels copy/pick
    class _StubRaw:
        # Expose les noms de canaux attendus par pick_channels
        def __init__(self, ch_names: list[str]) -> None:
            # Stocke la liste des canaux actuels
            self.ch_names = ch_names
            # Mémorise l'argument reçu par pick pour assertion
            self.picked_with: object | None = None

        # Retourne une copie indépendante pour vérifier qu'une copie est faite
        def copy(self) -> "_StubRaw":
            # Retourne une copie avec les mêmes canaux
            return _StubRaw(list(self.ch_names))

        # Simule une sélection stricte des canaux demandés
        def pick(self, channels: list[str]) -> "_StubRaw":
            # Refuse None pour tuer les mutants pick(None)
            assert channels is not None
            # Mémorise l'argument reçu pour verrouiller l'appel
            self.picked_with = channels
            # Applique la sélection sur la copie uniquement
            self.ch_names = list(channels)
            # Retourne self pour chaînage
            return self

    # Construit un Raw stub avec deux canaux valides
    raw = _StubRaw(["C3", "C4"])
    # Applique une sélection mono-canal
    picked = viz.pick_channels(raw, ["C3"])
    # Verrouille que l'objet retourné n'est pas l'original
    assert picked is not raw
    # Verrouille le fait que l'original n'a pas été modifié
    assert raw.ch_names == ["C3", "C4"]
    # Verrouille la réduction des canaux sur la copie
    assert picked.ch_names == ["C3"]
    # Verrouille l'argument transmis à pick
    assert picked.picked_with == ["C3"]


# Vérifie que filter_recording forwarde les paramètres attendus
# vers apply_bandpass_filter sans altération
def test_filter_recording_forwards_expected_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tue les mutants qui suppriment/altèrent method, freq_band ou pad_duration."""

    # Construit un Raw synthétique pour alimenter la fonction
    raw = _build_dummy_raw()
    # Prépare un objet sentinelle pour valider le retour
    sentinel = raw.copy()
    # Capture les paramètres reçus par le spy
    seen: dict[str, object] = {}

    # Définit un spy strict sur apply_bandpass_filter
    def _spy_apply_bandpass_filter(raw_arg: mne.io.Raw, **kwargs: object) -> mne.io.Raw:
        # Mémorise l'argument raw pour l'asserter ensuite
        seen["raw"] = raw_arg
        # Mémorise tous les kwargs pour vérifier leur présence
        seen.update(kwargs)
        # Retourne la sentinelle pour valider le retour de filter_recording
        return sentinel

    # Remplace le filtre réel par le spy pour éviter MNE en test
    monkeypatch.setattr(viz, "apply_bandpass_filter", _spy_apply_bandpass_filter)

    # Appelle la fonction avec des valeurs non-default pour tuer les None
    filtered = viz.filter_recording(
        raw,
        method="iir",
        freq_band=(1.0, 2.0),
        pad_duration=0.25,
    )

    # Verrouille le retour pour détecter tout bypass
    assert filtered is sentinel
    # Verrouille le raw transmis au filtre
    assert seen["raw"] is raw
    # Verrouille les kwargs exacts pour tuer method=None / suppression
    assert seen["method"] == "iir"
    assert seen["freq_band"] == (1.0, 2.0)
    assert seen["pad_duration"] == 0.25


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
    assert called["subject"] == "S010"
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


# Vérifie que plot_raw_vs_filtered crée les dossiers et garde le layout attendu
def test_plot_raw_vs_filtered_creates_dirs_and_layout_is_stable(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tue les mutants qui altèrent layout, labels et appels plot()."""

    # Construit des Raw synthétiques pour éviter le dataset
    raw = _build_dummy_raw(sfreq=64.0, duration=0.25)
    # Produit un Raw filtré cheap pour l'appel
    filtered = _mock_filter(raw)
    # Prépare des métadonnées minimales pour le titre fallback
    metadata = {"subject": "S01", "run": "R02"}
    # Fixe la bande de fréquence pour alimenter le titre automatique
    freq_band = (8.0, 40.0)
    # Prépare une configuration minimale pour le plot
    config = viz.VisualizationConfig(
        channels=None,
        output_dir=tmp_path,
        filter_method="fir",
        freq_band=freq_band,
        pad_duration=0.5,
        title=None,
    )
    # Prépare une configuration minimale pour le plot
    config = viz.VisualizationConfig(
        channels=None,
        output_dir=tmp_path,
        filter_method="fir",
        freq_band=freq_band,
        pad_duration=0.5,
        title=None,
    )
    # Cible un chemin dont le parent n'existe pas pour tester parents=True
    output_path = tmp_path / "nested" / "deep" / "fig.png"
    # Capture les arguments de subplots et les appels faits aux axes
    seen: dict[str, object] = {}

    # Définit un axe enregistreur pour verrouiller les appels plot/labels
    class _Axis:
        # Initialise les buffers d'appels pour une assertion post-exécution
        def __init__(self, name: str) -> None:
            # Nomme l'axe pour faciliter le debug en cas d'échec
            self.name = name
            # Capture tous les appels plot pour tuer args/kwargs mutés
            self.plot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture les appels legend pour tuer loc=None / mauvais subplot
            self.legend_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture les appels set_ylabel pour tuer None / texte altéré
            self.ylabel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture les appels set_title pour tuer None / texte altéré
            self.title_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture les appels set_xlabel pour tuer None / texte altéré
            self.xlabel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture les appels fill_between pour verrouiller l'enveloppe
            self.fill_between_calls: list[
                tuple[tuple[object, ...], dict[str, object]]
            ] = []
            # Capture les appels grid pour verrouiller la grille légère
            self.grid_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        # Enregistre le tracé pour valider times, data et label
        def plot(self, *args: object, **kwargs: object) -> list[object]:
            # Stocke args/kwargs pour assertions strictes après exécution
            self.plot_calls.append((args, dict(kwargs)))
            return [object()]

        # Enregistre la légende pour valider loc et l'axe utilisé
        def legend(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier le loc exact
            self.legend_calls.append((args, dict(kwargs)))

        # Enregistre le label Y pour valider le texte et l'axe ciblé
        def set_ylabel(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier le texte exact
            self.ylabel_calls.append((args, dict(kwargs)))

        # Enregistre le titre d'axe pour valider le texte et l'axe ciblé
        def set_title(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier le texte exact
            self.title_calls.append((args, dict(kwargs)))

        # Enregistre le label X pour valider le texte et l'axe ciblé
        def set_xlabel(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier le texte exact
            self.xlabel_calls.append((args, dict(kwargs)))

        # Enregistre l'enveloppe pour valider mean ± std
        def fill_between(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier la bande d'écart-type
            self.fill_between_calls.append((args, dict(kwargs)))

        # Enregistre les grilles pour valider l'intensité visuelle
        def grid(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour valider les paramètres de grille
            self.grid_calls.append((args, dict(kwargs)))

    # Définit une figure minimaliste pour capturer suptitle/savefig
    class _Fig:
        # Initialise le conteneur d'arguments suptitle
        def __init__(self) -> None:
            # Capture le dernier titre global appliqué
            self.suptitle_value: object | None = None
            # Marque l'appel tight_layout pour tuer le bypass
            self.tight_layout_called = False
            self.tight_layout_args: tuple[object, ...] = ()
            self.tight_layout_kwargs: dict[str, object] = {}
            self.legend_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            # Capture le chemin de sauvegarde pour tuer savefig mal routé
            self.savefig_path: object | None = None
            # Capture les appels text pour verrouiller le sous-titre
            self.text_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        # Capture le titre global pour vérification
        def suptitle(self, value: object) -> None:
            # Stocke le titre reçu pour assertion
            self.suptitle_value = value

        # Capture tight_layout pour éviter qu'il soit supprimé
        def tight_layout(self, *args: object, **kwargs: object) -> None:
            # Marque l'appel afin de détecter un bypass
            self.tight_layout_called = True
            self.tight_layout_args = args
            self.tight_layout_kwargs = dict(kwargs)

        def legend(self, *args: object, **kwargs: object) -> None:
            self.legend_calls.append((args, dict(kwargs)))

        # Capture les textes libres pour vérifier le sous-titre
        def text(self, *args: object, **kwargs: object) -> None:
            # Stocke args/kwargs pour vérifier le contenu ajouté
            self.text_calls.append((args, dict(kwargs)))

        # Simule l'écriture de la figure pour valider le chemin
        def savefig(self, path: Path) -> None:
            # Capture le chemin transmis pour assertion
            self.savefig_path = path
            # Crée un fichier pour permettre les assertions d'existence
            Path(path).write_bytes(b"png")

    # Définit un spy sur plt.subplots pour verrouiller args/kwargs
    def _spy_subplots(*args: object, **kwargs: object) -> tuple[_Fig, list[_Axis]]:
        # Capture args/kwargs pour tuer les mutants sharex/figsize/nrows
        seen["args"] = args
        seen["kwargs"] = kwargs
        # Crée la figure capturée pour assertions post-call
        fig = _Fig()
        # Stocke la figure utilisée par le code
        seen["fig"] = fig
        # Crée deux axes distincts pour verrouiller l'indexation axes[0]/axes[1]
        raw_axis = _Axis("raw")
        filtered_axis = _Axis("filtered")
        # Expose les axes pour assertions post-exécution
        seen["raw_axis"] = raw_axis
        seen["filtered_axis"] = filtered_axis
        # Retourne deux axes comme attendu par le code
        return fig, [raw_axis, filtered_axis]

    # Remplace subplots et close pour éviter la dépendance matplotlib complète
    monkeypatch.setattr(viz.plt, "subplots", _spy_subplots)
    monkeypatch.setattr(viz.plt, "close", lambda fig: seen.update({"closed": fig}))

    # Capture l'implémentation réelle AVANT patch pour éviter la récursion
    original_write_text = Path.write_text

    # Intercepte l'écriture du sidecar JSON pour verrouiller encoding + payload
    def _spy_write_text(
        self: Path, data: str, encoding: str | None = None, errors: str | None = None
    ) -> int:
        # Ne capture que l'écriture du sidecar de CE test
        if self == output_path.with_suffix(".json"):
            # Capture le payload exact pour tuer json.dumps muté
            seen["sidecar_data"] = data
            # Capture args/kwargs pour tuer encoding=None / encoding supprimé / UTF-8
            seen["sidecar_args"] = ()
            seen["sidecar_kwargs"] = {"encoding": encoding, "errors": errors}
        # Délègue à l'implémentation réelle pour écrire sur disque
        return original_write_text(self, data, encoding=encoding, errors=errors)

    # Patch Path.write_text une seule fois (sinon récursion)
    monkeypatch.setattr(Path, "write_text", _spy_write_text)

    # Exécute la génération de la figure avec title=None pour forcer le fallback
    returned = viz.plot_raw_vs_filtered(
        raw=raw,
        filtered=filtered,
        output_path=output_path,
        config=config,
        metadata=metadata,
    )

    # Verrouille le chemin retourné
    assert returned == output_path
    # Verrouille la création des répertoires parents (parents=True requis)
    assert output_path.parent.exists()
    # Verrouille la création du fichier PNG via savefig spy
    assert output_path.exists()
    # Verrouille la création du sidecar JSON
    assert output_path.with_suffix(".json").exists()

    # Verrouille subplots(2, 1, sharex=True, sharey=False, figsize=...)
    assert seen["args"] == (2, 1)
    assert cast(dict[str, object], seen["kwargs"])["sharex"] is True
    assert cast(dict[str, object], seen["kwargs"])["sharey"] is False
    assert cast(dict[str, object], seen["kwargs"])["figsize"] == (
        viz.FIGURE_WIDTH,
        viz.FIGURE_ROW_HEIGHT * 2,
    )

    # Verrouille le titre fallback (title=None) basé sur la bande de fréquence
    fig_used = cast(_Fig, seen["fig"])
    assert fig_used.suptitle_value == "EEG – Comparaison brut vs filtré (8-40 Hz)"

    # Verrouille l'ajout du sous-titre sujet/run
    assert len(fig_used.text_calls) == 1
    text_args, text_kwargs = fig_used.text_calls[0]
    assert text_args == (
        0.5,
        0.94,
        "Sujet S01 • Run R02 • 2 canaux",
    )
    assert text_kwargs["ha"] == "center"
    assert text_kwargs["fontsize"] == "small"

    # Verrouille l'application de tight_layout pour éviter les figures tassées
    assert fig_used.tight_layout_called is True
    assert fig_used.tight_layout_kwargs["rect"] == (0.0, 0.0, 1.0, 0.92)

    # Verrouille l'absence de légende globale pour le style vertical
    assert fig_used.legend_calls == []
    # Verrouille le chemin passé à savefig pour tuer les redirections
    assert fig_used.savefig_path == output_path
    # Verrouille la fermeture de la figure pour tuer le bypass close
    assert seen["closed"] is fig_used

    # Récupère les buffers capturés pour valider les appels matplotlib
    raw_axis = cast(_Axis, seen["raw_axis"])
    filtered_axis = cast(_Axis, seen["filtered_axis"])

    # Capture times/data attendus pour comparer les arguments plot()
    expected_times = raw.times
    expected_raw_data = raw.get_data()
    expected_filtered_data = filtered.get_data()

    # Calcule la moyenne et l'écart-type attendus pour C3/C4
    expected_raw_mean = expected_raw_data.mean(axis=0)
    expected_raw_std = expected_raw_data.std(axis=0)
    expected_filtered_mean = expected_filtered_data.mean(axis=0)
    expected_filtered_std = expected_filtered_data.std(axis=0)
    # Capture la couleur attendue pour la région centrale
    expected_color = viz.REGION_COLORS["Central"]

    # Verrouille les appels plot bruts: deux canaux + moyenne
    assert len(raw_axis.plot_calls) == 3
    for idx in range(2):
        plot_args, plot_kwargs = raw_axis.plot_calls[idx]
        assert len(plot_args) == 2
        assert np.array_equal(np.asarray(plot_args[0]), np.asarray(expected_times))
        assert np.array_equal(
            np.asarray(plot_args[1]),
            np.asarray(expected_raw_data[idx]),
        )
        assert plot_kwargs == {
            "color": expected_color,
            "alpha": 0.2,
            "linewidth": 0.6,
        }

    # Verrouille le tracé de moyenne brute
    mean_args, mean_kwargs = raw_axis.plot_calls[2]
    assert np.array_equal(np.asarray(mean_args[0]), np.asarray(expected_times))
    assert np.array_equal(np.asarray(mean_args[1]), np.asarray(expected_raw_mean))
    assert mean_kwargs == {"color": expected_color, "linewidth": 1.6}

    # Verrouille l'enveloppe brute mean ± std
    assert len(raw_axis.fill_between_calls) == 1
    fill_args, fill_kwargs = raw_axis.fill_between_calls[0]
    assert np.array_equal(np.asarray(fill_args[0]), np.asarray(expected_times))
    assert np.array_equal(
        np.asarray(fill_args[1]),
        np.asarray(expected_raw_mean - expected_raw_std),
    )
    assert np.array_equal(
        np.asarray(fill_args[2]),
        np.asarray(expected_raw_mean + expected_raw_std),
    )
    assert fill_kwargs == {"color": expected_color, "alpha": 0.2}

    # Verrouille les appels plot filtrés: deux canaux + moyenne
    assert len(filtered_axis.plot_calls) == 3
    for idx in range(2):
        plot_args, plot_kwargs = filtered_axis.plot_calls[idx]
        assert len(plot_args) == 2
        assert np.array_equal(np.asarray(plot_args[0]), np.asarray(expected_times))
        assert np.array_equal(
            np.asarray(plot_args[1]),
            np.asarray(expected_filtered_data[idx]),
        )
        assert plot_kwargs == {
            "color": expected_color,
            "alpha": 0.2,
            "linewidth": 0.6,
        }

    # Verrouille le tracé de moyenne filtrée
    mean_args, mean_kwargs = filtered_axis.plot_calls[2]
    assert np.array_equal(np.asarray(mean_args[0]), np.asarray(expected_times))
    assert np.array_equal(
        np.asarray(mean_args[1]),
        np.asarray(expected_filtered_mean),
    )
    assert mean_kwargs == {"color": expected_color, "linewidth": 1.6}

    # Verrouille l'enveloppe filtrée mean ± std
    assert len(filtered_axis.fill_between_calls) == 1
    fill_args, fill_kwargs = filtered_axis.fill_between_calls[0]
    assert np.array_equal(np.asarray(fill_args[0]), np.asarray(expected_times))
    assert np.array_equal(
        np.asarray(fill_args[1]),
        np.asarray(expected_filtered_mean - expected_filtered_std),
    )
    assert np.array_equal(
        np.asarray(fill_args[2]),
        np.asarray(expected_filtered_mean + expected_filtered_std),
    )
    assert fill_kwargs == {"color": expected_color, "alpha": 0.2}

    # Verrouille l'absence de légende sur les axes
    assert raw_axis.legend_calls == []
    assert filtered_axis.legend_calls == []

    # Verrouille le label Y brut pour tuer None / texte altéré
    assert len(raw_axis.ylabel_calls) == 1
    raw_ylabel_args, _ = raw_axis.ylabel_calls[0]
    assert raw_ylabel_args == ("Amplitude (a.u.)",)

    # Verrouille le label Y filtré pour tuer None / texte altéré
    assert len(filtered_axis.ylabel_calls) == 1
    filt_ylabel_args, _ = filtered_axis.ylabel_calls[0]
    assert filt_ylabel_args == ("Amplitude filtrée (a.u.)",)

    # Verrouille le titre brut pour tuer None / texte altéré
    assert len(raw_axis.title_calls) == 1
    raw_title_args, _ = raw_axis.title_calls[0]
    assert raw_title_args == ("Signal brut — Central",)

    # Verrouille le titre filtré pour tuer None / texte altéré
    assert len(filtered_axis.title_calls) == 1
    filt_title_args, _ = filtered_axis.title_calls[0]
    assert filt_title_args == ("Signal filtré 8-40 Hz — Central",)

    # Verrouille le label X sur la dernière ligne
    assert raw_axis.xlabel_calls == []
    assert len(filtered_axis.xlabel_calls) == 1
    filt_xlabel_args, _ = filtered_axis.xlabel_calls[0]
    assert filt_xlabel_args == ("Temps (s)",)

    # Verrouille la grille majeure + mineure sur les deux axes
    assert len(raw_axis.grid_calls) == 2
    assert raw_axis.grid_calls[0][1] == {
        "which": "major",
        "axis": "both",
        "linewidth": 0.4,
        "alpha": 0.8,
    }
    assert raw_axis.grid_calls[1][1] == {
        "which": "minor",
        "axis": "both",
        "linewidth": 0.2,
        "alpha": 0.5,
    }
    assert len(filtered_axis.grid_calls) == 2
    assert filtered_axis.grid_calls[0][1] == {
        "which": "major",
        "axis": "both",
        "linewidth": 0.4,
        "alpha": 0.8,
    }
    assert filtered_axis.grid_calls[1][1] == {
        "which": "minor",
        "axis": "both",
        "linewidth": 0.2,
        "alpha": 0.5,
    }

    # Verrouille l'usage explicite de encoding="utf-8" (tue mutmut_82/84/96)
    sidecar_kwargs = cast(dict[str, object], seen["sidecar_kwargs"])
    assert cast(tuple[object, ...], seen["sidecar_args"]) == ()
    assert "encoding" in sidecar_kwargs
    assert sidecar_kwargs["encoding"] == "utf-8"

    # Verrouille le JSON exact (tue mutmut_88/89/91/92/93/94)
    expected_sidecar = json.dumps({"metadata": metadata}, indent=2)
    assert cast(str, seen["sidecar_data"]) == expected_sidecar


def test_plot_raw_vs_filtered_allows_existing_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tue les mutants qui omettent exist_ok=True sur mkdir()."""

    # Construit des Raw synthétiques pour éviter le dataset
    raw = _build_dummy_raw(sfreq=64.0, duration=0.25)
    # Produit un Raw filtré cheap pour l'appel
    filtered = _mock_filter(raw)
    # Prépare des métadonnées minimales pour le fallback de titre
    metadata = {"subject": "S01", "run": "R02"}
    # Fixe la bande de fréquence pour alimenter le titre automatique
    freq_band = (8.0, 40.0)
    # Prépare une configuration minimale pour le plot
    config = viz.VisualizationConfig(
        channels=None,
        output_dir=tmp_path,
        filter_method="fir",
        freq_band=freq_band,
        pad_duration=0.5,
        title=None,
    )

    # Cible un chemin dont le parent existera déjà avant l'appel
    output_path = tmp_path / "already_there" / "fig.png"
    # Crée explicitement le parent pour forcer le chemin "directory exists"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Verrouille que le dossier existe avant plot (sinon le test ne tue rien)
    assert output_path.parent.exists()

    # Définit un axe minimaliste pour absorber les appels matplotlib
    class _Axis:
        # Ignore les tracés pour isoler mkdir() et savefig()
        def plot(self, *_: object, **__: object) -> list[object]:
            return [object()]

        # Ignore fill_between pour isoler mkdir() et savefig()
        def fill_between(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore la grille pour isoler mkdir() et savefig()
        def grid(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore la légende pour isoler mkdir() et savefig()
        def legend(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore set_ylabel pour isoler mkdir() et savefig()
        def set_ylabel(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore set_title pour isoler mkdir() et savefig()
        def set_title(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore set_xlabel pour isoler mkdir() et savefig()
        def set_xlabel(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

    # Définit une figure minimaliste pour éviter matplotlib réel
    class _Fig:
        def legend(self, *_: object, **__: object) -> None:
            return None

        # Ignore suptitle pour isoler mkdir() et savefig()
        def suptitle(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore text pour isoler mkdir() et savefig()
        def text(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Ignore tight_layout pour isoler mkdir() et savefig()
        def tight_layout(self, *_: object, **__: object) -> None:
            # No-op : test centré sur la création de dossier
            return None

        # Simule l'écriture du PNG pour conserver la sémantique de sortie
        def savefig(self, path: Path) -> None:
            # Écrit un PNG minimal pour valider l'existence du fichier
            Path(path).write_bytes(b"png")

    # Remplace subplots pour renvoyer nos stubs
    def _spy_subplots(*_: object, **__: object) -> tuple[_Fig, list[_Axis]]:
        # Retourne une figure stub + deux axes stubs comme attendu
        return _Fig(), [_Axis(), _Axis()]

    # Patch subplots et close pour éviter l'UI matplotlib
    monkeypatch.setattr(viz.plt, "subplots", _spy_subplots)
    # Patch close pour éviter les dépendances internes matplotlib
    monkeypatch.setattr(viz.plt, "close", lambda *_: None)

    # Exécute plot_raw_vs_filtered avec un parent déjà existant
    returned = viz.plot_raw_vs_filtered(
        raw=raw,
        filtered=filtered,
        output_path=output_path,
        config=config,
        metadata=metadata,
    )

    # Verrouille le chemin retourné
    assert returned == output_path
    # Verrouille que le PNG est bien écrit malgré le parent déjà présent
    assert output_path.exists()
    # Verrouille que le sidecar JSON est bien écrit malgré le parent déjà présent
    assert output_path.with_suffix(".json").exists()


# Vérifie que main convertit une erreur en code de sortie explicite
def test_main_exits_on_visualize_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # Force visualize_run à échouer pour tester la conversion en SystemExit
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
    # Verrouille l'impression de l'erreur pour tuer print(None)
    captured = capsys.readouterr()
    # Verrouille le message exact produit par print(error)
    assert captured.out == "boom\n"


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
    typed_preprocessing.load_physionet_raw = lambda path, reference="average": (
        dummy_raw.copy(),
        {"path": str(path), "reference": reference},
    )
    # Injecte le module factice dans sys.modules pour l'exécution runpy
    monkeypatch.setitem(sys.modules, "tpv.preprocessing", fake_preprocessing)
    # Prépare un répertoire data/root minimal pour satisfaire load_recording
    data_root = tmp_path / "data"
    # Crée le dossier sujet pour respecter la hiérarchie attendue
    (data_root / "S020").mkdir(parents=True)
    # Crée un fichier EDF vide pour passer le check d'existence
    (data_root / "S020" / "R09.edf").write_bytes(b"")
    # Configure argv pour pointer vers le dataset factice et un output temporaire
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "20",
            "9",
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
def test_visualize_run_smoke(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verrouille les paramètres forwardés par visualize_run()."""

    # Construit un Raw factice pour simuler une acquisition Physionet
    raw = _build_dummy_raw(sfreq=64.0, duration=0.5)
    # Construit des métadonnées pour vérifier la propagation jusqu'au plot
    metadata = {
        "sampling_rate": float(raw.info["sfreq"]),
        "channel_names": raw.ch_names,
        "montage": "mock",
        "path": "ignored",
        "subject": "S001",
        "run": "R01",
    }
    # Capture les appels internes pour tuer les mutants None/paramètres perdus
    seen: dict[str, object] = {}
    # Prépare un chemin EDF factice pour verrouiller build_recording_path
    recording_path = tmp_path / "S001" / "S001R01.edf"
    # Verrouille le chemin de sortie construit depuis output_dir + sujet/run
    expected_output_path = tmp_path / "raw_vs_filtered_S001_R01.png"

    # Prépare la configuration pour limiter le nombre d'arguments transmis
    config = viz.VisualizationConfig(
        channels=["C3"],
        output_dir=tmp_path,
        filter_method="fir",
        freq_band=(8.0, 40.0),
        pad_duration=0.1,
        title="Smoke Test",
    )

    # Spy build_recording_path pour tuer recording_path=None / run=None
    def _spy_build_recording_path(data_root: Path, subject: str, run: str) -> Path:
        # Trace l'appel pour détecter un bypass complet
        seen["build_recording_path_called"] = True
        # Verrouille le data_root transmis par visualize_run
        assert data_root == Path("data")
        # Verrouille le subject forwardé
        assert subject == "S001"
        # Verrouille le run forwardé
        assert run == "R01"
        # Retourne un chemin factice pour l'étape suivante du pipeline
        return recording_path

    # Spy load_recording pour tuer load_recording(None) / mauvais chemin
    def _spy_load_recording(path: Path) -> tuple[mne.io.Raw, dict]:
        # Trace l'argument reçu pour validation
        seen["load_recording_path"] = path
        # Verrouille le chemin attendu produit par build_recording_path
        assert path == recording_path
        # Retourne un Raw et une copie des métadonnées pour la suite
        return raw, dict(metadata)

    # Spy pick_channels pour tuer pick_channels(raw, None)
    def _spy_pick_channels(
        input_raw: mne.io.Raw, channels: list[str] | None
    ) -> mne.io.Raw:
        # Trace l'argument channels pour validation stricte
        seen["pick_channels_channels"] = channels
        # Verrouille le Raw issu du loader
        assert input_raw is raw
        # Verrouille la liste de canaux demandée par config
        assert channels == ["C3"]
        # Retourne le Raw inchangé pour garder un pipeline simple en test
        return input_raw

    # Spy filter_recording pour tuer method/freq_band/pad_duration=None
    def _spy_filter_recording(
        input_raw: mne.io.Raw,
        method: str,
        freq_band: tuple[float, float],
        pad_duration: float,
    ) -> mne.io.Raw:
        # Trace les paramètres reçus pour assertions
        seen["filter_args"] = (method, freq_band, pad_duration)
        # Verrouille le Raw forwardé depuis pick_channels
        assert input_raw is raw
        # Verrouille la méthode de filtrage
        assert method == "fir"
        # Verrouille la bande de fréquences
        assert freq_band == (8.0, 40.0)
        # Verrouille la durée de padding
        assert pad_duration == 0.1
        # Retourne un Raw filtré factice pour la phase de plot
        filtered_obj = _mock_filter(input_raw)
        seen["filtered_obj"] = filtered_obj
        return filtered_obj

    # Spy plot_raw_vs_filtered pour tuer title=None / metadata=None
    def _spy_plot_raw_vs_filtered(
        input_raw: mne.io.Raw,
        input_filtered: mne.io.Raw,
        output_path: Path,
        config: viz.VisualizationConfig,
        plot_metadata: dict,
    ) -> Path:
        # Verrouille raw pour tuer plot_raw_vs_filtered(None, ...)
        assert input_raw is raw
        # Verrouille filtered pour tuer plot_raw_vs_filtered(..., None, ...)
        assert "filtered_obj" in seen
        assert input_filtered is seen["filtered_obj"]
        # Verrouille le type attendu pour éviter des sentinelles non Raw
        assert hasattr(input_raw, "get_data")
        assert hasattr(input_filtered, "get_data")

        # Trace le chemin de sortie pour vérification
        seen["plot_output_path"] = output_path
        # Verrouille le chemin de sortie construit depuis config + sujet/run
        assert output_path == expected_output_path
        # Verrouille le titre forwardé depuis config
        assert config.title == "Smoke Test"
        # Verrouille la bande de fréquence forwardée depuis config
        assert config.freq_band == (8.0, 40.0)
        # Verrouille les métadonnées forwardées depuis load_recording
        assert plot_metadata == metadata
        # Crée le répertoire parent pour simuler l'écriture réelle
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Écrit un PNG minimal pour conserver la sémantique de sortie
        output_path.write_bytes(b"png")
        # Écrit un sidecar JSON pour conserver la sémantique de sortie
        output_path.with_suffix(".json").write_text(
            json.dumps({"metadata": plot_metadata}, indent=2),
            encoding="utf-8",
        )
        # Retourne le chemin pour valider la valeur renvoyée par visualize_run
        return output_path

    # Monkeypatch les briques internes pour observer tous les appels de pipeline
    monkeypatch.setattr(viz, "build_recording_path", _spy_build_recording_path)
    monkeypatch.setattr(viz, "load_recording", _spy_load_recording)
    monkeypatch.setattr(viz, "pick_channels", _spy_pick_channels)
    monkeypatch.setattr(viz, "filter_recording", _spy_filter_recording)
    monkeypatch.setattr(viz, "plot_raw_vs_filtered", _spy_plot_raw_vs_filtered)

    # Exécute la visualisation avec une sélection de canal unique
    output_path = viz.visualize_run(
        data_root=Path("data"),
        subject="S001",
        run="R01",
        config=config,
    )

    # Vérifie que build_recording_path est bien invoqué par le pipeline
    assert seen["build_recording_path_called"] is True
    # Vérifie que load_recording reçoit le chemin attendu
    assert seen["load_recording_path"] == recording_path
    # Vérifie que pick_channels reçoit bien la liste demandée
    assert seen["pick_channels_channels"] == ["C3"]
    # Vérifie que filter_recording reçoit tous les paramètres non mutés
    assert seen["filter_args"] == ("fir", (8.0, 40.0), 0.1)
    # Vérifie que plot_raw_vs_filtered reçoit le chemin de sortie attendu
    assert seen["plot_output_path"] == expected_output_path

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
        parser.description == "Charge un run Physionet, applique le filtre 8-40 Hz, "
        "et enregistre un plot brut vs filtré."
    )

    # Récupère les actions positionnelles attendues
    subject_action = _get_action(parser, "subject")
    run_action = _get_action(parser, "run")

    # Verrouille l'aide exacte de l'argument subject
    assert subject_action.help == "Identifiant du sujet ex: S001 ou 9"
    # Verrouille l'aide exacte de l'argument run
    assert run_action.help == "Identifiant du run ex: R01 ou 10"

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

    # Récupère les options ciblées par les mutants (freq-band/pad-duration/title)
    freq_band_action = _get_action(parser, "freq_band")
    pad_duration_action = _get_action(parser, "pad_duration")
    title_action = _get_action(parser, "title")

    # Verrouille nargs pour tuer les suppressions/variantes implicites
    assert freq_band_action.nargs == 2
    # Verrouille type pour tuer type=None et la suppression de type=float
    assert freq_band_action.type is float
    # Verrouille default pour détecter toute dérive de la valeur nominale
    assert tuple(freq_band_action.default) == (8.0, 40.0)
    # Verrouille metavar pour tuer metavar=None / valeurs altérées
    assert freq_band_action.metavar == ("LOW", "HIGH")
    # Verrouille help exact pour tuer help=None / variantes de casse/texte
    assert freq_band_action.help == "Bande passe-bas/passe-haut en Hz"

    # Verrouille type pour tuer type=None et la suppression de type=float
    assert pad_duration_action.type is float
    # Verrouille default pour détecter toute dérive de la valeur nominale
    assert pad_duration_action.default == PAD_DURATION_DEFAULT
    # Verrouille help exact pour tuer help=None / variantes de texte
    assert pad_duration_action.help == "Durée de padding réfléchissant en secondes"

    # Verrouille le default runtime (même si absent, argparse met None)
    assert title_action.default is None
    # Verrouille help exact pour tuer help=None / variantes de texte
    assert title_action.help == "Titre personnalisé pour la figure"


def test_build_parser_parses_numeric_overrides_as_floats() -> None:
    """Tue les mutants qui retirent type=float sur freq-band et pad-duration."""

    # Construit le parseur depuis la fonction de prod
    parser = visualize_raw_filtered.build_parser()
    # Parse avec overrides numériques pour vérifier la conversion argparse
    args = parser.parse_args(
        [
            "S001",
            "R02",
            "--freq-band",
            "5",
            "55",
            "--pad-duration",
            "1.5",
        ]
    )
    # Verrouille la conversion en floats (échoue si type=float est retiré)
    assert tuple(args.freq_band) == (5.0, 55.0)
    # Verrouille le type élémentaire pour éviter les strings silencieuses
    assert all(isinstance(v, float) for v in args.freq_band)
    # Verrouille le type de pad_duration pour éviter la string silencieuse
    assert isinstance(args.pad_duration, float)
    # Verrouille la valeur exacte pour éviter les dérives de parsing
    assert args.pad_duration == 1.5


def test_build_parser_title_passes_explicit_default_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tue le mutant qui supprime default=None sur --title (kwargs non transmis)."""

    # Capture les kwargs de add_argument pour l'option --title
    captured: list[dict[str, object]] = []

    # Sauvegarde l'implémentation réelle avant patch pour éviter la récursion
    original_add_argument = argparse.ArgumentParser.add_argument

    # Définit un spy qui enregistre uniquement l'appel concernant --title
    def _spy_add_argument(
        self: argparse.ArgumentParser, *args: Any, **kwargs: Any
    ) -> argparse.Action:
        # Enregistre les kwargs uniquement pour l'option --title
        if args and args[0] == "--title":
            captured.append(dict(kwargs))
        # Délègue à l'implémentation réelle pour construire le parser
        return original_add_argument(self, *args, **kwargs)

    # Patch add_argument pour observer les paramètres effectivement transmis
    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", _spy_add_argument)

    # Construit le parser pour déclencher tous les add_argument
    _ = viz.build_parser()

    # Verrouille que l'option --title est déclarée exactement une fois
    assert len(captured) == 1
    # Verrouille la présence du kwarg default (tue la suppression)
    assert "default" in captured[0]
    # Verrouille la valeur explicite default=None
    assert captured[0]["default"] is None
