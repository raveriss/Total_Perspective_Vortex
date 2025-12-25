# Importe time pour suivre le budget temporel d'extraction
from collections import OrderedDict
from typing import cast
from time import perf_counter
from typing import Any, Mapping, Sequence

# Importe numpy pour générer des signaux et des tenseurs de test
import numpy as np

# Importe pytest pour vérifier les erreurs et les approximations
import pytest

# Importe mne pour créer des epochs synthétiques conformes à l'API
from mne import EpochsArray, create_info

# Importe le module complet pour monkeypatcher les helpers internes
import tpv.features as features_module

# Importe l'extracteur procédural, la classe scikit-learn et les helpers Welch
from tpv.features import ExtractFeatures, _prepare_welch_parameters, extract_features

# Définit une constante pour le budget temps afin d'éviter les magic numbers
MAX_EXTRACTION_SECONDS = 0.25
DEFAULT_WELCH_LENGTH = 50
NPERSEG_WITH_OVERLAP = 20
MINIMUM_POSITIVE_NPERSEG = 1
EXCESSIVE_OVERLAP = 25
NEGATIVE_OVERLAP = -5
EXPECTED_OVERLAP_CAP = NPERSEG_WITH_OVERLAP - 1
EMPTY_WINDOW = ""
NON_INTEGER_OVERLAP = 1.5
NON_INTEGER_NPERSEG = 16.2


def _build_epochs(
    n_epochs: int, n_channels: int, n_times: int, sfreq: float
) -> EpochsArray:
    """Crée des epochs synthétiques avec des noms de canal explicites."""

    # Génère un bruit gaussien pour éviter des signaux dégénérés
    data = np.random.default_rng(seed=0).standard_normal(
        (n_epochs, n_channels, n_times)
    )
    # Déclare des noms de canaux pour suivre l'ordre des étiquettes
    ch_names = [f"C{idx}" for idx in range(n_channels)]
    # Construit l'info MNE avec la fréquence d'échantillonnage imposée
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    # Retourne les epochs prêt à être traités par l'extracteur
    return EpochsArray(data, info)


def test_extract_features_welch_respects_window_and_shape() -> None:
    """Welch doit produire des bandes aplaties et étiquetées par canal."""

    # Prépare des epochs de test avec plusieurs canaux
    epochs = _build_epochs(n_epochs=3, n_channels=2, n_times=256, sfreq=128.0)
    # Demande une fenêtre rectangulaire pour stabiliser le test
    features, labels = extract_features(
        epochs,
        config={"method": "welch", "window": "boxcar", "nperseg": 128},
    )
    # Vérifie que la matrice est bien de taille essais x (canaux * bandes)
    assert features.shape == (3, 8)
    # Vérifie que les étiquettes reflètent l'ordre canal puis bande
    assert labels == [
        "C0_theta",
        "C0_alpha",
        "C0_beta",
        "C0_gamma",
        "C1_theta",
        "C1_alpha",
        "C1_beta",
        "C1_gamma",
    ]


def test_extract_features_alpha_sine_dominates_alpha_band() -> None:
    """Un sinus alpha doit produire plus d'énergie dans la bande alpha."""

    # Fixe les paramètres temporels pour centrer le sinus sur 10 Hz
    sfreq = 128.0
    # Construit un axe temporel régulier pour le sinus
    times = np.arange(0, 2.0, 1.0 / sfreq)
    # Génère un sinus de 10 Hz pour la première composante
    alpha_signal = np.sin(2 * np.pi * 10.0 * times)
    # Empile le sinus et du bruit sur deux canaux
    data = np.stack(
        [alpha_signal, np.random.default_rng(seed=1).standard_normal(times.size)],
    )
    # Réplique le signal sur plusieurs essais pour renforcer la moyenne
    epochs_data = np.stack([data, data])
    # Crée l'info correspondante pour MNE
    info = create_info(ch_names=["C0", "C1"], sfreq=sfreq, ch_types="eeg")
    # Bâtit les epochs à partir du tenseur préparé
    epochs = EpochsArray(epochs_data, info)
    # Exécute l'extraction avec la méthode par défaut
    features, _ = extract_features(epochs)
    # Reshape pour retrouver la structure essais x canaux x bandes
    reshaped = features.reshape(2, 2, 4)
    # Vérifie que le canal alpha présente une énergie maximale dans la bande alpha
    assert reshaped[0, 0, 1] == pytest.approx(np.max(reshaped[0, 0]))


def test_compute_fft_features_removes_dc_component() -> None:
    """La FFT doit retirer la composante DC pour un signal constant."""

    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"dc": (0.0, 1.0)},
    )
    constant_offset = np.full((1, 1, 64), 3.5)

    transformed = extractor.transform(constant_offset)

    assert np.array_equal(transformed, np.zeros_like(transformed))


def test_compute_fft_features_preserves_two_sided_energy() -> None:
    """Le repliement symétrique doit conserver l'énergie d'un sinus pur."""

    sfreq = 64.0
    n_times = 64
    times = np.arange(n_times) / sfreq
    sine = np.sin(2 * np.pi * 8.0 * times)
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"alpha": (7.5, 8.5)},
    )

    transformed = extractor.transform(sine.reshape(1, 1, -1))

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_compute_fft_features_handles_zero_signal_with_normalization() -> None:
    """Un signal nul doit rester nul même avec normalisation active."""

    extractor = ExtractFeatures(
        sfreq=128.0,
        feature_strategy="fft",
        normalize=True,
        bands={"beta": (13.0, 30.0)},
    )
    zeros = np.zeros((2, 3, 32))

    transformed = extractor.transform(zeros)

    assert np.array_equal(transformed, np.zeros_like(transformed))


def test_compute_fft_features_calls_rfft_with_axis_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_fft_features doit fixer axis=-1 pour verrouiller l'API FFT."""

    # Capture l'implémentation réelle pour conserver le comportement numérique
    original_rfft = features_module.np.fft.rfft

    def _rfft_spy(
        data: np.ndarray,
        n: int | None = None,
        axis: int | None = None,
        norm: str | None = None,
    ) -> np.ndarray:
        """Vérifie que axis est explicitement fourni à -1."""

        # Verrouille axis afin de détecter les mutations qui le suppriment
        assert axis == -1
        # Délègue le calcul pour préserver la chaîne de features
        return original_rfft(data, n=n, axis=axis, norm=norm)

    # Monkeypatch la FFT pour observer l'argument axis
    monkeypatch.setattr(features_module.np.fft, "rfft", _rfft_spy)

    # Définit une config FFT minimale pour déclencher la FFT réelle
    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"alpha": (7.5, 8.5)},
    )
    # Construit un tenseur stable pour exécuter la branche FFT
    data = np.random.default_rng(seed=11).standard_normal((1, 1, 16))

    # Exécute la transformation pour déclencher l'appel FFT spy
    transformed = extractor.transform(data)

    # Vérifie la forme pour garantir l'exécution complète du pipeline
    assert transformed.shape == (1, 1)


def test_compute_fft_features_forces_float_dtype_in_power_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le tenseur power doit être construit via np.asarray(..., dtype=float)."""

    # Capture l'implémentation réelle afin de déléguer le calcul
    original_asarray = features_module.np.asarray

    def _asarray_spy(array: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        """Vérifie que dtype est explicitement forcé à float."""

        # Extrait dtype depuis kwargs ou args pour couvrir les appels futurs
        dtype_value = kwargs.get("dtype")
        # Récupère dtype depuis args si l'implémentation bascule en positionnel
        if dtype_value is None and args:
            dtype_value = args[0]
        # Verrouille dtype pour détecter les mutations et refactors non désirés
        assert dtype_value is float
        # Délègue à numpy pour conserver le comportement attendu
        return original_asarray(array, *args, **kwargs)

    # Monkeypatch np.asarray pour vérifier la stabilité du contrat dtype
    monkeypatch.setattr(features_module.np, "asarray", _asarray_spy)

    # Prépare un extracteur FFT pour exécuter la construction de power
    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"alpha": (7.5, 8.5)},
    )
    # Utilise un signal aléatoire stable pour éviter les dégénérescences
    data = np.random.default_rng(seed=12).standard_normal((1, 1, 16))

    # Exécute transform pour activer np.asarray spy dans _compute_fft_features
    transformed = extractor.transform(data)

    # Vérifie que la sortie reste une matrice aplatie essais x features
    assert transformed.shape == (1, 1)


def test_compute_fft_features_calls_typing_cast_with_ndarray_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La conversion typing.cast doit cibler np.ndarray pour verrouiller le contrat."""

    # Prépare un conteneur pour valider le type cible passé à cast
    captured_targets: list[object] = []

    def _cast_spy(target: object, value: object) -> object:
        """Capture le type cible et renvoie la valeur inchangée."""

        # Stocke le type demandé pour l'assertion postérieure
        captured_targets.append(target)
        # Retourne la valeur comme typing.cast le fait en runtime
        return value

    # Remplace cast dans le module afin d'observer l'argument target
    monkeypatch.setattr(features_module, "cast", _cast_spy)

    # Prépare un extracteur FFT standard pour atteindre le return cast(...)
    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"alpha": (7.5, 8.5)},
    )
    # Construit une entrée simple pour exécuter le pipeline complet
    data = np.random.default_rng(seed=13).standard_normal((1, 1, 16))

    # Exécute la transformation pour déclencher l'appel cast spy
    transformed = extractor.transform(data)

    # Vérifie que cast a été appelé avec np.ndarray comme type cible
    assert captured_targets[-1] is np.ndarray
    # Vérifie que la sortie reste cohérente malgré le spy
    assert transformed.shape == (1, 1)


def test_compute_fft_features_skips_two_sided_restoration_when_single_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le repliement symétrique ne doit pas s'exécuter quand rfft renvoie 1 bin."""

    # Capture l'implémentation réelle afin de produire le tenseur de base
    original_asarray = features_module.np.asarray
    # Définit un conteneur pour exposer l'instance spy à l'assertion finale
    spy_holder: dict[str, object] = {}

    class _ReconstructionSpyArray(np.ndarray):
        """ndarray spy qui détecte les indexations 1:-1 et -1 du repliement."""

        # Stocke un drapeau pour savoir si la branche repliement s'exécute
        saw_reconstruction_index: bool

        def __new__(cls, base: np.ndarray) -> "_ReconstructionSpyArray":
            """Construit une vue spy pour intercepter __getitem__."""

            # Crée une vue afin d'hériter des opérations numpy standard
            obj = original_asarray(base).view(cls)
            # Initialise le drapeau à faux avant le pipeline de features
            obj.saw_reconstruction_index = False
            # Retourne l'instance instrumentée
            return obj

        def __getitem__(self, item: object) -> object:
            """Marque les indexations propres au repliement de spectre réel."""

            # Normalise la clé en tuple pour inspecter les axes
            key = item if isinstance(item, tuple) else (item,)
            # Détecte l'indexation du repliement (slice 1:-1 ou bin -1)
            for element in key:
                if isinstance(element, slice):
                    if element.start == 1 and element.stop == -1:
                        self.saw_reconstruction_index = True
                if isinstance(element, int):
                    if element == -1:
                        self.saw_reconstruction_index = True
            # Délègue l'accès à numpy pour conserver le comportement attendu
            return super().__getitem__(item)

    def _asarray_spy(array: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        """Retourne un ndarray instrumenté pour détecter le repliement."""

        # Construit le tableau de base avec numpy pour respecter les conversions
        base = original_asarray(array, *args, **kwargs)
        # Construit une instance spy afin d'initialiser les attributs de suivi
        spy = _ReconstructionSpyArray(base)
        # Expose l'instance pour l'assertion après transform
        spy_holder["power"] = spy
        # Retourne l'array instrumenté au code de production
        return spy

    # Monkeypatch np.asarray pour injecter l'array spy dans _compute_fft_features
    monkeypatch.setattr(features_module.np, "asarray", _asarray_spy)

    # Fixe n_times=1 pour produire rfft avec un seul bin fréquentiel
    data = np.array([[[1.0]]], dtype=float)
    # Cible le bin DC afin d'exercer la sélection de bande
    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"dc": (0.0, 0.0)},
    )

    # Exécute la transformation qui ne doit pas toucher power[..., 1:-1] ou [..., -1]
    transformed = extractor.transform(data)

    # Récupère l'instance spy afin d'inspecter le drapeau
    spy_power = cast(_ReconstructionSpyArray, spy_holder["power"])
    # Valide que la branche repliement n'a pas été exécutée pour un seul bin
    assert spy_power.saw_reconstruction_index is False
    # Valide que la sortie est bien une matrice essai x feature
    assert transformed.shape == (1, 1)


def test_compute_fft_features_restores_last_bin_energy_for_three_samples() -> None:
    """Pour n_times=3, le dernier bin doit être doublé pour conserver l'énergie."""

    # Fixe un sfreq compatible avec un bin utile à 1 Hz
    sfreq = 3.0
    # Fixe un nombre impair qui ne produit que 2 bins rfft (0 et f=sfreq/3)
    n_times = 3
    # Construit l'axe temporel discret cohérent
    times = np.arange(n_times) / sfreq
    # Génère un sinus aligné sur le dernier bin rfft
    sine = np.sin(2 * np.pi * 1.0 * times)
    # Instancie l'extracteur FFT sur une bande centrée sur ce bin
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"last_bin": (1.0, 1.0)},
    )

    # Exécute la transformation (1 essai, 1 canal, n_times)
    transformed = extractor.transform(sine.reshape(1, 1, -1))

    # Vérifie que l'énergie est conservée avec la correction du dernier bin
    assert transformed[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_compute_fft_features_does_not_double_nyquist_bin_for_even_length() -> None:
    """Pour n_times pair, le bin Nyquist ne doit pas être doublé."""

    # Fixe sfreq afin que Nyquist tombe sur une fréquence entière
    sfreq = 4.0
    # Fixe une longueur paire qui expose le bin Nyquist unique
    n_times = 4
    # Construit l'axe temporel discret
    times = np.arange(n_times) / sfreq
    # Utilise un cosinus Nyquist pour éviter la nullité d'un sinus à pi
    cosine_nyquist = np.cos(2 * np.pi * (sfreq / 2.0) * times)
    # Instancie un extracteur FFT ciblant uniquement le bin Nyquist
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"nyquist": (sfreq / 2.0, sfreq / 2.0)},
    )

    # Exécute la transformation avec un seul essai et un seul canal
    transformed = extractor.transform(cosine_nyquist.reshape(1, 1, -1))

    # Vérifie que l'énergie Nyquist n'est pas doublée (sinon facteur 2)
    assert transformed[0, 0] == pytest.approx(1.0, rel=1e-3)


def test_compute_fft_features_returns_flattened_matrix_with_multiple_bands() -> None:
    """La sortie FFT doit s'aplatir en (n_epochs, n_channels * n_bands)."""

    # Fixe un signal multi-canaux et multi-bandes pour verrouiller reshape
    data = np.random.default_rng(seed=14).standard_normal((2, 3, 32))
    # Définit deux bandes distinctes pour tester l'aplatissement
    bands = {"theta": (4.0, 7.0), "alpha": (8.0, 12.0)}
    # Construit l'extracteur FFT avec deux bandes
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="fft",
        normalize=False,
        bands=bands,
    )

    # Exécute la transformation pour produire la matrice de features
    transformed = extractor.transform(data)

    # Vérifie la forme aplatie attendue
    assert transformed.shape == (2, 3 * len(bands))



def test_compute_fft_features_reshapes_with_minus_one_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_fft_features doit utiliser -1 pour aplatir (canaux * bandes)."""

    class FakeStacked:
        """Expose une API minimale pour contrôler l'appel à reshape."""

        def __init__(self, shape: tuple[int, int, int]) -> None:
            # Stocke la forme pour imiter un ndarray
            self.shape = shape

        def reshape(self, first_dim: int, second_dim: int) -> np.ndarray:
            """Valide que l'aplatissement utilise strictement l'idiome -1."""

            if first_dim != self.shape[0]:
                raise AssertionError("FFT features must preserve sample dimension.")
            if second_dim != -1:
                raise AssertionError("FFT features must reshape with -1.")
            return np.zeros((first_dim, self.shape[1] * self.shape[2]))

    def _spy_stack(*args: Any, **kwargs: Any) -> Any:
        """Retourne un faux tenseur empilé pour vérifier l'argument reshape."""

        # Capture la liste des matrices (n_epochs, n_channels) empilées en bandes
        band_blocks: Any = args[0]
        # Récupère l'axis fourni (positional ou keyword)
        axis_provided = len(args) >= 2 or "axis" in kwargs
        axis_value: Any = args[1] if len(args) >= 2 else kwargs.get("axis")
        # Force la présence explicite de axis pour éviter les mutants équivalents
        assert axis_provided, "np.stack doit recevoir axis explicitement."
        # Impose l'axe 2 pour obtenir (epochs, channels, bands)
        assert axis_value == 2
        # Déduit la forme empilée attendue à partir du premier bloc
        first_block = band_blocks[0]
        return FakeStacked(
            (first_block.shape[0], first_block.shape[1], len(band_blocks))
        )

    # Intercepte np.stack dans le module features pour contrôler reshape
    monkeypatch.setattr(features_module.np, "stack", _spy_stack)

    # Fixe un signal multi-canaux et multi-bandes pour déclencher le reshape final
    data = np.random.default_rng(seed=15).standard_normal((2, 3, 32))
    # Définit deux bandes distinctes pour produire un empilement non ambigu
    bands = {"theta": (4.0, 7.0), "alpha": (8.0, 12.0)}
    # Construit l'extracteur FFT avec normalisation désactivée
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="fft",
        normalize=False,
        bands=bands,
    )

    # Exécute la transformation pour exercer le reshape final
    transformed = extractor.transform(data)

    # Vérifie la forme aplatie attendue (et le spy verrouille le -1)
    assert transformed.shape == (2, 3 * len(bands))


def test_compute_fft_features_centers_signal_per_channel_not_global_mean() -> None:
    """Le centrage FFT doit retirer la DC par canal, pas via une moyenne globale."""

    extractor = ExtractFeatures(
        sfreq=32.0,
        feature_strategy="fft",
        normalize=False,
        bands={"dc": (0.0, 0.0)},
    )

    channel0 = np.full(64, 2.0)
    channel1 = np.full(64, -2.0)
    signal = np.stack([channel0, channel1], axis=0)
    X = signal.reshape(1, 2, 64)

    transformed = extractor.transform(X)

    assert transformed.shape == (1, 2)
    assert np.array_equal(transformed, np.zeros_like(transformed))


def test_compute_fft_features_restores_energy_for_odd_length_interior_bin() -> None:
    """Le repliement doit doubler les bins internes pour n_times impair."""

    sfreq = 5.0
    n_times = 5
    times = np.arange(n_times) / sfreq
    sine = np.sin(2 * np.pi * 1.0 * times)
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"bin_1": (1.0, 1.0)},
    )

    transformed = extractor.transform(sine.reshape(1, 1, -1))

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_compute_fft_features_restores_energy_for_odd_length_last_bin() -> None:
    """Le repliement doit doubler le dernier bin quand n_times est impair."""

    sfreq = 5.0
    n_times = 5
    times = np.arange(n_times) / sfreq
    sine = np.sin(2 * np.pi * 2.0 * times)
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"bin_last": (2.0, 2.0)},
    )

    transformed = extractor.transform(sine.reshape(1, 1, -1))

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_compute_fft_features_includes_band_edges_when_equal_to_bin() -> None:
    """Le masque de bande FFT doit inclure freqs==low et freqs==high."""

    sfreq = 64.0
    n_times = 64
    times = np.arange(n_times) / sfreq
    sine = np.sin(2 * np.pi * 10.0 * times)
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize=False,
        bands={"edge": (10.0, 10.0)},
    )

    transformed = extractor.transform(sine.reshape(1, 1, -1))

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == pytest.approx(0.5, rel=1e-3)


def test_extract_features_wavelet_emphasizes_alpha_band() -> None:
    """La voie wavelet doit concentrer l'énergie sur la bande centrale alpha."""

    # Prépare un signal dominé par une fréquence alpha et du bruit multicanal
    sfreq = 64.0
    # Calibre la durée pour contenir plusieurs périodes alpha
    times = np.arange(0.0, 1.0, 1.0 / sfreq)
    # Génère un sinus alpha sur le premier canal et un bruit sur les autres
    alpha_signal = np.sin(2 * np.pi * 10.0 * times)
    # Empile trois canaux pour vérifier la stabilité inter-canaux
    data = np.stack(
        [
            alpha_signal,
            np.random.default_rng(seed=2).standard_normal(times.size),
            np.random.default_rng(seed=3).standard_normal(times.size),
        ]
    )
    # Réplique le signal sur un seul essai pour isoler la réponse spectrale
    epochs_data = np.expand_dims(data, axis=0)
    # Construit les métadonnées MNE nécessaires à extract_features
    info = create_info(ch_names=["C0", "C1", "C2"], sfreq=sfreq, ch_types="eeg")
    # Instancie des epochs MNE prêts pour l'extraction
    epochs = EpochsArray(epochs_data, info)
    # Exécute l'extraction en mode wavelet pour capter l'énergie alpha
    features, labels = extract_features(epochs, config={"method": "wavelet"})
    # Reshape pour retrouver la structure essais x canaux x bandes
    reshaped = features.reshape(1, 3, 4)
    # Vérifie que l'énergie alpha domine les autres bandes sur le canal ciblé
    assert reshaped[0, 0, 1] == pytest.approx(np.max(reshaped[0, 0]))
    # Vérifie que les étiquettes restent cohérentes avec les canaux
    assert labels[0] == "C0_theta"
    # Instancie la classe wavelet pour couvrir la branche scikit-learn
    extractor = ExtractFeatures(
        sfreq=sfreq, feature_strategy="wavelet", normalize=False
    )
    # Applique transform pour vérifier que les coefficients sont non nuls
    transformed = extractor.transform(epochs.get_data())
    # Vérifie que la matrice contient de l'énergie et respecte la forme attendue
    assert transformed.shape == (1, 12)
    # Vérifie que la composante alpha reste dominante après l'appel orienté classe
    assert transformed.reshape(1, 3, 4)[0, 0, 1] == pytest.approx(
        np.max(transformed.reshape(1, 3, 4)[0, 0])
    )


def test_extract_features_wavelet_rejects_empty_bands() -> None:
    """La configuration wavelet doit refuser explicitement l'absence de bandes."""

    # Prépare des epochs synthétiques minimaux
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    # Vérifie que l'absence de bandes entraîne une erreur explicite
    with pytest.raises(
        ValueError, match="At least one frequency band must be provided"
    ):
        extract_features(epochs, config={"method": "wavelet", "bands": []})


def test_extract_features_wavelet_handles_overlapping_bands() -> None:
    """Les bandes wavelet peuvent se chevaucher tout en préservant la forme."""

    # Prépare un signal centré sur 10 Hz pour favoriser les bandes alpha
    sfreq = 128.0
    times = np.arange(0.0, 1.0, 1.0 / sfreq)
    alpha_signal = np.sin(2 * np.pi * 10.0 * times)
    epochs = EpochsArray(
        np.expand_dims(np.stack([alpha_signal, alpha_signal]), axis=0),
        create_info(ch_names=["C0", "C1"], sfreq=sfreq, ch_types="eeg"),
    )
    # Définit deux bandes qui se chevauchent autour de 10 Hz et une bande distante
    custom_bands = [
        ("alpha_wide", (8.0, 12.0)),
        ("alpha_narrow", (9.0, 11.0)),
        ("beta", (20.0, 30.0)),
    ]
    features, labels = extract_features(
        epochs, config={"method": "wavelet", "bands": custom_bands}
    )
    # Vérifie que la forme reflète les bandes personnalisées et les canaux
    assert features.shape == (1, 6)
    assert labels == [
        "C0_alpha_wide",
        "C0_alpha_narrow",
        "C0_beta",
        "C1_alpha_wide",
        "C1_alpha_narrow",
        "C1_beta",
    ]
    # Les deux bandes alpha devraient dominer la bande beta sur chaque canal
    reshaped = features.reshape(1, 2, 3)
    assert reshaped[0, 0, 0] > reshaped[0, 0, 2]
    assert reshaped[0, 0, 1] > reshaped[0, 0, 2]
    assert reshaped[0, 1, 0] > reshaped[0, 1, 2]
    assert reshaped[0, 1, 1] > reshaped[0, 1, 2]


def test_compute_wavelet_features_validates_wavelet_name_type() -> None:
    """_compute_wavelet_features doit rejeter un nom de wavelet non textuel."""

    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        strategy_config={"wavelet": ["morlet"]},
    )
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)

    with pytest.raises(
        ValueError,
        match=r"^Wavelet selection must be a non-empty string \(e\.g\., 'morlet'\)\.$",
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_validates_wavelet_type_before_band_power_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_wavelet_features doit valider la wavelet avant le calcul de bandes."""

    # Prépare un extracteur avec une wavelet invalide dans la stratégie
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        strategy_config={"wavelet": ["morlet"]},
    )
    # Prépare des epochs minimales pour déclencher la stratégie wavelet
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)

    def forbidden_band_powers(
        data: np.ndarray,  # noqa: ARG001
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Échoue si l'appel passe la validation de wavelet."""

        raise AssertionError("Band powers should not be computed when wavelet is invalid.")

    # Empêche la couche inférieure de masquer un défaut de validation amont
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_band_powers",
        forbidden_band_powers,
    )

    with pytest.raises(
        ValueError,
        match=r"^Wavelet selection must be a non-empty string \(e\.g\., 'morlet'\)\.$",
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_rejects_unsupported_wavelet_before_band_power_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_wavelet_features doit refuser une wavelet non supportée."""

    # Prépare un extracteur configuré avec une wavelet inconnue
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        strategy_config={"wavelet": "mexican_hat"},
    )
    # Prépare des epochs minimales pour déclencher la stratégie wavelet
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)

    def forbidden_band_powers(
        data: np.ndarray,  # noqa: ARG001
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Échoue si la validation de wavelet n'a pas stoppé l'appel."""

        raise AssertionError("Band powers should not be computed for unsupported wavelet.")

    # Empêche la couche inférieure de jeter l'erreur à la place de la classe
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_band_powers",
        forbidden_band_powers,
    )

    with pytest.raises(
        ValueError,
        match=r"^Unsupported wavelet: 'mexican_hat'\. Use 'morlet'\.$",
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_passes_lowercase_default_wavelet_to_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le défaut 'morlet' doit rester en minuscules pour la validation."""

    # Prépare un extracteur sans config wavelet pour activer le défaut
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        strategy_config={},
    )
    # Prépare des epochs minimales pour déclencher la stratégie wavelet
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Prépare un conteneur pour tracer l'argument exact fourni au normaliseur
    captured_wavelets: list[Any] = []

    def spy_normalize_wavelet_name(wavelet_name: Any) -> str:
        """Capture l'argument pour détecter un défaut modifié."""

        captured_wavelets.append(wavelet_name)
        return "morlet"

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Retourne une sortie 3D cohérente sans relancer la validation wavelet."""

        return np.zeros((data.shape[0], data.shape[1], len(band_ranges)))

    # Espionne la validation pour vérifier la valeur défaut utilisée
    monkeypatch.setattr(
        features_module,
        "_normalize_wavelet_name",
        spy_normalize_wavelet_name,
    )
    # Simule le calcul pour que seul l'appel au normaliseur soit testé
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_band_powers",
        fake_band_powers,
    )

    _ = extractor.transform(epochs.get_data())

    # Vérifie que le défaut est exactement 'morlet' et non une variante
    assert captured_wavelets == ["morlet"]


def test_compute_wavelet_features_rejects_invalid_band_power_dimensions_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le message d'erreur 3D doit rester stable et explicite."""

    # Prépare des epochs de référence pour une forme non ambiguë
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Prépare un extracteur wavelet sans normalisation pour isoler l'erreur
    extractor = ExtractFeatures(sfreq=64.0, feature_strategy="wavelet", normalize=False)

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Retourne un tableau 2D pour déclencher le garde-fou ndim."""

        return np.zeros((data.shape[0], data.shape[1]))

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^Wavelet band powers must return a 3D array "
            r"\(epochs, channels, bands\) to preserve feature naming\.$"
        ),
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_rejects_mismatched_band_power_shape_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le message d'erreur de mismatch de bandes doit rester stable."""

    # Prépare des epochs de référence pour une forme non ambiguë
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Force deux bandes pour rendre le mismatch détectable
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        bands={"alpha": (8.0, 12.0), "beta": (13.0, 30.0)},
    )

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Retourne un tenseur avec une seule bande pour provoquer le mismatch."""

        return np.zeros((data.shape[0], data.shape[1], 1))

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^Wavelet band powers shape mismatch: expected 2 bands but received 1\. "
            r"Check the number of central frequencies and band definitions\.$"
        ),
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_rejects_epoch_channel_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_wavelet_features doit refuser la perte d'un axe epoch ou canal."""

    # Prépare des epochs de référence pour un contrôle des dimensions
    epochs = _build_epochs(n_epochs=2, n_channels=1, n_times=32, sfreq=64.0)
    # Prépare un extracteur wavelet sans normalisation pour isoler l'erreur
    extractor = ExtractFeatures(sfreq=64.0, feature_strategy="wavelet", normalize=False)

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        """Modifie uniquement l'axe epoch pour vérifier le test de cohérence."""

        return np.zeros((data.shape[0] + 1, data.shape[1], len(band_ranges)))

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    with pytest.raises(
        ValueError,
        match=r"^Wavelet band powers must preserve epoch and channel dimensions\.$",
    ):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_reshapes_with_minus_one_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_wavelet_features doit utiliser -1 pour aplatir les bandes."""

    # Prépare des epochs de référence pour une forme non ambiguë
    epochs = _build_epochs(n_epochs=2, n_channels=3, n_times=32, sfreq=64.0)
    # Prépare un extracteur avec la config par défaut des bandes
    extractor = ExtractFeatures(sfreq=64.0, feature_strategy="wavelet", normalize=False)

    class FakeStacked:
        """Expose une API minimale pour contrôler l'appel à reshape."""

        def __init__(self, shape: tuple[int, int, int]) -> None:
            # Stocke la forme pour imiter un ndarray
            self.shape = shape
            # Calcule ndim pour satisfaire les gardes de validation
            self.ndim = len(shape)

        def reshape(self, first_dim: int, second_dim: int) -> np.ndarray:
            """Valide que l'aplatissement utilise l'idiome -1."""

            if second_dim != -1:
                raise AssertionError("Wavelet features must reshape with -1.")
            return np.zeros((first_dim, self.shape[1] * self.shape[2]))

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,  # noqa: ARG001
        band_ranges: Mapping[str, tuple[float, float]],  # noqa: ARG001,E501
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> Any:
        """Retourne un faux tenseur 3D pour contrôler le reshape."""

        return FakeStacked((data.shape[0], data.shape[1], len(band_ranges)))

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    features = extractor.transform(epochs.get_data())

    # Valide la forme aplatie attendue pour éviter une régression silencieuse
    assert features.shape == (2, 3 * len(extractor.band_labels))


def test_compute_wavelet_band_powers_rejects_misordered_band() -> None:
    """Les bandes wavelet doivent imposer low < high pour chaque intervalle."""

    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    with pytest.raises(ValueError, match="low < high"):
        extract_features(
            epochs,
            config={
                "method": "wavelet",
                "bands": [("alpha", (12.0, 8.0))],
            },
        )

def test_compute_wavelet_band_powers_forwards_wavelet_width_to_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La config wavelet_width doit être propagée aux coefficients."""

    # Fixe une largeur non défaut pour détecter les régressions de forwarding
    custom_width = 9.25
    # Prépare des données minimales (epochs, canaux, temps)
    data = np.zeros((1, 1, 8), dtype=float)
    # Définit une bande valide pour déclencher le calcul wavelet
    band_ranges = {"alpha": (8.0, 12.0)}
    # Prépare un conteneur pour capturer la largeur réellement utilisée
    captured_cycles: List[float] = []

    def _fake_coefficients(
        channel_values: np.ndarray,
        central_frequencies: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        """Capture wavelet_cycles sans dépendre d'une convolution FFT."""

        # Enregistre la valeur transmise pour valider l'API interne
        captured_cycles.append(wavelet_cycles)
        # Retourne un tenseur cohérent (bandes, temps) pour poursuivre le flux
        return np.zeros(
            (len(central_frequencies), channel_values.size),
            dtype=np.complex128,
        )

    # Substitue le calcul réel pour isoler la propagation des paramètres
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_coefficients",
        _fake_coefficients,
    )

    # Exécute le calcul des puissances avec une largeur explicite
    _ = features_module._compute_wavelet_band_powers(
        data,
        band_ranges=band_ranges,
        sfreq=64.0,
        config={"wavelet_width": custom_width},
    )

    # Vérifie que la largeur configurée est bien utilisée en interne
    assert captured_cycles == [custom_width]


def test_compute_wavelet_band_powers_defaults_to_six_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'absence de wavelet_width doit tomber sur le défaut attendu (6.0)."""

    # Fixe la valeur défaut contractuelle pour stabiliser le comportement
    default_width = 6.0
    # Prépare des données minimales (epochs, canaux, temps)
    data = np.zeros((1, 1, 8), dtype=float)
    # Définit une bande valide pour déclencher le calcul wavelet
    band_ranges = {"alpha": (8.0, 12.0)}
    # Prépare un conteneur pour capturer la largeur réellement utilisée
    captured_cycles: List[float] = []

    def _fake_coefficients(
        channel_values: np.ndarray,
        central_frequencies: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        """Capture wavelet_cycles sans dépendre d'une convolution FFT."""

        # Enregistre la valeur transmise pour valider le défaut
        captured_cycles.append(wavelet_cycles)
        # Retourne un tenseur cohérent (bandes, temps) pour poursuivre le flux
        return np.zeros(
            (len(central_frequencies), channel_values.size),
            dtype=np.complex128,
        )

    # Substitue le calcul réel pour isoler la valeur défaut
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_coefficients",
        _fake_coefficients,
    )

    # Exécute le calcul des puissances sans largeur explicite
    _ = features_module._compute_wavelet_band_powers(
        data,
        band_ranges=band_ranges,
        sfreq=64.0,
        config={},
    )

    # Vérifie que la valeur défaut est bien appliquée
    assert captured_cycles == [default_width]


def test_compute_wavelet_band_powers_uses_lowercase_default_wavelet_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'absence de clé wavelet doit tomber sur le défaut canonique 'morlet'."""

    # Prépare des données minimales (epochs, canaux, temps)
    data = np.zeros((1, 1, 8), dtype=float)
    # Définit une bande valide pour déclencher le calcul wavelet
    band_ranges = {"alpha": (8.0, 12.0)}
    # Prépare un conteneur pour capturer la valeur transmise au normaliseur
    captured_wavelet: list[str] = []
    # Sauvegarde le normaliseur réel pour préserver la validation contractuelle
    original_normalize = features_module._normalize_wavelet_name

    def _spy_normalize(wavelet_name: Any) -> str:
        """Capture la valeur brute pour empêcher un changement de défaut implicite."""

        # Enregistre la valeur transmise pour figer le choix par défaut
        captured_wavelet.append(cast(str, wavelet_name))
        # Délègue au normaliseur réel pour conserver les erreurs attendues
        return original_normalize(wavelet_name)

    def _fake_coefficients(
        channel_values: np.ndarray,
        central_frequencies: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        """Retourne des coefficients nuls pour éviter une convolution FFT coûteuse."""

        # Retourne un tenseur (bandes, temps) cohérent pour poursuivre le flux
        return np.zeros(
            (len(central_frequencies), channel_values.size),
            dtype=np.complex128,
        )

    # Espionne le normaliseur pour figer le défaut de wavelet (mutmut_9)
    monkeypatch.setattr(features_module, "_normalize_wavelet_name", _spy_normalize)
    # Neutralise le calcul des coefficients pour garder le test rapide et stable
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_coefficients",
        _fake_coefficients,
    )

    # Exécute le calcul des puissances sans config wavelet explicite
    _ = features_module._compute_wavelet_band_powers(
        data,
        band_ranges=band_ranges,
        sfreq=64.0,
        config={},
    )

    # Vérifie que le défaut transmis est bien le canonique 'morlet'
    assert captured_wavelet == ["morlet"]


def test_compute_wavelet_band_powers_rejects_configured_unknown_wavelet_name() -> None:
    """_compute_wavelet_band_powers doit valider config['wavelet'] explicitement."""

    # Prépare des données minimales pour isoler la validation de config
    data = np.zeros((1, 1, 8), dtype=float)
    # Définit une bande valide pour ne pas échouer avant la validation wavelet
    band_ranges = {"alpha": (8.0, 12.0)}
    # Force un nom invalide pour vérifier la lecture de la clé 'wavelet'
    config = {"wavelet": "mexican_hat"}

    # Vérifie qu'une wavelet non supportée déclenche bien une erreur dédiée
    with pytest.raises(
        ValueError,
        match=r"^Unsupported wavelet: 'mexican_hat'\. Use 'morlet'\.$",
    ):
        features_module._compute_wavelet_band_powers(
            data,
            band_ranges=band_ranges,
            sfreq=64.0,
            config=config,
        )


def test_compute_wavelet_band_powers_rejects_equal_band_edges() -> None:
    """Les bandes wavelet doivent rejeter les bornes identiques (low == high)."""

    # Prépare des données minimales (epochs, canaux, temps)
    data = np.zeros((1, 1, 8), dtype=float)
    # Définit une bande invalide (low == high) qui doit lever une erreur
    band_ranges = {"alpha": (8.0, 8.0)}

    # Vérifie que la validation impose strictement low < high
    with pytest.raises(ValueError, match="low < high"):
        features_module._compute_wavelet_band_powers(
            data,
            band_ranges=band_ranges,
            sfreq=64.0,
            config={},
        )


def test_compute_wavelet_band_powers_accepts_single_sample_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Une sortie coefficients de taille 1 ne doit pas être traitée comme vide."""

    # Prépare la plus petite entrée possible (1 epoch, 1 canal, 1 échantillon)
    data = np.zeros((1, 1, 1), dtype=float)
    # Définit une seule bande pour produire un seul coefficient
    band_ranges = {"alpha": (8.0, 12.0)}

    def _single_coefficients(
        channel_values: np.ndarray,
        central_frequencies: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        """Retourne un tenseur (1, 1) pour couvrir le cas size == 1."""

        # Force un seul coefficient non nul pour différencier de l'absence de données
        return np.ones((1, 1), dtype=np.complex128)

    # Substitue le calcul réel pour forcer coefficients.size == 1
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_coefficients",
        _single_coefficients,
    )

    # Calcule les puissances wavelet pour ce cas limite
    band_powers = features_module._compute_wavelet_band_powers(
        data,
        band_ranges=band_ranges,
        sfreq=64.0,
        config={},
    )

    # Vérifie que la forme (epochs, canaux, bandes) est conservée
    assert band_powers.shape == (1, 1, 1)
    # Vérifie que l'énergie est calculée (|1|^2 == 1) et non rejetée
    assert np.allclose(band_powers[0, 0, 0], 1.0)


def test_compute_wavelet_band_powers_uses_magnitude_squared_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La puissance doit être la moyenne temporelle de |coeffs|^2 par bande."""

    # Prépare des données minimales (la valeur n'importe pas car on mocke les coeffs)
    data = np.zeros((1, 1, 3), dtype=float)
    # Définit deux bandes pour vérifier une vectorisation correcte
    band_ranges = {"band0": (8.0, 12.0), "band1": (12.0, 16.0)}

    def _known_coefficients(
        channel_values: np.ndarray,
        central_frequencies: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        """Retourne des coefficients déterministes pour valider la formule."""

        # Assure une matrice (bandes, temps) alignée sur l'appelant
        return np.array(
            [
                [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                [2.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j],
            ],
            dtype=np.complex128,
        )

    # Substitue le calcul réel pour rendre l'énergie testable analytiquement
    monkeypatch.setattr(
        features_module,
        "_compute_wavelet_coefficients",
        _known_coefficients,
    )

    # Calcule les puissances wavelet pour les coefficients déterministes
    band_powers = features_module._compute_wavelet_band_powers(
        data,
        band_ranges=band_ranges,
        sfreq=64.0,
        config={},
    )

    # Fixe les puissances attendues: |1+i|^2=2 et |2|^2=4
    expected = np.array([2.0, 4.0], dtype=float)
    # Valide la moyenne temporelle par bande en sortie
    np.testing.assert_allclose(band_powers[0, 0, :], expected)




def test_compute_wavelet_band_powers_handles_zero_energy_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Les signaux nuls doivent produire des puissances nulles sans erreur."""

    epochs = _build_epochs(n_epochs=2, n_channels=2, n_times=32, sfreq=32.0)
    # Force des données nulles pour obtenir une énergie nulle
    epochs._data[...] = 0.0

    captured_coeffs: list[np.ndarray] = []

    def _capture_coeffs(
        channel_values: np.ndarray,
        central_freqs: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        coeffs = original_compute_coeffs(
            channel_values, central_freqs, sfreq, wavelet_cycles
        )
        captured_coeffs.append(coeffs)
        return coeffs

    original_compute_coeffs = features_module._compute_wavelet_coefficients
    monkeypatch.setattr(
        features_module, "_compute_wavelet_coefficients", _capture_coeffs
    )

    features, _ = extract_features(epochs, config={"method": "wavelet"})
    # Toutes les features doivent rester nulles car l'entrée est nulle
    assert np.array_equal(features, np.zeros_like(features))
    # Les coefficients doivent avoir été calculés pour chaque canal
    assert (
        len(captured_coeffs) == epochs.get_data().shape[0] * epochs.get_data().shape[1]
    )
    assert all(
        coeff.shape[-1] == epochs.get_data().shape[-1] for coeff in captured_coeffs
    )


def test_compute_wavelet_band_powers_maps_bands_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La CWT doit respecter l'ordre des bandes et la fréquence centrale calculée."""

    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    custom_bands = [("beta", (20.0, 30.0)), ("theta", (4.0, 7.0))]

    captured_frequencies: list[list[float]] = []

    def _capture_coeffs(
        channel_values: np.ndarray,
        central_freqs: Sequence[float],
        sfreq: float,
        wavelet_cycles: float,
    ) -> np.ndarray:
        captured_frequencies.append(list(central_freqs))
        return original_compute_coeffs(
            channel_values, central_freqs, sfreq, wavelet_cycles
        )

    original_compute_coeffs = features_module._compute_wavelet_coefficients
    monkeypatch.setattr(
        features_module, "_compute_wavelet_coefficients", _capture_coeffs
    )

    features, labels = extract_features(
        epochs, config={"method": "wavelet", "bands": custom_bands}
    )

    assert labels == ["C0_beta", "C0_theta"]
    assert captured_frequencies == [[25.0, 5.5]]
    # Vérifie que les puissances sont produites dans le même ordre que les bandes
    reshaped = features.reshape(1, 1, 2)
    assert reshaped.shape == (1, 1, 2)




def test_compute_wavelet_coefficients_stacks_band_axis_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le stacking doit spécifier axis=0 pour garantir l'axe bandes stable."""

    # Construit un signal canal minimal pour déclencher le calcul
    channel_values = np.arange(8, dtype=float)
    # Définit deux fréquences centrales pour produire deux bandes
    central_frequencies = [10.0, 20.0]
    # Fixe la fréquence d'échantillonnage utilisée pour les wavelets
    sfreq = 100.0
    # Fixe le nombre de cycles pour stabiliser la largeur temporelle
    wavelet_cycles = 6.0

    # Sauvegarde l'implémentation numpy réelle pour l'appeler depuis le spy
    original_stack = features_module.np.stack

    def _spy_stack(*args: Any, **kwargs: Any) -> np.ndarray:
        """Capture l'argument axis pour détecter sa présence explicite."""

        # Détecte un axis positional pour couvrir les futurs refactors
        axis_provided = len(args) >= 2
        # Extrait l'axis positional quand il est présent
        axis_value: Any = args[1] if axis_provided else None
        # Détecte un axis keyword pour la signature actuelle
        if "axis" in kwargs:
            # Force la présence de l'axis pour éviter un mutant équivalent
            axis_provided = True
            # Extrait l'axis keyword pour valider la dimension band
            axis_value = kwargs["axis"]
        # Échoue si l'axis n'est pas explicitement fourni à np.stack
        assert axis_provided, "np.stack doit recevoir axis explicitement."
        # Impose l'axe 0 pour conserver la matrice (bandes, temps)
        assert axis_value == 0
        # Délègue à numpy pour produire un résultat réel exploitable
        return original_stack(*args, **kwargs)

    # Espionne np.stack pour tuer les mutants supprimant axis=0
    monkeypatch.setattr(features_module.np, "stack", _spy_stack)

    def _fake_fftconvolve(
        in1: np.ndarray,
        in2: np.ndarray,
        mode: str = "same",
    ) -> np.ndarray:
        """Retourne le signal tel quel pour stabiliser la convolution."""

        # Retourne un complexe pour rester compatible avec les wavelets
        return in1.astype(complex)

    # Neutralise scipy pour rendre le test déterministe et rapide
    monkeypatch.setattr(features_module.signal, "fftconvolve", _fake_fftconvolve)

    # Calcule les coefficients pour déclencher l'empilement final
    coeffs = features_module._compute_wavelet_coefficients(
        channel_values,
        central_frequencies,
        sfreq,
        wavelet_cycles,
    )

    # Vérifie la forme attendue (bandes, temps) du tenseur final
    assert coeffs.shape == (2, 8)


def test_compute_wavelet_coefficients_builds_expected_morlet_wavelets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_wavelet_coefficients doit construire une Morlet centrée attendue."""

    # Fixe une fréquence d'échantillonnage standard pour l'EEG
    sfreq = 64.0
    # Fixe une largeur de wavelet classique pour la Morlet
    wavelet_cycles = 6.0
    # Choisit une longueur impaire pour obtenir un centre exact à t=0
    n_times = 11
    # Utilise un signal nul car on ne teste ici que la wavelet générée
    channel_values = np.zeros(n_times, dtype=float)
    # Inclut une fréquence nulle pour verrouiller le plancher 1e-9
    central_frequencies = [0.0, 10.0]

    # Capture les wavelets envoyées à fftconvolve pour vérification exacte
    captured_wavelets: list[np.ndarray] = []
    # Capture le mode de convolution attendu pour préserver la longueur
    captured_modes: list[str] = []

    def _fake_fftconvolve(
        x: np.ndarray,
        wavelet: np.ndarray,
        *,
        mode: str,
    ) -> np.ndarray:
        # Stocke la wavelet telle qu'elle est construite par la fonction
        captured_wavelets.append(np.asarray(wavelet))
        # Stocke le mode pour vérifier l'alignement demandé
        captured_modes.append(mode)
        # Retourne un résultat de forme correcte pour laisser la fonction finir
        return cast(np.ndarray, np.zeros_like(x, dtype=complex))

    # Remplace fftconvolve pour observer la wavelet sans dépendre du calcul FFT
    monkeypatch.setattr(features_module.signal, "fftconvolve", _fake_fftconvolve)

    # Calcule les coefficients en déclenchant la construction des wavelets
    coeffs = features_module._compute_wavelet_coefficients(
        channel_values=channel_values,
        central_frequencies=central_frequencies,
        sfreq=sfreq,
        wavelet_cycles=wavelet_cycles,
    )

    # Vérifie que la forme (bandes, temps) est respectée
    assert coeffs.shape == (len(central_frequencies), n_times)
    # Vérifie que l'implémentation demande une convolution "same"
    assert captured_modes == ["same", "same"]
    # Vérifie que la wavelet est construite pour chaque fréquence centrale
    assert len(captured_wavelets) == len(central_frequencies)

    # Reproduit l'axe temporel centré attendu par la spécification actuelle
    centered_times = np.arange(n_times) - (n_times - 1) / 2
    # Fixe l'index central qui doit correspondre à t=0
    center_index = int((n_times - 1) / 2)

    for built_wavelet, central_frequency in zip(captured_wavelets, central_frequencies):
        # Verrouille le plancher numérique pour éviter les divisions par zéro
        safe_frequency = max(central_frequency, 1e-9)
        # Verrouille la formule sigma attendue (cycles * sfreq / f)
        sigma = wavelet_cycles * sfreq / safe_frequency
        # Verrouille l'enveloppe gaussienne centrée (t^2 / (2*sigma^2))
        envelope = np.exp(-(centered_times**2) / (2 * sigma**2))
        # Verrouille l'oscillation complexe e^{j 2π f t / sfreq}
        oscillation = np.exp(2j * np.pi * safe_frequency * centered_times / sfreq)
        # Verrouille la combinaison attendue (produit, pas division)
        expected_wavelet = envelope * oscillation

        # Compare la wavelet complexe complète (tue les mutations de phase)
        assert np.allclose(built_wavelet, expected_wavelet, rtol=1e-12, atol=1e-12)
        # Verrouille que le centre vaut 1+0j (enveloppe=1 et oscillation=1)
        assert built_wavelet[center_index] == pytest.approx(1.0 + 0.0j)



def test_extract_features_wavelet_rejects_unknown_wavelet_name() -> None:
    """La configuration doit remonter une erreur pour les wavelets non supportées."""

    # Prépare des epochs synthétiques pour alimenter la fonction
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    # Vérifie qu'une wavelet inconnue est explicitement rejetée
    with pytest.raises(ValueError, match="Unsupported wavelet"):
        extract_features(epochs, config={"method": "wavelet", "wavelet": "mexican_hat"})


def test_compute_wavelet_features_rejects_mismatched_band_power_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La validation doit refuser une forme incohérente des puissances wavelet."""

    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy="wavelet",
        normalize=False,
        bands={"alpha": (8.0, 12.0), "beta": (13.0, 30.0)},
    )

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,
        band_ranges: Mapping[str, tuple[float, float]],
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        return np.zeros((data.shape[0], data.shape[1], 1))

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    with pytest.raises(ValueError, match="expected 2 bands but received 1"):
        extractor.transform(epochs.get_data())


def test_compute_wavelet_features_rejects_invalid_band_power_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La validation doit refuser des coefficients qui perdent des axes nécessaires."""

    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    extractor = ExtractFeatures(sfreq=64.0, feature_strategy="wavelet", normalize=False)

    def fake_band_powers(
        data: np.ndarray,
        sfreq: float,
        band_ranges: Mapping[str, tuple[float, float]],
        config: Mapping[str, Any],  # noqa: ARG001,E501
    ) -> np.ndarray:
        return np.zeros((data.shape[0], data.shape[1]))  # perdu la dimension bande

    monkeypatch.setattr(
        features_module, "_compute_wavelet_band_powers", fake_band_powers
    )

    with pytest.raises(ValueError, match="must return a 3D array"):
        extractor.transform(epochs.get_data())


def test_extract_features_returns_zeros_when_band_mask_empty() -> None:
    """Une bande hors spectre doit produire des puissances nulles."""

    # Prépare des epochs avec une fréquence d'échantillonnage limitée
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    # Demande une bande trop haute pour être couverte par la FFT
    features, labels = extract_features(
        epochs, config={"method": "welch", "bands": [("void", (100.0, 120.0))]}
    )
    # Vérifie que la bande inexistante génère uniquement des zéros
    assert np.array_equal(features, np.zeros((1, 1)))
    # Vérifie que l'étiquette reflète la bande personnalisée
    assert labels == ["C0_void"]

def test_compute_welch_band_powers_forwards_sanitized_parameters_to_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_welch_band_powers doit forwarder la config Welch à SciPy."""

    # Fixe une sfreq non triviale pour détecter la perte de l'argument fs.
    expected_sfreq = 128.0
    # Prépare une forme (epochs, channels, times) non ambiguë.
    data = np.random.default_rng(seed=11).standard_normal((2, 3, 128))
    # Force des valeurs non défaut pour rendre chaque omission détectable.
    config: dict[str, object] = {
        "window": "boxcar",
        "nperseg": 64,
        "noverlap": 16,
        "average": "median",
        "scaling": "spectrum",
    }
    # Cible une bande qui inclut 10 Hz dans le fake freqs.
    band_ranges = {"alpha": (8.0, 12.0)}
    # Prépare un conteneur pour vérifier l'appel.
    captured: dict[str, object] = {}

    def _fake_welch(x: np.ndarray, *args: object, **kwargs: object):
        # Capture args/kwargs pour diagnostiquer les régressions.
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Fournit une grille fréquentielle contrôlée et stable.
        freqs = np.array([0.0, 10.0, 20.0])
        # Fournit une PSD non nulle pour détecter les masques cassés.
        psd = np.full((x.shape[0], x.shape[1], freqs.size), 3.0)
        return freqs, psd

    # Remplace SciPy Welch pour rendre l'appel observable.
    monkeypatch.setattr(features_module.signal, "welch", _fake_welch)
    # Exécute la fonction testée avec la configuration imposée.
    powers = features_module._compute_welch_band_powers(
        data,
        sfreq=expected_sfreq,
        band_ranges=band_ranges,
        config=config,
    )

    # Vérifie que fs est bien transmis (mutmut_19).
    assert tuple(captured["args"])[0] == pytest.approx(expected_sfreq)
    # Vérifie que window est bien transmis (mutmut_20).
    assert dict(captured["kwargs"])["window"] == "boxcar"
    # Vérifie que nperseg provient de n_times (mutmut_2, mutmut_3, mutmut_13, 21).
    assert dict(captured["kwargs"])["nperseg"] == 64
    # Vérifie que noverlap est bien transmis (mutmut_14, 22).
    assert dict(captured["kwargs"])["noverlap"] == 16
    # Verrouille l’axe temporel explicitement (mutmut_23).
    assert dict(captured["kwargs"])["axis"] == -1
    # Vérifie que average est bien transmis (mutmut_24).
    assert dict(captured["kwargs"])["average"] == "median"
    # Vérifie que scaling est bien transmis (mutmut_25).
    assert dict(captured["kwargs"])["scaling"] == "spectrum"
    # Vérifie que le masque de bande produit une énergie non nulle (mutmut_29, 34).
    assert powers.shape == (2, 3, 1)
    assert np.all(powers > 0.0)


def test_compute_welch_band_powers_includes_low_edge_frequency_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le masque doit inclure la borne basse (>= low) quand freqs == low."""

    # Prépare une entrée minimale pour déclencher le calcul.
    data = np.zeros((1, 1, 16))
    # Force un nperseg valide pour éviter les chemins implicites.
    config: dict[str, object] = {"nperseg": 16}
    # Cible une bande dont la borne basse tombe exactement sur freqs.
    band_ranges = {"edge_low": (10.0, 10.0001)}

    def _fake_welch(_x: np.ndarray, *_: object, **__: object):
        # Force une unique fréquence exactement sur la borne basse.
        freqs = np.array([10.0])
        # Force une PSD non nulle uniquement sur ce bin.
        psd = np.array([[[7.0]]])
        return freqs, psd

    # Remplace SciPy Welch pour contrôler freqs/psd.
    monkeypatch.setattr(features_module.signal, "welch", _fake_welch)
    # Calcule les puissances de bande sur la grille contrôlée.
    powers = features_module._compute_welch_band_powers(
        data,
        sfreq=100.0,
        band_ranges=band_ranges,
        config=config,
    )
    # La borne basse inclusive doit capter le bin à 10 Hz (mutmut_31).
    assert powers.shape == (1, 1, 1)
    assert powers[0, 0, 0] == pytest.approx(7.0)


def test_compute_welch_band_powers_includes_high_edge_frequency_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le masque doit inclure la borne haute (<= high) quand freqs == high."""

    # Prépare une entrée minimale pour déclencher le calcul.
    data = np.zeros((1, 1, 16))
    # Force un nperseg valide pour éviter les chemins implicites.
    config: dict[str, object] = {"nperseg": 16}
    # Cible une bande dont la borne haute tombe exactement sur freqs.
    band_ranges = {"edge_high": (9.9999, 10.0)}

    def _fake_welch(_x: np.ndarray, *_: object, **__: object):
        # Force une unique fréquence exactement sur la borne haute.
        freqs = np.array([10.0])
        # Force une PSD non nulle uniquement sur ce bin.
        psd = np.array([[[7.0]]])
        return freqs, psd

    # Remplace SciPy Welch pour contrôler freqs/psd.
    monkeypatch.setattr(features_module.signal, "welch", _fake_welch)
    # Calcule les puissances de bande sur la grille contrôlée.
    powers = features_module._compute_welch_band_powers(
        data,
        sfreq=100.0,
        band_ranges=band_ranges,
        config=config,
    )
    # La borne haute inclusive doit capter le bin à 10 Hz (mutmut_32).
    assert powers.shape == (1, 1, 1)
    assert powers[0, 0, 0] == pytest.approx(7.0)


def test_compute_welch_features_flattens_with_negative_one_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_welch_features doit aplatir via reshape(..., -1) uniquement."""

    # Fixe une forme 3D cohérente avec 2 epochs, 3 canaux, 4 bandes
    stacked_shape = (2, 3, 4)

    class _StackedSpy:
        """Spy minimal qui échoue si reshape n'utilise pas -1."""

        # Expose la forme attendue pour reproduire un tenseur (epochs, channels, bands)
        shape = stacked_shape

        def reshape(self, n_epochs: int, n_features: int) -> np.ndarray:
            """Valide les arguments reshape et retourne une matrice plate."""

            # Verrouille que l'aplatissement conserve le nombre d'epochs
            assert n_epochs == stacked_shape[0]
            # Interdit toute dimension négative hors -1 (tue mutmut_16)
            assert n_features == -1
            # Retourne une matrice cohérente (epochs, channels * bands)
            return np.zeros(
                (n_epochs, stacked_shape[1] * stacked_shape[2]),
                dtype=float,
            )

    def _fake_welch_band_powers(
        _X: np.ndarray,
        _sfreq: float,
        _band_ranges: dict[str, tuple[float, float]],
        _config: dict[str, object],
    ) -> _StackedSpy:
        """Retourne un tenseur spy pour isoler l'assertion reshape."""

        # Retourne un spy pour capturer l'appel reshape de _compute_welch_features
        return _StackedSpy()

    # Remplace le calcul Welch pour ne tester que l'aplatissement final
    monkeypatch.setattr(
        features_module,
        "_compute_welch_band_powers",
        _fake_welch_band_powers,
    )

    # Prépare une entrée valide qui déclenche _compute_welch_features
    X = np.zeros((stacked_shape[0], stacked_shape[1], 8), dtype=float)
    # Construit un extracteur configuré sur la stratégie Welch
    extractor = ExtractFeatures(
        sfreq=128.0,
        feature_strategy="welch",
        normalize=False,
    )

    # Exécute transform pour appeler _compute_welch_features puis reshape spy
    transformed = extractor.transform(X)

    # Vérifie que l'aplatissement produit bien (epochs, channels * bands)
    assert transformed.shape == (
        stacked_shape[0],
        stacked_shape[1] * stacked_shape[2],
    )


def test_extract_features_wrapper_rejects_unknown_strategy() -> None:
    """La classe scikit-learn doit refuser les stratégies non supportées."""

    # Construit un tenseur conforme pour l'appel
    tensor = np.zeros((1, 2, 16))
    # Instancie un extracteur avec une stratégie inconnue
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="invalid")
    # Vérifie que l'appelant reçoit une erreur explicite
    with pytest.raises(ValueError):
        extractor.transform(tensor)


def test_extract_features_constructor_defaults() -> None:
    """Le constructeur doit appliquer les valeurs par défaut documentées."""

    # Instancie l'extracteur avec uniquement la fréquence d'échantillonnage
    extractor = ExtractFeatures(sfreq=128.0)
    # Vérifie que la fréquence est stockée en flottant pour les calculs
    assert extractor.sfreq == pytest.approx(128.0)
    # Vérifie que la stratégie par défaut reste la FFT
    assert extractor.feature_strategy == "fft"
    # Vérifie que la normalisation est activée par défaut
    assert extractor.normalize is True
    # Vérifie que les bandes par défaut couvrent les quatre intervalles attendus
    assert extractor.band_labels == ["theta", "alpha", "beta", "gamma"]


def test_extract_features_constructor_stores_custom_attributes() -> None:
    """Le constructeur doit persister les attributs fournis par l'appelant."""

    # Instancie l'extracteur avec une stratégie et une normalisation custom
    extractor = ExtractFeatures(
        sfreq=256.0, feature_strategy="wavelet", normalize=False
    )
    # Vérifie que la fréquence d'échantillonnage reste fidèle à l'entrée
    assert extractor.sfreq == pytest.approx(256.0)
    # Vérifie que la stratégie demandée est bien conservée
    assert extractor.feature_strategy == "wavelet"
    # Vérifie que la désactivation de la normalisation est persistée
    assert extractor.normalize is False
    # Vérifie que la liste des bandes reste alignée avec la constante
    assert extractor.band_labels == ["theta", "alpha", "beta", "gamma"]


def test_extract_features_fit_behaves_like_transformer() -> None:
    """fit doit suivre le contrat scikit-learn et laisser l'état cohérent."""

    extractor = ExtractFeatures(sfreq=128.0, normalize=False)
    data = np.zeros((1, 2, 8))

    fitted = extractor.fit(data)

    assert fitted is extractor
    transformed = extractor.transform(data)
    assert transformed.shape == (1, 8)
    assert extractor.band_labels == ["theta", "alpha", "beta", "gamma"]


def test_extract_features_constructor_exposes_bands_and_strategy_config() -> None:
    """Le constructeur doit exposer bands/strategy_config pour get_params."""

    # Fixe une fréquence valide pour éviter les variations de cast
    sfreq = 128.0
    # Prépare des bandes custom pour détecter une perte d'attribut
    custom_bands = OrderedDict(
        [
            ("alpha", (8.0, 12.0)),
            ("beta", (13.0, 30.0)),
        ]
    )
    # Prépare une config custom pour détecter une perte d'attribut
    strategy_config = {
        "wavelet_width": 5.0,
        "nperseg": 64,
    }

    # Instancie l'extracteur avec des paramètres non-default
    extractor = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy="welch",
        normalize=False,
        bands=custom_bands,
        strategy_config=strategy_config,
    )

    # Vérifie que bands reste exposé tel quel (contrat get_params)
    assert extractor.bands is custom_bands
    # Vérifie que strategy_config reste exposé tel quel (contrat get_params)
    assert extractor.strategy_config is strategy_config

    # Récupère les paramètres scikit-learn pour valider l'exposition
    params = extractor.get_params(deep=False)
    # Vérifie que get_params relaie bands sans le perdre
    assert params["bands"] is custom_bands
    # Vérifie que get_params relaie strategy_config sans le perdre
    assert params["strategy_config"] is strategy_config


def test_extract_features_wrapper_rejects_incorrect_shape() -> None:
    """La classe scikit-learn doit refuser une dimension d'entrée erronée."""

    # Prépare un extracteur pour tester la validation de dimension
    extractor = ExtractFeatures(sfreq=128.0)
    # Prépare un tableau bidimensionnel pour déclencher la validation
    bad_shape = np.zeros((2, 16))
    # Vérifie que la méthode transforme lève une erreur de forme
    with pytest.raises(ValueError):
        extractor.transform(bad_shape)


def test_extract_features_transform_rejects_extra_dimensions() -> None:
    """La méthode transform doit refuser les tenseurs de rang supérieur."""

    extractor = ExtractFeatures(sfreq=128.0)
    too_many_dims = np.zeros((1, 2, 3, 4))

    with pytest.raises(
        ValueError,
        match=r"^X must have shape \(n_samples, n_channels, n_times\)$",
    ):
        extractor.transform(too_many_dims)


def test_extract_features_transform_rejects_empty_epochs() -> None:
    """Une extraction sans epoch doit remonter une erreur explicite."""

    extractor = ExtractFeatures(sfreq=128.0)
    empty_epochs = np.zeros((0, 2, 4))

    with pytest.raises(
        ValueError,
        match=r"^X must contain at least one epoch\.$",
    ):        extractor.transform(empty_epochs)


def test_extract_features_transform_normalizes_per_epoch_with_eps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """normalize=True doit normaliser par epoch (axis=1) avec eps et (raw-mean)/std."""

    # Force normalize pour couvrir le chemin de normalisation
    extractor = ExtractFeatures(sfreq=128.0, normalize=True)
    # Fournit un X valide (ndim=3, n_epochs>0) pour passer les gardes
    X = np.zeros((2, 1, 1), dtype=float)

    # Construit des features brutes avec std=1e-12 par epoch
    # (ainsi, eps=1e-12 est déterminant et tue le mutant "- eps")
    raw_features = np.array(
        [
            [0.0, 2e-12],
            [3e-12, 5e-12],
        ],
        dtype=float,
    )

    # Remplace le calcul réel pour isoler la logique de normalisation
    def _fake_compute_features(_: np.ndarray) -> np.ndarray:
        return raw_features

    monkeypatch.setattr(extractor, "_compute_features", _fake_compute_features)

    # Calcule la sortie normalisée
    features = extractor.transform(X)

    # Reconstruit la normalisation attendue (par epoch, axis=1)
    mean = raw_features.mean(axis=1, keepdims=True)
    std = raw_features.std(axis=1, keepdims=True) + extractor.NORMALIZATION_EPS
    expected = (raw_features - mean) / std

    # Verrouille la formule exacte (tue axis=None / mean global / +mean / -eps)
    np.testing.assert_allclose(features, expected, rtol=0.0, atol=1e-12)


def test_extract_features_wrapper_rejects_unknown_method() -> None:
    """La fonction procédurale doit refuser une méthode non supportée."""

    # Prépare des epochs minimaux pour l'appel procédural
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Vérifie que la fonction signale la méthode inconnue
    with pytest.raises(ValueError):
        extract_features(epochs, config={"method": "invalid"})


def test_extract_features_transform_preserves_band_order_deterministically() -> None:
    """L'ordre des bandes doit rester déterministe dans transform."""

    sfreq = 64.0
    times = np.arange(0.0, 1.0, 1.0 / sfreq)
    theta_signal = np.sin(2 * np.pi * 6.0 * times)
    data = theta_signal.reshape(1, 1, -1)
    ordered_bands = OrderedDict(
        [
            ("gamma", (31.0, 45.0)),
            ("theta", (4.0, 7.0)),
        ]
    )
    extractor = ExtractFeatures(
        sfreq=sfreq, feature_strategy="fft", normalize=False, bands=ordered_bands
    )

    transformed_first = extractor.transform(data)
    transformed_second = extractor.transform(data.copy())

    assert np.array_equal(transformed_first, transformed_second)
    reshaped = transformed_first.reshape(1, 1, 2)
    assert reshaped[0, 0, 1] == pytest.approx(np.max(reshaped[0, 0]))


def test_extract_features_rejects_empty_band_configuration() -> None:
    """Une configuration sans bandes doit être rejetée explicitement."""

    # Prépare des epochs synthétiques pour déclencher la validation
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Vérifie que l'absence de bandes lève une erreur explicite
    with pytest.raises(ValueError, match=r"^At least one frequency band must be provided\.$"):
        extract_features(epochs, config={"method": "welch", "bands": []})


def test_extract_features_numeric_stability() -> None:
    """Welch doit produire des valeurs finies et non négatives."""

    # Prépare un bruit blanc pour sonder la stabilité numérique
    epochs = _build_epochs(n_epochs=2, n_channels=1, n_times=512, sfreq=256.0)
    # Exécute l'extraction pour récupérer les PSD de bande
    features, _ = extract_features(epochs, config={"method": "welch"})
    # Vérifie l'absence de NaN ou d'infini
    assert np.isfinite(features).all()
    # Vérifie l'absence de valeurs négatives après la moyenne de puissance
    assert (features >= 0).all()


def test_extract_features_respects_time_budget() -> None:
    """L'extraction Welch doit rester sous un budget temps raisonnable."""

    # Construit des epochs plus longs pour sonder la performance temporelle
    epochs = _build_epochs(n_epochs=5, n_channels=4, n_times=512, sfreq=256.0)
    # Mesure l'instant initial pour contrôler la durée d'extraction
    start = perf_counter()
    # Lance l'extraction avec une configuration Welch recouvrante
    features, labels = extract_features(
        epochs, config={"method": "welch", "nperseg": 256, "noverlap": 128}
    )
    # Calcule la durée écoulée pour valider le budget temporel
    elapsed = perf_counter() - start
    # Vérifie que le nombre de colonnes correspond bien aux étiquettes
    assert features.shape[1] == len(labels)
    # Vérifie que l'extraction reste sous un quart de seconde
    assert elapsed < MAX_EXTRACTION_SECONDS


def test_prepare_welch_parameters_caps_segment_and_overlap() -> None:
    """Les bornes doivent limiter la taille de fenêtre et le recouvrement."""

    # Prépare une configuration qui dépasse la longueur disponible
    config = {
        "window": "flattop",
        "nperseg": 128,
        "noverlap": 127,
        "average": "median",
        "scaling": "spectrum",
    }
    # Calcule les paramètres effectifs pour une série courte
    window, effective_nperseg, effective_noverlap, average, scaling = (
        _prepare_welch_parameters(config, n_times=64)
    )
    # Vérifie que la fenêtre demandée est transmise intacte
    assert window == "flattop"
    # Vérifie que la taille de segment est bornée par la durée réelle
    # Fixe la borne maximale pour rendre l'assertion lisible
    expected_nperseg = 64
    # Garantit que la fenêtre ne dépasse pas la longueur totale
    assert effective_nperseg == expected_nperseg
    # Vérifie que le recouvrement est borné à une fenêtre strictement positive
    # Fixe le recouvrement maximal autorisé juste sous la fenêtre
    expected_noverlap = 63
    # Garantit que le recouvrement garde une fenêtre strictement positive
    assert effective_noverlap == expected_noverlap
    # Vérifie que la stratégie d'agrégation personnalisée est préservée
    assert average == "median"
    # Vérifie que l'option de mise à l'échelle personnalisée est préservée
    assert scaling == "spectrum"


def test_prepare_welch_parameters_defaults_when_overlap_missing() -> None:
    """Les valeurs par défaut doivent être utilisées sans recouvrement fourni."""

    # Prépare une configuration minimale pour sonder les valeurs implicites
    config: dict[str, object] = {}
    # Calcule les paramètres avec une longueur limitée et aucun recouvrement
    (
        window,
        effective_nperseg,
        effective_noverlap,
        average,
        scaling,
    ) = _prepare_welch_parameters(config, n_times=50)
    # Vérifie que la fenêtre par défaut est la fenêtre Hann lissée
    assert window == "hann"
    # Vérifie que la taille de segment par défaut couvre toute la série
    # Stocke la longueur maximale pour l'utiliser dans l'assertion
    default_length = DEFAULT_WELCH_LENGTH
    # Confirme que la fenêtre par défaut couvre toute la série disponible
    assert effective_nperseg == default_length
    # Vérifie qu'aucun recouvrement n'est défini sans instruction explicite
    assert effective_noverlap is None
    # Vérifie que la moyenne par défaut correspond à l'option SciPy standard
    assert average == "mean"
    # Vérifie que l'échelle par défaut correspond à la densité spectrale
    assert scaling == "density"


def test_prepare_welch_parameters_rejects_invalid_window_and_lengths() -> None:
    """Les paramètres Welch doivent valider fenêtres et tailles fournies."""

    with pytest.raises(
        ValueError,
        match=r"^Welch window name must be a non-empty string\.$",
    ):
        _prepare_welch_parameters({"window": EMPTY_WINDOW}, n_times=32)
    with pytest.raises(
        ValueError,
        match=r"^Welch nperseg must be an integer or None\.$",
    ):
        _prepare_welch_parameters({"nperseg": NON_INTEGER_NPERSEG}, n_times=64)


def test_prepare_welch_parameters_preserves_minimum_positive_nperseg() -> None:
    """nperseg==1 doit rester valide pour éviter un gonflement silencieux."""

    config = {"nperseg": 1, "noverlap": 0}

    _, effective_nperseg, effective_noverlap, _, _ = _prepare_welch_parameters(
        config, n_times=8
    )

    assert effective_nperseg == 1
    assert effective_noverlap == 0


def test_prepare_welch_parameters_replaces_non_positive_nperseg() -> None:
    """nperseg non positif doit basculer sur la longueur totale disponible."""

    # Prépare des configurations pathologiques avec nperseg nul ou négatif
    configs = [{"nperseg": 0}, {"nperseg": -16}]
    # Calcule les paramètres effectifs pour une longueur fixe
    for config in configs:
        _, effective_nperseg, effective_noverlap, _, _ = _prepare_welch_parameters(
            config, n_times=DEFAULT_WELCH_LENGTH
        )
        # Vérifie que la taille de fenêtre est restaurée à la longueur totale
        assert effective_nperseg == DEFAULT_WELCH_LENGTH
        # Vérifie qu'en l'absence de recouvrement valide, la valeur reste None
        assert effective_noverlap is None


def test_prepare_welch_parameters_caps_overlap_above_window() -> None:
    """Le recouvrement doit être borné par la taille effective de fenêtre."""

    # Prépare un cas où le recouvrement dépasse largement la fenêtre
    config = {"nperseg": NPERSEG_WITH_OVERLAP, "noverlap": EXCESSIVE_OVERLAP}
    _, effective_nperseg, effective_noverlap, _, _ = _prepare_welch_parameters(
        config, n_times=40
    )
    # Vérifie que nperseg respecte la borne la plus restrictive
    assert effective_nperseg == NPERSEG_WITH_OVERLAP
    # Vérifie que le recouvrement est borné juste sous la fenêtre
    assert effective_noverlap == EXPECTED_OVERLAP_CAP
    # Vérifie qu'un recouvrement négatif est remis à zéro
    _, _, non_negative_overlap, _, _ = _prepare_welch_parameters(
        {"nperseg": NPERSEG_WITH_OVERLAP, "noverlap": NEGATIVE_OVERLAP}, n_times=40
    )
    assert non_negative_overlap == 0
    # Vérifie qu'un recouvrement non entier est explicitement rejeté
    with pytest.raises(
        ValueError,
        match=r"^Welch noverlap must be a non-negative integer or None\.$",
    ):        _prepare_welch_parameters(
            {"nperseg": NPERSEG_WITH_OVERLAP, "noverlap": NON_INTEGER_OVERLAP},
            n_times=40,
        )
    # Vérifie qu'un recouvrement égal à la fenêtre
    # laisse une fenêtre strictement positive
    _, _, overlap_equal_window, _, _ = _prepare_welch_parameters(
        {"nperseg": NPERSEG_WITH_OVERLAP, "noverlap": NPERSEG_WITH_OVERLAP},
        n_times=NPERSEG_WITH_OVERLAP,
    )
    assert overlap_equal_window == EXPECTED_OVERLAP_CAP


def test_extract_features_delegates_to_sklearn_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_features doit instancier ExtractFeatures et réutiliser son transform."""

    # Prépare des epochs minimaux pour forcer le chemin de délégation
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=8, sfreq=64.0)
    captured_init: dict[str, object] = {}
    captured_data: dict[str, np.ndarray] = {}

    class DummyExtractor:
        def __init__(
            self,
            *,
            sfreq: float,
            feature_strategy: str,
            normalize: bool,
            bands: dict[str, tuple[float, float]],
            strategy_config: dict[str, object],
        ) -> None:
            captured_init["init"] = {
                "sfreq": sfreq,
                "feature_strategy": feature_strategy,
                "normalize": normalize,
                "bands": bands,
                "strategy_config": strategy_config,
            }
            self._bands = bands

        def transform(self, data: np.ndarray) -> np.ndarray:
            captured_data["data"] = np.asarray(data)
            return np.full((data.shape[0], data.shape[1] * len(self._bands)), 2.5)

    monkeypatch.setattr("tpv.features.ExtractFeatures", DummyExtractor)

    features, labels = extract_features(epochs, config={"method": "welch"})

    assert captured_init["init"] == {
        "sfreq": pytest.approx(64.0),
        "feature_strategy": "welch",
        "normalize": False,
        "bands": {
            "theta": (4.0, 7.0),
            "alpha": (8.0, 12.0),
            "beta": (13.0, 30.0),
            "gamma": (31.0, 45.0),
        },
        "strategy_config": {},
    }
    assert np.array_equal(captured_data["data"], epochs.get_data())
    assert np.array_equal(features, np.full((1, 4), 2.5))
    assert labels == ["C0_theta", "C0_alpha", "C0_beta", "C0_gamma"]


def test_extract_features_default_method_is_lowercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La stratégie par défaut doit rester normalisée pour l'API interne."""

    # Prépare des epochs synthétiques pour déclencher le chemin par défaut
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=8, sfreq=64.0)
    captured_init: dict[str, object] = {}

    class DummyExtractor:
        def __init__(
            self,
            *,
            sfreq: float,
            feature_strategy: str,
            normalize: bool,
            bands: Mapping[str, tuple[float, float]],
            strategy_config: Mapping[str, object],
        ) -> None:
            """Capture l'initialisation sans appliquer de normalisation implicite."""

            # Capture la stratégie pour vérifier la valeur transmise par défaut
            captured_init["feature_strategy"] = feature_strategy
            # Stocke la configuration des bandes pour dimensionner la sortie
            self._bands = bands

        def transform(self, data: np.ndarray) -> np.ndarray:
            """Retourne une matrice compatible avec le nombre de features attendues."""

            # Calcule le nombre de features à partir des canaux et des bandes
            n_features = data.shape[1] * len(self._bands)
            # Retourne une matrice n_epochs x n_features stable et déterministe
            return np.zeros((data.shape[0], n_features), dtype=float)

    monkeypatch.setattr(features_module, "ExtractFeatures", DummyExtractor)

    # Appelle sans méthode pour vérifier la valeur par défaut transmise
    extract_features(epochs, config={})

    assert captured_init["feature_strategy"] == "welch"



def test_extract_features_requests_list_default_for_channel_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Le fallback sur les noms de canaux doit rester une liste vide explicite."""

    import builtins

    # Construit des epochs MNE pour exposer une info standardisée
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=8, sfreq=64.0)
    sentinel = object()
    seen_default: dict[str, object] = {}

    real_getattr = builtins.getattr

    def spy_getattr(obj: object, name: str, default: object = sentinel) -> object:
        """Intercepte l'appel ciblé sans casser les autres getattr."""

        # Capture uniquement la résolution des noms de canaux
        if obj is epochs.info and name == "ch_names":
            seen_default["default"] = default
        # Respecte la signature standard lorsqu'aucun défaut n'est fourni
        if default is sentinel:
            return real_getattr(obj, name)
        # Délègue à getattr avec valeur par défaut explicitée
        return real_getattr(obj, name, default)

    monkeypatch.setattr(builtins, "getattr", spy_getattr)

    class DummyExtractor:
        def __init__(
            self,
            *,
            sfreq: float,
            feature_strategy: str,
            normalize: bool,
            bands: Mapping[str, tuple[float, float]],
            strategy_config: Mapping[str, object],
        ) -> None:
            """Ignore le contenu et retourne une matrice minimale."""

            # Stocke les bandes pour dimensionner correctement la sortie
            self._bands = bands

        def transform(self, data: np.ndarray) -> np.ndarray:
            """Retourne une matrice cohérente pour permettre l'étiquetage."""

            # Calcule le nombre de colonnes à partir des canaux et des bandes
            n_features = data.shape[1] * len(self._bands)
            # Retourne une matrice déterministe pour stabiliser les assertions
            return np.zeros((data.shape[0], n_features), dtype=float)

    monkeypatch.setattr(features_module, "ExtractFeatures", DummyExtractor)

    # Lance l'extraction pour déclencher la capture de getattr sur ch_names
    extract_features(epochs, config={"method": "welch"})

    default_value = seen_default["default"]

    assert isinstance(default_value, list)
    assert default_value == []


def test_extract_features_propagates_normalize_and_strips_control_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """normalize/bands/method ne doivent pas fuiter dans strategy_config."""

    # Prépare des epochs synthétiques pour exercer la lecture de la config
    epochs = _build_epochs(n_epochs=1, n_channels=2, n_times=16, sfreq=64.0)
    captured_init: dict[str, object] = {}

    class DummyExtractor:
        def __init__(
            self,
            *,
            sfreq: float,
            feature_strategy: str,
            normalize: bool,
            bands: Mapping[str, tuple[float, float]],
            strategy_config: Mapping[str, object],
        ) -> None:
            """Capture les paramètres transmis pour valider l'assemblage."""

            # Capture les paramètres pour vérifier la propagation exacte
            captured_init["sfreq"] = sfreq
            captured_init["feature_strategy"] = feature_strategy
            captured_init["normalize"] = normalize
            captured_init["bands"] = dict(bands)
            captured_init["strategy_config"] = dict(strategy_config)

        def transform(self, data: np.ndarray) -> np.ndarray:
            """Retourne une matrice conforme pour permettre l'étiquetage."""

            # Calcule le nombre de features à partir des canaux et des bandes
            n_features = data.shape[1] * len(cast(dict[str, tuple[float, float]], captured_init["bands"]))
            # Renvoie une sortie non nulle pour valider le passage de données
            return np.full((data.shape[0], n_features), 1.0, dtype=float)

    monkeypatch.setattr(features_module, "ExtractFeatures", DummyExtractor)

    # Définit une configuration riche pour vérifier le nettoyage des champs
    config: dict[str, object] = {
        "method": "welch",
        "normalize": True,
        "bands": [("mu", (8.0, 12.0))],
        "nperseg": 16,
    }

    features, labels = extract_features(epochs, config=config)

    assert captured_init["sfreq"] == pytest.approx(64.0)
    assert captured_init["feature_strategy"] == "welch"
    assert captured_init["normalize"] is True
    assert captured_init["bands"] == {"mu": (8.0, 12.0)}
    assert captured_init["strategy_config"] == {"nperseg": 16}
    assert features.shape == (1, 2)
    assert labels == ["C0_mu", "C1_mu"]


def test_extract_features_supports_epochs_without_channel_names_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'absence de ch_names doit tomber sur une stratégie de nommage stable."""

    # Prépare un objet epochs minimal sans attribut ch_names sur info
    data = np.ones((1, 1, 8), dtype=float)

    class FakeEpochs:
        def __init__(self, data_: np.ndarray, sfreq: float) -> None:
            """Expose uniquement l'API utilisée par extract_features."""

            # Stocke les données pour simuler Epochs.get_data
            self._data = data_
            # Expose sfreq via un mapping sans fournir ch_names
            self.info = {"sfreq": sfreq}

        def get_data(self) -> np.ndarray:
            """Retourne les données brutes pour extraire les features."""

            # Retourne la vue directe pour garder un test simple
            return self._data

    captured_labels: dict[str, list[str]] = {}

    class DummyExtractor:
        def __init__(
            self,
            *,
            sfreq: float,
            feature_strategy: str,
            normalize: bool,
            bands: Mapping[str, tuple[float, float]],
            strategy_config: Mapping[str, object],
        ) -> None:
            """Stocke les bandes pour dimensionner la sortie."""

            # Stocke les bandes afin de reproduire le nombre de colonnes attendu
            self._bands = bands

        def transform(self, data_in: np.ndarray) -> np.ndarray:
            """Retourne une matrice constante adaptée à la dimension attendue."""

            # Calcule le nombre de features à partir des canaux et des bandes
            n_features = data_in.shape[1] * len(self._bands)
            # Renvoie une matrice constante pour faciliter l'assertion
            return np.full((data_in.shape[0], n_features), 3.0, dtype=float)

    monkeypatch.setattr(features_module, "ExtractFeatures", DummyExtractor)

    # Déclenche l'extraction avec un epoch minimal pour exercer le fallback
    features, labels = extract_features(FakeEpochs(data, sfreq=64.0), config={"method": "welch"})
    captured_labels["labels"] = labels

    assert np.array_equal(features, np.full((1, 4), 3.0))
    assert captured_labels["labels"] == ["ch0_theta", "ch0_alpha", "ch0_beta", "ch0_gamma"]


def test_extract_features_uses_explicit_false_default_for_normalize_pop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_features doit appeler pop('normalize', False) explicitement."""

    # Prépare des epochs minimaux pour déclencher le chemin de délégation
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=8, sfreq=64.0)

    # Capture le défaut fourni à pop pour distinguer False de None
    sentinel = object()

    class SpyDict(dict):
        """Dict espion qui enregistre le défaut de pop('normalize', ...)."""

        # Stocke le dernier défaut observé pour la clé normalize
        normalize_default: object = sentinel

        def pop(self, key: object, default: object = sentinel):  # type: ignore[override]
            if key == "normalize":
                SpyDict.normalize_default = default
            if default is sentinel:
                return super().pop(key)  # type: ignore[arg-type]
            return super().pop(key, default)  # type: ignore[arg-type]

    # Injecte le constructeur dict pour préserver l'observabilité des appels
    monkeypatch.setattr(features_module, "dict", SpyDict, raising=False)

    class DummyExtractor:
        """Extracteur minimal pour éviter les dépendances lourdes."""

        def __init__(
            self,
            sfreq: float,
            bands: dict[str, tuple[float, float]],
            feature_strategy: str,
            normalize: bool,
            strategy_config: dict[str, object] | None = None,
        ) -> None:
            self._bands = bands

        def transform(self, data: np.ndarray) -> np.ndarray:
            return np.full((data.shape[0], data.shape[1] * len(self._bands)), 1.0)

    monkeypatch.setattr("tpv.features.ExtractFeatures", DummyExtractor)

    extract_features(epochs, config={"method": "welch"})

    assert SpyDict.normalize_default is not sentinel
    assert SpyDict.normalize_default is False


def test_extract_features_propagates_transform_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L'erreur levée par ExtractFeatures.transform doit remonter telle quelle."""

    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=8, sfreq=64.0)

    class FailingExtractor:
        def __init__(self, **_: object) -> None:
            pass

        def transform(self, data: np.ndarray) -> np.ndarray:  # noqa: ARG002
            raise RuntimeError("delegated failure")

    monkeypatch.setattr("tpv.features.ExtractFeatures", FailingExtractor)

    with pytest.raises(RuntimeError, match="delegated failure"):
        extract_features(epochs, config={"method": "welch"})


def test_compute_features_accepts_family_selection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_features doit chaîner les familles dans l'ordre demandé."""

    # Prépare des tenseurs de sortie distincts pour tracer l'ordre de concaténation
    X = np.ones((2, 1, 4))
    call_order: list[str] = []

    def fake_fft(self: ExtractFeatures, data: np.ndarray) -> np.ndarray:  # noqa: ARG001
        call_order.append("fft")
        return np.full((data.shape[0], 1), 1.0)

    def fake_welch(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        call_order.append("welch")
        return np.full((data.shape[0], 1), 2.0)

    def fake_wavelet(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        call_order.append("wavelet")
        return np.full((data.shape[0], 1), 3.0)

    monkeypatch.setattr(ExtractFeatures, "_compute_fft_features", fake_fft)
    monkeypatch.setattr(ExtractFeatures, "_compute_welch_features", fake_welch)
    monkeypatch.setattr(ExtractFeatures, "_compute_wavelet_features", fake_wavelet)

    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy=["fft", "welch", "wavelet"],
        normalize=False,
    )

    features = extractor._compute_features(X)

    assert call_order == ["fft", "welch", "wavelet"]
    expected_row = np.array([1.0, 2.0, 3.0])
    expected = np.vstack([expected_row, expected_row])
    assert np.array_equal(features, expected)




def test_compute_features_rejects_mismatched_sample_count_vs_first_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_features doit aligner toutes les familles sur le 1er bloc."""

    # Fixe un tenseur d'entrée minimal pour produire des features factices.
    X = np.ones((2, 1, 4))

    def fake_fft(self: ExtractFeatures, data: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.full((2, 1), 1.0)

    def fake_welch(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return np.full((3, 1), 2.0)

    def fake_wavelet(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return np.full((3, 1), 3.0)

    monkeypatch.setattr(ExtractFeatures, "_compute_fft_features", fake_fft)
    monkeypatch.setattr(ExtractFeatures, "_compute_welch_features", fake_welch)
    monkeypatch.setattr(ExtractFeatures, "_compute_wavelet_features", fake_wavelet)

    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy=["fft", "welch", "wavelet"],
        normalize=False,
    )

    with pytest.raises(
        ValueError,
        match=r"^All feature families must yield the same sample count\.$",
    ):
        extractor._compute_features(X)


def test_compute_features_rejects_mismatched_sample_count_in_middle_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_features doit vérifier chaque famille, y compris l'intermédiaire."""

    # Fixe un tenseur d'entrée minimal pour produire des features factices.
    X = np.ones((2, 1, 4))

    def fake_fft(self: ExtractFeatures, data: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return np.full((2, 1), 1.0)

    def fake_welch(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return np.full((3, 1), 2.0)

    def fake_wavelet(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        return np.full((2, 1), 3.0)

    monkeypatch.setattr(ExtractFeatures, "_compute_fft_features", fake_fft)
    monkeypatch.setattr(ExtractFeatures, "_compute_welch_features", fake_welch)
    monkeypatch.setattr(ExtractFeatures, "_compute_wavelet_features", fake_wavelet)

    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy=["fft", "welch", "wavelet"],
        normalize=False,
    )

    with pytest.raises(
        ValueError,
        match=r"^All feature families must yield the same sample count\.$",
    ):
        extractor._compute_features(X)


def test_compute_features_concatenates_two_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_features doit concaténer deux familles (pas court-circuiter)."""

    # Prépare un tenseur minimal pour déclencher exactement deux handlers
    X = np.ones((2, 1, 4))
    # Trace l'ordre d'appel pour s'assurer des handlers utilisés
    call_order: list[str] = []

    def fake_fft(self: ExtractFeatures, data: np.ndarray) -> np.ndarray:  # noqa: ARG001
        call_order.append("fft")
        return np.full((data.shape[0], 1), 1.0)

    def fake_welch(
        self: ExtractFeatures, data: np.ndarray
    ) -> np.ndarray:  # noqa: ARG001
        call_order.append("welch")
        return np.full((data.shape[0], 1), 2.0)

    monkeypatch.setattr(ExtractFeatures, "_compute_fft_features", fake_fft)
    monkeypatch.setattr(ExtractFeatures, "_compute_welch_features", fake_welch)

    extractor = ExtractFeatures(
        sfreq=64.0,
        feature_strategy=["fft", "welch"],
        normalize=False,
    )

    features = extractor._compute_features(X)

    assert call_order == ["fft", "welch"]
    expected_row = np.array([1.0, 2.0])
    expected = np.vstack([expected_row, expected_row])
    assert np.array_equal(features, expected)


def test_compute_features_rejects_empty_family_selection() -> None:
    """Une sélection vide doit être refusée explicitement."""

    extractor = ExtractFeatures(sfreq=64.0, feature_strategy=[], normalize=False)

    with pytest.raises(
        ValueError,
        match=r"^feature_strategy must include at least one feature family\.$",
    ):        extractor._compute_features(np.ones((1, 1, 4)))


def test_compute_features_rejects_unknown_family() -> None:
    """Une famille inconnue doit lever une erreur explicite avant concaténation."""

    extractor = ExtractFeatures(
        sfreq=64.0, feature_strategy=["fft", "unknown"], normalize=False
    )

    with pytest.raises(ValueError, match="Unsupported feature_strategy"):
        extractor._compute_features(np.ones((1, 1, 4)))
