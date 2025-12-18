# Importe time pour suivre le budget temporel d'extraction
from time import perf_counter

# Importe numpy pour générer des signaux et des tenseurs de test
import numpy as np

# Importe pytest pour vérifier les erreurs et les approximations
import pytest

# Importe mne pour créer des epochs synthétiques conformes à l'API
from mne import EpochsArray, create_info

# Importe l'extracteur procédural, la classe scikit-learn et les helpers Welch
from tpv.features import (
    ExtractFeatures,
    _prepare_welch_parameters,
    extract_features,
)

# Définit une constante pour le budget temps afin d'éviter les magic numbers
MAX_EXTRACTION_SECONDS = 0.25


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


def test_extract_features_wrapper_rejects_unknown_strategy() -> None:
    """La classe scikit-learn doit refuser les stratégies non supportées."""

    # Construit un tenseur conforme pour l'appel
    tensor = np.zeros((1, 2, 16))
    # Instancie un extracteur avec une stratégie inconnue
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="invalid")
    # Vérifie que l'appelant reçoit une erreur explicite
    with pytest.raises(ValueError):
        extractor.transform(tensor)


def test_extract_features_wrapper_rejects_incorrect_shape() -> None:
    """La classe scikit-learn doit refuser une dimension d'entrée erronée."""

    # Prépare un extracteur pour tester la validation de dimension
    extractor = ExtractFeatures(sfreq=128.0)
    # Prépare un tableau bidimensionnel pour déclencher la validation
    bad_shape = np.zeros((2, 16))
    # Vérifie que la méthode transforme lève une erreur de forme
    with pytest.raises(ValueError):
        extractor.transform(bad_shape)


def test_extract_features_wrapper_rejects_unknown_method() -> None:
    """La fonction procédurale doit refuser une méthode non supportée."""

    # Prépare des epochs minimaux pour l'appel procédural
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Vérifie que la fonction signale la méthode inconnue
    with pytest.raises(ValueError):
        extract_features(epochs, config={"method": "invalid"})


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
    assert effective_nperseg == 64
    # Vérifie que le recouvrement est borné à une fenêtre strictement positive
    assert effective_noverlap == 63
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
    assert effective_nperseg == 50
    # Vérifie qu'aucun recouvrement n'est défini sans instruction explicite
    assert effective_noverlap is None
    # Vérifie que la moyenne par défaut correspond à l'option SciPy standard
    assert average == "mean"
    # Vérifie que l'échelle par défaut correspond à la densité spectrale
    assert scaling == "density"
