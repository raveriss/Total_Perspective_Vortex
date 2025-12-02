"""Tests de réduction de dimension TPV."""

# Vérifie que numpy est disponible pour les constructions matricielles
import numpy as np

# Importe le réducteur TPV à valider
from tpv.dimensionality import TPVDimReducer

# Définit le seuil minimal de variance principale acceptable
PCA_VARIANCE_THRESHOLD = 0.8

# Définit la tolérance d'acceptation pour les valeurs propres négatives
EIGENVALUE_TOLERANCE = 1e-8


# Vérifie que la projection CSP reste orthogonale
def test_csp_projection_orthogonality():
    # Prépare un générateur pour des données reproductibles
    rng = np.random.default_rng(0)
    # Fixe le nombre d'essais par classe
    trials_per_class = 8
    # Fixe le nombre de canaux simulés
    channels = 4
    # Fixe la longueur temporelle des essais
    time_points = 16
    # Génère des essais pour la première classe avec faible variance
    class_a = rng.standard_normal((trials_per_class, channels, time_points)) * 0.5
    # Génère des essais pour la seconde classe avec une variance accentuée
    class_b = rng.standard_normal((trials_per_class, channels, time_points)) * 1.5
    # Concatène les essais des deux classes
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les étiquettes correspondantes
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Instancie le réducteur CSP avec régularisation faible
    reducer = TPVDimReducer(method="csp", n_components=channels, regularization=1e-3)
    # Apprend la matrice de projection CSP
    reducer.fit(trials, labels)
    # Récupère la matrice de projection apprise
    projection = reducer.w_matrix
    # Vérifie la forme de la matrice obtenue
    assert projection.shape == (channels, channels)
    # Construit la covariance moyenne de la première classe
    cov_a = np.zeros((channels, channels))
    # Parcourt chaque essai de la première classe
    for trial in class_a:
        # Calcule la covariance normalisée par la trace
        trial_cov = trial @ trial.T
        # Normalise la covariance pour stabiliser les échelles
        trial_cov /= np.trace(trial_cov)
        # Accumule la covariance sur la classe
        cov_a += trial_cov
    # Moyenne la covariance de la classe A
    cov_a /= float(trials_per_class)
    # Construit la covariance moyenne de la seconde classe
    cov_b = np.zeros((channels, channels))
    # Parcourt chaque essai de la seconde classe
    for trial in class_b:
        # Calcule la covariance normalisée pour l'essai
        trial_cov = trial @ trial.T
        # Normalise la covariance pour homogénéiser les essais
        trial_cov /= np.trace(trial_cov)
        # Accumule la covariance de la seconde classe
        cov_b += trial_cov
    # Moyenne la covariance de la classe B
    cov_b /= float(trials_per_class)
    # Ajoute la régularisation définie sur le modèle
    composite = cov_a + cov_b + reducer.regularization * np.eye(channels)
    # Calcule le produit qui doit se rapprocher de l'identité
    identity_candidate = projection.T @ composite @ projection
    # Vérifie l'orthogonalité dans l'espace de covariance régularisée
    assert np.allclose(identity_candidate, np.eye(channels), atol=5e-3)


# Vérifie que PCA capture l'essentiel de la variance
def test_pca_explained_variance():
    # Prépare un générateur pour des données déterministes
    rng = np.random.default_rng(1)
    # Fixe le nombre d'échantillons
    samples = 200
    # Génère une composante principale dominante
    dominant = rng.standard_normal(samples) * 3.0
    # Génère deux composantes de bruit plus faibles
    noise1 = rng.standard_normal(samples) * 0.3
    noise2 = rng.standard_normal(samples) * 0.2
    # Assemble les composantes pour former des observations 3D
    observations = np.column_stack([dominant + noise1, noise1, noise2])
    # Instancie le réducteur PCA pour deux composantes
    reducer = TPVDimReducer(method="pca", n_components=2, regularization=1e-4)
    # Apprend la projection PCA
    reducer.fit(observations)
    # Calcule la part de variance captée par chaque composante
    variance_ratio = reducer.eigenvalues_ / np.sum(reducer.eigenvalues_)
    # Vérifie que la première composante dépasse la majorité de la variance
    assert variance_ratio[0] > PCA_VARIANCE_THRESHOLD
    # Vérifie que la deuxième composante reste positive
    assert variance_ratio[1] > 0


# Vérifie que la régularisation stabilise les covariances singulières
def test_regularization_stabilizes_singular_covariance():
    # Prépare un générateur aléatoire pour fixer les données
    rng = np.random.default_rng(2)
    # Fixe le nombre d'échantillons à analyser
    samples = 120
    # Génère une caractéristique aléatoire
    feature = rng.standard_normal(samples)
    # Duplique la caractéristique pour créer une covariance singulière
    duplicated = np.column_stack([feature, feature])
    # Instancie le PCA avec une régularisation pour lever la singularité
    reducer = TPVDimReducer(method="pca", regularization=1e-3)
    # Apprend la projection malgré la singularité initiale
    reducer.fit(duplicated)
    # Transforme les données pour vérifier la stabilité numérique
    transformed = reducer.transform(duplicated)
    # Vérifie l'absence de valeurs infinies dans la projection
    assert np.isfinite(transformed).all()
    # Vérifie que les valeurs propres restent non négatives sous tolérance
    assert np.all(reducer.eigenvalues_ >= -EIGENVALUE_TOLERANCE)
