"""Tests de réduction de dimension TPV."""

# Vérifie que numpy est disponible pour les constructions matricielles
import numpy as np

# Importe pytest pour vérifier les exceptions attendues
import pytest

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


# Vérifie que la validation de méthode refuse les options inconnues
def test_reducer_rejects_unknown_method():
    # Instancie un réducteur avec une méthode invalide
    reducer = TPVDimReducer(method="invalid")
    # Prépare des données tabulaires minimales pour déclencher la validation
    data = np.ones((2, 2))
    # Vérifie que l'appel à fit soulève une erreur explicite
    with pytest.raises(ValueError, match="method must be"):
        # Appelle fit pour déclencher la validation de la méthode
        reducer.fit(data)


# Vérifie que CSP exige des labels fournis par l'utilisateur
def test_csp_requires_labels():
    # Instancie le réducteur en mode CSP
    reducer = TPVDimReducer(method="csp")
    # Prépare des données structurées en essais simulés
    trials = np.ones((4, 2, 3))
    # Vérifie que l'absence de labels déclenche une erreur dédiée
    with pytest.raises(ValueError, match="required for CSP"):
        # Appelle fit sans labels pour couvrir le contrôle de présence
        reducer.fit(trials)


# Vérifie que CSP refuse un nombre de classes différent de deux
def test_csp_rejects_more_than_two_classes():
    # Instancie un réducteur CSP avec configuration par défaut
    reducer = TPVDimReducer(method="csp")
    # Prépare des essais avec trois étiquettes distinctes
    trials = np.ones((6, 2, 3))
    # Crée des labels non binaires pour déclencher le contrôle
    labels = np.array([0, 1, 2, 0, 1, 2])
    # Vérifie que fit refuse une cardinalité différente de deux
    with pytest.raises(ValueError, match="exactly two"):
        # Appelle fit avec des labels invalides
        reducer.fit(trials, labels)


# Vérifie la branche CSP sans limitation de composantes
def test_csp_retains_all_components_by_default():
    # Prépare un générateur déterministe pour la reproductibilité
    rng = np.random.default_rng(3)
    # Fixe le nombre d'essais par classe
    trials_per_class = 4
    # Fixe le nombre de canaux à projeter
    channels = 3
    # Génère des essais bruités pour la première classe
    class_a = rng.standard_normal((trials_per_class, channels, 5))
    # Génère des essais bruités pour la seconde classe
    class_b = rng.standard_normal((trials_per_class, channels, 5))
    # Concatène les essais pour former l'ensemble complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires associés
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Instancie le réducteur CSP sans préciser n_components
    reducer = TPVDimReducer(method="csp")
    # Apprend la projection complète
    reducer.fit(trials, labels)
    # Vérifie que toutes les composantes sont conservées
    assert reducer.w_matrix.shape == (channels, channels)
    # Vérifie que le vecteur des valeurs propres couvre toutes les composantes
    assert reducer.eigenvalues_.shape[0] == channels


# Vérifie que transform requiert un apprentissage préalable
def test_transform_requires_fitted_model():
    # Instancie un réducteur PCA simple
    reducer = TPVDimReducer(method="pca")
    # Prépare des données tabulaires pour l'appel
    data = np.ones((2, 2))
    # Vérifie que transform lève une erreur sans fit
    with pytest.raises(ValueError, match="must be fitted"):
        # Appelle transform pour couvrir la validation d'entraînement
        reducer.transform(data)


# Vérifie la projection des données trial x channel x time
def test_transform_handles_trial_dimension():
    # Prépare un générateur aléatoire pour créer des essais
    rng = np.random.default_rng(4)
    # Fixe le nombre d'essais par classe
    trials_per_class = 2
    # Fixe le nombre de canaux
    channels = 2
    # Fixe le nombre de points temporels
    time_points = 3
    # Génère des essais pour la première classe
    class_a = rng.standard_normal((trials_per_class, channels, time_points))
    # Génère des essais pour la seconde classe
    class_b = rng.standard_normal((trials_per_class, channels, time_points))
    # Concatène les essais des deux classes
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires correspondants
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Instancie le réducteur CSP pour deux composantes
    reducer = TPVDimReducer(method="csp", n_components=2)
    # Apprend la projection pour obtenir w_matrix
    reducer.fit(trials, labels)
    # Transforme les essais pour couvrir la branche 3D
    projected = reducer.transform(trials)
    # Vérifie que la forme correspond aux attentes trial x comp x time
    assert projected.shape == (trials.shape[0], 2, time_points)


# Vérifie qu'une dimension inattendue est explicitement refusée
def test_transform_rejects_invalid_dimension():
    # Instancie un réducteur PCA et l'entraîne
    reducer = TPVDimReducer(method="pca")
    # Prépare des données d'entraînement simples
    training = np.eye(2)
    # Apprend la projection sur les données
    reducer.fit(training)
    # Prépare un tenseur 4D pour déclencher l'erreur
    invalid_input = np.ones((1, 2, 2, 2))
    # Vérifie que transform lève une erreur sur dimension inattendue
    with pytest.raises(ValueError, match="2D or 3D"):
        # Appelle transform pour couvrir la levée de ValueError
        reducer.transform(invalid_input)


# Vérifie que la sauvegarde est interdite sans apprentissage préalable
def test_save_requires_fitted_model(tmp_path):
    # Instancie un réducteur PCA sans entraînement
    reducer = TPVDimReducer(method="pca")
    # Définit le chemin de sauvegarde temporaire
    target = tmp_path / "dim.joblib"
    # Vérifie que save refuse d'écrire sans fit préalable
    with pytest.raises(ValueError, match="Cannot save"):
        # Appelle save pour déclencher la validation
        reducer.save(target)


# Vérifie que l'estimation de covariance signale l'absence d'essais
def test_average_covariance_rejects_empty_trials():
    # Instancie un réducteur PCA pour accéder à la méthode privée
    reducer = TPVDimReducer(method="pca")
    # Crée un tableau vide d'essais avec trois axes
    empty_trials = np.empty((0, 2, 2))
    # Vérifie que l'appel lève une ValueError explicite
    with pytest.raises(ValueError, match="No trials"):
        # Appelle la méthode interne pour couvrir la validation
        reducer._average_covariance(empty_trials)
