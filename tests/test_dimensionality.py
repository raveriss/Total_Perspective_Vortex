"""Tests unitaires pour la réduction de dimension TPV."""

# Importe numpy pour générer des données synthétiques
# Utilise Path pour manipuler les chemins de manière sûre
from pathlib import Path

# Importe joblib pour instrumenter le chargement persistant
import joblib

# Charge numpy pour créer des matrices de test
import numpy as np

# Importe pytest pour vérifier les exceptions attendues
import pytest

# Importe le réducteur pour valider son comportement
from tpv.dimensionality import TPVDimReducer

# Fige l'écart de variance minimal attendu entre les classes
VARIANCE_GAP_THRESHOLD = 0.5


# Vérifie que la PCA produit une base orthonormée et reconstructible
def test_pca_projection_is_orthonormal_and_reconstructs():
    # Crée une matrice de données centrée artificiellement
    X = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    # Instancie le réducteur en mode PCA avec deux composantes explicites
    reducer = TPVDimReducer(method="pca", n_components=2)
    # Apprend la matrice de projection à partir de X
    reducer.fit(X)
    # Calcule l'orthogonalité en vérifiant W^T W = I
    identity = reducer.w_matrix.T @ reducer.w_matrix
    # Vérifie que la matrice est proche de l'identité
    np.testing.assert_allclose(identity, np.eye(identity.shape[0]), atol=1e-6)
    # Projette les données puis reconstruit dans l'espace initial
    projected = reducer.transform(X)
    # Recompose en appliquant l'inverse orthonormée
    reconstructed = projected @ reducer.w_matrix.T + reducer.mean_
    # Vérifie que la reconstruction approche les données originales
    np.testing.assert_allclose(reconstructed, X, atol=1e-6)


# Vérifie que le CSP sépare des classes aux covariances distinctes
def test_csp_separates_covariances():
    # Crée un générateur déterministe pour stabiliser le test
    rng = np.random.default_rng(42)
    # Définit un nombre de canaux et d'échantillons temporels
    n_channels = 2
    # Fixe la durée en échantillons pour chaque essai
    n_times = 50
    # Génère des essais pour la classe 0 avec variance dominante sur le canal 0
    class0 = rng.standard_normal((20, n_channels, n_times))
    # Amplifie le premier canal pour augmenter sa variance relative
    class0[:, 0, :] *= 2.0
    # Génère des essais pour la classe 1 avec variance dominante sur le canal 1
    class1 = rng.standard_normal((20, n_channels, n_times))
    # Amplifie le second canal pour créer un contraste de variance
    class1[:, 1, :] *= 2.0
    # Concatène les essais en un seul tableau
    X = np.concatenate([class0, class1], axis=0)
    # Crée les étiquettes correspondantes
    y = np.array([0] * class0.shape[0] + [1] * class1.shape[0])
    # Instancie le réducteur en mode CSP avec deux composantes
    reducer = TPVDimReducer(method="csp", n_components=2)
    # Apprend la matrice de projection CSP
    reducer.fit(X, y)
    # Projette les essais complets pour préserver la dynamique temporelle
    projected0 = reducer.transform(class0)
    # Projette la classe 1 pour comparaison
    projected1 = reducer.transform(class1)
    # Calcule la variance spatio-temporelle sur la première composante
    var0 = np.var(projected0[:, 0, :])
    # Calcule la variance sur la même composante pour l'autre classe
    var1 = np.var(projected1[:, 0, :])
    # Vérifie que les variances sont nettement différentes
    assert abs(var0 - var1) > VARIANCE_GAP_THRESHOLD


# Vérifie que la matrice peut être sauvegardée puis rechargée
def test_save_and_load_projection(tmp_path: Path):
    # Crée des données simples pour la PCA
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    # Initialise le réducteur
    reducer = TPVDimReducer(method="pca")
    # Apprend la projection
    reducer.fit(X)
    # Prépare un chemin temporaire pour la sauvegarde
    target = tmp_path / "projection.joblib"
    # Sauvegarde la matrice apprise
    reducer.save(target)
    # Crée un nouveau réducteur vide
    restored = TPVDimReducer(method="pca")
    # Charge la matrice précédemment sauvegardée
    restored.load(target)
    # Vérifie que les matrices sont identiques
    np.testing.assert_allclose(restored.w_matrix, reducer.w_matrix)
    # Vérifie que la moyenne est préservée
    np.testing.assert_allclose(restored.mean_, reducer.mean_)


# Vérifie que la méthode inconnue est refusée dès l'apprentissage
def test_invalid_method_rejected():
    # Crée un jeu de données minimal pour déclencher l'erreur
    X = np.zeros((2, 2))
    # Instancie le réducteur avec une méthode non supportée
    reducer = TPVDimReducer(method="unknown")
    # Vérifie que fit lève une ValueError pour méthode invalide
    with pytest.raises(ValueError):
        # Lance l'apprentissage pour atteindre la validation de méthode
        reducer.fit(X)


# Vérifie que CSP exige la présence des labels y
def test_csp_requires_labels():
    # Crée des essais fictifs à deux canaux et un temps
    X = np.zeros((2, 2, 1))
    # Instancie le réducteur CSP sans fournir y
    reducer = TPVDimReducer(method="csp")
    # Vérifie que fit lève une ValueError en absence de labels
    with pytest.raises(ValueError):
        # Lance fit pour déclencher la vérification des labels
        reducer.fit(X)


# Vérifie que CSP refuse plus de deux classes
def test_csp_rejects_multiclass():
    # Crée trois essais pour simuler trois classes distinctes
    X = np.zeros((3, 2, 1))
    # Assigne trois labels différents pour dépasser la limite
    y = np.array([0, 1, 2])
    # Instancie le réducteur CSP
    reducer = TPVDimReducer(method="csp")
    # Vérifie que fit lève une ValueError pour classes multiples
    with pytest.raises(ValueError):
        # Lance fit pour atteindre la validation du nombre de classes
        reducer.fit(X, y)


# Vérifie que transform nécessite un modèle entraîné
def test_transform_requires_fit():
    # Crée une entrée tabulaire simple
    X = np.zeros((1, 2))
    # Instancie le réducteur sans apprentissage préalable
    reducer = TPVDimReducer(method="pca")
    # Vérifie que transform lève une ValueError si fit n'est pas appelé
    with pytest.raises(ValueError):
        # Appelle transform pour déclencher la protection
        reducer.transform(X)


# Vérifie que transform refuse une dimension inattendue
def test_transform_rejects_invalid_dimension():
    # Crée une entrée 4D pour déclencher l'erreur
    X = np.zeros((1, 2, 3, 4))
    # Instancie et entraîne le réducteur pour permettre transform
    reducer = TPVDimReducer(method="pca")
    # Lance fit pour initialiser la matrice de projection
    reducer.fit(np.zeros((2, 2)))
    # Remplace la moyenne par un scalaire neutre pour éviter le broadcasting
    reducer.mean_ = np.array(0.0)
    # Vérifie que transform lève une ValueError sur dimension 4D
    with pytest.raises(ValueError, match="X must be 2D or 3D for transform"):
        # Appelle transform avec la mauvaise dimensionnalité
        reducer.transform(X)


# Vérifie que save refuse d'opérer avant l'entraînement avec message clair
def test_save_requires_fitted_model(tmp_path: Path):
    # Prépare un chemin temporaire pour la sauvegarde
    target = tmp_path / "projection.joblib"
    # Instancie un réducteur PCA sans apprentissage
    reducer = TPVDimReducer(method="pca")
    # Vérifie que save lève une ValueError en absence de matrice
    with pytest.raises(ValueError, match="^Cannot save before fitting the model$"):
        # Tente de sauvegarder sans avoir appelé fit
        reducer.save(target)


# Vérifie que la covariance moyenne refuse un ensemble vide
def test_average_covariance_rejects_empty_trials():
    # Instancie un réducteur pour accéder à la méthode interne
    reducer = TPVDimReducer(method="csp")
    # Crée un tableau vide d'essais pour provoquer l'erreur
    empty_trials = np.zeros((0, 2, 2))
    # Vérifie que la fonction interne lève une ValueError explicite
    with pytest.raises(
        ValueError, match="^No trials provided for covariance estimation$"
    ):
        # Appelle directement la méthode de covariance moyenne
        reducer._average_covariance(empty_trials)


# Vérifie que load transmet fidèlement le chemin fourni
def test_load_uses_provided_path(monkeypatch, tmp_path: Path):
    # Crée un chemin temporaire pour simuler la source
    target = tmp_path / "projection.joblib"
    # Prépare une matrice de projection factice
    expected_w = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Prépare une moyenne factice pour vérifier la restauration
    expected_mean = np.array([0.5, -0.5])
    # Capture le chemin utilisé par joblib.load
    captured = {}

    # Définit un faux loader qui enregistre le chemin reçu
    def fake_load(path: str):
        # Mémorise le chemin pour vérification ultérieure
        captured["path"] = path
        # Retourne un contenu cohérent pour simuler la persistance
        return {"w_matrix": expected_w, "mean": expected_mean}

    # Remplace joblib.load par la version instrumentée
    monkeypatch.setattr(joblib, "load", fake_load)
    # Instancie un réducteur vide prêt à charger
    reducer = TPVDimReducer(method="pca")
    # Charge les données via le faux loader
    reducer.load(target)
    # Vérifie que le chemin fourni a été transmis à joblib.load
    assert captured["path"] == str(target)
    # Vérifie la restauration de la matrice de projection
    np.testing.assert_allclose(reducer.w_matrix, expected_w)
    # Vérifie la restauration de la moyenne
    np.testing.assert_allclose(reducer.mean_, expected_mean)


# Vérifie que la covariance moyenne agrège l'information de chaque essai
def test_average_covariance_accumulates_trials():
    # Crée un premier essai avec énergie sur l'axe diagonal
    trial_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Crée un second essai avec énergie mixte pour tester l'accumulation
    trial_b = np.array([[1.0, 1.0], [1.0, 1.0]])
    # Empile les essais pour simuler deux répétitions
    trials = np.stack([trial_a, trial_b])
    # Calcule la covariance normalisée du premier essai
    cov_a = trial_a @ trial_a.T
    # Normalise pour suivre la logique de la méthode
    cov_a /= np.trace(cov_a)
    # Calcule la covariance normalisée du second essai
    cov_b = trial_b @ trial_b.T
    # Normalise pour comparer sur une base commune
    cov_b /= np.trace(cov_b)
    # Calcule la moyenne attendue sur les deux essais
    expected = (cov_a + cov_b) / trials.shape[0]
    # Instancie le réducteur pour accéder à la méthode de covariance
    reducer = TPVDimReducer(method="csp")
    # Calcule la covariance moyenne via la méthode
    averaged = reducer._average_covariance(trials)
    # Vérifie que la moyenne correspond à l'accumulation attendue
    np.testing.assert_allclose(averaged, expected)
