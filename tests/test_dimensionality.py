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
# Fige le nombre de composantes cibles pour certains tests CSP
CSP_COMPONENTS = 2
# Fige le nombre de canaux des tests CSP à trois dimensions
CSP_CHANNELS = 3


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


# Vérifie que les valeurs propres sont initialisées à None avant entraînement
def test_eigenvalues_start_as_none():
    # Instancie un réducteur sans déclencher fit
    reducer = TPVDimReducer()
    # Vérifie que les valeurs propres ne sont pas préremplies
    assert reducer.eigenvalues_ is None


# Vérifie que la configuration par défaut déclenche bien la PCA complète
def test_default_pca_runs_with_all_components():
    # Crée des données simples avec deux caractéristiques distinctes
    X = np.array([[1.0, 2.0], [3.0, 0.0], [0.0, -1.0]])
    # Instancie le réducteur sans préciser la méthode pour tester la valeur par défaut
    reducer = TPVDimReducer()
    # Apprend la projection pour valider l'absence d'erreur sur la valeur par défaut
    reducer.fit(X)
    # Vérifie que la méthode par défaut reste la PCA attendue
    assert reducer.method == "pca"
    # Vérifie que toutes les composantes sont conservées en absence de coupe
    assert reducer.w_matrix.shape[1] == X.shape[1]


# Vérifie que la PCA par défaut enregistre toutes les valeurs propres
def test_default_pca_records_all_eigenvalues():
    # Crée des données simples avec dispersion sur deux axes
    X = np.array([[1.0, -1.0], [2.0, 0.5], [-0.5, 2.0]])
    # Instancie un réducteur PCA sans limiter les composantes
    reducer = TPVDimReducer(method="pca")
    # Apprend la projection complète
    reducer.fit(X)
    # Vérifie que les valeurs propres existent après apprentissage
    assert reducer.eigenvalues_ is not None
    # Vérifie que chaque composante possède une valeur propre enregistrée
    assert reducer.eigenvalues_.shape[0] == X.shape[1]


# Vérifie que le paramètre n_components est respecté pendant la PCA
def test_pca_honours_requested_component_count():
    # Crée un jeu de données bidimensionnel pour le découpage des composantes
    X = np.array([[2.0, 1.0], [0.0, 3.0], [-1.0, -2.0]])
    # Instancie un réducteur PCA limité à une seule composante
    reducer = TPVDimReducer(method="pca", n_components=1)
    # Apprend la matrice de projection contrainte
    reducer.fit(X)
    # Vérifie que la matrice de projection ne conserve qu'une colonne
    assert reducer.w_matrix.shape == (X.shape[1], 1)


# Vérifie que la moyenne PCA est calculée sur chaque feature indépendamment
def test_pca_mean_is_featurewise():
    # Crée des données où les moyennes par colonne diffèrent
    X = np.array([[0.0, 3.0], [2.0, 1.0], [4.0, -1.0]])
    # Instancie un réducteur PCA
    reducer = TPVDimReducer(method="pca")
    # Apprend la projection pour remplir la moyenne interne
    reducer.fit(X)
    # Vérifie que la moyenne correspond exactement à la moyenne par colonne
    np.testing.assert_allclose(reducer.mean_, np.mean(X, axis=0))


# Vérifie que la covariance centrée reproduit les vecteurs propres attendus
def test_pca_covariance_matches_manual_eigendecomposition():
    # Crée des données asymétriques pour distinguer les composantes principales
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 0.0], [-1.0, -3.0]])
    # Instancie un réducteur PCA à deux composantes
    reducer = TPVDimReducer(method="pca", n_components=2)
    # Apprend la matrice de projection à partir des données
    reducer.fit(X)
    # Calcule manuellement la covariance centrée pour établir la référence
    centered = X - np.mean(X, axis=0)
    # Calcule la covariance manuelle alignée sur l'implémentation
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    # Extrait les vecteurs propres triés par variance décroissante
    eigvals, eigvecs = np.linalg.eigh(covariance)
    # Trie les vecteurs pour correspondre à l'ordre attendu
    order = np.argsort(eigvals)[::-1]
    # Réordonne les vecteurs pour comparer colonne par colonne
    expected_vecs = eigvecs[:, order]
    # Compare chaque colonne en neutralisant l'ambiguïté de signe
    for idx in range(expected_vecs.shape[1]):
        # Vérifie la correspondance absolue des vecteurs propres
        np.testing.assert_allclose(
            np.abs(reducer.w_matrix[:, idx]), np.abs(expected_vecs[:, idx]), atol=1e-6
        )


# Vérifie que les valeurs propres stockées reflètent la variance projetée
def test_pca_eigenvalues_match_projected_variance():
    # Crée des données avec variances distinctes par direction
    X = np.array([[3.0, 0.0], [1.0, 2.0], [-2.0, -1.0], [0.0, -3.0]])
    # Instancie un réducteur PCA à deux composantes
    reducer = TPVDimReducer(method="pca", n_components=2)
    # Apprend la projection pour remplir les valeurs propres
    reducer.fit(X)
    # Projette les données dans l'espace des composantes principales
    projected = reducer.transform(X)
    # Calcule la variance empirique sur chaque composante projetée
    projected_variance = np.var(projected, axis=0, ddof=1)
    # Vérifie la concordance avec les valeurs propres stockées
    np.testing.assert_allclose(projected_variance, reducer.eigenvalues_)


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
    reducer = TPVDimReducer(method="csp", n_components=CSP_COMPONENTS)
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


# Vérifie que CSP respecte le nombre de composantes demandé
def test_csp_honours_component_count():
    # Crée un tenseur simple à deux canaux et deux essais par classe
    class0 = np.array([np.diag([4.0, 1.0]), np.diag([4.0, 1.0])])
    # Crée des essais pour la classe 1 avec diagonale inversée
    class1 = np.array([np.diag([1.0, 3.0]), np.diag([1.0, 3.0])])
    # Concatène les essais
    X = np.concatenate([class0, class1], axis=0)
    # Crée les labels correspondants
    y = np.array([0, 0, 1, 1])
    # Instancie un réducteur CSP demandant deux composantes
    reducer = TPVDimReducer(method="csp", n_components=2)
    # Apprend la projection
    reducer.fit(X, y)
    # Vérifie que le nombre de colonnes correspond à la demande
    assert reducer.w_matrix.shape[1] == CSP_COMPONENTS
    # Vérifie que les valeurs propres sont renseignées pour chaque composante
    assert reducer.eigenvalues_.shape[0] == CSP_COMPONENTS
    # Vérifie que l'ordre des valeurs propres est décroissant
    assert reducer.eigenvalues_[0] >= reducer.eigenvalues_[1]


# Vérifie que CSP tronque réellement les composantes demandées
def test_csp_truncates_components_when_requested():
    # Crée des essais à trois canaux pour forcer un découpage
    class0 = np.stack([np.eye(3)] * 2)
    # Crée des essais pour la classe 1 avec énergie sur le troisième canal
    class1 = np.stack([np.diag([1.0, 1.0, 3.0])] * 2)
    # Concatène les essais des deux classes
    X = np.concatenate([class0, class1], axis=0)
    # Crée les labels associés
    y = np.array([0, 0, 1, 1])
    # Instancie un réducteur CSP limité à une seule composante
    reducer = TPVDimReducer(method="csp", n_components=1)
    # Apprend la projection réduite
    reducer.fit(X, y)
    # Vérifie que la matrice ne conserve qu'une colonne
    assert reducer.w_matrix.shape == (3, 1)
    # Vérifie que seule une valeur propre est conservée
    assert reducer.eigenvalues_.shape == (1,)


# Vérifie que chaque classe alimente correctement le calcul CSP
def test_csp_uses_class_specific_covariances(monkeypatch):
    # Crée deux essais pour la classe 0 avec structure diagonale
    class0 = np.array([np.eye(2), np.eye(2)])
    # Crée deux essais pour la classe 1 avec diagonale amplifiée
    class1 = np.array([2.0 * np.eye(2), 2.0 * np.eye(2)])
    # Empile les essais des deux classes
    X = np.concatenate([class0, class1], axis=0)
    # Associe les labels correspondants
    y = np.array([0, 0, 1, 1])
    # Conserve l'implémentation originale pour la délégation
    original_average = TPVDimReducer._average_covariance
    # Capture les arguments fournis à la méthode interne
    captured: list[np.ndarray] = []

    # Déclare un wrapper pour enregistrer les données passées
    def recording_average(self, trials: np.ndarray):
        # Stocke les essais reçus pour vérification ultérieure
        captured.append(trials.copy())
        # Délègue au comportement d'origine
        return original_average(self, trials)

    # Remplace la méthode interne par le wrapper enregistré
    monkeypatch.setattr(TPVDimReducer, "_average_covariance", recording_average)
    # Instancie le réducteur CSP
    reducer = TPVDimReducer(method="csp", n_components=CSP_COMPONENTS)
    # Apprend la projection en déclenchant les appels enregistrés
    reducer.fit(X, y)
    # Vérifie que le premier appel concerne uniquement la classe 0
    np.testing.assert_allclose(captured[0], class0)
    # Vérifie que le second appel concerne uniquement la classe 1
    np.testing.assert_allclose(captured[1], class1)


# Vérifie que la covariance composite est bien la somme des deux classes
def test_csp_constructs_composite_covariance(monkeypatch):
    # Crée des essais distincts pour identifier chaque classe
    class0 = np.array([np.eye(2), np.eye(2)])
    # Crée des essais à variance doublée pour la classe 1
    class1 = np.array([2.0 * np.eye(2), 2.0 * np.eye(2)])
    # Concatène les essais pour former l'entrée complète
    X = np.concatenate([class0, class1], axis=0)
    # Définit les labels associés
    y = np.array([0, 0, 1, 1])
    # Prépare des covariances synthétiques pour chaque classe
    cov0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Prépare la covariance attendue pour la seconde classe
    cov1 = np.array([[2.0, 0.0], [0.0, 2.0]])
    # Capture les arguments transmis à linalg.eigh
    captured: dict[str, np.ndarray] = {}

    # Remplace la moyenne de covariance pour injecter les matrices de test
    def fake_average(self, trials: np.ndarray):
        # Retourne cov0 si les essais correspondent à la classe 0
        if np.array_equal(trials, class0):
            return cov0
        # Retourne cov1 si les essais correspondent à la classe 1
        if np.array_equal(trials, class1):
            return cov1
        # Signale tout appel inattendu pour sécuriser le test
        raise AssertionError("Unexpected trials passed to _average_covariance")

    # Remplace linalg.eigh pour enregistrer les matrices reçues
    def fake_eigh(cov_a: np.ndarray, composite: np.ndarray):
        # Stocke la covariance de la première classe
        captured["cov_a"] = cov_a.copy()
        # Stocke la covariance composite calculée
        captured["composite"] = composite.copy()
        # Retourne des valeurs propres stables pour terminer fit
        return np.array([1.0, 0.5]), np.eye(2)

    # Applique les monkeypatches nécessaires
    monkeypatch.setattr(TPVDimReducer, "_average_covariance", fake_average)
    monkeypatch.setattr("tpv.dimensionality.linalg.eigh", fake_eigh)
    # Instancie le réducteur CSP
    reducer = TPVDimReducer(method="csp", n_components=2)
    # Déclenche fit pour construire la covariance composite
    reducer.fit(X, y)
    # Vérifie que cov_a correspond à la première classe
    np.testing.assert_allclose(captured["cov_a"], cov0)
    # Vérifie que la covariance composite est bien la somme attendue
    np.testing.assert_allclose(captured["composite"], cov0 + cov1)
    # Vérifie que tout autre appel non référencé déclenche une alerte
    with pytest.raises(AssertionError, match="Unexpected trials"):
        # Force un appel inattendu pour couvrir la branche de garde
        fake_average(reducer, np.zeros((1, 1, 1)))


# Vérifie que CSP conserve toutes les composantes par défaut
def test_csp_defaults_to_full_component_set():
    # Crée des essais simples sur trois canaux
    class0 = np.stack([np.eye(CSP_CHANNELS)] * 2)
    # Crée des essais avec variance dominante sur le troisième canal
    class1 = np.stack([np.diag([1.0, 1.0, 4.0])] * 2)
    # Concatène les essais pour former X
    X = np.concatenate([class0, class1], axis=0)
    # Crée les labels associés
    y = np.array([0, 0, 1, 1])
    # Instancie un réducteur CSP sans n_components explicite
    reducer = TPVDimReducer(method="csp")
    # Apprend la projection en conservant toutes les composantes
    reducer.fit(X, y)
    # Vérifie que la matrice conserve une colonne par canal
    assert reducer.w_matrix.shape == (CSP_CHANNELS, CSP_CHANNELS)
    # Vérifie que les valeurs propres couvrent toutes les composantes
    assert reducer.eigenvalues_.shape[0] == CSP_CHANNELS


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
    # Vérifie que les valeurs propres sont également restaurées
    np.testing.assert_allclose(restored.eigenvalues_, reducer.eigenvalues_)


# Vérifie que la méthode inconnue est refusée dès l'apprentissage
def test_invalid_method_rejected():
    # Crée un jeu de données minimal pour déclencher l'erreur
    X = np.zeros((2, 2))
    # Instancie le réducteur avec une méthode non supportée
    reducer = TPVDimReducer(method="unknown")
    # Vérifie que fit lève une ValueError pour méthode invalide avec message clair
    with pytest.raises(ValueError, match="^method must be 'pca' or 'csp'$"):
        # Lance l'apprentissage pour atteindre la validation de méthode
        reducer.fit(X)


# Vérifie que CSP exige la présence des labels y
def test_csp_requires_labels():
    # Crée des essais fictifs à deux canaux et un temps
    X = np.zeros((2, 2, 1))
    # Instancie le réducteur CSP sans fournir y
    reducer = TPVDimReducer(method="csp")
    # Vérifie que fit lève une ValueError en absence de labels
    with pytest.raises(ValueError, match="^y is required for CSP$"):
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
    with pytest.raises(ValueError, match="^CSP requires exactly two classes$"):
        # Lance fit pour atteindre la validation du nombre de classes
        reducer.fit(X, y)


# Vérifie que transform nécessite un modèle entraîné
def test_transform_requires_fit():
    # Crée une entrée tabulaire simple
    X = np.zeros((1, 2))
    # Instancie le réducteur sans apprentissage préalable
    reducer = TPVDimReducer(method="pca")
    # Vérifie que transform lève une ValueError si fit n'est pas appelé
    with pytest.raises(
        ValueError, match="^The model must be fitted before calling transform$"
    ):
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
    with pytest.raises(ValueError, match="^X must be 2D or 3D for transform$"):
        # Appelle transform avec la mauvaise dimensionnalité
        reducer.transform(X)


# Vérifie que le centrage est bien soustrait avant projection en 2D et 3D
def test_transform_subtracts_mean_for_all_supported_shapes():
    # Crée une matrice 2D simple pour vérifier le centrage
    X2d = np.array([[2.0, -1.0], [0.0, 1.0]])
    # Définit une moyenne arbitraire pour vérifier la soustraction
    mean = np.array([1.0, -2.0])
    # Instancie un réducteur avec matrice identité pour isoler le centrage
    reducer = TPVDimReducer(method="pca")
    # Fixe une matrice de projection identité pour suivre directement le centrage
    reducer.w_matrix = np.eye(2)
    # Positionne la moyenne attendue pour l'opération
    reducer.mean_ = mean
    # Applique la transformation 2D pour observer le décalage
    projected_2d = reducer.transform(X2d)
    # Vérifie que la moyenne a été soustraite composante par composante
    np.testing.assert_allclose(projected_2d, X2d - mean)
    # Crée un tenseur 3D pour vérifier le centrage sur les essais
    X3d = np.stack([X2d, X2d + 1.0])
    # Applique la transformation 3D en conservant le même centrage
    projected_3d = reducer.transform(X3d)
    # Calcule l'attendu en soustrayant la moyenne sur chaque essai
    expected_3d = np.stack([X2d - mean, X2d + 1.0 - mean])
    # Vérifie que chaque essai est centré correctement après projection
    np.testing.assert_allclose(projected_3d, expected_3d)


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
    # Prépare des valeurs propres factices pour compléter la restauration
    expected_eig = np.array([2.0, 1.0])
    # Capture le chemin utilisé par joblib.load
    captured = {}

    # Définit un faux loader qui enregistre le chemin reçu
    def fake_load(path: str):
        # Mémorise le chemin pour vérification ultérieure
        captured["path"] = path
        # Retourne un contenu cohérent pour simuler la persistance
        return {"w_matrix": expected_w, "mean": expected_mean, "eig": expected_eig}

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
    # Vérifie la restauration des valeurs propres
    np.testing.assert_allclose(reducer.eigenvalues_, expected_eig)


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
