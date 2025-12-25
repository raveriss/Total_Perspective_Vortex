# Importe NumPy pour générer des données synthétiques stables
import numpy as np

# Importe pytest pour vérifier les erreurs attendues
import pytest

# Importe le réducteur dimensionnel de TPV à vérifier
from tpv.dimensionality import TPVDimReducer

# Définit un seuil de domination pour la variance expliquée
DOMINANT_VARIANCE_THRESHOLD = 0.9


def test_init_defaults_lock_api_contract() -> None:
    """Le constructeur doit exposer des défauts stables pour l'API."""

    # Instancie sans argument pour valider le contrat des valeurs par défaut
    reducer = TPVDimReducer()
    # Verrouille la méthode par défaut attendue par la pipeline
    assert reducer.method == "csp"
    # Verrouille l'absence de limitation de composantes par défaut
    assert reducer.n_components is None
    # Verrouille la régularisation nulle par défaut pour éviter un biais caché
    assert reducer.regularization == 0.0
    # Verrouille l'état non entraîné pour empêcher un usage avant fit
    assert reducer.w_matrix is None
    # Verrouille l'absence de moyenne avant l'apprentissage PCA
    assert reducer.mean_ is None
    # Verrouille l'absence de valeurs propres avant l'apprentissage
    assert reducer.eigenvalues_ is None


def test_csp_returns_log_variances_and_orthogonality() -> None:
    """CSP doit générer des filtres orthogonaux et séparer l'énergie."""

    # Prépare un générateur déterministe pour les données
    rng = np.random.default_rng(0)
    # Fixe le nombre d'essais par classe
    trials_per_class = 10
    # Fixe le nombre de canaux simulés
    channels = 3
    # Fixe le nombre d'échantillons temporels
    time_points = 64
    # Génère des essais faiblement énergétiques pour la classe A
    class_a = rng.standard_normal((trials_per_class, channels, time_points)) * 0.5
    # Génère des essais fortement énergétiques pour la classe B
    class_b = rng.standard_normal((trials_per_class, channels, time_points)) * 2.0
    # Concatène les essais pour former le jeu complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires associés
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Instancie le réducteur CSP avec régularisation légère
    reducer = TPVDimReducer(method="csp", n_components=channels, regularization=1e-3)
    # Apprend la matrice de projection sur les essais
    reducer.fit(trials, labels)
    # Transforme les essais pour obtenir des log-variances par composante
    projected = reducer.transform(trials)
    # Vérifie la forme des features renvoyées
    assert projected.shape == (trials_per_class * 2, channels)
    # Calcule la moyenne des features par classe pour vérifier la séparation
    mean_a = projected[:trials_per_class].mean(axis=0)
    # Calcule la moyenne pour la seconde classe
    mean_b = projected[trials_per_class:].mean(axis=0)
    # Vérifie que la classe énergique produit des variances plus élevées
    assert np.all(mean_b > mean_a)
    # Reconstruit la matrice composite pour vérifier l'orthogonalité
    composite = reducer._regularize_matrix(
        reducer._average_covariance(class_a) + reducer._average_covariance(class_b)
    )
    # Vérifie que la matrice W apprise est disponible avant le produit
    assert reducer.w_matrix is not None
    # Calcule le produit attendu proche de l'identité
    identity_candidate = reducer.w_matrix.T @ composite @ reducer.w_matrix
    # Vérifie l'orthogonalité dans l'espace régularisé
    assert np.allclose(identity_candidate, np.eye(channels), atol=1e-2)


# Vérifie que la moyenne de covariance reste normalisée et régularisée
def test_average_covariance_regularizes_diagonal() -> None:
    # Instancie un réducteur CSP avec régularisation diagonale
    reducer = TPVDimReducer(method="csp", regularization=0.1)
    # Construit deux essais orthogonaux pour isoler les contributions
    trials = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    # Calcule la covariance moyenne sur les essais synthétiques
    averaged = reducer._average_covariance(trials)
    # Vérifie que la matrice résultante conserve la diagonale attendue
    assert np.allclose(np.diag(averaged), [0.6, 0.6])
    # Vérifie que les termes hors diagonale restent nuls après moyenne
    assert averaged[0, 1] == pytest.approx(0.0)


def test_average_covariance_normalizes_by_trace_and_divides_by_trial_count() -> None:
    """La covariance moyenne doit être normalisée par trace et moyennée par essais."""

    # Instancie un réducteur CSP sans régularisation pour isoler la moyenne
    reducer = TPVDimReducer(method="csp", regularization=0.0)
    # Construit un essai dont la trace de covariance vaut 4
    trial_high_trace = np.array([[2.0, 0.0], [0.0, 0.0]])
    # Construit un essai dont la trace de covariance vaut 9
    trial_other_trace = np.array([[0.0, 0.0], [0.0, 3.0]])
    # Construit un essai identité dont la trace de covariance vaut 2
    trial_identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Empile trois essais pour casser l'égalité essais == canaux
    trials = np.stack([trial_high_trace, trial_other_trace, trial_identity], axis=0)
    # Calcule la covariance moyenne à partir des essais synthétiques
    averaged = reducer._average_covariance(trials)
    # Verrouille la forme pour confirmer le calcul canal x canal
    assert averaged.shape == (2, 2)
    # Vérifie que la diagonale est exactement celle attendue après normalisation
    assert np.allclose(np.diag(averaged), [0.5, 0.5], atol=1e-12)
    # Vérifie que l'hors diagonale reste nul sur ces essais orthogonaux
    assert averaged[0, 1] == pytest.approx(0.0)
    # Vérifie la symétrie attendue d'une covariance moyenne
    assert averaged[1, 0] == pytest.approx(0.0)


def test_regularize_matrix_makes_explicit_copy_and_does_not_share_memory(
    monkeypatch,
) -> None:
    """La régularisation doit retourner une copie indépendante de l'entrée."""

    # Importe le module pour patcher np.array au bon endroit
    import tpv.dimensionality as dimensionality_module

    # Conserve la fonction array d'origine pour déléguer sans changer NumPy
    original_array = dimensionality_module.np.array
    # Prépare une capture pour verrouiller l'usage explicite de copy=True
    captured: dict[str, object] = {}

    # Définit un espion pour détecter si copy=True est passé explicitement
    def spy_array(obj, *args, **kwargs):
        # Marque la présence de l'argument copy pour tuer l'appel implicite
        captured["has_copy_kw"] = "copy" in kwargs
        # Conserve la valeur de copy pour tuer copy=False
        captured["copy_value"] = kwargs.get("copy")
        # Délègue à NumPy pour préserver le comportement normal
        return original_array(obj, *args, **kwargs)

    # Patche np.array dans le module afin d'observer l'appel interne
    monkeypatch.setattr(dimensionality_module.np, "array", spy_array)

    # Instancie un réducteur avec régularisation nulle pour isoler la copie
    reducer = TPVDimReducer(method="pca", regularization=0.0)
    # Construit une matrice ndarray pour permettre un partage mémoire si copy=False
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Applique la régularisation pour déclencher la copie interne
    regularized = reducer._regularize_matrix(matrix)

    # Verrouille que copy est passé explicitement pour rendre l'intention stable
    assert captured.get("has_copy_kw") is True
    # Verrouille copy=True pour empêcher toute aliasing avec l'entrée
    assert captured.get("copy_value") is True
    # Verrouille l'absence de partage mémoire pour éviter une modification indirecte
    assert not np.shares_memory(regularized, matrix)
    # Vérifie que modifier la sortie ne peut pas impacter l'entrée
    regularized[0, 0] = 999.0
    # Verrouille que l'entrée reste inchangée malgré une mutation de la sortie
    assert matrix[0, 0] == pytest.approx(1.0)


def test_regularize_matrix_does_not_inject_identity_when_regularization_is_zero() -> (
    None
):
    """La branche d'injection doit rester inactive quand la régularisation vaut zéro."""

    # Instancie un réducteur avec régularisation exactement nulle
    reducer = TPVDimReducer(method="pca", regularization=0.0)
    # Construit une matrice entière pour rendre visible une addition float inutile
    matrix = np.array([[1, 0], [0, 1]], dtype=int)
    # Applique la régularisation pour vérifier l'absence d'addition in-place
    regularized = reducer._regularize_matrix(matrix)
    # Verrouille l'égalité pour confirmer qu'aucune modification n'est appliquée
    assert np.array_equal(regularized, matrix)
    # Verrouille le dtype pour détecter une injection inutile même à facteur zéro
    assert regularized.dtype == matrix.dtype


def test_regularize_matrix_builds_identity_from_first_dimension(monkeypatch) -> None:
    """La taille de l'identité doit dépendre de shape[0] et pas de shape[1]."""

    # Importe le module pour patcher np.eye au bon endroit
    import tpv.dimensionality as dimensionality_module

    # Conserve la fonction eye d'origine pour déléguer sans modifier NumPy
    original_eye = dimensionality_module.np.eye
    # Prépare une capture pour verrouiller l'argument de np.eye
    captured: dict[str, object] = {}

    # Définit un espion pour enregistrer la taille demandée pour l'identité
    def spy_eye(n):
        # Conserve n pour détecter shape[0] versus shape[1]
        captured["n"] = n
        # Délègue à NumPy pour préserver la forme et le dtype attendus
        return original_eye(n)

    # Patche np.eye dans le module afin d'observer l'appel interne
    monkeypatch.setattr(dimensionality_module.np, "eye", spy_eye)

    # Instancie un réducteur avec régularisation positive pour forcer l'identité
    reducer = TPVDimReducer(method="pca", regularization=0.1)
    # Construit une matrice non carrée pour rendre observable le choix de dimension
    matrix = np.ones((2, 3))
    # Vérifie que l'addition échoue sur une matrice non carrée
    with pytest.raises(ValueError):
        reducer._regularize_matrix(matrix)
    # Verrouille que np.eye utilise la première dimension de la matrice
    assert captured.get("n") == matrix.shape[0]


def test_pca_explained_variance_and_projection_shape() -> None:
    """PCA doit ordonner les composantes par variance et rester orthonormé."""

    # Prépare un générateur déterministe pour les observations
    rng = np.random.default_rng(1)
    # Fixe le nombre d'échantillons
    samples = 120
    # Génère une composante principale dominante
    dominant = rng.standard_normal(samples) * 4.0
    # Génère deux composantes de bruit réduites
    noise1 = rng.standard_normal(samples) * 0.2
    noise2 = rng.standard_normal(samples) * 0.1
    # Assemble les observations en matrice (samples x features)
    observations = np.column_stack([dominant + noise1, noise1, noise2])
    # Instancie le PCA en gardant deux composantes
    reducer = TPVDimReducer(method="pca", n_components=2, regularization=1e-4)
    # Apprend la projection sur les observations
    reducer.fit(observations)
    # Transforme les observations pour obtenir les scores
    transformed = reducer.transform(observations)
    # Vérifie la forme de la projection obtenue
    assert transformed.shape == (samples, 2)
    # Vérifie que les vecteurs propres sont orthonormés
    assert reducer.w_matrix is not None
    assert np.allclose(reducer.w_matrix.T @ reducer.w_matrix, np.eye(2), atol=1e-6)
    # Calcule le ratio de variance expliquée
    assert reducer.eigenvalues_ is not None
    variance_ratio = reducer.eigenvalues_ / reducer.eigenvalues_.sum()
    # Vérifie que la première composante capte la majorité de la variance
    assert variance_ratio[0] > DOMINANT_VARIANCE_THRESHOLD
    # Vérifie que la seconde composante reste strictement positive
    assert variance_ratio[1] > 0


def test_pca_regularization_stops_nan_on_duplicate_features() -> None:
    """La régularisation doit stabiliser une covariance singulière."""

    # Prépare une caractéristique unique répliquée deux fois
    base = np.linspace(0, 1, num=50)
    # Construit une matrice avec deux colonnes identiques
    duplicated = np.column_stack([base, base])
    # Instancie le PCA avec régularisation pour éviter les nan
    reducer = TPVDimReducer(method="pca", regularization=1e-3)
    # Apprend la projection malgré la singularité de la covariance
    reducer.fit(duplicated)
    # Transforme les données pour vérifier la stabilité numérique
    transformed = reducer.transform(duplicated)
    # Vérifie l'absence de valeurs non finies
    assert np.isfinite(transformed).all()


def test_regularized_covariance_matches_sample_covariance_without_regularization() -> (
    None
):
    """La covariance doit être normalisée par (n_samples - 1) avant régularisation."""

    # Instancie un réducteur PCA sans régularisation pour isoler la formule
    reducer = TPVDimReducer(method="pca", regularization=0.0)
    # Construit une matrice centrée (n_samples=3, n_features=2)
    centered = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    # Calcule la covariance régularisée pour exercer la normalisation interne
    covariance = reducer._regularized_covariance(centered)
    # Calcule X^T X attendu pour verrouiller l'échelle de normalisation
    xtx = centered.T @ centered
    # Calcule la covariance échantillon attendue avec (n_samples - 1)
    expected = xtx / (centered.shape[0] - 1)
    # Vérifie l'égalité numérique pour détecter toute dérive de facteur
    assert np.allclose(covariance, expected)


def test_pca_fit_records_feature_mean_and_is_translation_invariant() -> None:
    """PCA doit centrer par feature et rester invariant à une translation."""

    # Prépare un générateur déterministe pour stabiliser les tirages
    rng = np.random.default_rng(11)
    # Fixe un nombre d'échantillons suffisant pour stabiliser la covariance
    samples = 240
    # Fixe des échelles distinctes pour rendre les valeurs propres séparées
    scales = np.array([3.0, 1.0, 0.2])
    # Génère une matrice base avec variances distinctes par feature
    base = rng.standard_normal((samples, 3)) * scales
    # Définit un offset constant pour tester l'invariance par translation
    offset = np.array([100.0, -50.0, 10.0])
    # Construit un second jeu translaté sans changer la covariance centrée
    shifted = base + offset
    # Instancie un réducteur PCA sans coupe de composantes
    reference = TPVDimReducer(method="pca", regularization=1e-6)
    # Apprend la projection PCA sur le jeu de base
    reference.fit(base)
    # Vérifie que la moyenne est bien calculée et stockée
    assert reference.mean_ is not None
    # Vérifie que la moyenne stockée correspond à une moyenne par feature
    assert reference.mean_.shape == (3,)
    # Vérifie que la moyenne stockée correspond à NumPy axis=0
    assert np.allclose(reference.mean_, np.mean(base, axis=0))
    # Vérifie que les valeurs propres sont exposées après l'apprentissage
    assert reference.eigenvalues_ is not None
    # Vérifie que le spectre complet est conservé sans n_components
    assert reference.eigenvalues_.shape == (3,)
    # Vérifie que le spectre est trié par ordre décroissant
    assert np.all(np.diff(reference.eigenvalues_) <= 0)
    # Vérifie que la matrice de projection est disponible après fit
    assert reference.w_matrix is not None
    # Vérifie que la matrice conserve toutes les composantes
    assert reference.w_matrix.shape == (3, 3)
    # Apprend un second PCA sur le jeu translaté
    translated = TPVDimReducer(method="pca", regularization=1e-6)
    # Apprend la projection PCA sur le jeu translaté
    translated.fit(shifted)
    # Vérifie que les valeurs propres restent invariantes sous translation
    assert translated.eigenvalues_ is not None
    # Vérifie que le spectre correspond à celui du jeu non translaté
    assert np.allclose(translated.eigenvalues_, reference.eigenvalues_, atol=1e-6)
    # Vérifie que la matrice de projection est disponible après fit
    assert translated.w_matrix is not None
    # Calcule le produit scalaire colonne à colonne pour détecter les flips de signe
    column_dots = np.sum(reference.w_matrix * translated.w_matrix, axis=0)
    # Calcule l'alignement absolu pour tolérer un changement de signe
    column_alignment = np.abs(column_dots)
    # Vérifie que les colonnes restent alignées quand les valeurs propres sont séparées
    assert np.all(column_alignment > 0.99)
    # Calcule les scores sur le jeu de base
    base_scores = reference.transform(base)
    # Calcule les scores sur le jeu translaté
    shifted_scores = translated.transform(shifted)
    # Calcule les signes pour aligner les composantes entre les deux PCA
    signs = np.sign(column_dots)
    # Remplace les zéros numériques par un signe neutre
    signs[signs == 0] = 1
    # Aligne les signes pour comparer les scores de manière stable
    aligned_shifted_scores = shifted_scores * signs
    # Vérifie que la translation ne change pas les scores centrés
    assert np.allclose(aligned_shifted_scores, base_scores, atol=1e-5)


def test_save_and_load_roundtrip(tmp_path) -> None:
    """La sérialisation joblib doit préserver la projection apprise."""

    # Prépare des observations aléatoires pour le PCA
    rng = np.random.default_rng(5)
    # Génère une matrice d'observations tabulaires
    observations = rng.standard_normal((40, 3))
    # Instancie et entraîne le réducteur PCA
    reducer = TPVDimReducer(method="pca", n_components=2)
    # Apprend la projection sur les observations fournies
    reducer.fit(observations)
    # Transforme les observations pour servir de référence
    reference = reducer.transform(observations)
    # Définit le chemin de sauvegarde dans le répertoire temporaire
    target = tmp_path / "dim_reducer.joblib"
    # Sauvegarde le modèle entraîné
    reducer.save(target)
    # Crée une nouvelle instance pour charger le modèle
    restored = TPVDimReducer(method="pca")
    # Charge les paramètres depuis le fichier joblib
    restored.load(target)
    # Applique la transformation avec le modèle restauré
    loaded_projection = restored.transform(observations)
    # Vérifie que la projection rechargée correspond à la référence
    assert np.allclose(loaded_projection, reference)


def test_load_restores_expected_keys_from_payload(monkeypatch, tmp_path) -> None:
    """load doit restaurer eig, method, n_components et regularization via les bonnes clés."""

    # Importe le module pour patcher joblib.load au bon endroit
    import tpv.dimensionality as dimensionality_module

    # Construit une matrice de projection stable pour valider la restauration
    w_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Construit une moyenne explicite pour valider la restauration
    mean = np.array([0.5, -0.25])
    # Construit des valeurs propres explicites pour valider la clé "eig"
    eig = np.array([2.0, 1.0])
    # Fixe une méthode distincte de l'état initial pour détecter data.get(self.method)
    method = "pca"
    # Fixe un n_components distinct de l'état initial pour détecter data.get(self.n_components)
    n_components = 2
    # Fixe une régularisation distincte de l'état initial pour détecter data.get(self.regularization)
    regularization = 1e-3

    # Prépare un payload minimal et exact attendu par load
    payload = {
        "w_matrix": w_matrix,
        "mean": mean,
        "eig": eig,
        "method": method,
        "n_components": n_components,
        "regularization": regularization,
    }

    # Capture le chemin reçu pour verrouiller le str(path) passé à joblib.load
    captured: dict[str, object] = {}

    # Remplace joblib.load pour éviter l'I/O et renvoyer un payload contrôlé
    def fake_load(path: str):
        # Conserve le path pour s'assurer que load convertit bien en str
        captured["path"] = path
        # Retourne le payload exact pour tester la lecture des clés
        return payload

    # Patche joblib.load dans le module où TPVDimReducer l'utilise
    monkeypatch.setattr(dimensionality_module.joblib, "load", fake_load)

    # Instancie avec des valeurs initiales différentes pour rendre les mutations visibles
    restored = TPVDimReducer(method="csp", n_components=7, regularization=0.25)
    # Définit une destination factice pour exercer la conversion en str
    target = tmp_path / "loaded.joblib"
    # Charge le modèle depuis le payload patché
    restored.load(target)

    # Verrouille que load passe bien une chaîne à joblib.load
    assert captured.get("path") == str(target)
    # Verrouille la restauration de w_matrix via la clé attendue
    assert restored.w_matrix is not None
    assert np.allclose(restored.w_matrix, w_matrix)
    # Verrouille la restauration de mean via la clé attendue
    assert restored.mean_ is not None
    assert np.allclose(restored.mean_, mean)
    # Verrouille la restauration de eig via la clé exacte "eig"
    assert restored.eigenvalues_ is not None
    assert np.allclose(restored.eigenvalues_, eig)
    # Verrouille la restauration de method via la clé exacte "method"
    assert restored.method == method
    # Verrouille la restauration de n_components via la clé exacte "n_components"
    assert restored.n_components == n_components
    # Verrouille la restauration de regularization via la clé exacte "regularization"
    assert restored.regularization == pytest.approx(regularization)


def test_load_keeps_existing_defaults_when_optional_keys_missing(
    monkeypatch, tmp_path
) -> None:
    """load doit conserver les valeurs courantes si method/n_components/regularization sont absents."""

    # Importe le module pour patcher joblib.load au bon endroit
    import tpv.dimensionality as dimensionality_module

    # Prépare une instance avec des valeurs non triviales pour détecter les None
    reducer = TPVDimReducer(method="pca", n_components=3, regularization=0.125)
    # Construit une matrice minimale pour rendre load atteignable
    reducer.w_matrix = np.eye(2)
    # Construit un payload volontairement incomplet sur les champs optionnels
    payload = {
        "w_matrix": np.eye(2),
        "mean": np.array([0.0, 0.0]),
        # Omet "method" pour forcer le fallback vers l'état courant
        # Omet "n_components" pour forcer le fallback vers l'état courant
        # Omet "regularization" pour forcer le fallback vers l'état courant
        # Omet "eig" car il est optionnel quand on ne veut pas l'exposer
    }

    # Remplace joblib.load pour éviter l'I/O et renvoyer le payload incomplet
    monkeypatch.setattr(dimensionality_module.joblib, "load", lambda _: payload)

    # Charge depuis un chemin factice pour exercer la branche load
    reducer.load(tmp_path / "missing_keys.joblib")

    # Verrouille que l'absence de "method" ne force pas method à None
    assert reducer.method == "pca"
    # Verrouille que l'absence de "n_components" ne force pas n_components à None
    assert reducer.n_components == 3
    # Verrouille que l'absence de "regularization" ne force pas regularization à None
    assert reducer.regularization == pytest.approx(0.125)


def test_save_serializes_expected_payload_keys_and_values(
    tmp_path, monkeypatch
) -> None:
    """save doit sérialiser un payload stable pour permettre load."""

    # Importe le module pour patcher joblib.dump au bon endroit
    import tpv.dimensionality as dimensionality_module

    # Prépare des observations déterministes pour stabiliser la projection
    rng = np.random.default_rng(11)
    # Génère une matrice tabulaire suffisante pour apprendre PCA
    observations = rng.standard_normal((30, 4))
    # Fixe une régularisation non nulle pour verrouiller sa persistance
    regularization = 1e-3
    # Instancie un PCA avec composantes contrôlées pour stabiliser la sérialisation
    reducer = TPVDimReducer(method="pca", n_components=2, regularization=regularization)
    # Apprend la projection pour rendre save atteignable
    reducer.fit(observations)
    # Prépare une capture pour inspecter l'appel à joblib.dump
    captured = {}

    # Remplace joblib.dump pour éviter l'I/O et capturer le payload exact
    def fake_dump(payload, path) -> None:
        # Conserve le payload pour verrouiller le contrat de clés et de valeurs
        captured["payload"] = payload
        # Conserve le chemin pour verrouiller la conversion explicite en str
        captured["path"] = path

    # Patche joblib.dump dans le module où TPVDimReducer l'utilise
    monkeypatch.setattr(dimensionality_module.joblib, "dump", fake_dump)
    # Définit une destination de sauvegarde contrôlée
    target = tmp_path / "dim_reducer_payload.joblib"
    # Exécute la sauvegarde afin de capturer les données sérialisées
    reducer.save(target)
    # Récupère le payload capturé pour assertions structurantes
    payload = captured.get("payload")
    # Vérifie que la sérialisation produit bien un dictionnaire
    assert isinstance(payload, dict)
    # Verrouille strictement les clés attendues par load et par l'API
    assert set(payload.keys()) == {
        "w_matrix",
        "mean",
        "eig",
        "method",
        "n_components",
        "regularization",
    }
    # Verrouille la méthode persistée pour éviter des incohérences au rechargement
    assert payload["method"] == "pca"
    # Verrouille le nombre de composantes persisté
    assert payload["n_components"] == 2
    # Verrouille la régularisation persistée
    assert payload["regularization"] == pytest.approx(regularization)
    # Vérifie que la matrice de projection apprise est bien persistée
    assert reducer.w_matrix is not None
    assert np.allclose(payload["w_matrix"], reducer.w_matrix)
    # Vérifie que la moyenne PCA apprise est bien persistée
    assert reducer.mean_ is not None
    assert np.allclose(payload["mean"], reducer.mean_)
    # Vérifie que les valeurs propres apprises sont bien persistées
    assert reducer.eigenvalues_ is not None
    assert np.allclose(payload["eig"], reducer.eigenvalues_)
    # Verrouille que save passe bien une chaîne à joblib.dump
    assert captured.get("path") == str(target)


def test_validation_guards_raise_errors_for_invalid_calls() -> None:
    """Les garde-fous doivent empêcher les appels incohérents."""

    # Prépare une matrice tabulaire pour tester les validations PCA
    tabular = np.ones((4, 3))
    # Prépare un tenseur tridimensionnel pour les validations CSP
    trials = np.ones((2, 2, 5))
    # Instancie PCA avec une méthode incorrecte pour vérifier la validation
    invalid_method = TPVDimReducer(method="unknown")
    # Vérifie que la méthode inconnue est rejetée
    with pytest.raises(ValueError, match=r"^method must be 'pca' or 'csp'$"):
        invalid_method.fit(tabular)
    # Instancie PCA pour déclencher l'erreur de dimension attendue
    pca = TPVDimReducer(method="pca")
    # Vérifie que PCA refuse des données non tabulaires
    with pytest.raises(ValueError, match=r"^PCA expects a 2D array$"):
        pca.fit(trials)
    # Instancie CSP pour tester l'absence de labels
    csp_missing_labels = TPVDimReducer(method="csp")
    # Vérifie que CSP réclame explicitement les labels
    with pytest.raises(ValueError, match=r"^y is required for CSP$"):
        csp_missing_labels.fit(trials)
    # Instancie CSP pour tester la dimension incorrecte
    csp_wrong_shape = TPVDimReducer(method="csp")
    # Vérifie que CSP refuse une entrée qui n'est pas 3D
    with pytest.raises(ValueError, match=r"^CSP expects a 3D array$"):
        csp_wrong_shape.fit(tabular, np.array([0, 1, 0, 1]))
    # Instancie CSP pour tester le nombre de classes invalide
    csp_many_classes = TPVDimReducer(method="csp")
    # Construit des labels comprenant trois classes distinctes
    too_many_labels = np.array([0, 1, 2])
    # Vérifie que CSP exige exactement deux classes
    with pytest.raises(ValueError, match=r"^CSP requires exactly two classes$"):
        csp_many_classes.fit(trials[:3], too_many_labels)
    # Instancie CSP pour vérifier la protection sur la sauvegarde
    csp_unsaved = TPVDimReducer(method="csp")
    # Vérifie que la sauvegarde échoue si fit n'a pas été appelé
    with pytest.raises(ValueError, match=r"^Cannot save before fitting the model$"):
        csp_unsaved.save("/tmp/unused.joblib")
    # Instancie CSP pour vérifier la protection sur la transformation
    csp_unfitted = TPVDimReducer(method="csp")
    # Vérifie que transform refuse d'être appelé avant fit
    with pytest.raises(ValueError) as excinfo:
        csp_unfitted.transform(trials)
    # Verrouille le message pour détecter toute régression silencieuse
    assert str(excinfo.value) == "The model must be fitted before calling transform"


def test_csp_handles_default_component_count_and_empty_trials_guard() -> None:
    """CSP doit couvrir les branches par défaut et refuser les essais vides."""

    # Prépare un générateur déterministe pour reproduire les résultats
    rng = np.random.default_rng(7)
    # Fixe le nombre de canaux pour aligner l'attente des composantes
    channels = 2
    # Génère des essais pour la classe A
    class_a = rng.standard_normal((5, channels, 16))
    # Génère des essais pour la classe B
    class_b = rng.standard_normal((5, channels, 16))
    # Concatène les essais pour former le jeu complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Crée les labels binaires associés
    labels = np.array([0] * 5 + [1] * 5)
    # Instancie le CSP sans spécifier n_components pour couvrir la branche par défaut
    reducer = TPVDimReducer(method="csp")
    # Apprend la matrice de projection en mode par défaut
    reducer.fit(trials, labels)
    # Vérifie que la matrice de projection conserve toutes les composantes
    assert reducer.w_matrix is not None
    assert reducer.w_matrix.shape[1] == channels
    # Vérifie que les valeurs propres sont exposées après fit en mode par défaut
    assert reducer.eigenvalues_ is not None
    # Vérifie que le spectre conserve toutes les composantes en mode par défaut
    assert reducer.eigenvalues_.shape == (channels,)
    # Vérifie que le spectre est trié par ordre décroissant
    assert np.all(np.diff(reducer.eigenvalues_) <= 0)
    # Transforme les essais pour préparer un appel avec une dimension invalide
    _ = reducer.transform(trials)
    # Vérifie que transform refuse une dimension inattendue
    with pytest.raises(ValueError) as excinfo:
        reducer.transform(np.ones(5))
    # Verrouille le message pour détecter les mutations sur le contrat d'erreur
    assert str(excinfo.value) == "X must be 2D or 3D for transform"
    # Instancie un nouvel objet pour tester la covariance vide
    empty_guard = TPVDimReducer(method="csp")
    # Vérifie que la moyenne de covariance échoue sur un tableau vide
    with pytest.raises(
        ValueError, match=r"^No trials provided for covariance estimation$"
    ):
        empty_guard._average_covariance(np.empty((0, channels, 4)))


def test_csp_truncates_components_and_eigenvalues_when_requested() -> None:
    """CSP doit tronquer W et eigenvalues_ quand n_components est défini."""

    # Prépare un générateur déterministe pour stabiliser les essais
    rng = np.random.default_rng(21)
    # Fixe un nombre de canaux supérieur au nombre de composantes demandé
    channels = 4
    # Fixe un nombre d'essais par classe pour stabiliser les covariances
    trials_per_class = 6
    # Fixe un nombre d'échantillons temporels suffisant pour la variance
    time_points = 32
    # Génère les essais pour la première classe
    class_a = rng.standard_normal((trials_per_class, channels, time_points))
    # Génère les essais pour la seconde classe
    class_b = rng.standard_normal((trials_per_class, channels, time_points))
    # Concatène les essais pour former un tenseur complet
    trials = np.concatenate([class_a, class_b], axis=0)
    # Construit des labels binaires pour correspondre aux deux classes
    labels = np.array([0] * trials_per_class + [1] * trials_per_class)
    # Fixe un nombre de composantes inférieur au nombre de canaux
    n_components = 2
    # Instancie un CSP avec une coupe explicite des composantes
    reducer = TPVDimReducer(
        method="csp", n_components=n_components, regularization=1e-3
    )
    # Apprend la projection CSP sur les essais synthétiques
    reducer.fit(trials, labels)
    # Vérifie que la matrice de projection est disponible après fit
    assert reducer.w_matrix is not None
    # Vérifie que la matrice est tronquée selon n_components
    assert reducer.w_matrix.shape == (channels, n_components)
    # Vérifie que les valeurs propres sont exposées après fit
    assert reducer.eigenvalues_ is not None
    # Vérifie que les valeurs propres sont tronquées selon n_components
    assert reducer.eigenvalues_.shape == (n_components,)
    # Vérifie que le spectre reste trié par ordre décroissant
    assert np.all(np.diff(reducer.eigenvalues_) <= 0)
    # Transforme les essais pour vérifier la forme finale des features
    projected = reducer.transform(trials)
    # Vérifie que la projection renvoie bien n_components features
    assert projected.shape == (trials.shape[0], n_components)


def test_transform_applies_mean_centering_by_subtraction() -> None:
    """La transformation doit soustraire la moyenne apprise avant projection."""

    # Instancie un réducteur PCA pour tester la branche tabulaire 2D
    reducer = TPVDimReducer(method="pca")
    # Fixe une matrice identité pour isoler l'effet du centrage
    reducer.w_matrix = np.eye(2)
    # Fixe une moyenne non nulle pour rendre le centrage observable
    reducer.mean_ = np.array([1.0, -2.0])
    # Construit des observations simples pour une vérification exacte
    X = np.array([[1.0, 3.0], [2.0, 1.0]])
    # Applique la transformation pour observer le centrage effectif
    transformed = reducer.transform(X)
    # Vérifie que le centrage est une soustraction et non une addition
    assert np.allclose(transformed, X - reducer.mean_)


def test_transform_csp_eps_stabilizes_zero_variance_and_uses_float_finfo(
    monkeypatch,
) -> None:
    """CSP doit rester fini sur variance nulle et appeler finfo(float)."""

    # Conserve finfo d'origine pour déléguer sans modifier NumPy
    original_finfo = np.finfo
    # Trace les dtypes afin de verrouiller l'argument passé à finfo
    seen_dtypes: list[object] = []

    # Définit un espion pour capturer le dtype utilisé par finfo
    def spy_finfo(dtype):
        # Archive le dtype pour détecter la mutation float -> None
        seen_dtypes.append(dtype)
        # Délègue à finfo original pour préserver la valeur eps
        return original_finfo(dtype)

    # Patch finfo afin d'observer l'appel effectué par transform
    monkeypatch.setattr(np, "finfo", spy_finfo)

    # Instancie un réducteur CSP pour tester la branche 3D
    reducer = TPVDimReducer(method="csp")
    # Fixe une matrice identité pour générer des sorties constantes
    reducer.w_matrix = np.eye(2)
    # Prépare des essais nuls pour obtenir une variance strictement nulle
    trials = np.zeros((2, 2, 8))

    # Force NumPy à lever si log reçoit une valeur négative ou invalide
    with np.errstate(invalid="raise", divide="raise"):
        # Transforme les essais pour valider la stabilisation par +eps
        features = reducer.transform(trials)

    # Verrouille le dtype de finfo pour empêcher l'appel implicite finfo(None)
    assert seen_dtypes == [float]
    # Vérifie que la sortie reste finie malgré la variance nulle
    assert np.isfinite(features).all()
