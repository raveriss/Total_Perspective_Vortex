"""Réduction de dimension pour TPV."""

# Garantit l'accès aux types manipulables par joblib et pathlib
import os

# Offre une persistance simple de la matrice de projection
import joblib

# Garantit que numpy est disponible pour le calcul matriciel
import numpy as np

# Garantit l'accès aux décompositions hermitiennes et aux filtres passe-bande
from scipy import linalg, signal

# Assure l'intégration avec les API scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin

# Fige le nombre de classes attendu pour le CSP pour lever les ambiguïtés
EXPECTED_CSP_CLASSES = 2
# Fige la dimension tabulaire standard pour différencier l'entrée
TABLE_DIMENSION = 2
# Fige la dimension trial x channel x time attendue pour le CSP
TRIAL_DIMENSION = 3

# Le banc cible les rythmes mu et bêta associés aux tâches motrices.
DEFAULT_FILTER_BANK_BANDS = (
    (7.0, 12.0),
    (10.0, 15.0),
    (12.0, 18.0),
    (15.0, 22.0),
    (18.0, 26.0),
    (22.0, 30.0),
)
# Trois fenêtres capturent le début, le centre et la fin de la réponse motrice.
DEFAULT_FILTER_BANK_WINDOWS = ((0.0, 2.0), (0.75, 2.75), (1.5, 3.5))
FILTER_BANK_ORDER = 4


# Centralise la stabilisation numérique utilisée par les deux réducteurs.
def _regularize_covariance(covariance: np.ndarray, regularization: float) -> np.ndarray:
    """Retourne une copie régularisée d'une matrice de covariance."""

    # La copie interdit une mutation discrète des covariances appelantes.
    regularized_covariance = np.array(covariance, copy=True)
    if regularization < 0.0 or regularization > 1.0:
        raise ValueError("regularization must be between 0 and 1")
    if regularization > 0.0:
        if (
            covariance.ndim != TABLE_DIMENSION
            or covariance.shape[0] != covariance.shape[1]
        ):
            raise ValueError("covariance must be a square matrix")
        # Le shrinkage conserve l'échelle d'une covariance normalisée par sa trace.
        average_variance = float(np.trace(covariance)) / covariance.shape[0]
        identity_target = average_variance * np.eye(covariance.shape[0])
        regularized_covariance = (
            1.0 - regularization
        ) * regularized_covariance + regularization * identity_target
    # Le même résultat alimente désormais TPVDimReducer et CSP.
    return regularized_covariance


# Centralise la covariance normalisée commune aux implémentations CSP/CSSP.
def _compute_average_covariance(
    trials: np.ndarray, regularization: float
) -> np.ndarray:
    """Moyenne les covariances normalisées d'une collection d'essais."""

    # Une classe vide rendrait la moyenne et le CSP indéfinis.
    if trials.size == 0:
        # Conserve le diagnostic historique attendu par les appelants.
        raise ValueError("No trials provided for covariance estimation")
    # Prépare une matrice carrée alignée sur le nombre de canaux.
    covariance_sum = np.zeros((trials.shape[1], trials.shape[1]))
    # Chaque essai contribue de manière égale indépendamment de son énergie.
    for trial in trials:
        # Le produit canal par canal produit la covariance spatiale brute.
        trial_covariance = trial @ trial.T
        # La trace rend les essais comparables avant leur moyenne.
        trial_covariance /= np.trace(trial_covariance)
        # L'accumulation évite de construire un tenseur intermédiaire volumineux.
        covariance_sum += trial_covariance
    # Divise une seule fois pour conserver exactement l'algorithme précédent.
    average_covariance = np.asarray(covariance_sum / trials.shape[0])
    # Applique la même régularisation dans les deux classes publiques.
    return _regularize_covariance(average_covariance, regularization)


# Centralise le tri décroissant utilisé par PCA, CSP et CSSP.
def _select_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_components: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trie les paires propres et conserve les premières composantes."""

    order = np.argsort(eigenvalues)[::-1]
    sorted_values = eigenvalues[order]
    sorted_vectors = eigenvectors[:, order]
    if n_components is not None:
        sorted_values = sorted_values[:n_components]
        sorted_vectors = sorted_vectors[:, :n_components]
    return sorted_values, sorted_vectors


# Résout une seule fois le problème généralisé commun à CSP et CSSP.
def _solve_csp_filters(
    trials: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    n_components: int | None,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne les valeurs propres et filtres spatiaux CSP ordonnés."""

    cov_a = _compute_average_covariance(trials[y == classes[0]], regularization)
    cov_b = _compute_average_covariance(trials[y == classes[1]], regularization)
    composite = cov_a + cov_b
    eigenvalues, eigenvectors = linalg.eigh(cov_a, composite)
    # Les deux extrêmes portent respectivement l'information des deux classes.
    order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
    sorted_values = eigenvalues[order]
    sorted_vectors = eigenvectors[:, order]
    if n_components is not None:
        sorted_values = sorted_values[:n_components]
        sorted_vectors = sorted_vectors[:, :n_components]
    return np.asarray(sorted_values), np.asarray(sorted_vectors)


# Centralise la projection temporelle W^T X utilisée par les deux transformeurs.
def _project_trials(w_matrix: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Projette des essais et restitue l'ordre trial, composante, temps."""

    projected = np.tensordot(w_matrix.T, trials, axes=([1], [1]))
    return np.asarray(np.moveaxis(projected, 1, 0))


class TPVDimReducer(BaseEstimator, TransformerMixin):
    """Réducteur de dimension via PCA ou CSP."""

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def __init__(
        self,
        method: str = "csp",
        n_components: int | None = None,
        regularization: float = 0.0,
    ):
        # Conserve la méthode demandée par l'utilisateur
        self.method = method
        # Conserve le nombre de composantes souhaité
        self.n_components = n_components
        # Conserve la régularisation ajoutée aux covariances
        self.regularization = regularization
        # Initialise la matrice de projection à None avant apprentissage
        self.w_matrix: np.ndarray | None
        # Initialise la moyenne pour la centration éventuelle
        self.mean_: np.ndarray | None
        # Prépare le stockage des valeurs propres pour validation et débogage
        self.eigenvalues_: np.ndarray | None
        # Prépare le stockage des valeurs singulières pour l'option SVD
        self.singular_values_: np.ndarray | None
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None
        # Positionne None avant calcul des valeurs singulières
        self.singular_values_ = None

    # Apprend la matrice de projection à partir des données et des labels
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp", "svd"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca', 'csp', or 'svd'")
        # Délègue le calcul aux helpers spécialisés pour réduire la complexité
        if self.method == "pca":
            self._fit_pca(X)
        elif self.method == "svd":
            self._fit_svd(X)
        else:
            self._fit_csp(X, y)
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Transforme les données en appliquant la matrice de projection apprise
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is not None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = X - self.mean_
        # Applique la projection aux données tabulaires classiques
        if X.ndim == TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai via la formule partagée avec CSP/CSSP.
            reordered = _project_trials(self.w_matrix, X)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Enregistre la matrice de projection pour réutilisation future
    def save(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "singular_values": self.singular_values_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Charge la matrice de projection depuis un fichier joblib
    def load(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure les valeurs singulières éventuelles
        self.singular_values_ = data.get("singular_values")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def _average_covariance(self, trials: np.ndarray) -> np.ndarray:
        # Délègue la formule partagée sans changer la valeur numérique obtenue.
        return _compute_average_covariance(trials, self.regularization)

    # Calcule une covariance régularisée pour les données tabulaires
    def _regularized_covariance(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def _regularize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Délègue la stabilisation au helper partagé avec la classe CSP.
        return _regularize_covariance(matrix, self.regularization)

    # Applique l'apprentissage PCA sur des données tabulaires
    def _fit_pca(self, X: np.ndarray) -> None:
        """Apprend la projection PCA via la covariance régularisée."""

        # Vérifie que les données sont tabulaires pour PCA
        if X.ndim != TABLE_DIMENSION:
            # Informe que PCA attend des données échantillon x feature
            raise ValueError("PCA expects a 2D array")
        # Centre les données pour une covariance cohérente
        self.mean_ = np.mean(X, axis=0)
        # Calcule les données centrées pour la covariance
        centered = X - self.mean_
        # Calcule la covariance avec régularisation diagonale
        covariance = self._regularized_covariance(centered)
        # Extrait les vecteurs propres pour définir la projection
        eigvals, eigvecs = np.linalg.eigh(covariance)
        # Trie et tronque les mêmes paires propres via le helper partagé.
        eigvals, sorted_vecs = _select_eigenpairs(eigvals, eigvecs, self.n_components)
        # Stocke la matrice de projection apprise
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres associées pour vérification externe
        self.eigenvalues_ = eigvals
        # Réinitialise les valeurs singulières quand PCA est utilisé
        self.singular_values_ = None

    # Applique l'apprentissage SVD sur des données tabulaires
    def _fit_svd(self, X: np.ndarray) -> None:
        """Apprend la projection SVD via la covariance centrée."""

        # Vérifie que les données sont tabulaires pour la SVD
        if X.ndim != TABLE_DIMENSION:
            # Informe que SVD attend des données échantillon x feature
            raise ValueError("SVD expects a 2D array")
        # Centre les données pour stabiliser la covariance
        self.mean_ = np.mean(X, axis=0)
        # Calcule les données centrées pour la SVD
        centered = X - self.mean_
        # Calcule une SVD maison à partir de la covariance centrée
        _u_matrix, singular_values, v_matrix = self._svd_from_covariance(centered)
        # Limite le nombre de composantes si demandé
        if self.n_components is not None:
            # Tronque la base de projection au nombre souhaité
            v_matrix = v_matrix[:, : self.n_components]
            # Tronque les valeurs singulières
            singular_values = singular_values[: self.n_components]
        # Stocke la matrice de projection apprise
        self.w_matrix = v_matrix
        # Stocke les valeurs singulières pour inspection
        self.singular_values_ = singular_values
        # Dérive les valeurs propres de la covariance à partir des singulières
        self.eigenvalues_ = (
            singular_values**2 / (centered.shape[0] - 1)
            if centered.shape[0] > 1
            else singular_values**2
        )

    # Applique l'apprentissage CSP sur des essais EEG 3D
    def _fit_csp(self, X: np.ndarray, y: np.ndarray | None) -> None:
        """Apprend la projection CSP à partir des essais et labels."""

        # Vérifie la présence des étiquettes pour la méthode CSP
        if y is None:
            # Informe que CSP requiert des labels binaires
            raise ValueError("y is required for CSP")
        # Valide la dimension trial x channel x time attendue
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP demande des essais temporels bruts
            raise ValueError("CSP expects a 3D array")
        # Identifie les classes présentes pour contrôler le problème
        classes = np.unique(y)
        # Valide que seules deux classes sont fournies
        if classes.size != EXPECTED_CSP_CLASSES:
            # Empêche un calcul CSP invalide avec plus de deux classes
            raise ValueError("CSP requires exactly two classes")
        # Résout la formule CSP partagée sans modifier son ordre numérique.
        eigvals, sorted_vecs = _solve_csp_filters(
            X, y, classes, self.n_components, self.regularization
        )
        # Stocke la matrice de projection CSP
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres pour inspection éventuelle
        self.eigenvalues_ = eigvals
        # Réinitialise les valeurs singulières pour CSP
        self.singular_values_ = None

    # Reconstruit une SVD via la covariance pour conserver un contrôle local
    def _svd_from_covariance(
        self, centered: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit une SVD maison via la covariance centrée."""

        # Vérifie que les données sont bien 2D pour la SVD
        if centered.ndim != TABLE_DIMENSION:
            # Signale une incohérence de dimension avant le calcul
            raise ValueError("SVD expects a 2D array")
        # Calcule la covariance régularisée pour stabiliser le spectre
        covariance = self._regularized_covariance(centered)
        # Résout la décomposition en valeurs propres de la covariance
        eigvals, eigvecs = np.linalg.eigh(covariance)
        # Trie les valeurs propres en ordre décroissant
        order = np.argsort(eigvals)[::-1]
        # Réordonne les vecteurs propres en conséquence
        eigvecs = eigvecs[:, order]
        # Force les valeurs propres négatives à zéro pour stabilité numérique
        eigvals = np.clip(eigvals[order], 0.0, None)
        # Convertit les valeurs propres en valeurs singulières de X
        singular_values = np.sqrt(eigvals * max(centered.shape[0] - 1, 1))
        # Construit V (vecteurs propres) pour la projection
        v_matrix = eigvecs
        # Calcule U en normalisant les projections sur V
        u_matrix = centered @ v_matrix
        # Évite les divisions par zéro en masquant les valeurs singulières nulles
        nonzero = singular_values > 0
        if np.any(nonzero):
            # Normalise chaque colonne de U par la valeur singulière associée
            u_matrix[:, nonzero] = u_matrix[:, nonzero] / singular_values[nonzero]
        # Retourne les matrices U, S, V pour inspection éventuelle
        return u_matrix, singular_values, v_matrix


# Implémente un transformeur CSP/CSSP pour les signaux EEG
class CSP(BaseEstimator, TransformerMixin):
    """Transformeur CSP/CSSP appliquant W^T X = X_CSP sur X ∈ R^{d×N}."""

    # Déclare le constructeur pour configurer CSP ou CSSP
    def __init__(
        self,
        n_components: int | None = None,
        regularization: float = 0.0,
        method: str = "csp",
        cssp_lag: int = 1,
        return_log_variance: bool = True,
    ):
        # Conserve le nombre de composantes souhaité
        self.n_components = n_components
        # Conserve la régularisation appliquée aux covariances
        self.regularization = regularization
        # Conserve la variante de l'algorithme à appliquer
        self.method = method
        # Conserve le décalage temporel pour CSSP
        self.cssp_lag = cssp_lag
        # Conserve le choix de sortie log-variance pour le classifieur
        self.return_log_variance = return_log_variance
        # Prépare la matrice de projection avant apprentissage
        self.w_matrix: np.ndarray | None
        # Prépare le stockage des valeurs propres pour inspection
        self.eigenvalues_: np.ndarray | None
        # Positionne None avant le calcul des filtres
        self.w_matrix = None
        # Positionne None avant la résolution du problème généralisé
        self.eigenvalues_ = None

    # Apprend les filtres spatiaux CSP/CSSP à partir des essais
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Refuse l'appel sans labels pour éviter un apprentissage invalide
        if y is None:
            # Explique que CSP exige des labels de classe
            raise ValueError("y is required for CSP/CSSP")
        # Valide la dimension trial x channel x time attendue
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP/CSSP attend des essais 3D
            raise ValueError("CSP/CSSP expects a 3D array")
        # Valide la variante CSP/CSSP demandée
        if self.method not in {"csp", "cssp"}:
            # Signale un paramètre de méthode non supporté
            raise ValueError("method must be 'csp' or 'cssp'")
        # Identifie les classes présentes pour vérifier le binaire
        classes = np.unique(y)
        # Valide que le problème est strictement binaire
        if classes.size != EXPECTED_CSP_CLASSES:
            # Empêche le calcul CSP/CSSP hors contrat binaire
            raise ValueError("CSP/CSSP requires exactly two classes")
        # Prépare les essais selon la variante choisie
        trials = X
        # Applique l'augmentation CSSP si demandée
        if self.method == "cssp":
            # Transforme les essais en espace élargi temps-décalé
            trials = self._augment_cssp_trials(trials)
        # Résout la formule commune après l'éventuelle augmentation CSSP.
        eigvals, sorted_vecs = _solve_csp_filters(
            trials, y, classes, self.n_components, self.regularization
        )
        # Stocke la matrice de filtres spatiaux
        self.w_matrix = sorted_vecs
        # Stocke les valeurs propres pour inspection ultérieure
        self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Applique les filtres CSP/CSSP pour projeter les signaux
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Vérifie la disponibilité des filtres appris
        if self.w_matrix is None:
            # Interdit la projection avant l'apprentissage
            raise ValueError("The CSP/CSSP model must be fitted before transform")
        # Valide la dimension attendue des essais
        if X.ndim != TRIAL_DIMENSION:
            # Informe que CSP/CSSP attend des essais 3D
            raise ValueError("CSP/CSSP expects a 3D array")
        # Prépare les essais selon la variante choisie
        trials = X
        # Applique l'augmentation CSSP si demandée
        if self.method == "cssp":
            # Transforme les essais en espace élargi temps-décalé
            trials = self._augment_cssp_trials(trials)
        # Projette les essais via la formule partagée avec TPVDimReducer.
        reordered = _project_trials(self.w_matrix, trials)
        # Renvoie les signaux projetés si demandé
        if not self.return_log_variance:
            # Retourne directement W^T X pour les étapes suivantes
            return np.asarray(reordered)
        # Calcule la variance par composante pour chaque essai
        variances = np.var(reordered, axis=2)
        # Stabilise la variance via le log pour les classifieurs
        return np.asarray(np.log(variances + np.finfo(float).eps))

    # Calcule la moyenne des covariances pour une classe d'essais
    def _average_covariance(self, trials: np.ndarray) -> np.ndarray:
        # Délègue la formule partagée sans changer la valeur numérique obtenue.
        return _compute_average_covariance(trials, self.regularization)

    # Ajoute une régularisation diagonale pour stabiliser les covariances
    def _regularize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Délègue la stabilisation au helper partagé avec TPVDimReducer.
        return _regularize_covariance(matrix, self.regularization)

    # Construit les essais augmentés pour CSSP via un retard temporel
    def _augment_cssp_trials(self, trials: np.ndarray) -> np.ndarray:
        # Valide un lag strictement positif pour éviter un doublon
        if self.cssp_lag < 1:
            # Signale que le lag CSSP doit être positif
            raise ValueError("cssp_lag must be >= 1")
        # Refuse un lag supérieur à la durée pour garder des segments valides
        if trials.shape[2] <= self.cssp_lag:
            # Signale que le lag est trop grand pour la fenêtre
            raise ValueError("cssp_lag is too large for the trial length")
        # Extrait le signal sans les derniers échantillons
        base = trials[:, :, : -self.cssp_lag]
        # Extrait le signal décalé pour capturer la dynamique temporelle
        delayed = trials[:, :, self.cssp_lag :]
        # Concatène les canaux originaux et décalés pour CSSP
        return np.concatenate([base, delayed], axis=1)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Extrait des log-variances CSP sur plusieurs bandes et fenêtres."""

    def __init__(  # noqa: PLR0913 - paramètres exposés à GridSearchCV
        self,
        sfreq: float = 160.0,
        bands: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK_BANDS,
        windows: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK_WINDOWS,
        n_components: int = 2,
        regularization: float = 0.1,
        filter_mode: str = "zero_phase",
        time_origin: float = 0.0,
        baseline_window: tuple[float, float] | None = None,
        run_adaptive: bool = True,
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.windows = windows
        self.n_components = n_components
        self.regularization = regularization
        self.filter_mode = filter_mode
        self.time_origin = time_origin
        self.baseline_window = baseline_window
        self.run_adaptive = run_adaptive
        self.filters_: list[np.ndarray] | None = None
        self.eigenvalues_: list[np.ndarray] | None = None
        self.w_matrix: np.ndarray | None = None
        self.filters_per_block_: int | None = None

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        trials = np.asarray(X, dtype=float)
        if trials.ndim != TRIAL_DIMENSION:
            raise ValueError("FilterBankCSP expects a 3D array")
        if self.sfreq <= 0.0:
            raise ValueError("sfreq must be positive")
        if self.n_components < 1 or self.n_components > trials.shape[1]:
            raise ValueError("n_components must be between 1 and n_channels")
        self._validate_bands(self.sfreq / 2.0)
        self._validate_windows(trials.shape[2] / self.sfreq)
        if self.filter_mode not in {"zero_phase", "causal"}:
            raise ValueError("filter_mode must be 'zero_phase' or 'causal'")
        return trials

    def _validate_bands(self, nyquist: float) -> None:
        """Refuse une banque vide ou des bornes fréquentielles invalides."""

        if not self.bands or any(
            low <= 0.0 or low >= high or high >= nyquist for low, high in self.bands
        ):
            raise ValueError("bands must lie strictly between 0 and Nyquist")

    def _validate_windows(self, duration: float) -> None:
        """Vérifie que chaque fenêtre reste dans la durée d'un essai."""

        if not self.windows or any(
            start < self.time_origin
            or start >= stop
            or stop > self.time_origin + duration
            for start, stop in self.windows
        ):
            raise ValueError("windows must lie inside the epoch duration")
        if self.baseline_window is not None:
            start, stop = self.baseline_window
            if (
                start < self.time_origin
                or start >= stop
                or stop > self.time_origin + duration
            ):
                raise ValueError("baseline_window must lie inside the epoch duration")

    def _filter_trials(
        self, trials: np.ndarray, band: tuple[float, float]
    ) -> np.ndarray:
        sos = signal.butter(
            FILTER_BANK_ORDER,
            band,
            btype="bandpass",
            fs=self.sfreq,
            output="sos",
        )
        if self.filter_mode == "causal":
            return np.asarray(signal.sosfilt(sos, trials, axis=2))
        return np.asarray(signal.sosfiltfilt(sos, trials, axis=2))

    def _slice_window(
        self, trials: np.ndarray, window: tuple[float, float]
    ) -> np.ndarray:
        start = int(round((window[0] - self.time_origin) * self.sfreq))
        stop = int(round((window[1] - self.time_origin) * self.sfreq))
        return trials[:, :, start:stop]

    def _valid_run_groups(self, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Conserve les runs ayant les deux classes dans le pli d'entraînement."""

        valid_groups = [
            group
            for group in np.unique(groups)
            if np.unique(labels[groups == group]).size == EXPECTED_CSP_CLASSES
        ]
        if not valid_groups:
            raise ValueError("no run group contains both classes")
        return np.asarray(valid_groups)

    def _fit_filter_blocks(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        classes: np.ndarray,
        unique_groups: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Apprend les blocs CSP par bande, fenêtre et groupe autorisé."""

        filters: list[np.ndarray] = []
        eigenvalues: list[np.ndarray] = []
        for band in self.bands:
            filtered = self._filter_trials(trials, band)
            for window in self.windows:
                windowed = self._slice_window(filtered, window)
                for group in unique_groups:
                    group_mask = (
                        groups == group
                        if self.run_adaptive
                        else np.ones(labels.shape, dtype=bool)
                    )
                    values, vectors = _solve_csp_filters(
                        windowed[group_mask],
                        labels[group_mask],
                        classes,
                        self.n_components,
                        self.regularization,
                    )
                    eigenvalues.append(values)
                    filters.append(vectors)
        return filters, eigenvalues

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        run_groups: np.ndarray | None = None,
    ):
        trials = self._validate_input(X)
        if y is None:
            raise ValueError("y is required for FilterBankCSP")
        labels = np.asarray(y)
        if labels.shape != (trials.shape[0],):
            raise ValueError("y must contain one label per trial")
        classes = np.unique(labels)
        if classes.size != EXPECTED_CSP_CLASSES:
            raise ValueError("FilterBankCSP requires exactly two classes")
        groups = (
            np.zeros(trials.shape[0], dtype=int)
            if run_groups is None
            else np.asarray(run_groups)
        )
        if groups.shape != labels.shape:
            raise ValueError("run_groups must contain one value per trial")
        unique_groups = (
            self._valid_run_groups(labels, groups)
            if self.run_adaptive
            else np.asarray([0])
        )

        filters, eigenvalues = self._fit_filter_blocks(
            trials,
            labels,
            groups,
            classes,
            unique_groups,
        )
        self.filters_ = filters
        self.eigenvalues_ = eigenvalues
        self.w_matrix = np.stack(filters, axis=0)
        self.filters_per_block_ = int(unique_groups.size)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        trials = self._validate_input(X)
        if self.filters_ is None:
            raise ValueError("FilterBankCSP must be fitted before transform")
        if self.filters_per_block_ is None:
            raise ValueError("FilterBankCSP fitted state is incomplete")
        expected_filters = len(self.bands) * len(self.windows) * self.filters_per_block_
        if len(self.filters_) != expected_filters:
            raise ValueError("fitted filter bank is inconsistent with its parameters")

        features: list[np.ndarray] = []
        filter_index = 0
        for band in self.bands:
            filtered = self._filter_trials(trials, band)
            for window in self.windows:
                windowed = self._slice_window(filtered, window)
                for _filter in range(self.filters_per_block_):
                    projected = _project_trials(self.filters_[filter_index], windowed)
                    variances = np.var(projected, axis=2)
                    if self.baseline_window is None:
                        values = variances
                    else:
                        baseline = self._slice_window(filtered, self.baseline_window)
                        baseline_projected = _project_trials(
                            self.filters_[filter_index], baseline
                        )
                        baseline_variances = np.var(baseline_projected, axis=2)
                        values = variances / (baseline_variances + np.finfo(float).eps)
                    features.append(np.log(values + np.finfo(float).eps))
                    filter_index += 1
        return np.concatenate(features, axis=1)


class CovarianceTangentSpace(BaseEstimator, TransformerMixin):
    """Projette des covariances SPD dans un espace tangent log-euclidien.

    Pour chaque essai ``X``, la covariance régularisée ``C`` est blanchie par
    la référence ``G`` apprise sur le train, puis ``log(G^-1/2 C G^-1/2)`` est
    vectorisé. Les termes hors diagonale sont multipliés par ``sqrt(2)`` afin
    de préserver le produit scalaire de Frobenius.
    """

    def __init__(self, regularization: float = 0.1) -> None:
        self.regularization = regularization
        self.reference_: np.ndarray | None = None
        self.n_channels_in_: int | None = None

    def _covariances(self, X: np.ndarray) -> np.ndarray:
        trials = np.asarray(X, dtype=float)
        if trials.ndim != TRIAL_DIMENSION:
            raise ValueError("CovarianceTangentSpace expects a 3D array")
        if not np.isfinite(trials).all():
            raise ValueError("CovarianceTangentSpace requires finite trials")
        covariances = []
        for trial in trials:
            centered = trial - np.mean(trial, axis=1, keepdims=True)
            covariance = centered @ centered.T / max(1, trial.shape[1] - 1)
            covariance = _regularize_covariance(covariance, self.regularization)
            covariance += np.finfo(float).eps * np.eye(covariance.shape[0])
            covariances.append(covariance)
        return np.asarray(covariances)

    @staticmethod
    def _symmetric_function(matrix: np.ndarray, function: object) -> np.ndarray:
        values, vectors = linalg.eigh(matrix)
        safe_values = np.maximum(values, np.finfo(float).eps)
        transformed = function(safe_values)  # type: ignore[operator]
        return np.asarray((vectors * transformed) @ vectors.T)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        del y
        covariances = self._covariances(X)
        logs = [
            self._symmetric_function(covariance, np.log) for covariance in covariances
        ]
        mean_log = np.mean(logs, axis=0)
        self.reference_ = np.asarray(linalg.expm(mean_log))
        self.n_channels_in_ = covariances.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None or self.n_channels_in_ is None:
            raise ValueError("CovarianceTangentSpace must be fitted before transform")
        covariances = self._covariances(X)
        if covariances.shape[1] != self.n_channels_in_:
            raise ValueError("X has a different number of channels than during fit")
        inverse_sqrt = self._symmetric_function(self.reference_, lambda x: x**-0.5)
        upper = np.triu_indices(self.n_channels_in_)
        off_diagonal = upper[0] != upper[1]
        vectors = []
        for covariance in covariances:
            whitened = inverse_sqrt @ covariance @ inverse_sqrt
            tangent = self._symmetric_function(whitened, np.log)
            vector = tangent[upper]
            vector[off_diagonal] *= np.sqrt(2.0)
            vectors.append(vector)
        return np.asarray(vectors)
