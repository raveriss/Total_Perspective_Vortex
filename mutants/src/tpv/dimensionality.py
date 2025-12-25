"""Réduction de dimension pour TPV."""

# Garantit l'accès aux types manipulables par joblib et pathlib
import os

# Offre une persistance simple de la matrice de projection
import joblib

# Garantit que numpy est disponible pour le calcul matriciel
import numpy as np

# Garantit l'accès aux décompositions hermitiennes généralisées
from scipy import linalg

# Assure l'intégration avec les API scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin

# Fige le nombre de classes attendu pour le CSP pour lever les ambiguïtés
EXPECTED_CSP_CLASSES = 2
# Fige la dimension tabulaire standard pour différencier l'entrée
TABLE_DIMENSION = 2
# Fige la dimension trial x channel x time attendue pour le CSP
TRIAL_DIMENSION = 3
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg is not None:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result


class TPVDimReducer(BaseEstimator, TransformerMixin):
    """Réducteur de dimension via PCA ou CSP."""

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_orig(
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_1(
        self,
        method: str = "XXcspXX",
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_2(
        self,
        method: str = "CSP",
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_3(
        self,
        method: str = "csp",
        n_components: int | None = None,
        regularization: float = 1.0,
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_4(
        self,
        method: str = "csp",
        n_components: int | None = None,
        regularization: float = 0.0,
    ):
        # Conserve la méthode demandée par l'utilisateur
        self.method = None
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_5(
        self,
        method: str = "csp",
        n_components: int | None = None,
        regularization: float = 0.0,
    ):
        # Conserve la méthode demandée par l'utilisateur
        self.method = method
        # Conserve le nombre de composantes souhaité
        self.n_components = None
        # Conserve la régularisation ajoutée aux covariances
        self.regularization = regularization
        # Initialise la matrice de projection à None avant apprentissage
        self.w_matrix: np.ndarray | None
        # Initialise la moyenne pour la centration éventuelle
        self.mean_: np.ndarray | None
        # Prépare le stockage des valeurs propres pour validation et débogage
        self.eigenvalues_: np.ndarray | None
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_6(
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
        self.regularization = None
        # Initialise la matrice de projection à None avant apprentissage
        self.w_matrix: np.ndarray | None
        # Initialise la moyenne pour la centration éventuelle
        self.mean_: np.ndarray | None
        # Prépare le stockage des valeurs propres pour validation et débogage
        self.eigenvalues_: np.ndarray | None
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_7(
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = ""
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_8(
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = ""
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = None

    # Déclare le constructeur pour choisir la méthode et le nombre de composantes
    def xǁTPVDimReducerǁ__init____mutmut_9(
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
        # Positionne None pour refléter l'absence d'apprentissage initial
        self.w_matrix = None
        # Positionne None pour éviter un centrage tant que fit n'est pas appelé
        self.mean_ = None
        # Positionne None avant calcul des valeurs propres
        self.eigenvalues_ = ""
    
    xǁTPVDimReducerǁ__init____mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁ__init____mutmut_1': xǁTPVDimReducerǁ__init____mutmut_1, 
        'xǁTPVDimReducerǁ__init____mutmut_2': xǁTPVDimReducerǁ__init____mutmut_2, 
        'xǁTPVDimReducerǁ__init____mutmut_3': xǁTPVDimReducerǁ__init____mutmut_3, 
        'xǁTPVDimReducerǁ__init____mutmut_4': xǁTPVDimReducerǁ__init____mutmut_4, 
        'xǁTPVDimReducerǁ__init____mutmut_5': xǁTPVDimReducerǁ__init____mutmut_5, 
        'xǁTPVDimReducerǁ__init____mutmut_6': xǁTPVDimReducerǁ__init____mutmut_6, 
        'xǁTPVDimReducerǁ__init____mutmut_7': xǁTPVDimReducerǁ__init____mutmut_7, 
        'xǁTPVDimReducerǁ__init____mutmut_8': xǁTPVDimReducerǁ__init____mutmut_8, 
        'xǁTPVDimReducerǁ__init____mutmut_9': xǁTPVDimReducerǁ__init____mutmut_9
    }
    
    def __init__(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁ__init____mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁ__init____mutmut_mutants"), args, kwargs, self)
        return result 
    
    __init__.__signature__ = _mutmut_signature(xǁTPVDimReducerǁ__init____mutmut_orig)
    xǁTPVDimReducerǁ__init____mutmut_orig.__name__ = 'xǁTPVDimReducerǁ__init__'

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_orig(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_1(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_2(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"XXpcaXX", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_3(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"PCA", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_4(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "XXcspXX"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_5(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "CSP"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_6(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError(None)
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_7(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("XXmethod must be 'pca' or 'csp'XX")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_8(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("METHOD MUST BE 'PCA' OR 'CSP'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_9(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method != "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_10(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "XXpcaXX":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_11(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "PCA":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_12(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim == TABLE_DIMENSION:
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_13(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError(None)
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_14(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("XXPCA expects a 2D arrayXX")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_15(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("pca expects a 2d array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_16(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA EXPECTS A 2D ARRAY")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_17(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = None
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_18(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(None, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_19(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=None)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_20(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_21(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, )
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_22(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=1)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_23(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = None
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_24(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X + self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(centered)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_25(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = None
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_26(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
            # Vérifie que les données sont tabulaires pour PCA
            if X.ndim != TABLE_DIMENSION:
                # Informe que PCA attend des données échantillon x feature
                raise ValueError("PCA expects a 2D array")
            # Centre les données pour une covariance cohérente
            self.mean_ = np.mean(X, axis=0)
            # Calcule les données centrées pour la covariance
            centered = X - self.mean_
            # Calcule la covariance avec régularisation diagonale
            covariance = self._regularized_covariance(None)
            # Extrait les vecteurs propres pour définir la projection
            eigvals, eigvecs = np.linalg.eigh(covariance)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_27(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            eigvals, eigvecs = None
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_28(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            eigvals, eigvecs = np.linalg.eigh(None)
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_29(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = None
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_30(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(None)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_31(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::+1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_32(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-2]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_33(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = None
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_34(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_35(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = None
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_36(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = None
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_37(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = None
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_38(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = None
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_39(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = None
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_40(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is not None:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_41(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError(None)
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_42(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("XXy is required for CSPXX")
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_43(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for csp")
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_44(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("Y IS REQUIRED FOR CSP")
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_45(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim == TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("CSP expects a 3D array")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_46(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError(None)
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_47(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("XXCSP expects a 3D arrayXX")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_48(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("csp expects a 3d array")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_49(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("CSP EXPECTS A 3D ARRAY")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(y)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_50(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("CSP expects a 3D array")
            # Identifie les classes présentes pour contrôler le problème
            classes = None
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_51(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
            # Vérifie la présence des étiquettes pour la méthode CSP
            if y is None:
                # Informe que CSP requiert des labels binaires
                raise ValueError("y is required for CSP")
            # Valide la dimension trial x channel x time attendue
            if X.ndim != TRIAL_DIMENSION:
                # Informe que CSP demande des essais temporels bruts
                raise ValueError("CSP expects a 3D array")
            # Identifie les classes présentes pour contrôler le problème
            classes = np.unique(None)
            # Valide que seules deux classes sont fournies
            if classes.size != EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_52(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            if classes.size == EXPECTED_CSP_CLASSES:
                # Empêche un calcul CSP invalide avec plus de deux classes
                raise ValueError("CSP requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_53(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
                raise ValueError(None)
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_54(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
                raise ValueError("XXCSP requires exactly two classesXX")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_55(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
                raise ValueError("csp requires exactly two classes")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_56(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
                raise ValueError("CSP REQUIRES EXACTLY TWO CLASSES")
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_57(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = None
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_58(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(None)
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_59(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y != classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_60(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[1]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_61(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = None
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_62(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(None)
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_63(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y != classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_64(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[2]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_65(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = None
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_66(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a - cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_67(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = None
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_68(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(None)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_69(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = None
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_70(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(None, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_71(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, None)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_72(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_73(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, )
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_74(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = None
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_75(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(None)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_76(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::+1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_77(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-2]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_78(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = None
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_79(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_80(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = None
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_81(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = None
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_82(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = None
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_83(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = None
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = eigvals
        # Retourne l'instance pour chaînage scikit-learn
        return self

    # Apprend la matrice de projection à partir des données et des labels
    def xǁTPVDimReducerǁfit__mutmut_84(self, X: np.ndarray, y: np.ndarray | None = None):
        # Sécurise le choix de méthode pour éviter des branches invalides
        if self.method not in {"pca", "csp"}:
            # Informe clairement l'utilisateur en cas de paramètre invalide
            raise ValueError("method must be 'pca' or 'csp'")
        # Applique la méthode PCA si spécifiée
        if self.method == "pca":
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
            # Trie les composantes par variance décroissante
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres selon l'importance
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes si demandé
            if self.n_components is not None:
                # Sélectionne uniquement les premières composantes utiles
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque aussi la liste des valeurs propres
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est demandée
                eigvals = eigvals[order]
            # Stocke la matrice de projection apprise
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres associées pour vérification externe
            self.eigenvalues_ = eigvals
        else:
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
            # Agrège les covariances des essais pour la première classe
            cov_a = self._average_covariance(X[y == classes[0]])
            # Agrège les covariances des essais pour la seconde classe
            cov_b = self._average_covariance(X[y == classes[1]])
            # Combine les covariances pour le problème généralisé
            composite = cov_a + cov_b
            # Ajoute une régularisation pour stabiliser l'inversion implicite
            composite = self._regularize_matrix(composite)
            # Résout le problème généralisé pour maximiser la séparation
            eigvals, eigvecs = linalg.eigh(cov_a, composite)
            # Trie les vecteurs par valeurs propres décroissantes
            order = np.argsort(eigvals)[::-1]
            # Réordonne les vecteurs propres pour prioriser les extrêmes
            sorted_vecs = eigvecs[:, order]
            # Limite le nombre de composantes selon la demande
            if self.n_components is not None:
                # Sélectionne la tranche désirée de composantes
                sorted_vecs = sorted_vecs[:, : self.n_components]
                # Tronque également les valeurs propres associées
                eigvals = eigvals[order][: self.n_components]
            else:
                # Conserve toutes les valeurs propres si aucune coupe n'est appliquée
                eigvals = eigvals[order]
            # Stocke la matrice de projection CSP
            self.w_matrix = sorted_vecs
            # Stocke les valeurs propres pour inspection éventuelle
            self.eigenvalues_ = None
        # Retourne l'instance pour chaînage scikit-learn
        return self
    
    xǁTPVDimReducerǁfit__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁfit__mutmut_1': xǁTPVDimReducerǁfit__mutmut_1, 
        'xǁTPVDimReducerǁfit__mutmut_2': xǁTPVDimReducerǁfit__mutmut_2, 
        'xǁTPVDimReducerǁfit__mutmut_3': xǁTPVDimReducerǁfit__mutmut_3, 
        'xǁTPVDimReducerǁfit__mutmut_4': xǁTPVDimReducerǁfit__mutmut_4, 
        'xǁTPVDimReducerǁfit__mutmut_5': xǁTPVDimReducerǁfit__mutmut_5, 
        'xǁTPVDimReducerǁfit__mutmut_6': xǁTPVDimReducerǁfit__mutmut_6, 
        'xǁTPVDimReducerǁfit__mutmut_7': xǁTPVDimReducerǁfit__mutmut_7, 
        'xǁTPVDimReducerǁfit__mutmut_8': xǁTPVDimReducerǁfit__mutmut_8, 
        'xǁTPVDimReducerǁfit__mutmut_9': xǁTPVDimReducerǁfit__mutmut_9, 
        'xǁTPVDimReducerǁfit__mutmut_10': xǁTPVDimReducerǁfit__mutmut_10, 
        'xǁTPVDimReducerǁfit__mutmut_11': xǁTPVDimReducerǁfit__mutmut_11, 
        'xǁTPVDimReducerǁfit__mutmut_12': xǁTPVDimReducerǁfit__mutmut_12, 
        'xǁTPVDimReducerǁfit__mutmut_13': xǁTPVDimReducerǁfit__mutmut_13, 
        'xǁTPVDimReducerǁfit__mutmut_14': xǁTPVDimReducerǁfit__mutmut_14, 
        'xǁTPVDimReducerǁfit__mutmut_15': xǁTPVDimReducerǁfit__mutmut_15, 
        'xǁTPVDimReducerǁfit__mutmut_16': xǁTPVDimReducerǁfit__mutmut_16, 
        'xǁTPVDimReducerǁfit__mutmut_17': xǁTPVDimReducerǁfit__mutmut_17, 
        'xǁTPVDimReducerǁfit__mutmut_18': xǁTPVDimReducerǁfit__mutmut_18, 
        'xǁTPVDimReducerǁfit__mutmut_19': xǁTPVDimReducerǁfit__mutmut_19, 
        'xǁTPVDimReducerǁfit__mutmut_20': xǁTPVDimReducerǁfit__mutmut_20, 
        'xǁTPVDimReducerǁfit__mutmut_21': xǁTPVDimReducerǁfit__mutmut_21, 
        'xǁTPVDimReducerǁfit__mutmut_22': xǁTPVDimReducerǁfit__mutmut_22, 
        'xǁTPVDimReducerǁfit__mutmut_23': xǁTPVDimReducerǁfit__mutmut_23, 
        'xǁTPVDimReducerǁfit__mutmut_24': xǁTPVDimReducerǁfit__mutmut_24, 
        'xǁTPVDimReducerǁfit__mutmut_25': xǁTPVDimReducerǁfit__mutmut_25, 
        'xǁTPVDimReducerǁfit__mutmut_26': xǁTPVDimReducerǁfit__mutmut_26, 
        'xǁTPVDimReducerǁfit__mutmut_27': xǁTPVDimReducerǁfit__mutmut_27, 
        'xǁTPVDimReducerǁfit__mutmut_28': xǁTPVDimReducerǁfit__mutmut_28, 
        'xǁTPVDimReducerǁfit__mutmut_29': xǁTPVDimReducerǁfit__mutmut_29, 
        'xǁTPVDimReducerǁfit__mutmut_30': xǁTPVDimReducerǁfit__mutmut_30, 
        'xǁTPVDimReducerǁfit__mutmut_31': xǁTPVDimReducerǁfit__mutmut_31, 
        'xǁTPVDimReducerǁfit__mutmut_32': xǁTPVDimReducerǁfit__mutmut_32, 
        'xǁTPVDimReducerǁfit__mutmut_33': xǁTPVDimReducerǁfit__mutmut_33, 
        'xǁTPVDimReducerǁfit__mutmut_34': xǁTPVDimReducerǁfit__mutmut_34, 
        'xǁTPVDimReducerǁfit__mutmut_35': xǁTPVDimReducerǁfit__mutmut_35, 
        'xǁTPVDimReducerǁfit__mutmut_36': xǁTPVDimReducerǁfit__mutmut_36, 
        'xǁTPVDimReducerǁfit__mutmut_37': xǁTPVDimReducerǁfit__mutmut_37, 
        'xǁTPVDimReducerǁfit__mutmut_38': xǁTPVDimReducerǁfit__mutmut_38, 
        'xǁTPVDimReducerǁfit__mutmut_39': xǁTPVDimReducerǁfit__mutmut_39, 
        'xǁTPVDimReducerǁfit__mutmut_40': xǁTPVDimReducerǁfit__mutmut_40, 
        'xǁTPVDimReducerǁfit__mutmut_41': xǁTPVDimReducerǁfit__mutmut_41, 
        'xǁTPVDimReducerǁfit__mutmut_42': xǁTPVDimReducerǁfit__mutmut_42, 
        'xǁTPVDimReducerǁfit__mutmut_43': xǁTPVDimReducerǁfit__mutmut_43, 
        'xǁTPVDimReducerǁfit__mutmut_44': xǁTPVDimReducerǁfit__mutmut_44, 
        'xǁTPVDimReducerǁfit__mutmut_45': xǁTPVDimReducerǁfit__mutmut_45, 
        'xǁTPVDimReducerǁfit__mutmut_46': xǁTPVDimReducerǁfit__mutmut_46, 
        'xǁTPVDimReducerǁfit__mutmut_47': xǁTPVDimReducerǁfit__mutmut_47, 
        'xǁTPVDimReducerǁfit__mutmut_48': xǁTPVDimReducerǁfit__mutmut_48, 
        'xǁTPVDimReducerǁfit__mutmut_49': xǁTPVDimReducerǁfit__mutmut_49, 
        'xǁTPVDimReducerǁfit__mutmut_50': xǁTPVDimReducerǁfit__mutmut_50, 
        'xǁTPVDimReducerǁfit__mutmut_51': xǁTPVDimReducerǁfit__mutmut_51, 
        'xǁTPVDimReducerǁfit__mutmut_52': xǁTPVDimReducerǁfit__mutmut_52, 
        'xǁTPVDimReducerǁfit__mutmut_53': xǁTPVDimReducerǁfit__mutmut_53, 
        'xǁTPVDimReducerǁfit__mutmut_54': xǁTPVDimReducerǁfit__mutmut_54, 
        'xǁTPVDimReducerǁfit__mutmut_55': xǁTPVDimReducerǁfit__mutmut_55, 
        'xǁTPVDimReducerǁfit__mutmut_56': xǁTPVDimReducerǁfit__mutmut_56, 
        'xǁTPVDimReducerǁfit__mutmut_57': xǁTPVDimReducerǁfit__mutmut_57, 
        'xǁTPVDimReducerǁfit__mutmut_58': xǁTPVDimReducerǁfit__mutmut_58, 
        'xǁTPVDimReducerǁfit__mutmut_59': xǁTPVDimReducerǁfit__mutmut_59, 
        'xǁTPVDimReducerǁfit__mutmut_60': xǁTPVDimReducerǁfit__mutmut_60, 
        'xǁTPVDimReducerǁfit__mutmut_61': xǁTPVDimReducerǁfit__mutmut_61, 
        'xǁTPVDimReducerǁfit__mutmut_62': xǁTPVDimReducerǁfit__mutmut_62, 
        'xǁTPVDimReducerǁfit__mutmut_63': xǁTPVDimReducerǁfit__mutmut_63, 
        'xǁTPVDimReducerǁfit__mutmut_64': xǁTPVDimReducerǁfit__mutmut_64, 
        'xǁTPVDimReducerǁfit__mutmut_65': xǁTPVDimReducerǁfit__mutmut_65, 
        'xǁTPVDimReducerǁfit__mutmut_66': xǁTPVDimReducerǁfit__mutmut_66, 
        'xǁTPVDimReducerǁfit__mutmut_67': xǁTPVDimReducerǁfit__mutmut_67, 
        'xǁTPVDimReducerǁfit__mutmut_68': xǁTPVDimReducerǁfit__mutmut_68, 
        'xǁTPVDimReducerǁfit__mutmut_69': xǁTPVDimReducerǁfit__mutmut_69, 
        'xǁTPVDimReducerǁfit__mutmut_70': xǁTPVDimReducerǁfit__mutmut_70, 
        'xǁTPVDimReducerǁfit__mutmut_71': xǁTPVDimReducerǁfit__mutmut_71, 
        'xǁTPVDimReducerǁfit__mutmut_72': xǁTPVDimReducerǁfit__mutmut_72, 
        'xǁTPVDimReducerǁfit__mutmut_73': xǁTPVDimReducerǁfit__mutmut_73, 
        'xǁTPVDimReducerǁfit__mutmut_74': xǁTPVDimReducerǁfit__mutmut_74, 
        'xǁTPVDimReducerǁfit__mutmut_75': xǁTPVDimReducerǁfit__mutmut_75, 
        'xǁTPVDimReducerǁfit__mutmut_76': xǁTPVDimReducerǁfit__mutmut_76, 
        'xǁTPVDimReducerǁfit__mutmut_77': xǁTPVDimReducerǁfit__mutmut_77, 
        'xǁTPVDimReducerǁfit__mutmut_78': xǁTPVDimReducerǁfit__mutmut_78, 
        'xǁTPVDimReducerǁfit__mutmut_79': xǁTPVDimReducerǁfit__mutmut_79, 
        'xǁTPVDimReducerǁfit__mutmut_80': xǁTPVDimReducerǁfit__mutmut_80, 
        'xǁTPVDimReducerǁfit__mutmut_81': xǁTPVDimReducerǁfit__mutmut_81, 
        'xǁTPVDimReducerǁfit__mutmut_82': xǁTPVDimReducerǁfit__mutmut_82, 
        'xǁTPVDimReducerǁfit__mutmut_83': xǁTPVDimReducerǁfit__mutmut_83, 
        'xǁTPVDimReducerǁfit__mutmut_84': xǁTPVDimReducerǁfit__mutmut_84
    }
    
    def fit(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁfit__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁfit__mutmut_mutants"), args, kwargs, self)
        return result 
    
    fit.__signature__ = _mutmut_signature(xǁTPVDimReducerǁfit__mutmut_orig)
    xǁTPVDimReducerǁfit__mutmut_orig.__name__ = 'xǁTPVDimReducerǁfit'

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_orig(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_1(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is not None:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_2(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError(None)
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_3(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("XXThe model must be fitted before calling transformXX")
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_4(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("the model must be fitted before calling transform")
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_5(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("THE MODEL MUST BE FITTED BEFORE CALLING TRANSFORM")
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_6(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = X - self.mean_
        # Applique la projection aux données tabulaires classiques
        if X.ndim == TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_7(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is not None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = None
        # Applique la projection aux données tabulaires classiques
        if X.ndim == TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_8(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is not None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = X + self.mean_
        # Applique la projection aux données tabulaires classiques
        if X.ndim == TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_9(self, X: np.ndarray) -> np.ndarray:
        # Vérifie que la matrice de projection est disponible
        if self.w_matrix is None:
            # Empêche l'usage avant apprentissage
            raise ValueError("The model must be fitted before calling transform")
        # Applique le centrage si une moyenne a été calculée
        if self.mean_ is not None:
            # Centre les données d'entrée pour cohérence avec l'apprentissage
            X = X - self.mean_
        # Applique la projection aux données tabulaires classiques
        if X.ndim != TABLE_DIMENSION:
            # Calcule la projection linéaire des données 2D
            return np.asarray(X @ self.w_matrix)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_10(self, X: np.ndarray) -> np.ndarray:
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
            return np.asarray(None)
        # Gère explicitement les données trial x channel x time
        if X.ndim == TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_11(self, X: np.ndarray) -> np.ndarray:
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
        if X.ndim != TRIAL_DIMENSION:
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_12(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = None
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_13(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(None, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_14(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, None, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_15(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=None)
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_16(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_17(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_18(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, )
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_19(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([2], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_20(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [2]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_21(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = None
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_22(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(None, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_23(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, None, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_24(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, None)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_25(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_26(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_27(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, )
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_28(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 2, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_29(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 1)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_30(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = None
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_31(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(None, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_32(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=None)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_33(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_34(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, )
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_35(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=3)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_36(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(None)
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_37(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(None))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_38(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances - np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_39(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(None).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X must be 2D or 3D for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_40(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError(None)

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_41(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("XXX must be 2D or 3D for transformXX")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_42(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("x must be 2d or 3d for transform")

    # Transforme les données en appliquant la matrice de projection apprise
    def xǁTPVDimReducerǁtransform__mutmut_43(self, X: np.ndarray) -> np.ndarray:
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
            # Projette chaque essai en conservant la dynamique temporelle
            projected = np.tensordot(self.w_matrix.T, X, axes=([1], [1]))
            # Réordonne les axes pour revenir à trial x composante x temps
            reordered = np.moveaxis(projected, 1, 0)
            # Calcule la variance par composante pour résumer chaque essai
            variances = np.var(reordered, axis=2)
            # Retourne la variance logarithmique pour stabiliser la distribution
            return np.asarray(np.log(variances + np.finfo(float).eps))
        # Refuse les dimensions inattendues pour maintenir la clarté
        raise ValueError("X MUST BE 2D OR 3D FOR TRANSFORM")
    
    xǁTPVDimReducerǁtransform__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁtransform__mutmut_1': xǁTPVDimReducerǁtransform__mutmut_1, 
        'xǁTPVDimReducerǁtransform__mutmut_2': xǁTPVDimReducerǁtransform__mutmut_2, 
        'xǁTPVDimReducerǁtransform__mutmut_3': xǁTPVDimReducerǁtransform__mutmut_3, 
        'xǁTPVDimReducerǁtransform__mutmut_4': xǁTPVDimReducerǁtransform__mutmut_4, 
        'xǁTPVDimReducerǁtransform__mutmut_5': xǁTPVDimReducerǁtransform__mutmut_5, 
        'xǁTPVDimReducerǁtransform__mutmut_6': xǁTPVDimReducerǁtransform__mutmut_6, 
        'xǁTPVDimReducerǁtransform__mutmut_7': xǁTPVDimReducerǁtransform__mutmut_7, 
        'xǁTPVDimReducerǁtransform__mutmut_8': xǁTPVDimReducerǁtransform__mutmut_8, 
        'xǁTPVDimReducerǁtransform__mutmut_9': xǁTPVDimReducerǁtransform__mutmut_9, 
        'xǁTPVDimReducerǁtransform__mutmut_10': xǁTPVDimReducerǁtransform__mutmut_10, 
        'xǁTPVDimReducerǁtransform__mutmut_11': xǁTPVDimReducerǁtransform__mutmut_11, 
        'xǁTPVDimReducerǁtransform__mutmut_12': xǁTPVDimReducerǁtransform__mutmut_12, 
        'xǁTPVDimReducerǁtransform__mutmut_13': xǁTPVDimReducerǁtransform__mutmut_13, 
        'xǁTPVDimReducerǁtransform__mutmut_14': xǁTPVDimReducerǁtransform__mutmut_14, 
        'xǁTPVDimReducerǁtransform__mutmut_15': xǁTPVDimReducerǁtransform__mutmut_15, 
        'xǁTPVDimReducerǁtransform__mutmut_16': xǁTPVDimReducerǁtransform__mutmut_16, 
        'xǁTPVDimReducerǁtransform__mutmut_17': xǁTPVDimReducerǁtransform__mutmut_17, 
        'xǁTPVDimReducerǁtransform__mutmut_18': xǁTPVDimReducerǁtransform__mutmut_18, 
        'xǁTPVDimReducerǁtransform__mutmut_19': xǁTPVDimReducerǁtransform__mutmut_19, 
        'xǁTPVDimReducerǁtransform__mutmut_20': xǁTPVDimReducerǁtransform__mutmut_20, 
        'xǁTPVDimReducerǁtransform__mutmut_21': xǁTPVDimReducerǁtransform__mutmut_21, 
        'xǁTPVDimReducerǁtransform__mutmut_22': xǁTPVDimReducerǁtransform__mutmut_22, 
        'xǁTPVDimReducerǁtransform__mutmut_23': xǁTPVDimReducerǁtransform__mutmut_23, 
        'xǁTPVDimReducerǁtransform__mutmut_24': xǁTPVDimReducerǁtransform__mutmut_24, 
        'xǁTPVDimReducerǁtransform__mutmut_25': xǁTPVDimReducerǁtransform__mutmut_25, 
        'xǁTPVDimReducerǁtransform__mutmut_26': xǁTPVDimReducerǁtransform__mutmut_26, 
        'xǁTPVDimReducerǁtransform__mutmut_27': xǁTPVDimReducerǁtransform__mutmut_27, 
        'xǁTPVDimReducerǁtransform__mutmut_28': xǁTPVDimReducerǁtransform__mutmut_28, 
        'xǁTPVDimReducerǁtransform__mutmut_29': xǁTPVDimReducerǁtransform__mutmut_29, 
        'xǁTPVDimReducerǁtransform__mutmut_30': xǁTPVDimReducerǁtransform__mutmut_30, 
        'xǁTPVDimReducerǁtransform__mutmut_31': xǁTPVDimReducerǁtransform__mutmut_31, 
        'xǁTPVDimReducerǁtransform__mutmut_32': xǁTPVDimReducerǁtransform__mutmut_32, 
        'xǁTPVDimReducerǁtransform__mutmut_33': xǁTPVDimReducerǁtransform__mutmut_33, 
        'xǁTPVDimReducerǁtransform__mutmut_34': xǁTPVDimReducerǁtransform__mutmut_34, 
        'xǁTPVDimReducerǁtransform__mutmut_35': xǁTPVDimReducerǁtransform__mutmut_35, 
        'xǁTPVDimReducerǁtransform__mutmut_36': xǁTPVDimReducerǁtransform__mutmut_36, 
        'xǁTPVDimReducerǁtransform__mutmut_37': xǁTPVDimReducerǁtransform__mutmut_37, 
        'xǁTPVDimReducerǁtransform__mutmut_38': xǁTPVDimReducerǁtransform__mutmut_38, 
        'xǁTPVDimReducerǁtransform__mutmut_39': xǁTPVDimReducerǁtransform__mutmut_39, 
        'xǁTPVDimReducerǁtransform__mutmut_40': xǁTPVDimReducerǁtransform__mutmut_40, 
        'xǁTPVDimReducerǁtransform__mutmut_41': xǁTPVDimReducerǁtransform__mutmut_41, 
        'xǁTPVDimReducerǁtransform__mutmut_42': xǁTPVDimReducerǁtransform__mutmut_42, 
        'xǁTPVDimReducerǁtransform__mutmut_43': xǁTPVDimReducerǁtransform__mutmut_43
    }
    
    def transform(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁtransform__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁtransform__mutmut_mutants"), args, kwargs, self)
        return result 
    
    transform.__signature__ = _mutmut_signature(xǁTPVDimReducerǁtransform__mutmut_orig)
    xǁTPVDimReducerǁtransform__mutmut_orig.__name__ = 'xǁTPVDimReducerǁtransform'

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_orig(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_1(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is not None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_2(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError(None)
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_3(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("XXCannot save before fitting the modelXX")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_4(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_5(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("CANNOT SAVE BEFORE FITTING THE MODEL")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_6(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            None,
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_7(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            None,
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_8(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_9(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_10(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "XXw_matrixXX": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_11(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "W_MATRIX": self.w_matrix,
                "mean": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_12(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "XXmeanXX": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_13(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "MEAN": self.mean_,
                "eig": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_14(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "XXeigXX": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_15(self, path: str | os.PathLike[str]) -> None:
        # Valide la disponibilité de la matrice pour éviter un dump vide
        if self.w_matrix is None:
            # Signale à l'utilisateur que fit doit précéder la sauvegarde
            raise ValueError("Cannot save before fitting the model")
        # Utilise joblib pour sérialiser la matrice et la moyenne
        joblib.dump(
            {
                "w_matrix": self.w_matrix,
                "mean": self.mean_,
                "EIG": self.eigenvalues_,
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_16(self, path: str | os.PathLike[str]) -> None:
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
                "XXmethodXX": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_17(self, path: str | os.PathLike[str]) -> None:
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
                "METHOD": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_18(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "XXn_componentsXX": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_19(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "N_COMPONENTS": self.n_components,
                "regularization": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_20(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "XXregularizationXX": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_21(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "REGULARIZATION": self.regularization,
            },
            str(path),
        )

    # Enregistre la matrice de projection pour réutilisation future
    def xǁTPVDimReducerǁsave__mutmut_22(self, path: str | os.PathLike[str]) -> None:
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
                "method": self.method,
                "n_components": self.n_components,
                "regularization": self.regularization,
            },
            str(None),
        )
    
    xǁTPVDimReducerǁsave__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁsave__mutmut_1': xǁTPVDimReducerǁsave__mutmut_1, 
        'xǁTPVDimReducerǁsave__mutmut_2': xǁTPVDimReducerǁsave__mutmut_2, 
        'xǁTPVDimReducerǁsave__mutmut_3': xǁTPVDimReducerǁsave__mutmut_3, 
        'xǁTPVDimReducerǁsave__mutmut_4': xǁTPVDimReducerǁsave__mutmut_4, 
        'xǁTPVDimReducerǁsave__mutmut_5': xǁTPVDimReducerǁsave__mutmut_5, 
        'xǁTPVDimReducerǁsave__mutmut_6': xǁTPVDimReducerǁsave__mutmut_6, 
        'xǁTPVDimReducerǁsave__mutmut_7': xǁTPVDimReducerǁsave__mutmut_7, 
        'xǁTPVDimReducerǁsave__mutmut_8': xǁTPVDimReducerǁsave__mutmut_8, 
        'xǁTPVDimReducerǁsave__mutmut_9': xǁTPVDimReducerǁsave__mutmut_9, 
        'xǁTPVDimReducerǁsave__mutmut_10': xǁTPVDimReducerǁsave__mutmut_10, 
        'xǁTPVDimReducerǁsave__mutmut_11': xǁTPVDimReducerǁsave__mutmut_11, 
        'xǁTPVDimReducerǁsave__mutmut_12': xǁTPVDimReducerǁsave__mutmut_12, 
        'xǁTPVDimReducerǁsave__mutmut_13': xǁTPVDimReducerǁsave__mutmut_13, 
        'xǁTPVDimReducerǁsave__mutmut_14': xǁTPVDimReducerǁsave__mutmut_14, 
        'xǁTPVDimReducerǁsave__mutmut_15': xǁTPVDimReducerǁsave__mutmut_15, 
        'xǁTPVDimReducerǁsave__mutmut_16': xǁTPVDimReducerǁsave__mutmut_16, 
        'xǁTPVDimReducerǁsave__mutmut_17': xǁTPVDimReducerǁsave__mutmut_17, 
        'xǁTPVDimReducerǁsave__mutmut_18': xǁTPVDimReducerǁsave__mutmut_18, 
        'xǁTPVDimReducerǁsave__mutmut_19': xǁTPVDimReducerǁsave__mutmut_19, 
        'xǁTPVDimReducerǁsave__mutmut_20': xǁTPVDimReducerǁsave__mutmut_20, 
        'xǁTPVDimReducerǁsave__mutmut_21': xǁTPVDimReducerǁsave__mutmut_21, 
        'xǁTPVDimReducerǁsave__mutmut_22': xǁTPVDimReducerǁsave__mutmut_22
    }
    
    def save(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁsave__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁsave__mutmut_mutants"), args, kwargs, self)
        return result 
    
    save.__signature__ = _mutmut_signature(xǁTPVDimReducerǁsave__mutmut_orig)
    xǁTPVDimReducerǁsave__mutmut_orig.__name__ = 'xǁTPVDimReducerǁsave'

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_orig(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_1(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = None
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_2(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(None)
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_3(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(None))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_4(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = None
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_5(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get(None)
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_6(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("XXw_matrixXX")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_7(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("W_MATRIX")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_8(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = None
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_9(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get(None)
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_10(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("XXmeanXX")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_11(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("MEAN")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_12(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = None
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_13(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get(None)
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_14(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("XXeigXX")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_15(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("EIG")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_16(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = None
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_17(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get(None, self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_18(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", None)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_19(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get(self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_20(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", )
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_21(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("XXmethodXX", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_22(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("METHOD", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_23(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = None
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_24(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get(None, self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_25(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", None)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_26(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get(self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_27(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", )
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_28(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("XXn_componentsXX", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_29(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("N_COMPONENTS", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_30(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = None

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_31(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get(None, self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_32(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", None)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_33(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get(self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_34(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("regularization", )

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_35(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("XXregularizationXX", self.regularization)

    # Charge la matrice de projection depuis un fichier joblib
    def xǁTPVDimReducerǁload__mutmut_36(self, path: str | os.PathLike[str]) -> None:
        # Récupère le contenu sérialisé pour restaurer le modèle
        data = joblib.load(str(path))
        # Restaure la matrice de projection sauvegardée
        self.w_matrix = data.get("w_matrix")
        # Restaure la moyenne si elle existe
        self.mean_ = data.get("mean")
        # Restaure les valeurs propres éventuelles
        self.eigenvalues_ = data.get("eig")
        # Restaure la méthode utilisée pour la projection
        self.method = data.get("method", self.method)
        # Restaure le nombre de composantes demandé
        self.n_components = data.get("n_components", self.n_components)
        # Restaure la régularisation appliquée aux covariances
        self.regularization = data.get("REGULARIZATION", self.regularization)
    
    xǁTPVDimReducerǁload__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁload__mutmut_1': xǁTPVDimReducerǁload__mutmut_1, 
        'xǁTPVDimReducerǁload__mutmut_2': xǁTPVDimReducerǁload__mutmut_2, 
        'xǁTPVDimReducerǁload__mutmut_3': xǁTPVDimReducerǁload__mutmut_3, 
        'xǁTPVDimReducerǁload__mutmut_4': xǁTPVDimReducerǁload__mutmut_4, 
        'xǁTPVDimReducerǁload__mutmut_5': xǁTPVDimReducerǁload__mutmut_5, 
        'xǁTPVDimReducerǁload__mutmut_6': xǁTPVDimReducerǁload__mutmut_6, 
        'xǁTPVDimReducerǁload__mutmut_7': xǁTPVDimReducerǁload__mutmut_7, 
        'xǁTPVDimReducerǁload__mutmut_8': xǁTPVDimReducerǁload__mutmut_8, 
        'xǁTPVDimReducerǁload__mutmut_9': xǁTPVDimReducerǁload__mutmut_9, 
        'xǁTPVDimReducerǁload__mutmut_10': xǁTPVDimReducerǁload__mutmut_10, 
        'xǁTPVDimReducerǁload__mutmut_11': xǁTPVDimReducerǁload__mutmut_11, 
        'xǁTPVDimReducerǁload__mutmut_12': xǁTPVDimReducerǁload__mutmut_12, 
        'xǁTPVDimReducerǁload__mutmut_13': xǁTPVDimReducerǁload__mutmut_13, 
        'xǁTPVDimReducerǁload__mutmut_14': xǁTPVDimReducerǁload__mutmut_14, 
        'xǁTPVDimReducerǁload__mutmut_15': xǁTPVDimReducerǁload__mutmut_15, 
        'xǁTPVDimReducerǁload__mutmut_16': xǁTPVDimReducerǁload__mutmut_16, 
        'xǁTPVDimReducerǁload__mutmut_17': xǁTPVDimReducerǁload__mutmut_17, 
        'xǁTPVDimReducerǁload__mutmut_18': xǁTPVDimReducerǁload__mutmut_18, 
        'xǁTPVDimReducerǁload__mutmut_19': xǁTPVDimReducerǁload__mutmut_19, 
        'xǁTPVDimReducerǁload__mutmut_20': xǁTPVDimReducerǁload__mutmut_20, 
        'xǁTPVDimReducerǁload__mutmut_21': xǁTPVDimReducerǁload__mutmut_21, 
        'xǁTPVDimReducerǁload__mutmut_22': xǁTPVDimReducerǁload__mutmut_22, 
        'xǁTPVDimReducerǁload__mutmut_23': xǁTPVDimReducerǁload__mutmut_23, 
        'xǁTPVDimReducerǁload__mutmut_24': xǁTPVDimReducerǁload__mutmut_24, 
        'xǁTPVDimReducerǁload__mutmut_25': xǁTPVDimReducerǁload__mutmut_25, 
        'xǁTPVDimReducerǁload__mutmut_26': xǁTPVDimReducerǁload__mutmut_26, 
        'xǁTPVDimReducerǁload__mutmut_27': xǁTPVDimReducerǁload__mutmut_27, 
        'xǁTPVDimReducerǁload__mutmut_28': xǁTPVDimReducerǁload__mutmut_28, 
        'xǁTPVDimReducerǁload__mutmut_29': xǁTPVDimReducerǁload__mutmut_29, 
        'xǁTPVDimReducerǁload__mutmut_30': xǁTPVDimReducerǁload__mutmut_30, 
        'xǁTPVDimReducerǁload__mutmut_31': xǁTPVDimReducerǁload__mutmut_31, 
        'xǁTPVDimReducerǁload__mutmut_32': xǁTPVDimReducerǁload__mutmut_32, 
        'xǁTPVDimReducerǁload__mutmut_33': xǁTPVDimReducerǁload__mutmut_33, 
        'xǁTPVDimReducerǁload__mutmut_34': xǁTPVDimReducerǁload__mutmut_34, 
        'xǁTPVDimReducerǁload__mutmut_35': xǁTPVDimReducerǁload__mutmut_35, 
        'xǁTPVDimReducerǁload__mutmut_36': xǁTPVDimReducerǁload__mutmut_36
    }
    
    def load(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁload__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁload__mutmut_mutants"), args, kwargs, self)
        return result 
    
    load.__signature__ = _mutmut_signature(xǁTPVDimReducerǁload__mutmut_orig)
    xǁTPVDimReducerǁload__mutmut_orig.__name__ = 'xǁTPVDimReducerǁload'

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_orig(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_1(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size != 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_2(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 1:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_3(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError(None)
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_4(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("XXNo trials provided for covariance estimationXX")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_5(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("no trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_6(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("NO TRIALS PROVIDED FOR COVARIANCE ESTIMATION")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_7(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = None
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_8(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros(None)
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_9(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[2], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_10(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[2]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_11(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = None
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_12(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov = np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_13(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov *= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_14(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(None)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_15(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum = trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_16(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum -= trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_17(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = None
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_18(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(None)
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_19(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum * trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_20(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[1])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(averaged)

    # Calcule la moyenne des matrices de covariance sur un ensemble d'essais
    def xǁTPVDimReducerǁ_average_covariance__mutmut_21(self, trials: np.ndarray) -> np.ndarray:
        # Valide la présence d'essais pour éviter une division par zéro
        if trials.size == 0:
            # Informe que des données sont nécessaires pour la covariance
            raise ValueError("No trials provided for covariance estimation")
        # Prépare une accumulation de covariances pour stabilité
        cov_sum = np.zeros((trials.shape[1], trials.shape[1]))
        # Parcourt chaque essai pour accumuler la covariance
        for trial in trials:
            # Calcule la covariance d'un essai avec normalisation par la trace
            trial_cov = trial @ trial.T
            # Normalise pour éviter des échelles dépendantes de l'énergie
            trial_cov /= np.trace(trial_cov)
            # Ajoute la covariance normalisée à l'accumulateur
            cov_sum += trial_cov
        # Calcule la moyenne en divisant par le nombre d'essais
        averaged = np.asarray(cov_sum / trials.shape[0])
        # Ajoute une régularisation diagonale pour stabilité numérique
        return self._regularize_matrix(None)
    
    xǁTPVDimReducerǁ_average_covariance__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁ_average_covariance__mutmut_1': xǁTPVDimReducerǁ_average_covariance__mutmut_1, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_2': xǁTPVDimReducerǁ_average_covariance__mutmut_2, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_3': xǁTPVDimReducerǁ_average_covariance__mutmut_3, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_4': xǁTPVDimReducerǁ_average_covariance__mutmut_4, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_5': xǁTPVDimReducerǁ_average_covariance__mutmut_5, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_6': xǁTPVDimReducerǁ_average_covariance__mutmut_6, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_7': xǁTPVDimReducerǁ_average_covariance__mutmut_7, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_8': xǁTPVDimReducerǁ_average_covariance__mutmut_8, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_9': xǁTPVDimReducerǁ_average_covariance__mutmut_9, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_10': xǁTPVDimReducerǁ_average_covariance__mutmut_10, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_11': xǁTPVDimReducerǁ_average_covariance__mutmut_11, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_12': xǁTPVDimReducerǁ_average_covariance__mutmut_12, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_13': xǁTPVDimReducerǁ_average_covariance__mutmut_13, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_14': xǁTPVDimReducerǁ_average_covariance__mutmut_14, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_15': xǁTPVDimReducerǁ_average_covariance__mutmut_15, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_16': xǁTPVDimReducerǁ_average_covariance__mutmut_16, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_17': xǁTPVDimReducerǁ_average_covariance__mutmut_17, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_18': xǁTPVDimReducerǁ_average_covariance__mutmut_18, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_19': xǁTPVDimReducerǁ_average_covariance__mutmut_19, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_20': xǁTPVDimReducerǁ_average_covariance__mutmut_20, 
        'xǁTPVDimReducerǁ_average_covariance__mutmut_21': xǁTPVDimReducerǁ_average_covariance__mutmut_21
    }
    
    def _average_covariance(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁ_average_covariance__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁ_average_covariance__mutmut_mutants"), args, kwargs, self)
        return result 
    
    _average_covariance.__signature__ = _mutmut_signature(xǁTPVDimReducerǁ_average_covariance__mutmut_orig)
    xǁTPVDimReducerǁ_average_covariance__mutmut_orig.__name__ = 'xǁTPVDimReducerǁ_average_covariance'

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_orig(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_1(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = None
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_2(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) * (centered.shape[0] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_3(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] + 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_4(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[1] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_5(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] - 2)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(covariance)

    # Calcule une covariance régularisée pour les données tabulaires
    def xǁTPVDimReducerǁ_regularized_covariance__mutmut_6(self, centered: np.ndarray) -> np.ndarray:
        # Calcule la covariance échantillon pour capturer la variance partagée
        covariance = (centered.T @ centered) / (centered.shape[0] - 1)
        # Ajoute une régularisation diagonale pour éviter les matrices singulières
        return self._regularize_matrix(None)
    
    xǁTPVDimReducerǁ_regularized_covariance__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁ_regularized_covariance__mutmut_1': xǁTPVDimReducerǁ_regularized_covariance__mutmut_1, 
        'xǁTPVDimReducerǁ_regularized_covariance__mutmut_2': xǁTPVDimReducerǁ_regularized_covariance__mutmut_2, 
        'xǁTPVDimReducerǁ_regularized_covariance__mutmut_3': xǁTPVDimReducerǁ_regularized_covariance__mutmut_3, 
        'xǁTPVDimReducerǁ_regularized_covariance__mutmut_4': xǁTPVDimReducerǁ_regularized_covariance__mutmut_4, 
        'xǁTPVDimReducerǁ_regularized_covariance__mutmut_5': xǁTPVDimReducerǁ_regularized_covariance__mutmut_5, 
        'xǁTPVDimReducerǁ_regularized_covariance__mutmut_6': xǁTPVDimReducerǁ_regularized_covariance__mutmut_6
    }
    
    def _regularized_covariance(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁ_regularized_covariance__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁ_regularized_covariance__mutmut_mutants"), args, kwargs, self)
        return result 
    
    _regularized_covariance.__signature__ = _mutmut_signature(xǁTPVDimReducerǁ_regularized_covariance__mutmut_orig)
    xǁTPVDimReducerǁ_regularized_covariance__mutmut_orig.__name__ = 'xǁTPVDimReducerǁ_regularized_covariance'

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_orig(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_1(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = None
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_2(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(None, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_3(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=None)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_4(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_5(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, )
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_6(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=False)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_7(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization >= 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_8(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 1:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_9(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized = self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_10(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized -= self.regularization * np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_11(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization / np.eye(matrix.shape[0])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_12(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(None)
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized

    # Ajoute une régularisation diagonale proportionnelle à l'identité
    def xǁTPVDimReducerǁ_regularize_matrix__mutmut_13(self, matrix: np.ndarray) -> np.ndarray:
        # Conserve la matrice initiale pour éviter de modifier l'entrée in place
        regularized = np.array(matrix, copy=True)
        # Ajoute la régularisation uniquement si elle est demandée
        if self.regularization > 0:
            # Injecte une identité scalaire proportionnelle pour stabiliser l'inversion
            regularized += self.regularization * np.eye(matrix.shape[1])
        # Retourne la matrice régularisée pour les calculs ultérieurs
        return regularized
    
    xǁTPVDimReducerǁ_regularize_matrix__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁTPVDimReducerǁ_regularize_matrix__mutmut_1': xǁTPVDimReducerǁ_regularize_matrix__mutmut_1, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_2': xǁTPVDimReducerǁ_regularize_matrix__mutmut_2, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_3': xǁTPVDimReducerǁ_regularize_matrix__mutmut_3, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_4': xǁTPVDimReducerǁ_regularize_matrix__mutmut_4, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_5': xǁTPVDimReducerǁ_regularize_matrix__mutmut_5, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_6': xǁTPVDimReducerǁ_regularize_matrix__mutmut_6, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_7': xǁTPVDimReducerǁ_regularize_matrix__mutmut_7, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_8': xǁTPVDimReducerǁ_regularize_matrix__mutmut_8, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_9': xǁTPVDimReducerǁ_regularize_matrix__mutmut_9, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_10': xǁTPVDimReducerǁ_regularize_matrix__mutmut_10, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_11': xǁTPVDimReducerǁ_regularize_matrix__mutmut_11, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_12': xǁTPVDimReducerǁ_regularize_matrix__mutmut_12, 
        'xǁTPVDimReducerǁ_regularize_matrix__mutmut_13': xǁTPVDimReducerǁ_regularize_matrix__mutmut_13
    }
    
    def _regularize_matrix(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁTPVDimReducerǁ_regularize_matrix__mutmut_orig"), object.__getattribute__(self, "xǁTPVDimReducerǁ_regularize_matrix__mutmut_mutants"), args, kwargs, self)
        return result 
    
    _regularize_matrix.__signature__ = _mutmut_signature(xǁTPVDimReducerǁ_regularize_matrix__mutmut_orig)
    xǁTPVDimReducerǁ_regularize_matrix__mutmut_orig.__name__ = 'xǁTPVDimReducerǁ_regularize_matrix'
