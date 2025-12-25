# Offre les protocoles scikit-learn pour créer un classifieur compatible
# Importe numpy pour calculer les centroids et distances
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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


class CentroidClassifier(BaseEstimator, ClassifierMixin):
    """Classifieur léger basé sur la distance aux centroïdes de classe."""

    # Conserve une marge minimale pour éviter les divisions par zéro
    EPSILON = 1e-12

    def xǁCentroidClassifierǁfit__mutmut_orig(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_1(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = None
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_2(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(None, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_3(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=None)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_4(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_5(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, )
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_6(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=False)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_7(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = None
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_8(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            None
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_9(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=None)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_10(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices != class_index].mean(axis=0)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_11(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=1)
                for class_index in range(len(self.classes_))
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self

    def xǁCentroidClassifierǁfit__mutmut_12(self, X: np.ndarray, y: np.ndarray):
        # Enregistre les classes uniques pour conserver l'ordre de prédiction
        self.classes_, indices = np.unique(y, return_inverse=True)
        # Calcule les centroïdes pour chaque classe afin de simplifier la décision
        self.centroids_ = np.vstack(
            [
                X[indices == class_index].mean(axis=0)
                for class_index in range(None)
            ]
        )
        # Retourne l'instance pour respecter l'API scikit-learn
        return self
    
    xǁCentroidClassifierǁfit__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁCentroidClassifierǁfit__mutmut_1': xǁCentroidClassifierǁfit__mutmut_1, 
        'xǁCentroidClassifierǁfit__mutmut_2': xǁCentroidClassifierǁfit__mutmut_2, 
        'xǁCentroidClassifierǁfit__mutmut_3': xǁCentroidClassifierǁfit__mutmut_3, 
        'xǁCentroidClassifierǁfit__mutmut_4': xǁCentroidClassifierǁfit__mutmut_4, 
        'xǁCentroidClassifierǁfit__mutmut_5': xǁCentroidClassifierǁfit__mutmut_5, 
        'xǁCentroidClassifierǁfit__mutmut_6': xǁCentroidClassifierǁfit__mutmut_6, 
        'xǁCentroidClassifierǁfit__mutmut_7': xǁCentroidClassifierǁfit__mutmut_7, 
        'xǁCentroidClassifierǁfit__mutmut_8': xǁCentroidClassifierǁfit__mutmut_8, 
        'xǁCentroidClassifierǁfit__mutmut_9': xǁCentroidClassifierǁfit__mutmut_9, 
        'xǁCentroidClassifierǁfit__mutmut_10': xǁCentroidClassifierǁfit__mutmut_10, 
        'xǁCentroidClassifierǁfit__mutmut_11': xǁCentroidClassifierǁfit__mutmut_11, 
        'xǁCentroidClassifierǁfit__mutmut_12': xǁCentroidClassifierǁfit__mutmut_12
    }
    
    def fit(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁCentroidClassifierǁfit__mutmut_orig"), object.__getattribute__(self, "xǁCentroidClassifierǁfit__mutmut_mutants"), args, kwargs, self)
        return result 
    
    fit.__signature__ = _mutmut_signature(xǁCentroidClassifierǁfit__mutmut_orig)
    xǁCentroidClassifierǁfit__mutmut_orig.__name__ = 'xǁCentroidClassifierǁfit'

    def xǁCentroidClassifierǁpredict__mutmut_orig(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_1(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = None
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_2(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(None, axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_3(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=None)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_4(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_5(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], )
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_6(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] + self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_7(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=3)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_8(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = None
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_9(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=None)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_10(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=2)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

    def xǁCentroidClassifierǁpredict__mutmut_11(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = None
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions
    
    xǁCentroidClassifierǁpredict__mutmut_mutants : ClassVar[MutantDict] = {
    'xǁCentroidClassifierǁpredict__mutmut_1': xǁCentroidClassifierǁpredict__mutmut_1, 
        'xǁCentroidClassifierǁpredict__mutmut_2': xǁCentroidClassifierǁpredict__mutmut_2, 
        'xǁCentroidClassifierǁpredict__mutmut_3': xǁCentroidClassifierǁpredict__mutmut_3, 
        'xǁCentroidClassifierǁpredict__mutmut_4': xǁCentroidClassifierǁpredict__mutmut_4, 
        'xǁCentroidClassifierǁpredict__mutmut_5': xǁCentroidClassifierǁpredict__mutmut_5, 
        'xǁCentroidClassifierǁpredict__mutmut_6': xǁCentroidClassifierǁpredict__mutmut_6, 
        'xǁCentroidClassifierǁpredict__mutmut_7': xǁCentroidClassifierǁpredict__mutmut_7, 
        'xǁCentroidClassifierǁpredict__mutmut_8': xǁCentroidClassifierǁpredict__mutmut_8, 
        'xǁCentroidClassifierǁpredict__mutmut_9': xǁCentroidClassifierǁpredict__mutmut_9, 
        'xǁCentroidClassifierǁpredict__mutmut_10': xǁCentroidClassifierǁpredict__mutmut_10, 
        'xǁCentroidClassifierǁpredict__mutmut_11': xǁCentroidClassifierǁpredict__mutmut_11
    }
    
    def predict(self, *args, **kwargs):
        result = _mutmut_trampoline(object.__getattribute__(self, "xǁCentroidClassifierǁpredict__mutmut_orig"), object.__getattribute__(self, "xǁCentroidClassifierǁpredict__mutmut_mutants"), args, kwargs, self)
        return result 
    
    predict.__signature__ = _mutmut_signature(xǁCentroidClassifierǁpredict__mutmut_orig)
    xǁCentroidClassifierǁpredict__mutmut_orig.__name__ = 'xǁCentroidClassifierǁpredict'

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Calcule les distances pour dériver des pseudo-probabilités normalisées
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Inverse les distances en similarités en protégeant contre les zéros
        similarities = 1.0 / (distances + self.EPSILON)
        # Normalise les similarités pour sommer à un
        probabilities: np.ndarray = similarities / similarities.sum(
            axis=1, keepdims=True
        )
        # Retourne les probabilités alignées sur l'ordre des classes
        return probabilities
