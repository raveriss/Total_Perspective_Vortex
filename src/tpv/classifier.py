# Offre les protocoles scikit-learn pour créer un classifieur compatible
# Importe numpy pour calculer les centroids et distances
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CentroidClassifier(BaseEstimator, ClassifierMixin):
    """Classifieur léger basé sur la distance aux centroïdes de classe."""

    # Conserve une marge minimale pour éviter les divisions par zéro
    EPSILON = 1e-12

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Calcule la distance euclidienne entre chaque échantillon et chaque centroïde
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        # Sélectionne l'indice de la classe avec la distance minimale
        closest_indices = distances.argmin(axis=1)
        # Remappe les indices vers les labels originaux pour préserver les classes
        predictions: np.ndarray = self.classes_[closest_indices]
        # Retourne un tableau numpy explicitement typé pour satisfaire mypy
        return predictions

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
