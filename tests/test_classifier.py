"""Tests essentiels du classifieur maison et du rapport de prédiction."""

# Fournit des observations déterministes pour les deux classes.
import numpy as np

# Vérifie le petit adaptateur de rapport utilisé par la CLI predict.
from scripts.predict import build_report

# Vérifie directement l'implémentation bonus conservée dans le rendu.
from tpv.classifier import CentroidClassifier


# Protège le bonus d'implémentation d'un classifieur personnel.
def test_centroid_classifier_predicts_nearest_class_center() -> None:
    """Sépare deux groupes simples à partir de leurs centroïdes appris."""

    # Place deux observations proches autour de chaque classe.
    training_features = np.array(
        [[0.0, 0.0], [0.2, 0.1], [9.8, 10.0], [10.0, 9.9]],
        dtype=float,
    )
    # Associe les deux premières observations à zéro et les autres à un.
    training_labels = np.array([0, 0, 1, 1], dtype=int)
    # Construit l'estimateur compatible avec l'API scikit-learn.
    classifier = CentroidClassifier()
    # Apprend uniquement les centres des deux classes.
    classifier.fit(training_features, training_labels)

    # Interroge un point proche de chaque centre appris.
    predictions = classifier.predict(np.array([[0.1, 0.0], [9.9, 10.1]]))

    # Le classifieur doit choisir la classe du centroïde le plus proche.
    assert np.array_equal(predictions, np.array([0, 1]))


# Protège l'intégration scikit-learn attendue par Pipeline.
def test_centroid_classifier_exposes_sklearn_parameters() -> None:
    """Conserve get_params/set_params hérités de BaseEstimator."""

    # Instancie le classifieur sans état appris.
    classifier = CentroidClassifier()
    # BaseEstimator doit pouvoir introspecter l'estimateur dans une Pipeline.
    assert classifier.get_params() == {}


# Protège les probabilités utilisées par l'agrégation des fenêtres Welch.
def test_centroid_classifier_probabilities_are_normalized() -> None:
    """Retourne une probabilité par classe dont la somme vaut un."""

    # Apprend deux centres volontairement éloignés.
    classifier = CentroidClassifier().fit(
        np.array([[0.0, 0.0], [10.0, 10.0]]),
        np.array([0, 1]),
    )
    # Interroge un point situé entre les deux centroïdes.
    probabilities = classifier.predict_proba(np.array([[5.0, 5.0]]))
    # Une colonne doit être produite pour chaque classe connue.
    assert probabilities.shape == (1, 2)
    # La normalisation est requise par l'agrégation probabiliste Welch.
    assert np.allclose(probabilities.sum(axis=1), np.ones(1))


# Protège la structure minimale consommée par la sortie predict.
def test_build_report_keeps_validation_accuracy_and_confusion() -> None:
    """Expose le score officiel sans recalcul dans l'adaptateur de rapport."""

    # Prépare un résultat équivalent à celui d'evaluate_run.
    result = {
        "run": "R03",
        "subject": "S001",
        "accuracy": 0.625,
        "reports": {"confusion": [[1, 0], [1, 0]]},
    }

    # Construit la vue utilisée par la CLI de prédiction.
    report = build_report(result)

    # Le score doit rester celui de la validation croisée en amont.
    assert report["global"] == 0.625
    # Le run doit rester identifiable pendant la soutenance.
    assert report["by_run"] == {"R03": 0.625}
    # La matrice permet toujours d'expliquer les prédictions affichées.
    assert report["confusion_matrix"] == [[1, 0], [1, 0]]
