"""Tests du tableau de bord d'évaluation."""

# Importe pathlib pour manipuler les chemins temporaires
from pathlib import Path

# Importe numpy pour vérifier les sorties numériques
import numpy as np

# Importe le module de visualisation à tester
from scripts import visualize_eval_dashboard as dashboard


# Vérifie que le dashboard se génère avec des données synthétiques
def test_generate_dashboard_creates_png(tmp_path: Path) -> None:
    """Valide la création d'un PNG de dashboard en sortie."""

    # Charge les données d'exemple fournies par le module
    y_true, y_pred, y_proba = dashboard.load_dashboard_data(None)
    # Définit le nom de fichier de sortie attendu
    output_name = "eval_dashboard.png"
    # Définit les classes pour la figure de test
    class_names = ["Classe 0", "Classe 1", "Classe 2", "Classe 3"]
    # Définit le nombre de bins de calibration
    bins = 10
    # Prépare la configuration de génération pour un dossier temporaire
    config = dashboard.DashboardConfig(None, tmp_path, output_name, class_names, bins)
    # Génère le dashboard et récupère le chemin produit
    output_path = dashboard.generate_dashboard(y_true, y_pred, y_proba, config)
    # Vérifie que le fichier PNG a bien été créé
    assert output_path.exists()
    # Vérifie que le fichier a une taille non nulle
    assert output_path.stat().st_size > 0


# Vérifie que l'ECE retourne un scalaire positif
def test_expected_calibration_error_scalar() -> None:
    """Valide le calcul d'un ECE scalaire pour un cas simple."""

    # Définit un jeu simple de confiances
    confidences = np.asarray([0.1, 0.4, 0.8, 0.9])
    # Définit la correction correspondante
    correct = np.asarray([0.0, 1.0, 1.0, 0.0])
    # Définit le nombre de bins utilisé pour l'ECE
    bins = 4
    # Calcule l'ECE pour vérifier le type de retour
    ece_value = dashboard.expected_calibration_error(confidences, correct, bins)
    # Vérifie que l'ECE est bien un float non négatif
    assert isinstance(ece_value, float)
    # Vérifie que l'ECE reste dans un intervalle plausible
    assert ece_value >= 0.0
