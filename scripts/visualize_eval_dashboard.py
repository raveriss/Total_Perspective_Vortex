"""Visualisation WBS 10.3 : tableau de bord d'évaluation modèle."""

# Force un backend sans affichage pour les environnements CI et serveurs
import matplotlib

# Active Agg pour générer les PNG sans dépendre d'un écran
matplotlib.use("Agg")

# Importe argparse pour exposer les options CLI du tableau de bord
import argparse

# Importe json pour lire les scores sérialisés en entrée
import json

# Importe pathlib pour gérer les chemins d'entrée/sortie
from pathlib import Path

# Importe typing pour gérer les casts explicites
from typing import Any, cast

# Importe colors pour accéder aux colormaps matplotlib
import matplotlib.colors as mcolors

# Importe pyplot pour tracer le tableau de bord
import matplotlib.pyplot as plt

# Importe numpy pour manipuler les tableaux de labels/probas
import numpy as np

# Importe gridspec pour structurer la mise en page du dashboard
from matplotlib import gridspec

# Importe Patch pour construire la légende des classes
from matplotlib.patches import Patch

# Importe l'isotonic regression pour recalibrer les probabilités
from sklearn.isotonic import IsotonicRegression

# Définit la taille par défaut de la figure en pouces
DEFAULT_FIGSIZE = (12, 8)

# Fixe le nombre de classes attendu par défaut
DEFAULT_CLASS_COUNT = 4

# Définit les couleurs associées aux classes pour cohérence visuelle
DEFAULT_CLASS_COLORS = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0"]

# Définit les libellés par défaut des classes
DEFAULT_CLASS_NAMES = tuple(f"Classe {idx}" for idx in range(DEFAULT_CLASS_COUNT))

# Définit le nombre de bins pour l'histogramme des confiances
DEFAULT_CONFIDENCE_BINS = 10

# Définit le seuil de contraste pour les annotations de matrice
CONFUSION_TEXT_THRESHOLD = 0.5

# Définit l'évitement de division par zéro pour les métriques
SAFE_DENOMINATOR = 1.0


# Regroupe les options CLI pour la génération du tableau de bord
class DashboardConfig:
    """Paramètres de génération du tableau de bord de performance."""

    # Initialise les options nécessaires au rendu du dashboard
    def __init__(self, input_path, output_dir, output_name, class_names, bins) -> None:
        # Stocke la source optionnelle de données en entrée
        self.input_path = input_path
        # Stocke le répertoire de sortie des figures
        self.output_dir = output_dir
        # Stocke le nom de fichier PNG à produire
        self.output_name = output_name
        # Stocke les labels de classes affichés dans la figure
        self.class_names = class_names
        # Stocke le nombre de bins de l'histogramme de confiance
        self.confidence_bins = bins


# Centralise la construction du parseur CLI
def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments pour la génération du dashboard."""

    # Décrit brièvement la finalité du script
    description = "Génère un tableau de bord de performance EEG."
    # Initialise le parseur avec une description claire
    parser = argparse.ArgumentParser(description=description)
    # Ajoute un chemin d'entrée optionnel pour les labels/probabilités
    parser.add_argument("--input", default=None, help="Chemin JSON/NPZ")
    # Ajoute le répertoire de sortie pour l'image générée
    parser.add_argument("--output-dir", default="docs/viz", help="Dossier de sortie")
    # Ajoute le nom de fichier cible pour l'image
    parser.add_argument("--output-name", default="eval_dashboard.png", help="PNG")
    # Définit les classes par défaut pour argparse
    default_classes = DEFAULT_CLASS_NAMES
    # Définit les bins par défaut pour argparse
    default_bins = DEFAULT_CONFIDENCE_BINS
    # Ajoute les noms de classes à afficher dans la figure
    parser.add_argument("--classes", nargs="*", default=default_classes, help="Classes")
    # Ajoute le nombre de bins pour l'histogramme de confiance
    parser.add_argument("--bins", type=int, default=default_bins, help="Bins")
    # Retourne le parseur prêt à l'emploi
    return parser


# Valide que les tableaux chargés ont une cohérence basique
def _validate_input_shapes(y_true, y_pred, y_proba) -> None:
    """Vérifie la cohérence des dimensions des labels/probabilités."""

    # Refuse les tailles différentes entre vérité et prédiction
    if y_true.shape[0] != y_pred.shape[0]:
        # Signale l'incohérence pour éviter des métriques fausses
        raise ValueError("y_true et y_pred doivent avoir la même longueur")
    # Refuse des probabilités non alignées sur les labels
    if y_true.shape[0] != y_proba.shape[0]:
        # Signale l'incohérence pour éviter un index hors borne
        raise ValueError("y_proba doit avoir autant de lignes que y_true")


# Charge un JSON avec y_true/y_pred/y_proba
def _load_json_payload(payload_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge les tableaux à partir d'un fichier JSON."""

    # Lit le contenu JSON pour restaurer les tableaux
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    # Construit les tableaux numpy à partir des listes JSON
    y_true = np.asarray(payload["y_true"], dtype=int)
    # Construit les prédictions à partir des listes JSON
    y_pred = np.asarray(payload["y_pred"], dtype=int)
    # Construit la matrice de probabilités à partir des listes JSON
    y_proba = np.asarray(payload["y_proba"], dtype=float)
    # Retourne les tableaux pour la génération du dashboard
    return y_true, y_pred, y_proba


# Charge un NPZ avec y_true/y_pred/y_proba
def _load_npz_payload(payload_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge les tableaux à partir d'un fichier NPZ."""

    # Ouvre l'archive NPZ pour lire les tableaux
    with np.load(payload_path) as archive:
        # Extrait y_true en tableau numpy
        y_true = np.asarray(archive["y_true"], dtype=int)
        # Extrait y_pred en tableau numpy
        y_pred = np.asarray(archive["y_pred"], dtype=int)
        # Extrait y_proba en tableau numpy
        y_proba = np.asarray(archive["y_proba"], dtype=float)
    # Retourne les tableaux pour la suite du pipeline
    return y_true, y_pred, y_proba


# Génère un exemple déterministe quand aucun fichier n'est fourni
def _generate_example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit un jeu synthétique cohérent pour le rendu visuel."""

    # Fixe un générateur déterministe pour stabiliser l'exemple
    rng = np.random.default_rng(42)
    # Définit un nombre d'epochs cohérent avec la démo
    sample_count = 80
    # Génère les labels de vérité terrain
    y_true = rng.integers(0, DEFAULT_CLASS_COUNT, size=sample_count)
    # Génère des probabilités brutes par classe
    y_proba = rng.random((sample_count, DEFAULT_CLASS_COUNT))
    # Normalise les probabilités pour qu'elles somment à 1
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    # Déduit les prédictions top-1 à partir des probas
    y_pred = np.argmax(y_proba, axis=1)
    # Retourne les tableaux synthétiques
    return y_true, y_pred, y_proba


# Charge les données d'entrée ou génère un exemple déterministe
def load_dashboard_data(input_path):
    """Charge les labels/probabilités depuis un fichier ou génère un exemple."""

    # Utilise un jeu d'exemple si aucun fichier n'est fourni
    if input_path is None:
        # Retourne les données synthétiques de démonstration
        return _generate_example_data()
    # Normalise le chemin pour éviter les erreurs relatives
    resolved_path = input_path.expanduser().resolve()
    # Refuse un chemin d'entrée inexistant
    if not resolved_path.exists():
        # Lève une erreur claire pour guider l'utilisateur
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {resolved_path}")
    # Charge le JSON si l'extension correspond
    if resolved_path.suffix.lower() == ".json":
        # Retourne les tableaux depuis le JSON
        return _load_json_payload(resolved_path)
    # Charge le NPZ si l'extension correspond
    if resolved_path.suffix.lower() == ".npz":
        # Retourne les tableaux depuis le NPZ
        return _load_npz_payload(resolved_path)
    # Refuse les formats inconnus pour rester explicite
    raise ValueError("Formats supportés : .json ou .npz")


# Calcule l'Expected Calibration Error (ECE) sur la confiance top-1
def expected_calibration_error(confidences, correct, bins) -> float:
    """Retourne l'ECE en comparant confiance et précision par bin."""

    # Initialise l'ECE à zéro avant accumulation
    ece = 0.0
    # Définit les bornes de bins uniformes entre 0 et 1
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    # Itère sur chaque intervalle pour agréger l'erreur
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        # Sélectionne les prédictions appartenant au bin courant
        in_bin = (confidences >= lower) & (confidences < upper)
        # Ignore les bins vides pour éviter des divisions par zéro
        if not np.any(in_bin):
            # Passe au bin suivant si aucune donnée
            continue
        # Calcule la précision observée dans le bin
        accuracy = float(np.mean(correct[in_bin]))
        # Calcule la confiance moyenne dans le bin
        confidence = float(np.mean(confidences[in_bin]))
        # Calcule le poids relatif du bin dans l'échantillon
        weight = float(np.mean(in_bin))
        # Accumule l'erreur de calibration pondérée
        ece += weight * abs(accuracy - confidence)
    # Retourne l'ECE final
    return ece


# Calibre les confidences via une isotonic regression sur la correction
def calibrate_confidences(confidences: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Applique une calibration isotonic sur la confiance top-1."""

    # Instancie le modèle isotonic pour réajuster les confiances
    model = IsotonicRegression(out_of_bounds="clip")
    # Ajuste le modèle sur les couples confiance/correction
    model.fit(confidences, correct)
    # Retourne les confiances recalibrées
    return np.asarray(model.transform(confidences))


# Calcule les points de la courbe de calibration (reliability diagram)
def compute_calibration_curve(confidences, correct, bins):
    """Retourne les moyennes de confiance et précision par bin."""

    # Prépare les bornes de bins entre 0 et 1
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    # Initialise les listes de moyenne de confiance
    mean_confidences: list[float] = []
    # Initialise les listes de précision par bin
    mean_accuracies: list[float] = []
    # Itère sur chaque bin pour construire la courbe
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        # Calcule le masque des prédictions dans le bin
        in_bin = (confidences >= lower) & (confidences < upper)
        # Ignore les bins vides pour éviter des points trompeurs
        if not np.any(in_bin):
            # Passe au bin suivant si aucune donnée
            continue
        # Calcule la confiance moyenne du bin
        mean_confidences.append(float(np.mean(confidences[in_bin])))
        # Calcule la précision moyenne du bin
        mean_accuracies.append(float(np.mean(correct[in_bin])))
    # Retourne les tableaux numpy prêts pour le tracé
    return np.asarray(mean_confidences), np.asarray(mean_accuracies)


# Calcule une accuracy simple sur les prédictions
def compute_accuracy(y_true, y_pred) -> float:
    """Retourne la proportion de prédictions correctes."""

    # Calcule la moyenne des prédictions correctes
    return float(np.mean(y_true == y_pred))


# Calcule la balanced accuracy en moyenne de recall par classe
def compute_balanced_accuracy(y_true, y_pred, labels) -> float:
    """Retourne la balanced accuracy pour les classes fournies."""

    # Initialise la liste des recalls par classe
    recalls: list[float] = []
    # Itère sur chaque label pour calculer le recall
    for label in labels:
        # Calcule le nombre de vrais positifs pour la classe
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        # Calcule le nombre total d'exemples de la classe
        true_total = int(np.sum(y_true == label))
        # Définit un dénominateur sûr pour éviter la division par zéro
        denom = float(true_total) if true_total > 0 else SAFE_DENOMINATOR
        # Ajoute le recall pour cette classe
        recalls.append(true_positive / denom)
    # Retourne la moyenne des recalls
    return float(np.mean(recalls)) if recalls else 0.0


# Calcule le F1 macro via précision et rappel par classe
def compute_f1_macro(y_true, y_pred, labels) -> float:
    """Retourne le F1 macro sur les classes fournies."""

    # Initialise la liste des F1 par classe
    f1_scores: list[float] = []
    # Itère sur chaque label pour calculer précision et rappel
    for label in labels:
        # Calcule le nombre de vrais positifs pour la classe
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        # Calcule le nombre de prédictions pour la classe
        pred_total = int(np.sum(y_pred == label))
        # Calcule le nombre de vrais exemples de la classe
        true_total = int(np.sum(y_true == label))
        # Définit un dénominateur sûr pour la précision
        precision_denom = float(pred_total) if pred_total > 0 else SAFE_DENOMINATOR
        # Définit un dénominateur sûr pour le rappel
        recall_denom = float(true_total) if true_total > 0 else SAFE_DENOMINATOR
        # Calcule la précision pour la classe
        precision = true_positive / precision_denom
        # Calcule le rappel pour la classe
        recall = true_positive / recall_denom
        # Calcule le dénominateur du F1
        f1_denom = precision + recall
        # Définit le F1 à zéro si la classe est absente
        f1 = 0.0 if f1_denom == 0.0 else 2.0 * precision * recall / f1_denom
        # Ajoute le F1 pour cette classe
        f1_scores.append(f1)
    # Retourne la moyenne des F1
    return float(np.mean(f1_scores)) if f1_scores else 0.0


# Produit la matrice de confusion normalisée par ligne
def compute_confusion_data(y_true, y_pred, labels):
    """Retourne la matrice de confusion brute et normalisée en %."""

    # Construit l'index des labels pour la matrice
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    # Initialise la matrice de confusion à zéro
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    # Itère sur les paires vraie/predite pour remplir la matrice
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        # Ignore les labels inconnus pour rester robuste
        if true_label not in label_to_index or pred_label not in label_to_index:
            continue
        # Récupère l'index ligne pour la vraie classe
        row = label_to_index[true_label]
        # Récupère l'index colonne pour la classe prédite
        col = label_to_index[pred_label]
        # Incrémente la cellule correspondante
        matrix[row, col] += 1
    # Calcule la somme par ligne pour normaliser
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Évite la division par zéro en remplaçant les lignes vides
    row_sums = np.where(row_sums == 0, 1, row_sums)
    # Calcule la matrice en pourcentage pour l'affichage
    percent = matrix / row_sums
    # Retourne les matrices brute et normalisée
    return matrix, percent


# Trace la matrice de confusion annotée avec pourcentages et comptes
def plot_confusion_matrix(axis, matrix, percent, class_names) -> None:
    """Dessine la matrice de confusion avec annotations lisibles."""

    # Trace la matrice en heatmap pour la lisibilité
    axis.imshow(percent, cmap="Blues", vmin=0.0, vmax=1.0)
    # Fixe les ticks X/Y avec les labels de classes
    axis.set_xticks(range(len(class_names)))
    # Applique les libellés sur l'axe X
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    # Fixe les ticks Y pour les classes vraies
    axis.set_yticks(range(len(class_names)))
    # Applique les libellés sur l'axe Y
    axis.set_yticklabels(class_names)
    # Ajoute un titre explicite pour la matrice
    axis.set_title("Matrice de confusion")
    # Ajoute un label d'axe pour la vérité terrain
    axis.set_ylabel("Vérité terrain")
    # Ajoute un label d'axe pour les prédictions
    axis.set_xlabel("Prédictions")
    # Calcule la taille pour itérer sur chaque cellule
    class_count = len(class_names)
    # Itère sur les cellules pour annoter les valeurs
    for row in range(class_count):
        # Itère sur les colonnes de la matrice
        for col in range(class_count):
            # Formate le texte avec pourcentage et compte brut
            label = f"{percent[row, col]:.0%}\n({matrix[row, col]})"
            # Choisit une couleur de texte contrastée
            color = "white" if percent[row, col] > CONFUSION_TEXT_THRESHOLD else "black"
            # Place le texte au centre de la cellule
            axis.text(col, row, label, ha="center", va="center", color=color)


# Trace l'histogramme de confiance et affiche l'ECE
def plot_confidence_histogram(axis, confidences, bins, ece) -> None:
    """Affiche l'histogramme de confiance avec annotation ECE."""

    # Définit les bornes de bins uniformes
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    # Trace l'histogramme des confiances
    axis.hist(confidences, bins=bin_edges, color="#5DA5DA", edgecolor="white")
    # Ajoute un titre explicite pour le histogramme
    axis.set_title("Histogramme des scores de confiance")
    # Ajoute un label d'axe pour la confiance
    axis.set_xlabel("Score de confiance")
    # Ajoute un label d'axe pour le nombre d'epochs
    axis.set_ylabel("Nombre d'epochs")
    # Définit le style d'encart pour l'ECE
    bbox_style = {"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
    # Définit la position X de l'annotation
    ece_x = 0.05
    # Définit la position Y de l'annotation
    ece_y = 0.85
    # Formate le texte d'ECE pour l'annotation
    ece_text = f"ECE = {ece:.02f}"
    # Prépare les options de style pour l'annotation
    text_kwargs = {"transform": axis.transAxes}
    # Ajoute l'alignement horizontal aux options texte
    text_kwargs["ha"] = "left"
    # Ajoute l'alignement vertical aux options texte
    text_kwargs["va"] = "center"
    # Ajoute le style de boîte aux options texte
    text_kwargs["bbox"] = bbox_style
    # Ajoute une annotation ECE dans le graphique
    axis.text(ece_x, ece_y, ece_text, **text_kwargs)


# Trace la séquence prédictions/vérité terrain le long des epochs
def plot_predictions_vs_truth(axis, y_true, y_pred, class_names) -> None:
    """Affiche une bande temporelle vérité vs prédictions."""

    # Empile vérité et prédictions pour une image 2xN
    stacked = np.vstack([y_true, y_pred])
    # Prépare le constructeur de colormap avec un cast explicite
    cmap_builder = cast(Any, mcolors.ListedColormap)
    # Construit une colormap discrète à partir des classes
    cmap = cmap_builder(DEFAULT_CLASS_COLORS)
    # Trace l'image des classes sur les epochs
    axis.imshow(stacked, aspect="auto", cmap=cmap)
    # Masque les ticks Y pour une présentation épurée
    axis.set_yticks([0, 1])
    # Ajoute les labels de lignes pour distinguer vérité/pred
    axis.set_yticklabels(["Vérité", "Prédiction"])
    # Ajoute un titre pour contextualiser le graphique
    axis.set_title("Prédictions vs Vérité terrain")
    # Ajoute un label d'axe X pour les epochs
    axis.set_xlabel("Epochs")
    # Initialise les handles de légende
    handles = []
    # Itère sur les classes pour créer les patches
    for color, name in zip(DEFAULT_CLASS_COLORS, class_names, strict=False):
        # Ajoute un patch de légende par classe
        handles.append(Patch(color=color, label=name))
    # Ajoute la légende sous le graphique
    axis.legend(handles=handles, loc="upper center", ncol=2, fontsize="small")


# Trace la courbe de calibration avant et après recalibration
def plot_calibration_curves(axis, raw_curve, cal_curve, raw_ece, cal_ece) -> None:
    """Affiche les courbes de calibration avant/après recalibration."""

    # Prépare l'étiquette de la courbe avant recalibration
    raw_label = f"Avant (ECE {raw_ece:.02f})"
    # Prépare l'étiquette de la courbe après recalibration
    calibrated_label = f"Après (ECE {cal_ece:.02f})"
    # Trace la diagonale idéale pour référence
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    # Prépare les X de la courbe avant recalibration
    raw_x = raw_curve[0]
    # Prépare les Y de la courbe avant recalibration
    raw_y = raw_curve[1]
    # Trace la courbe avant recalibration
    axis.plot(raw_x, raw_y, marker="o", color="#F15854", label=raw_label)
    # Prépare les X de la courbe après recalibration
    calibrated_x = cal_curve[0]
    # Prépare les Y de la courbe après recalibration
    calibrated_y = cal_curve[1]
    # Prépare les options de style pour la courbe recalibrée
    cal_style = {"marker": "o"}
    # Ajoute la couleur de la courbe recalibrée
    cal_style["color"] = "#5DA5DA"
    # Ajoute le label de la courbe recalibrée
    cal_style["label"] = calibrated_label
    # Trace la courbe après recalibration
    axis.plot(calibrated_x, calibrated_y, **cal_style)
    # Ajoute un titre explicite au sous-graphique
    axis.set_title("Courbes de calibration")
    # Ajoute un label d'axe X pour la confiance
    axis.set_xlabel("Score prédit")
    # Ajoute un label d'axe Y pour la précision
    axis.set_ylabel("Précision observée")
    # Ajoute une légende pour distinguer les courbes
    axis.legend(loc="lower right", fontsize="small")


# Génère la figure complète du tableau de bord
def generate_dashboard(y_true, y_pred, y_proba, config) -> Path:
    """Construit et sauvegarde le tableau de bord d'évaluation."""

    # Valide la cohérence des tailles d'entrée
    _validate_input_shapes(y_true, y_pred, y_proba)
    # Assure que la sortie existe avant sauvegarde
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Déduit le nombre de classes depuis les proba
    class_count = y_proba.shape[1]
    # Construit les labels numériques pour la matrice
    labels = list(range(class_count))
    # Calcule les métriques principales d'évaluation
    accuracy = compute_accuracy(y_true, y_pred)
    # Calcule la balanced accuracy utile aux classes déséquilibrées
    balanced_acc = compute_balanced_accuracy(y_true, y_pred, labels)
    # Calcule le F1 macro pour résumer les classes
    f1_macro = compute_f1_macro(y_true, y_pred, labels)
    # Calcule le nombre d'erreurs totales
    errors = int(np.sum(y_true != y_pred))
    # Calcule la matrice de confusion brute et en %
    matrix, percent = compute_confusion_data(y_true, y_pred, labels)
    # Calcule la confiance top-1 des prédictions
    confidences = np.max(y_proba, axis=1)
    # Calcule la correction binaire pour la calibration
    correct = (y_true == y_pred).astype(float)
    # Stocke le nombre de bins pour les métriques de calibration
    bins = config.confidence_bins
    # Calcule l'ECE avant recalibration
    raw_ece = expected_calibration_error(confidences, correct, bins)
    # Calibre les confidences via isotonic regression
    calibrated_confidences = calibrate_confidences(confidences, correct)
    # Calcule l'ECE après recalibration
    calibrated_ece = expected_calibration_error(calibrated_confidences, correct, bins)
    # Calcule les points de calibration avant recalibration
    raw_curve = compute_calibration_curve(confidences, correct, bins)
    # Calcule les points de calibration après recalibration
    calibrated_curve = compute_calibration_curve(calibrated_confidences, correct, bins)
    # Alias court pour la courbe recalibrée
    cal_curve = calibrated_curve

    # Crée la figure principale avec la taille définie
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    # Crée une grille pour organiser les sous-graphes
    layout = gridspec.GridSpec(3, 2, height_ratios=[0.3, 1.0, 1.0])
    # Ajoute un axe texte pour les métriques globales
    metrics_axis = fig.add_subplot(layout[0, :])
    # Masque les axes pour un rendu type header
    metrics_axis.axis("off")
    # Construit le fragment Accuracy pour l'entête
    acc_text = f"Accuracy:{accuracy:.2f}"
    # Construit le fragment Balanced Acc pour l'entête
    balanced_text = f"Balanced:{balanced_acc:.2f}"
    # Construit le fragment F1-macro pour l'entête
    f1_text = f"F1-macro:{f1_macro:.2f}"
    # Construit le fragment Errors pour l'entête
    errors_text = f"Errors:{errors}/{len(y_true)} epochs"
    # Construit le texte d'entête avec les métriques
    metrics_text = " | ".join([acc_text, balanced_text, f1_text, errors_text])
    # Ajoute le texte d'entête centré
    metrics_axis.text(0.5, 0.5, metrics_text, ha="center", va="center")

    # Ajoute l'axe de la matrice de confusion
    confusion_axis = fig.add_subplot(layout[1, 0])
    # Trace la matrice de confusion avec annotations
    plot_confusion_matrix(confusion_axis, matrix, percent, config.class_names)

    # Ajoute l'axe pour l'histogramme de confiance
    histogram_axis = fig.add_subplot(layout[1, 1])
    # Trace l'histogramme des confiances avec ECE
    plot_confidence_histogram(histogram_axis, confidences, bins, raw_ece)

    # Ajoute l'axe pour la comparaison vérité/pred
    prediction_axis = fig.add_subplot(layout[2, 0])
    # Trace la comparaison prédictions/vérité terrain
    plot_predictions_vs_truth(prediction_axis, y_true, y_pred, config.class_names)

    # Ajoute l'axe pour les courbes de calibration
    calibration_axis = fig.add_subplot(layout[2, 1])
    # Regroupe les arguments de calibration pour un appel compact
    calibration_args = (raw_curve, cal_curve, raw_ece, calibrated_ece)
    # Trace les courbes de calibration avant/après
    plot_calibration_curves(calibration_axis, *calibration_args)

    # Ajuste la mise en page globale
    fig.tight_layout()
    # Normalise le répertoire de sortie en objet Path
    output_dir = Path(config.output_dir)
    # Normalise le nom de fichier de sortie en chaîne
    output_name = str(config.output_name)
    # Construit le chemin de sortie complet
    output_path = output_dir / output_name
    # Sauvegarde la figure au format PNG
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire
    plt.close(fig)
    # Retourne le chemin de la figure générée
    return output_path


# Point d'entrée CLI pour générer le tableau de bord
def main() -> None:
    """Charge les données et génère le tableau de bord demandé."""

    # Construit le parseur pour récupérer les arguments CLI
    parser = build_parser()
    # Analyse les arguments fournis par l'utilisateur
    args = parser.parse_args()
    # Charge les données depuis le fichier ou l'exemple
    input_path = Path(args.input) if args.input else None
    # Charge les labels et probabilités
    y_true, y_pred, y_proba = load_dashboard_data(input_path)
    # Prépare le répertoire de sortie en objet Path
    output_dir = Path(args.output_dir)
    # Prépare le nom de fichier de sortie
    output_name = str(args.output_name)
    # Prépare les noms de classes pour l'affichage
    class_names = list(args.classes)
    # Prépare le nombre de bins pour l'histogramme
    bins = int(args.bins)
    # Prépare la configuration de génération
    config = DashboardConfig(input_path, output_dir, output_name, class_names, bins)
    # Génère le tableau de bord et récupère le chemin
    output_path = generate_dashboard(y_true, y_pred, y_proba, config)
    # Affiche le chemin pour faciliter le scripting
    print(f"Dashboard généré: {output_path}")


# Active l'exécution du main uniquement en invocation directe
if __name__ == "__main__":
    # Lance le script via le point d'entrée CLI
    main()
