# ruff: noqa: PLR0915
# Importe builtins pour espionner zip et verrouiller strict=False
import builtins

# Importe csv pour analyser les fichiers produits par le reporting
import csv

# Importe json pour inspecter le contenu du rapport sérialisé
import json

# Importe Path pour espionner les appels à open et verrouiller newline=""
from pathlib import Path

# Importe numpy pour générer des labels synthétiques
import numpy as np

# Importe le module predict pour monkeypatcher les dépendances internes
from scripts import predict as predict_module

# Importe les utilitaires internes pour la sérialisation des rapports
from scripts.predict import _stringify_label, _write_reports


def test_write_reports_serializes_expected_outputs(
    tmp_path, monkeypatch
):  # noqa: PLR0915
    """Vérifie la sérialisation complète des rapports de prédiction."""

    # Prépare le répertoire cible pour collecter les fichiers générés
    target_dir = tmp_path / "reports"
    # Crée le répertoire cible pour imiter le dossier d'artefacts
    target_dir.mkdir()
    # Construit des identifiants minimaux pour les rapports générés
    identifiers = {"subject": "S001", "run": "R01"}
    # Crée un vecteur de vérité terrain binaire
    y_true = np.array([0, 1], dtype=int)
    # Crée un vecteur de prédictions volontairement imparfait
    y_pred = np.array([0, 0], dtype=int)
    # Fixe une accuracy réaliste pour ce cas de test jouet
    accuracy = 0.5

    # Calcule les labels attendus pour verrouiller l'appel à confusion_matrix
    expected_labels = sorted(np.unique(np.concatenate((y_true, y_pred))).tolist())
    # Capture les paramètres passés à confusion_matrix pour tuer les mutants équivalents
    confusion_labels: list[object] = []
    # Préserve la fonction réelle pour produire la matrice attendue
    real_confusion_matrix = predict_module.confusion_matrix

    # Espionne confusion_matrix pour valider que labels est explicitement fourni
    def spy_confusion_matrix(y_true_arg, y_pred_arg, **kwargs):
        confusion_labels.append(kwargs.get("labels", "__missing__"))
        return real_confusion_matrix(y_true_arg, y_pred_arg, **kwargs)

    monkeypatch.setattr(predict_module, "confusion_matrix", spy_confusion_matrix)

    # Capture les kwargs passés à json.dumps pour verrouiller ensure_ascii et indent
    dumps_kwargs: list[dict[str, object]] = []
    # Préserve json.dumps pour obtenir un contenu de fichier valide
    real_json_dumps = predict_module.json.dumps

    # Espionne json.dumps pour valider la sérialisation UTF-8 et l'indentation
    def spy_json_dumps(*args, **kwargs):
        dumps_kwargs.append(dict(kwargs))
        return real_json_dumps(*args, **kwargs)

    monkeypatch.setattr(predict_module.json, "dumps", spy_json_dumps)

    # Capture le strict ciblé sur la boucle de prédictions pour figer le contrat
    zip_strict_values: list[object] = []
    # Préserve zip pour déléguer le comportement d'itération standard
    real_zip = builtins.zip

    # Espionne zip afin de garantir l'usage explicite de strict=False
    def spy_zip(*iterables, **kwargs):
        if len(iterables) == 2 and iterables[0] is y_true and iterables[1] is y_pred:
            zip_strict_values.append(kwargs.get("strict", "__missing__"))
        return real_zip(*iterables, **kwargs)

    monkeypatch.setattr(builtins, "zip", spy_zip)

    # Capture les paramètres newline utilisés pour les fichiers CSV écrits
    newline_values: list[tuple[str, object]] = []
    # Préserve Path.open pour ouvrir réellement les fichiers sur disque
    real_path_open = Path.open

    # Espionne Path.open pour verrouiller newline="" sur les deux CSV
    def spy_path_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode.startswith("w") and self.name in {
            "class_report.csv",
            "predictions.csv",
        }:
            newline_values.append((self.name, kwargs.get("newline", "__missing__")))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_path_open)

    # Génère les rapports en appelant l'utilitaire interne
    reports = _write_reports(target_dir, identifiers, y_true, y_pred, accuracy)

    # Verrouille les chemins de sortie pour éviter des régressions silencieuses
    assert reports["json_report"] == target_dir / "report.json"
    assert reports["class_report"] == target_dir / "class_report.csv"
    assert reports["csv_report"] == target_dir / "predictions.csv"

    # Vérifie que les clés structurantes sont retournées par l'API interne
    assert reports["confusion"] == [[1, 0], [1, 0]]
    assert reports["per_class_accuracy"] == {"0": 1.0, "1": 0.0}

    # Vérifie que labels est explicitement fourni à confusion_matrix
    assert confusion_labels == [expected_labels]

    # Vérifie que json.dumps force ensure_ascii=False et indent=2
    assert len(dumps_kwargs) == 1
    assert dumps_kwargs[0].get("ensure_ascii", "__missing__") is False
    assert dumps_kwargs[0].get("indent", "__missing__") == 2

    # Vérifie que zip est invoqué en spécifiant strict=False
    assert zip_strict_values == [False]

    # Vérifie que newline="" est propagé sur les deux ouvertures CSV en écriture
    assert sorted(newline_values) == [
        ("class_report.csv", ""),
        ("predictions.csv", ""),
    ]

    # Charge le contenu JSON pour inspecter les valeurs calculées
    report_json = json.loads(reports["json_report"].read_text())
    # Vérifie que le sujet est bien propagé dans le rapport
    assert report_json["subject"] == identifiers["subject"]
    # Vérifie que le run est bien propagé dans le rapport
    assert report_json["run"] == identifiers["run"]
    # Vérifie que l'accuracy est bien propagée dans le rapport
    assert report_json["accuracy"] == accuracy
    # Vérifie la matrice de confusion sérialisée attendue
    assert report_json["confusion_matrix"] == [[1, 0], [1, 0]]
    # Vérifie l'accuracy par classe, y compris la classe en échec
    assert report_json["per_class_accuracy"] == {"0": 1.0, "1": 0.0}
    # Vérifie que le champ samples reflète le jeu de test
    assert report_json["samples"] == 2
    # Ouvre le rapport par classe pour contrôler l'écriture CSV
    with reports["class_report"].open(newline="") as handle:
        # Charge le contenu CSV en mémoire pour vérification
        rows = list(csv.DictReader(handle))
    # Vérifie que deux lignes correspondant aux classes existent
    assert rows == [
        {"class": "0", "accuracy": "1.0", "support": "1"},
        {"class": "1", "accuracy": "0.0", "support": "1"},
    ]
    # Ouvre le CSV de prédictions détaillées
    with reports["csv_report"].open(newline="") as handle:
        # Charge chaque ligne pour inspecter les valeurs sérialisées
        prediction_rows = list(csv.DictReader(handle))
    # Vérifie que chaque ligne reprend les identifiants et les labels
    assert prediction_rows == [
        {
            "subject": "S001",
            "run": "R01",
            "index": "0",
            "y_true": "0",
            "y_pred": "0",
        },
        {
            "subject": "S001",
            "run": "R01",
            "index": "1",
            "y_true": "1",
            "y_pred": "0",
        },
    ]


def test_write_reports_handles_missing_true_class(tmp_path):
    """Vérifie la gestion d'une classe absente dans y_true."""

    # Prépare le répertoire pour ce scénario de classe manquante
    target_dir = tmp_path / "missing_class"
    # Crée le répertoire cible pour stocker les rapports produits
    target_dir.mkdir()
    # Construit les identifiants de sujet et de run pour le rapport
    identifiers = {"subject": "S002", "run": "R02"}
    # Simule un jeu de labels sans occurrence de la classe 1
    y_true = np.array([0, 0], dtype=int)
    # Simule des prédictions contenant une classe jamais observée
    y_pred = np.array([1, 1], dtype=int)
    # Fixe une accuracy nulle pour ce scénario défavorable
    accuracy = 0.0
    # Génère les rapports pour valider la robustesse du calcul
    reports = _write_reports(target_dir, identifiers, y_true, y_pred, accuracy)
    # Charge le rapport JSON pour inspecter la matrice de confusion
    report_json = json.loads(reports["json_report"].read_text())
    # Vérifie que la matrice inclut la classe absente côté vérité
    assert report_json["confusion_matrix"] == [[0, 2], [0, 0]]
    # Vérifie que l'accuracy de la classe sans support vaut bien 0.0
    assert report_json["per_class_accuracy"]["1"] == 0.0
    # Ouvre le rapport par classe pour vérifier le support nul
    with reports["class_report"].open(newline="") as handle:
        # Charge les lignes CSV pour inspection des supports
        rows = list(csv.DictReader(handle))
    # Vérifie que le support de la classe absente reste nul
    assert rows[1]["support"] == "0"


def test_stringify_label_coerces_float_integers() -> None:
    """Confirme la conversion des floats entiers en chaînes sans décimales."""

    # Vérifie que les floats entiers sont convertis en entiers textuels
    assert _stringify_label(np.float64(2.0)) == "2"
    # Vérifie que les labels non numériques restent inchangés
    assert _stringify_label("left") == "left"
