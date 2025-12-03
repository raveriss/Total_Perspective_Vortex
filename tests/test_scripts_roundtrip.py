# Importe json pour inspecter les manifestes produits
import json

# Importe numpy pour générer des données synthétiques
import numpy as np

# Importe PipelineConfig pour aligner les paramètres du pipeline
from tpv.pipeline import PipelineConfig

# Importe TrainingRequest pour formuler la demande d'entraînement
from scripts.train import TrainingRequest, run_training

# Importe evaluate_run pour vérifier la génération des rapports prédictifs
from scripts.predict import evaluate_run


# Vérifie qu'entraînement et prédiction produisent manifestes et rapports
def test_train_and_predict_produce_manifests_and_reports(tmp_path):
    """Valide l'intégration train/predict sur données jouets."""

    # Fixe le sujet et le run utilisés pour le scénario d'intégration
    subject = "S01"
    # Fixe l'identifiant du run pour cibler les fichiers numpy
    run = "R01"
    # Prépare le répertoire racine des données jouets
    data_dir = tmp_path / "data"
    # Prépare le répertoire racine des artefacts à valider
    artifacts_dir = tmp_path / "artifacts"
    # Prépare le sous-dossier spécifique au sujet
    subject_dir = data_dir / subject
    # Crée l'arborescence de données synthétiques
    subject_dir.mkdir(parents=True)
    # Initialise un générateur aléatoire pour des données stables
    rng = np.random.default_rng(0)
    # Génère six échantillons bidimensionnels pour l'entraînement
    X = rng.normal(size=(6, 2, 20))
    # Alterne les labels pour garantir deux classes équilibrées
    y = np.array([0, 1, 0, 1, 0, 1])
    # Sauvegarde les features dans la structure attendue par la CLI
    np.save(subject_dir / f"{run}_X.npy", X)
    # Sauvegarde les labels correspondants pour l'entraînement
    np.save(subject_dir / f"{run}_y.npy", y)
    # Construit une configuration minimale pour accélérer le test
    config = PipelineConfig(
        sfreq=64.0,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
    )
    # Construit la requête d'entraînement complète
    request = TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    # Lance l'entraînement pour produire modèle et manifeste
    train_result = run_training(request)
    # Récupère le chemin du manifeste généré
    manifest_path = train_result["manifest_path"]
    # Vérifie que le manifeste existe bien sur disque
    assert manifest_path.exists()
    # Charge le contenu JSON pour valider la structure
    manifest = json.loads(manifest_path.read_text())
    # Vérifie que le manifeste référence le bon sujet
    assert manifest["dataset"]["subject"] == subject
    # Vérifie que le manifeste référence le bon run
    assert manifest["dataset"]["run"] == run
    # Vérifie que la section des scores expose une liste
    assert isinstance(manifest["scores"]["cv_scores"], list)
    # Lance la prédiction pour produire les rapports demandés
    predict_result = evaluate_run(subject, run, data_dir, artifacts_dir)
    # Récupère les chemins de rapports générés par evaluate_run
    reports = predict_result["reports"]
    # Vérifie que le rapport JSON existe bien dans les artefacts
    assert reports["json_report"].exists()
    # Vérifie que le rapport CSV existe bien dans les artefacts
    assert reports["csv_report"].exists()
    # Charge le rapport JSON pour valider la matrice de confusion
    json_report = json.loads(reports["json_report"].read_text())
    # Vérifie que la matrice de confusion est bien un tableau imbriqué
    assert isinstance(json_report["confusion_matrix"], list)
    # Vérifie que le nombre d'échantillons loggé correspond aux données
    assert json_report["samples"] == len(y)
