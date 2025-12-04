# Importe Path pour manipuler les chemins du dépôt git
# Importe json pour inspecter les manifestes produits
import json
from pathlib import Path

# Importe numpy pour générer des données synthétiques
import numpy as np

# Importe le module train pour invoquer le main CLI sans ambiguïté
from scripts import train

# Importe evaluate_run pour vérifier la génération des rapports prédictifs
from scripts.predict import evaluate_run

# Importe _get_git_commit pour couvrir les branches de repli git
from scripts.train import TrainingRequest, _get_git_commit, run_training

# Importe PipelineConfig pour aligner les paramètres du pipeline
from tpv.pipeline import PipelineConfig


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


# Vérifie que l'entrée CLI produit bien un manifeste et un scaler dédié
def test_train_main_produces_manifest_and_scaler(tmp_path, monkeypatch):
    """Couvre le chemin CLI avec scaler explicite et manifeste attendu."""

    # Fige le sujet simulé pour préparer les chemins attendus
    subject = "S99"
    # Fige le run simulé pour contrôler le nommage des fichiers
    run = "R09"
    # Prépare le répertoire de données isolé pour le scénario CLI
    data_dir = tmp_path / "data"
    # Prépare le répertoire d'artefacts isolé pour vérifier la sortie
    artifacts_dir = tmp_path / "artifacts"
    # Construit l'arborescence du sujet pour placer les fichiers numpy
    subject_dir = data_dir / subject
    # Crée les dossiers pour accueillir les matrices synthétiques
    subject_dir.mkdir(parents=True)
    # Initialise un générateur déterministe pour stabiliser la CV
    rng = np.random.default_rng(42)
    # Génère des observations bidimensionnelles réalistes pour le test
    X = rng.normal(size=(6, 2, 20))
    # Alterne les étiquettes pour forcer la stratification tripartite
    y = np.array([0, 1, 0, 1, 0, 1])
    # Sauvegarde les features dans la structure attendue par la CLI
    np.save(subject_dir / f"{run}_X.npy", X)
    # Sauvegarde les labels alignés pour déclencher la CV
    np.save(subject_dir / f"{run}_y.npy", y)
    # Construit les arguments CLI en forçant un scaler explicite
    argv = [
        subject,
        run,
        "--data-dir",
        str(data_dir),
        "--artifacts-dir",
        str(artifacts_dir),
        "--sfreq",
        "64.0",
        "--feature-strategy",
        "fft",
        "--dim-method",
        "pca",
        "--scaler",
        "standard",
    ]
    # Bascule dans un répertoire neutre pour éviter les effets globaux
    monkeypatch.chdir(tmp_path)
    # Exécute le main CLI et vérifie le code retour de succès
    assert train.main(argv) == 0
    # Construit le chemin attendu du manifeste pour l'assertion
    manifest_path = artifacts_dir / subject / run / "manifest.json"
    # Vérifie que le manifeste CLI est bien présent sur disque
    assert manifest_path.exists()
    # Charge le JSON pour inspecter la section des artefacts
    manifest = json.loads(manifest_path.read_text())
    # Vérifie que le scaler sérialisé correspond exactement au chemin attendu
    assert manifest["artifacts"]["scaler"] == str(
        artifacts_dir / subject / run / "scaler.joblib"
    )


# Vérifie que _get_git_commit gère l'absence complète du dépôt
def test_get_git_commit_returns_unknown_without_repo(tmp_path, monkeypatch):
    """Couvre les branches de secours lorsque .git n'existe pas."""

    # Définit un répertoire sans initialisation git pour tester le repli
    naked_dir = tmp_path / "no_git"
    # Crée le répertoire isolé pour éviter tout artefact git
    naked_dir.mkdir()
    # Change le répertoire courant pour pointer vers l'emplacement vierge
    monkeypatch.chdir(naked_dir)
    # Vérifie que l'absence de HEAD renvoie la valeur inconnue attendue
    assert _get_git_commit() == "unknown"


# Vérifie que _get_git_commit gère une référence manquante
def test_get_git_commit_returns_unknown_without_ref(tmp_path, monkeypatch):
    """Valide le repli lorsque la référence HEAD est introuvable."""

    # Prépare l'emplacement git minimal pour simuler une référence brisée
    git_dir = tmp_path / "broken_git" / ".git"
    # Crée l'arborescence .git factice pour écrire le HEAD
    git_dir.mkdir(parents=True)
    # Déclare une référence HEAD vers une branche inexistante
    head_content = "ref: refs/heads/main"
    # Écrit la référence dans le fichier HEAD sans créer la cible
    (git_dir / "HEAD").write_text(head_content)
    # Bascule dans le dépôt factice pour invoquer la fonction
    monkeypatch.chdir(git_dir.parent)
    # Vérifie que l'absence du fichier de référence déclenche le repli
    assert _get_git_commit() == "unknown"


# Vérifie que _get_git_commit gère un HEAD vide
def test_get_git_commit_returns_unknown_with_empty_head(tmp_path, monkeypatch):
    """Assure le repli lorsque HEAD ne contient aucun hash."""

    # Prépare un dépôt minimal avec fichier HEAD vide
    git_dir = tmp_path / "empty_head" / ".git"
    # Crée l'arborescence git simulée pour manipuler HEAD
    git_dir.mkdir(parents=True)
    # Écrit un fichier HEAD vide pour déclencher la valeur de secours
    (git_dir / "HEAD").write_text("")
    # Bascule dans le dépôt simulé pour exécuter la fonction
    monkeypatch.chdir(git_dir.parent)
    # Vérifie que la valeur retournée correspond au repli attendu
    assert _get_git_commit() == "unknown"


# Vérifie que _get_git_commit retourne bien un hash dans un dépôt valide
def test_get_git_commit_returns_hash_in_repo(monkeypatch):
    """Couvre le chemin nominal lorsque HEAD pointe vers une référence valide."""

    # Identifie la racine du dépôt git pour simuler un appel utilisateur
    repo_root = Path(__file__).resolve().parent.parent
    # Force l'exécution dans le dépôt réel pour lire le HEAD courant
    monkeypatch.chdir(repo_root)
    # Appelle la récupération du hash pour exercer la branche nominale
    commit = _get_git_commit()
    # Vérifie que le hash est inconnu ou bien composé de caractères hexadécimaux
    assert commit == "unknown" or all(
        char in "0123456789abcdef" for char in commit.lower()
    )
