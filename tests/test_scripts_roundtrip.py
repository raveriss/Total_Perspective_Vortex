# Importe Path pour manipuler les chemins du dépôt git
# Importe json pour inspecter les manifestes produits
# Importe csv pour valider les manifestes tabulaires
import csv

# Importe json pour inspecter les manifestes sérialisés
import json
from pathlib import Path

# Importe numpy pour générer des données synthétiques
import numpy as np

# Importe pytest pour orchestrer les scénarios paramétrés
import pytest

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
    subject = "S001"
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
    # Récupère le chemin du manifeste CSV pour valider l'export tabulaire
    manifest_csv_path = train_result["manifest_csv_path"]
    # Vérifie que le manifeste CSV est bien créé
    assert manifest_csv_path.exists()
    # Ouvre le manifeste CSV pour comparer avec la version JSON
    with manifest_csv_path.open() as handle:
        # Construit un reader pour extraire la ligne unique
        reader = csv.DictReader(handle)
        # Récupère la ligne décrivant le run
        csv_manifest = next(reader)
    # Vérifie que le manifeste référence le bon sujet
    assert manifest["dataset"]["subject"] == subject
    # Vérifie que le manifeste référence le bon run
    assert manifest["dataset"]["run"] == run
    # Vérifie que le manifeste CSV référence bien le sujet
    assert csv_manifest["subject"] == subject
    # Vérifie que le manifeste CSV référence bien le run
    assert csv_manifest["run"] == run
    # Vérifie que le hash git est synchronisé entre JSON et CSV
    assert csv_manifest["git_commit"] == manifest["git_commit"]
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
    # Vérifie que le rapport par classe existe pour les diagnostics
    assert reports["class_report"].exists()
    # Charge le rapport JSON pour valider la matrice de confusion
    json_report = json.loads(reports["json_report"].read_text())
    # Vérifie que la matrice de confusion est bien un tableau imbriqué
    assert isinstance(json_report["confusion_matrix"], list)
    # Vérifie que l'accuracy par classe est bien renseignée
    assert set(json_report["per_class_accuracy"].keys()) == {"0", "1"}
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


def test_run_training_handles_two_splits_without_cv(tmp_path):
    """Valide la génération de manifeste quand la validation croisée est bypassée."""

    # Fixe le sujet synthétique pour construire l'arborescence attendue
    subject = "S02"
    # Fixe le run synthétique pour aligner le nommage des fichiers
    run = "R02"
    # Prépare le répertoire racine de données jouets
    data_dir = tmp_path / "data"
    # Prépare le répertoire racine des artefacts générés
    artifacts_dir = tmp_path / "artifacts"
    # Construit le sous-dossier dédié au sujet synthétique
    subject_dir = data_dir / subject
    # Crée l'arborescence des fichiers numpy pour le sujet cible
    subject_dir.mkdir(parents=True)
    # Initialise un générateur aléatoire pour des données stables
    rng = np.random.default_rng(123)
    # Génère quatre échantillons pour rester sous le seuil des trois splits
    X = rng.normal(size=(4, 2, 20))
    # Associe deux classes équilibrées pour conserver la stratification
    y = np.array([0, 1, 0, 1])
    # Sauvegarde les features dans le format attendu par la CLI
    np.save(subject_dir / f"{run}_X.npy", X)
    # Sauvegarde les labels alignés pour l'entraînement
    np.save(subject_dir / f"{run}_y.npy", y)
    # Construit une configuration minimale pour accélérer l'entraînement
    config = PipelineConfig(
        sfreq=64.0,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
    )
    # Prépare la requête d'entraînement avec les chemins simulés
    request = TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    # Lance l'entraînement pour générer le modèle et le manifeste
    result = run_training(request)
    # Vérifie que la validation croisée a été ignorée faute de splits suffisants
    assert result["cv_scores"].size == 0
    # Charge le manifeste pour inspecter les valeurs sérialisées
    manifest = json.loads(result["manifest_path"].read_text())
    # Vérifie que la liste des scores est bien vide dans le manifeste
    assert manifest["scores"]["cv_scores"] == []
    # Vérifie que la moyenne des scores est absente lorsque la CV est omise
    assert manifest["scores"]["cv_mean"] is None
    # Vérifie que le scaler reste absent lorsqu'aucun scaler n'est configuré
    assert manifest["artifacts"]["scaler"] is None


def test_run_training_aligns_cv_splits_with_min_class_count(tmp_path):
    """Valide que la CV suit l'effectif minimal détecté dans les labels."""

    # Fixe le sujet synthétique pour construire l'arborescence attendue
    subject = "S03"
    # Fixe le run synthétique pour aligner le nommage des fichiers
    run = "R03"
    # Définit le nombre attendu d'occurrences minoritaires pour la CV
    minority_class_count = 7
    # Prépare le répertoire racine de données jouets
    data_dir = tmp_path / "data"
    # Prépare le répertoire racine des artefacts générés
    artifacts_dir = tmp_path / "artifacts"
    # Construit le sous-dossier dédié au sujet synthétique
    subject_dir = data_dir / subject
    # Crée l'arborescence des fichiers numpy pour le sujet cible
    subject_dir.mkdir(parents=True)
    # Initialise un générateur aléatoire pour des données stables
    rng = np.random.default_rng(7)
    # Génère quatorze échantillons pour permettre sept splits stratifiés
    X = rng.normal(size=(14, 2, 20))
    # Construit des labels en bornant la classe minoritaire au compteur dédié
    y = np.array([0] * minority_class_count + [1] * minority_class_count)
    # Sauvegarde les features dans le format attendu par la CLI
    np.save(subject_dir / f"{run}_X.npy", X)
    # Sauvegarde les labels alignés pour l'entraînement
    np.save(subject_dir / f"{run}_y.npy", y)
    # Construit la configuration de pipeline simplifiée pour accélérer le test
    config = PipelineConfig(
        sfreq=64.0,
        feature_strategy="fft",
        normalize_features=True,
        dim_method="pca",
    )
    # Prépare la requête d'entraînement avec les chemins simulés
    request = TrainingRequest(
        subject=subject,
        run=run,
        pipeline_config=config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    # Lance l'entraînement pour générer le modèle et le manifeste
    result = run_training(request)
    # Vérifie que le nombre de scores reflète bien les sept occurrences minoritaires
    assert result["cv_scores"].size == minority_class_count
    # Charge le manifeste pour inspecter les valeurs sérialisées
    manifest = json.loads(result["manifest_path"].read_text())
    # Vérifie que la longueur des scores dans le manifeste correspond aux splits
    assert len(manifest["scores"]["cv_scores"]) == minority_class_count


# Vérifie que _load_data reconstruit les .npy corrompus via l'EDF
def test_load_data_rebuilds_after_corruption(tmp_path, monkeypatch):
    """Forcer la reconstruction quand np.load échoue sur un fichier corrompu."""

    # Prépare le sujet fictif pour isoler les chemins de test
    subject = "S077"
    # Prépare le run fictif pour déclencher la branche de reconstruction
    run = "R09"
    # Construit le répertoire racine des données simulées
    data_dir = tmp_path / "data"
    # Construit un répertoire brut factice pour respecter la signature
    raw_dir = tmp_path / "raw"
    # Compose le dossier sujet où seront placés les fichiers corrompus
    subject_dir = data_dir / subject
    # Crée l'arborescence pour simuler des fichiers déjà présents
    subject_dir.mkdir(parents=True)
    # Crée un fichier X illisible pour provoquer un ValueError lors du chargement
    (subject_dir / f"{run}_X.npy").write_text("corrupted payload")
    # Génère un y minimal pour respecter la convention de nommage
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    # Prépare les features que le stub va écrire lors de la reconstruction
    rebuilt_X = np.ones((2, 2, 2))
    # Prépare les labels régénérés pour valider la correspondance
    rebuilt_y = np.array([1, 0])
    # Trace les appels pour s'assurer que la reconstruction a été sollicitée
    calls: list[tuple[str, str]] = []

    # Déclare un stub pour remplacer la reconstruction EDF pendant le test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        # Archive les arguments pour vérifier la propagation des paramètres
        calls.append((subject_arg, run_arg))
        # Construit les chemins de sortie pour les fichiers régénérés
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        # Construit le chemin des labels pour rester cohérent avec _load_data
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        # Sauvegarde les features reconstruites pour remplacer le fichier corrompu
        np.save(features_path, rebuilt_X)
        # Sauvegarde les labels régénérés pour réaligner X et y
        np.save(labels_path, rebuilt_y)
        # Retourne les chemins pour respecter l'interface attendue
        return features_path, labels_path

    # Injecte le stub pour forcer le chemin de reconstruction
    monkeypatch.setattr(train, "_build_npy_from_edf", fake_build_npy)

    # Charge les données, ce qui doit déclencher la reconstruction simulée
    X, y = train._load_data(subject, run, data_dir, raw_dir)

    # Vérifie que la reconstruction a bien été invoquée pendant le chargement
    assert calls == [(subject, run)]
    # Vérifie que les features proviennent bien des fichiers reconstruits
    assert np.array_equal(X, rebuilt_X)
    # Vérifie que les labels proviennent bien des fichiers reconstruits
    assert np.array_equal(y, rebuilt_y)


# Vérifie que _load_data régénère pour toutes les variantes de fichiers invalides
@pytest.mark.parametrize(
    "scenario",
    (
        "missing_features",
        "features_not_3d",
        "labels_mismatch",
    ),
)
def test_load_data_rebuilds_invalid_numpy_payloads(tmp_path, monkeypatch, scenario):
    """Déclenche la reconstruction lorsque les .npy sont mal formés."""

    # Prépare le sujet fictif pour isoler les chemins des échantillons
    subject = "S088"
    # Prépare le run fictif pour cibler un run moteur documenté
    run = "R08"
    # Construit le répertoire racine des données synthétiques
    data_dir = tmp_path / "data"
    # Construit le répertoire racine des données brutes attendu par la signature
    raw_dir = tmp_path / "raw"
    # Construit le dossier du sujet pour déposer les .npy temporaires
    subject_dir = data_dir / subject
    # Crée l'arborescence nécessaire pour simuler les fichiers
    subject_dir.mkdir(parents=True)
    # Prépare des features synthétiques pour alimenter la reconstruction
    rebuilt_X = np.full((3, 2, 2), fill_value=7)
    # Prépare des labels synthétiques alignés sur les features reconstruites
    rebuilt_y = np.array([1, 0, 1])
    # Initialise un traceur d'appels pour vérifier la reconstruction
    calls: list[tuple[str, str]] = []

    # Déclare un stub de reconstruction pour simuler l'EDF absent du test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        # Archive les arguments reçus pour vérifier la propagation complète
        calls.append((subject_arg, run_arg))
        # Calcule le chemin des features reconstruites pour remplacer l'entrée
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        # Calcule le chemin des labels reconstruits pour réaligner les échantillons
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        # Sauvegarde les features simulées pour annuler les fichiers invalides
        np.save(features_path, rebuilt_X)
        # Sauvegarde les labels simulés pour aligner la longueur avec X
        np.save(labels_path, rebuilt_y)
        # Retourne les chemins sauvegardés pour respecter le contrat attendu
        return features_path, labels_path

    # Injecte le stub pour suivre les reconstructions demandées
    monkeypatch.setattr(train, "_build_npy_from_edf", fake_build_npy)

    # Injecte des fichiers invalides selon le scénario ciblé
    if scenario == "missing_features":
        # Enregistre uniquement y pour simuler un X absent
        np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    elif scenario == "features_not_3d":
        # Crée un X 2D pour déclencher la reconstruction sur la dimension attendue
        np.save(subject_dir / f"{run}_X.npy", np.ones((2, 4)))
        # Crée un y aligné sur le nombre d'échantillons de X 2D
        np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    elif scenario == "labels_mismatch":
        # Crée un X 3D valide pour isoler le désalignement des labels
        np.save(subject_dir / f"{run}_X.npy", np.ones((4, 2, 2)))
        # Crée un y plus court pour déclencher la régénération
        np.save(subject_dir / f"{run}_y.npy", np.array([0, 1, 1]))

    # Charge les données, ce qui doit forcer la reconstruction simulée
    X, y = train._load_data(subject, run, data_dir, raw_dir)

    # Vérifie que la reconstruction a été invoquée exactement une fois
    assert calls == [(subject, run)]
    # Vérifie que les features chargées correspondent au fichier reconstruit
    assert np.array_equal(X, rebuilt_X)
    # Vérifie que les labels chargés correspondent au fichier reconstruit
    assert np.array_equal(y, rebuilt_y)


# Vérifie que corrupted_reason est journalisé lorsqu'un np.load échoue
def test_load_data_logs_corrupted_reason(tmp_path, monkeypatch, capsys):
    """Capture le log de reconstruction quand np.load lève ValueError."""

    # Prépare le sujet factice pour isoler les chemins temporaires
    subject = "S099"
    # Prépare le run factice pour déclencher le code de reconstruction
    run = "R07"
    # Construit le répertoire racine des données synthétiques
    data_dir = tmp_path / "data"
    # Construit le répertoire racine des données brutes attendu par la signature
    raw_dir = tmp_path / "raw"
    # Construit le dossier sujet utilisé pour logguer la corruption
    subject_dir = data_dir / subject
    # Crée l'arborescence pour écrire les fichiers corrompus
    subject_dir.mkdir(parents=True)
    # Injecte une charge illisible pour provoquer un ValueError de np.load
    (subject_dir / f"{run}_X.npy").write_text("broken payload")
    # Écrit un y minimal pour permettre l'analyse du couple X/y
    np.save(subject_dir / f"{run}_y.npy", np.array([0, 1]))
    # Prépare des features reconstruites pour remplacer l'entrée corrompue
    rebuilt_X = np.zeros((2, 2, 2))
    # Prépare des labels reconstruits pour aligner les échantillons
    rebuilt_y = np.array([1, 1])

    # Déclare un stub pour simuler la reconstruction EDF pendant le test
    def fake_build_npy(subject_arg, run_arg, data_arg, raw_arg):
        # Construit le chemin des features reconstruites pour remplacer le X corrompu
        features_path = data_arg / subject_arg / f"{run_arg}_X.npy"
        # Construit le chemin des labels reconstruits pour aligner X et y
        labels_path = data_arg / subject_arg / f"{run_arg}_y.npy"
        # Sauvegarde les features reconstruites pour la suite du test
        np.save(features_path, rebuilt_X)
        # Sauvegarde les labels reconstruits pour permettre le chargement final
        np.save(labels_path, rebuilt_y)
        # Retourne les chemins sauvegardés pour respecter l'interface attendue
        return features_path, labels_path

    # Injecte le stub pour suivre la reconstruction forcée
    monkeypatch.setattr(train, "_build_npy_from_edf", fake_build_npy)

    # Charge les données pour déclencher la reconstruction et le log associé
    X, y = train._load_data(subject, run, data_dir, raw_dir)
    # Capture les sorties pour inspecter le message de corruption
    captured = capsys.readouterr().out

    # Vérifie que les features proviennent bien du fichier reconstruit
    assert np.array_equal(X, rebuilt_X)
    # Vérifie que les labels proviennent bien du fichier reconstruit
    assert np.array_equal(y, rebuilt_y)
    # Vérifie que le log mentionne explicitement la reconstruction forcée
    assert "Chargement numpy impossible" in captured
    # Vérifie que le log mentionne la régénération depuis l'EDF
    assert "Régénération depuis l'EDF" in captured


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
    """Couvre le chemin nominal lorsque HEAD pointe vers une ref valide."""

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


# Garantit la couverture lorsque HEAD stocke directement un hash détaché
def test_get_git_commit_returns_detached_hash(tmp_path, monkeypatch):
    """Couvre le cas HEAD contenant directement un hash détaché."""

    # Prépare une arborescence .git artificielle pour simuler un HEAD isolé
    git_dir = tmp_path / "detached" / ".git"
    # Crée les dossiers .git pour écrire un fichier HEAD détaché
    git_dir.mkdir(parents=True)
    # Construit un hash hexadécimal réaliste pour le scénario de test
    detached_hash = "abcde" * 8
    # Écrit le hash dans HEAD pour activer la branche sans référence symbolique
    (git_dir / "HEAD").write_text(detached_hash)
    # Change le répertoire courant pour cibler le dépôt simulé
    monkeypatch.chdir(git_dir.parent)
    # Vérifie que _get_git_commit retourne exactement le hash détaché attendu
    assert _get_git_commit() == detached_hash
