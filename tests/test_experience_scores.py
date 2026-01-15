# Valide le mapping run -> expérience pour le scoring global
from scripts import aggregate_experience_scores


# Vérifie que les runs sont correctement associés aux types attendus
def test_map_run_to_experience_handles_known_and_baseline() -> None:
    # Vérifie la correspondance pour un run de type T1
    assert aggregate_experience_scores._map_run_to_experience("R03") == "T1"
    # Vérifie la correspondance pour un run de type T2
    assert aggregate_experience_scores._map_run_to_experience("R04") == "T2"
    # Vérifie la correspondance pour un run de type T3
    assert aggregate_experience_scores._map_run_to_experience("R05") == "T3"
    # Vérifie la correspondance pour un run de type T4
    assert aggregate_experience_scores._map_run_to_experience("R06") == "T4"
    # Vérifie que les baselines sont ignorées
    assert aggregate_experience_scores._map_run_to_experience("R01") is None


# Vérifie le calcul des moyennes par sujet et la moyenne globale
def test_aggregate_experience_scores_averages_by_subject(tmp_path, monkeypatch) -> None:
    # Définit un dictionnaire de scores simulés par run
    scores = {
        # Regroupe les scores simulés pour le sujet complet
        "S001": {
            # Simule le score du run R03
            "R03": 0.80,
            # Simule le score du run R04
            "R04": 0.70,
            # Simule le score du run R05
            "R05": 0.90,
            # Simule le score du run R06
            "R06": 0.60,
            # Simule le score du run R07
            "R07": 0.70,
            # Simule le score du run R08
            "R08": 0.80,
            # Simule le score du run R09
            "R09": 0.85,
            # Simule le score du run R10
            "R10": 0.65,
            # Simule le score du run R11
            "R11": 0.75,
            # Simule le score du run R12
            "R12": 0.72,
            # Simule le score du run R13
            "R13": 0.88,
            # Simule le score du run R14
            "R14": 0.68,
        },
        # Regroupe les scores simulés pour le sujet incomplet
        "S002": {
            # Simule le score du run R03
            "R03": 0.50,
            # Simule le score du run R04
            "R04": 0.55,
            # Simule le score du run R05
            "R05": 0.60,
        },
    }

    # Définit un faux evaluate_run pour éviter l'I/O de données EEG
    def fake_evaluate_run(subject: str, run: str, data_dir, artifacts_dir):
        # Retourne un dictionnaire aligné sur l'API attendue
        return {"accuracy": scores[subject][run]}

    # Remplace la fonction d'évaluation pour isoler le calcul
    monkeypatch.setattr(
        # Cible le module predict utilisé par l'agrégateur
        aggregate_experience_scores.predict_cli,
        # Spécifie le nom de la fonction à patcher
        "evaluate_run",
        # Injecte la fonction factice
        fake_evaluate_run,
    )
    # Crée un dossier d'artefacts minimal pour S001 complet
    for run in scores["S001"].keys():
        # Construit le chemin du modèle simulé
        model_path = tmp_path / "artifacts" / "S001" / run / "model.joblib"
        # Crée les dossiers parents requis
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Écrit un fichier factice pour activer le run
        model_path.write_text("stub")
        # Construit le chemin de la matrice W simulée
        w_matrix_path = tmp_path / "artifacts" / "S001" / run / "w_matrix.joblib"
        # Écrit un fichier factice pour activer la matrice W
        w_matrix_path.write_text("stub")
    # Crée un dossier d'artefacts incomplet pour S002
    for run in scores["S002"].keys():
        # Construit le chemin du modèle simulé
        model_path = tmp_path / "artifacts" / "S002" / run / "model.joblib"
        # Crée les dossiers parents requis
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Écrit un fichier factice pour activer le run
        model_path.write_text("stub")
        # Construit le chemin de la matrice W simulée
        w_matrix_path = tmp_path / "artifacts" / "S002" / run / "w_matrix.joblib"
        # Écrit un fichier factice pour activer la matrice W
        w_matrix_path.write_text("stub")
    # Calcule le rapport en utilisant le dossier temporaire
    report = aggregate_experience_scores.aggregate_experience_scores(
        # Passe un répertoire data factice
        tmp_path / "data",
        # Passe le répertoire d'artefacts préparé
        tmp_path / "artifacts",
        # Désactive l'auto-train pour stabiliser le test
        False,
    )
    # Récupère l'entrée pour S001
    subject_entry = next(
        # Filtre sur le sujet S001
        entry
        for entry in report["subjects"]
        if entry["subject"] == "S001"
    )
    # Vérifie que S001 est éligible avec quatre moyennes
    assert subject_entry["eligible"] is True
    # Vérifie que la moyenne des quatre types est calculée
    assert subject_entry["mean_of_means"] is not None
    # Récupère l'entrée pour S002
    incomplete_entry = next(
        # Filtre sur le sujet S002
        entry
        for entry in report["subjects"]
        if entry["subject"] == "S002"
    )
    # Vérifie que S002 est inéligible sans quatre types
    assert incomplete_entry["eligible"] is False
    # Vérifie que la moyenne globale ne considère qu'un sujet complet
    assert report["eligible_subjects"] == 1
    # Vérifie que le bonus est calculé pour la moyenne globale
    assert report["bonus_points"] >= 1


# Vérifie que les bonus sont nuls lorsque la moyenne est absente
def test_compute_bonus_points_returns_zero_on_missing_mean() -> None:
    # Calcule le bonus pour une moyenne absente
    bonus = aggregate_experience_scores.compute_bonus_points(None)
    # Vérifie que le bonus est nul sans moyenne
    assert bonus == 0


# Vérifie la borne du bonus lorsque la moyenne atteint le seuil
def test_compute_bonus_points_returns_zero_at_threshold() -> None:
    # Référence le seuil pour éviter une ligne trop longue
    threshold = aggregate_experience_scores.BONUS_THRESHOLD
    # Calcule le bonus pour la moyenne exactement au seuil
    bonus = aggregate_experience_scores.compute_bonus_points(threshold)
    # Vérifie que le bonus est nul à la limite
    assert bonus == 0


# Vérifie que _discover_runs renvoie une liste vide si aucun artefact
def test_discover_runs_returns_empty_when_dir_missing(tmp_path) -> None:
    # Construit un chemin d'artefacts inexistant
    missing_dir = tmp_path / "artifacts"
    # Construit un chemin data inexistant pour le fallback
    missing_data_dir = tmp_path / "data"
    # Exécute la découverte sur un dossier absent
    runs = aggregate_experience_scores._discover_runs(
        missing_data_dir,
        missing_dir,
        False,
    )
    # Vérifie que la liste est vide
    assert runs == []


# Vérifie le formatage du tableau texte avec une ligne globale
def test_format_experience_table_renders_global_and_na() -> None:
    # Définit les moyennes par expérience pour le sujet complet
    means_complete = {
        # Fournit la moyenne pour l'expérience T1
        "T1": 0.5,
        # Fournit la moyenne pour l'expérience T2
        "T2": 0.6,
        # Fournit la moyenne pour l'expérience T3
        "T3": 0.7,
        # Fournit la moyenne pour l'expérience T4
        "T4": 0.8,
    }
    # Construit l'entrée complète pour le sujet S001
    subject_complete = {
        # Renseigne l'identifiant du sujet complet
        "subject": "S001",
        # Fournit les moyennes par expérience pour S001
        "means": means_complete,
        # Indique que le sujet est éligible
        "eligible": True,
        # Stocke la moyenne des moyennes pour S001
        "mean_of_means": 0.65,
    }
    # Définit les moyennes absentes pour le sujet incomplet
    means_incomplete = {
        # Marque l'absence de moyenne pour T1
        "T1": None,
        # Marque l'absence de moyenne pour T2
        "T2": None,
        # Marque l'absence de moyenne pour T3
        "T3": None,
        # Marque l'absence de moyenne pour T4
        "T4": None,
    }
    # Construit l'entrée incomplète pour le sujet S002
    subject_incomplete = {
        # Renseigne l'identifiant du sujet incomplet
        "subject": "S002",
        # Fournit des moyennes absentes pour S002
        "means": means_incomplete,
        # Indique que le sujet est inéligible
        "eligible": False,
        # Marque l'absence de moyenne globale pour S002
        "mean_of_means": None,
    }
    # Construit un rapport minimal avec un sujet complet et un incomplet
    # Annote explicitement le type pour satisfaire mypy
    report: dict[str, object] = {
        # Fournit les lignes sujet pour le tableau
        "subjects": [subject_complete, subject_incomplete],
        # Définit la moyenne globale attendue
        "global_mean": 0.65,
        # Définit le nombre de sujets éligibles
        "eligible_subjects": 1,
        # Définit un bonus arbitraire
        "bonus_points": 2,
    }
    # Formate le tableau texte à partir du rapport
    table = aggregate_experience_scores.format_experience_table(report)
    # Vérifie que la ligne globale est incluse
    assert "Global" in table
    # Vérifie que les valeurs absentes affichent n/a
    assert "n/a" in table


# Vérifie la ligne globale même sans moyenne calculée
def test_format_experience_table_renders_global_without_scores() -> None:
    # Construit un rapport sans sujets ni moyenne globale
    # Annote explicitement le type pour satisfaire mypy
    report: dict[str, object] = {
        # Laisse la liste des sujets vide pour simuler l'absence d'artefacts
        "subjects": [],
        # Indique qu'aucune moyenne globale n'est calculée
        "global_mean": None,
        # Fixe le nombre de sujets éligibles à zéro
        "eligible_subjects": 0,
        # Fixe le bonus à zéro en l'absence de score
        "bonus_points": 0,
    }
    # Formate le tableau texte à partir du rapport vide
    table = aggregate_experience_scores.format_experience_table(report)
    # Vérifie que la ligne globale est incluse
    assert "Global" in table
    # Vérifie que la moyenne globale absente affiche n/a
    assert "\tn/a\t" in table


# Vérifie l'écriture CSV des moyennes par expérience
def test_write_csv_outputs_expected_rows(tmp_path) -> None:
    # Définit les moyennes par expérience pour le sujet complet
    means_entry = {
        # Fournit la moyenne pour l'expérience T1
        "T1": 0.1,
        # Fournit la moyenne pour l'expérience T2
        "T2": 0.2,
        # Fournit la moyenne pour l'expérience T3
        "T3": 0.3,
        # Fournit la moyenne pour l'expérience T4
        "T4": 0.4,
    }
    # Construit l'entrée complète pour le sujet S001
    subject_entry = {
        # Renseigne l'identifiant du sujet complet
        "subject": "S001",
        # Fournit les moyennes par expérience pour S001
        "means": means_entry,
        # Indique que le sujet est éligible
        "eligible": True,
        # Stocke la moyenne des moyennes pour S001
        "mean_of_means": 0.25,
    }
    # Construit un rapport minimal avec un sujet complet
    report = {
        # Fournit les lignes sujet pour le tableau
        "subjects": [subject_entry],
        # Définit la moyenne globale attendue
        "global_mean": 0.25,
        # Définit le nombre de sujets éligibles
        "eligible_subjects": 1,
        # Définit un bonus arbitraire
        "bonus_points": 0,
    }
    # Définit le chemin de sortie CSV
    csv_path = tmp_path / "report.csv"
    # Écrit le fichier CSV via le helper de prod
    aggregate_experience_scores.write_csv(report, csv_path)
    # Lit le contenu du fichier généré
    content = csv_path.read_text(encoding="utf-8")
    # Vérifie que l'en-tête contient les colonnes attendues
    assert "subject,T1_mean,T2_mean,T3_mean,T4_mean,mean_of_means,eligible" in content
    # Vérifie que la ligne sujet contient la moyenne formatée
    assert "S001,0.100000,0.200000,0.300000,0.400000,0.250000,True" in content


# Vérifie que les chemins d'artefacts sont construits correctement
def test_artifact_paths_returns_expected_paths(tmp_path) -> None:
    # Définit un sujet et un run de référence
    subject = "S001"
    # Définit un run de référence
    run = "R03"
    # Calcule les chemins d'artefacts via l'utilitaire
    model_path, w_matrix_path = aggregate_experience_scores._artifact_paths(
        tmp_path, subject, run
    )
    # Vérifie que le chemin du modèle respecte la convention attendue
    assert model_path == tmp_path / subject / run / "model.joblib"
    # Vérifie que le chemin de la matrice W respecte la convention attendue
    assert w_matrix_path == tmp_path / subject / run / "w_matrix.joblib"


# Vérifie que l'accuracy est extraite d'un rapport JSON
def test_load_cached_accuracy_returns_value(tmp_path) -> None:
    # Définit un sujet et un run pour le cache
    subject = "S001"
    # Définit un run pour le cache
    run = "R03"
    # Construit le dossier d'artefacts cible
    target_dir = tmp_path / subject / run
    # Crée l'arborescence des artefacts
    target_dir.mkdir(parents=True, exist_ok=True)
    # Prépare un rapport JSON minimal avec une accuracy
    (target_dir / "report.json").write_text('{"accuracy": 0.85}', encoding="utf-8")
    # Charge l'accuracy via la fonction utilitaire
    accuracy = aggregate_experience_scores._load_cached_accuracy(tmp_path, subject, run)
    # Vérifie que l'accuracy est bien lue et convertie
    assert accuracy == 0.85


# Vérifie que l'absence de rapport retourne None
def test_load_cached_accuracy_returns_none_when_missing(tmp_path) -> None:
    # Définit un sujet et un run pour le cache
    subject = "S001"
    # Définit un run pour le cache
    run = "R03"
    # Charge l'accuracy sans rapport présent
    accuracy = aggregate_experience_scores._load_cached_accuracy(tmp_path, subject, run)
    # Vérifie que l'absence de rapport retourne None
    assert accuracy is None


# Vérifie l'append d'une accuracy dans la structure agrégée
def test_append_subject_score_accumulates_values() -> None:
    # Initialise un conteneur vide
    subject_scores: dict[str, dict[str, list[float]]] = {}
    # Ajoute une première valeur pour un sujet et une expérience
    aggregate_experience_scores._append_subject_score(subject_scores, "S001", "T1", 0.7)
    # Ajoute une seconde valeur pour le même sujet et expérience
    aggregate_experience_scores._append_subject_score(subject_scores, "S001", "T1", 0.8)
    # Vérifie que les deux valeurs sont conservées
    assert subject_scores["S001"]["T1"] == [0.7, 0.8]
