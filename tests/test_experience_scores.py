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
    # Crée un dossier d'artefacts incomplet pour S002
    for run in scores["S002"].keys():
        # Construit le chemin du modèle simulé
        model_path = tmp_path / "artifacts" / "S002" / run / "model.joblib"
        # Crée les dossiers parents requis
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Écrit un fichier factice pour activer le run
        model_path.write_text("stub")
    # Calcule le rapport en utilisant le dossier temporaire
    report = aggregate_experience_scores.aggregate_experience_scores(
        # Passe un répertoire data factice
        tmp_path / "data",
        # Passe le répertoire d'artefacts préparé
        tmp_path / "artifacts",
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
