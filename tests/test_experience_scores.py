# Sérialise les manifests et rapports reproductibles simulés.
import json

# Centralise l'aide de cast pour les tests typés mypy
from typing import cast

# Centralise NumPy pour les moyennes attendues
import numpy as np
import pytest

# Expose le type TrainingRequest pour les casts mypy
# Valide le mapping run -> expérience pour le scoring global
from scripts import aggregate_experience_scores
from scripts import train as train_module


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
    def fake_evaluate_run(subject: str, run: str, data_dir, artifacts_dir, options):
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
        # Transmet des options explicites pour l'agrégation
        aggregate_experience_scores.AggregationOptions(
            allow_auto_train=False,
            force_retrain=False,
            enable_grid_search=False,
            grid_search_splits=None,
            raw_dir=tmp_path / "data",
        ),
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
    # Vérifie que le seuil 0.75 est correctement évalué
    assert subject_entry["meets_threshold"] is (subject_entry["mean_of_means"] >= 0.75)
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
    # Récupère les moyennes globales par expérience
    global_means = report["global_experience_means"]
    # Construit la liste des moyennes disponibles dans l'ordre
    global_values = [
        global_means[experience]
        for experience in aggregate_experience_scores.EXPERIENCE_ORDER
    ]
    # Vérifie que toutes les moyennes sont présentes pour le global
    assert all(value is not None for value in global_values)
    # Calcule la moyenne globale attendue selon la formule T1..T4
    expected_global_mean = float(np.mean(global_values))
    # Vérifie que la moyenne globale correspond à la formule demandée
    assert report["global_mean"] == expected_global_mean
    # Vérifie que la moyenne globale par expérience est calculée
    assert report["global_experience_mean"] is not None
    # Vérifie que le bonus est calculé pour la moyenne globale
    assert report["bonus_points"] >= 0
    # Vérifie que le rapport documente la méthode d'évaluation officielle
    assert report["evaluation_method"] == "cross_validation"
    # Vérifie que ce rapport partiel ne peut pas être présenté comme complet
    assert report["complete"] is False
    # Vérifie que le format de preuve est explicitement versionné.
    assert report["schema_version"] == 1
    # Vérifie que la preuve énumère l'intégralité des sujets officiels.
    assert report["expected_subject_ids"] == [
        f"S{subject_index:03d}" for subject_index in range(1, 110)
    ]
    # Vérifie que la configuration demandée accompagne toujours les résultats.
    assert report["requested_configuration"]["feature_strategy"] == "fft"


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


# Protège la limite de cinq points imposée par la grille d'évaluation.
def test_compute_bonus_points_is_clamped_to_checklist_maximum() -> None:
    """Empêche un score parfait de produire plus de cinq points."""

    # Utilise la meilleure accuracy possible pour couvrir la borne haute.
    bonus = aggregate_experience_scores.compute_bonus_points(1.0)
    # La checklist ne permet jamais de dépasser cinq points de score.
    assert bonus == 5


@pytest.mark.parametrize(
    ("score", "expected_points"),
    [
        (0.779999, 0),
        (0.78, 1),
        (0.81, 2),
        (0.84, 3),
        (0.87, 4),
        (0.90, 5),
    ],
)
def test_compute_bonus_points_requires_each_complete_three_percent_step(
    score: float,
    expected_points: int,
) -> None:
    """WBS 7.4.3 / TPV-045: un palier incomplet ne rapporte aucun point."""

    assert aggregate_experience_scores.compute_bonus_points(score) == expected_points


def test_pooled_experiment_cv_builds_reproducible_subject_report(
    tmp_path, monkeypatch
) -> None:
    """Le nouveau chemin doit conserver plis, configuration et complétude."""

    results = {
        experience: {
            "runs": [],
            "n_trials": 45,
            "class_counts": {"0": 21, "1": 24},
            "cv_scores": [0.8] * 5,
            "cv_mean": 0.8,
        }
        for experience in aggregate_experience_scores.EXPERIENCE_ORDER
    }

    def fake_evaluate(_raw_dir, subjects, _config, progress, cache_dir):
        scores = {
            subject: {experience: [0.8] for experience in results}
            for subject in subjects
        }
        evidence = [
            {"subject": subject, "experiences": results} for subject in subjects
        ]
        for subject in subjects:
            progress(subject, results)
        assert cache_dir == tmp_path / "artifacts/evaluation/fbcsp_subjects"
        return scores, evidence

    monkeypatch.setattr(
        aggregate_experience_scores.tpv_evaluation,
        "evaluate_all_subjects",
        fake_evaluate,
    )
    monkeypatch.setattr(
        aggregate_experience_scores.train_cli,
        "_get_git_commit",
        lambda: "abc123",
    )
    options = aggregate_experience_scores.AggregationOptions(
        allow_auto_train=False,
        force_retrain=False,
        enable_grid_search=False,
        grid_search_splits=None,
        raw_dir=tmp_path / "data",
        pooled_experiment_cv=True,
        subject_limit=2,
    )

    report = aggregate_experience_scores.aggregate_experience_scores(
        tmp_path / "data",
        tmp_path / "artifacts",
        options,
    )

    assert report["eligible_subjects"] == 2
    assert report["complete"] is False
    assert report["schema_version"] == 3
    assert report["source_commits"] == ["abc123"]
    assert report["requested_configuration"]["pooled_experiment_cv"] is True
    assert report["requested_configuration"]["nested_cv"] is True
    assert report["requested_configuration"]["campaign_size"] == 2
    assert report["score_targets"] == [0.75, 0.81, 0.87, 0.9]
    assert report["evaluation_method"].startswith("nested_stratified_cv")
    assert len(report["evaluated_subjects"]) == 2


# Protège une évaluation complète lorsque seuls certains modèles existent déjà.
def test_discover_runs_merges_artifacts_with_available_data(tmp_path) -> None:
    """Complète les artefacts partiels par les runs présents dans data/."""

    # Prépare les deux racines inspectées par le moteur d'agrégation.
    data_dir = tmp_path / "data"
    # Sépare les modèles des données pour reproduire l'arborescence réelle.
    artifacts_dir = tmp_path / "artifacts"
    # Matérialise un premier run déjà entraîné.
    artifact_run_dir = artifacts_dir / "S001" / "R03"
    # Crée les parents nécessaires à l'artefact factice.
    artifact_run_dir.mkdir(parents=True)
    # Le modèle suffit à rendre ce run découvrable côté artefacts.
    (artifact_run_dir / "model.joblib").write_bytes(b"model")
    # Matérialise un second run disponible uniquement dans le dataset.
    data_subject_dir = data_dir / "S001"
    # Crée le dossier du sujet avant les tableaux numpy.
    data_subject_dir.mkdir(parents=True)
    # Une matrice et ses labels constituent un run disponible.
    np.save(data_subject_dir / "R04_X.npy", np.ones((2, 1, 2)))
    # Les labels alignés terminent le couple de fichiers attendu.
    np.save(data_subject_dir / "R04_y.npy", np.array([0, 1]))

    # Active le scan des données pour compléter les artefacts existants.
    runs = aggregate_experience_scores._discover_runs(
        data_dir,
        artifacts_dir,
        True,
    )

    # Les deux provenances doivent être fusionnées sans doublon.
    assert runs == [("S001", "R03"), ("S001", "R04")]


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


# Vérifie que _train_run construit une requête alignée sur les options
def test_train_run_builds_training_request(tmp_path, monkeypatch) -> None:
    # Prépare une liste pour capturer la requête d'entraînement
    captured: list[object] = []

    # Définit un resolve_sampling_rate déterministe pour le test
    def fake_resolve_sampling_rate(
        subject, run, raw_dir, requested_sfreq, eeg_reference
    ):
        # Retourne une fréquence factice pour vérifier l'appel
        return 128.0

    # Définit un run_training factice pour capturer la requête
    def fake_run_training(request):
        # Stocke la requête pour inspection
        captured.append(request)
        # Retourne un rapport minimal pour satisfaire l'appelant
        return {"cv_scores": []}

    # Remplace la résolution de fréquence pour isoler le test
    monkeypatch.setattr(
        aggregate_experience_scores.train_cli,
        "resolve_sampling_rate",
        fake_resolve_sampling_rate,
    )
    # Remplace l'entraînement pour éviter l'I/O
    monkeypatch.setattr(
        aggregate_experience_scores.train_cli,
        "run_training",
        fake_run_training,
    )
    # Prépare les options d'entraînement pour la simulation
    options = aggregate_experience_scores.AggregationOptions(
        allow_auto_train=True,
        force_retrain=False,
        enable_grid_search=True,
        grid_search_splits=3,
        raw_dir=tmp_path / "raw",
    )
    # Prépare le contexte d'entraînement complet
    context = aggregate_experience_scores.TrainingContext(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        options=options,
    )
    # Lance l'entraînement factice pour couvrir _train_run
    aggregate_experience_scores._train_run("S001", "R03", context)
    # Vérifie qu'une requête a été capturée
    assert len(captured) == 1
    # Récupère la requête capturée pour valider les champs
    request = cast(train_module.TrainingRequest, captured[0])
    # Vérifie le sujet de la requête
    assert request.subject == "S001"
    # Vérifie le run de la requête
    assert request.run == "R03"
    # Vérifie le drapeau grid search dans la requête
    assert request.enable_grid_search is True
    # Vérifie le nombre de splits configuré
    assert request.grid_search_splits == 3
    # Vérifie le chemin des données dans la requête
    assert request.data_dir == tmp_path / "data"
    # Vérifie le chemin des artefacts dans la requête
    assert request.artifacts_dir == tmp_path / "artifacts"
    # Vérifie le chemin raw transmis
    assert request.raw_dir == tmp_path / "raw"


# Vérifie que _collect_subject_scores déclenche le réentraînement forcé
def test_collect_subject_scores_force_retrain_triggers_train(
    tmp_path, monkeypatch
) -> None:
    # Prépare une liste pour suivre les appels d'entraînement
    called: list[tuple[str, str]] = []

    # Définit un faux _train_run pour marquer l'appel
    def fake_train_run(subject, run, context):
        # Enregistre le couple sujet/run pour validation
        called.append((subject, run))

    # Définit un evaluate_run factice pour retourner un score fixe
    def fake_evaluate_run(subject, run, data_dir, artifacts_dir, options):
        # Retourne une accuracy stable pour l'agrégation
        return {"accuracy": 0.8}

    # Remplace _train_run pour éviter l'entraînement réel
    monkeypatch.setattr(aggregate_experience_scores, "_train_run", fake_train_run)
    # Remplace evaluate_run pour éviter l'I/O
    monkeypatch.setattr(
        aggregate_experience_scores.predict_cli,
        "evaluate_run",
        fake_evaluate_run,
    )
    # Prépare des options qui forcent le réentraînement
    options = aggregate_experience_scores.AggregationOptions(
        allow_auto_train=True,
        force_retrain=True,
        enable_grid_search=False,
        grid_search_splits=None,
        raw_dir=tmp_path / "raw",
    )
    # Exécute la collecte sur un run simulé
    result = aggregate_experience_scores._collect_subject_scores(
        runs=[("S001", "R03")],
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        options=options,
    )
    # Vérifie que l'entraînement forcé a été déclenché
    assert called == [("S001", "R03")]
    # Vérifie que l'accuracy est bien enregistrée
    assert result["S001"]["T1"] == [0.8]


def test_collect_subject_scores_retrains_manifest_without_cv_score(
    tmp_path, monkeypatch
) -> None:
    """Un modèle final sans preuve CV ne doit jamais être réutilisé tel quel."""

    subject, run = "S106", "R05"
    target_dir = tmp_path / "artifacts" / subject / run
    target_dir.mkdir(parents=True)
    (target_dir / "model.joblib").write_text("stub", encoding="utf-8")
    (target_dir / "w_matrix.joblib").write_text("stub", encoding="utf-8")
    (target_dir / "manifest.json").write_text(
        '{"scores": {"cv_scores": [], "cv_mean": null}}', encoding="utf-8"
    )
    trained: list[tuple[str, str]] = []
    monkeypatch.setattr(
        aggregate_experience_scores,
        "_train_run",
        lambda current_subject, current_run, _context: trained.append(
            (current_subject, current_run)
        ),
    )
    monkeypatch.setattr(
        aggregate_experience_scores.predict_cli,
        "evaluate_run",
        lambda *_args, **_kwargs: {"accuracy": 0.5},
    )
    options = aggregate_experience_scores.AggregationOptions(
        allow_auto_train=True,
        force_retrain=False,
        enable_grid_search=False,
        grid_search_splits=None,
        raw_dir=tmp_path / "data",
    )

    scores = aggregate_experience_scores._collect_subject_scores(
        [(subject, run)], tmp_path / "data", tmp_path / "artifacts", options
    )

    assert trained == [(subject, run)]
    assert scores[subject]["T3"] == [0.5]


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
        # Indique que le seuil est atteint pour S001
        "meets_threshold": False,
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
        # Indique que le seuil est considéré non atteint
        "meets_threshold": False,
        # Marque l'absence de moyenne globale pour S002
        "mean_of_means": None,
    }
    # Construit un rapport minimal avec un sujet complet et un incomplet
    # Annote explicitement le type pour satisfaire mypy
    report: dict[str, object] = {
        # Fournit les lignes sujet pour le tableau
        "subjects": [subject_complete, subject_incomplete],
        # Définit les moyennes globales par expérience attendues
        "global_experience_means": means_complete,
        # Définit la moyenne des quatre moyennes attendue
        "global_experience_mean": 0.65,
        # Définit la moyenne globale attendue pour les sujets complets
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
        # Indique qu'aucune moyenne globale par expérience n'est calculée
        "global_experience_means": {},
        # Indique qu'aucune moyenne des quatre moyennes n'est calculée
        "global_experience_mean": None,
        # Indique qu'aucune moyenne globale des sujets complets n'est calculée
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
        # Indique que le sujet atteint le seuil 0.75
        "meets_threshold": False,
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
    assert (
        "subject,T1_mean,T2_mean,T3_mean,T4_mean,mean_of_means,eligible,"
        "meets_threshold_0p75"
    ) in content
    # Vérifie que la ligne sujet contient la moyenne formatée
    assert "S001,0.100000,0.200000,0.300000,0.400000,0.250000,True,False" in content


# Vérifie que main retourne un code d'erreur quand le seuil global échoue
def test_main_exits_nonzero_when_global_mean_below_threshold(
    monkeypatch, capsys
) -> None:
    # Prépare un rapport minimal avec un score global inférieur au seuil
    report = {
        # Fournit une liste de sujets vide pour simplifier le test
        "subjects": [],
        # Fournit des moyennes globales par expérience
        "global_experience_means": {
            "T1": 0.7,
            "T2": 0.7,
            "T3": 0.7,
            "T4": 0.7,
        },
        # Fournit la moyenne globale calculée
        "global_mean": 0.7,
        # Fournit le nombre de sujets éligibles
        "eligible_subjects": 0,
        # Fournit un bonus nul pour ce test
        "bonus_points": 0,
        # Fournit une liste vide de pires sujets
        "worst_subjects": [],
    }
    # Remplace l'agrégateur pour isoler le test du I/O
    monkeypatch.setattr(
        aggregate_experience_scores,
        "aggregate_experience_scores",
        lambda *_args, **_kwargs: report,
    )
    # Exécute la commande principale sans arguments
    exit_code = aggregate_experience_scores.main([])
    # Capture la sortie pour vérifier le message d'erreur
    captured = capsys.readouterr()
    # Vérifie que le code de sortie signale l'échec
    assert exit_code == 1
    # Vérifie que le message d'erreur mentionne GlobalMean
    assert "GlobalMean" in captured.out


# Refuse qu'un sous-ensemble de sujets soit présenté comme une preuve finale.
def test_main_exits_nonzero_when_report_is_incomplete(monkeypatch, capsys) -> None:
    """Exige les 109 sujets en l'absence de l'option exploratoire."""

    # Prépare un rapport au-dessus du seuil mais fondé sur un seul sujet.
    report = {
        "subjects": [],
        "global_experience_means": {
            "T1": 0.8,
            "T2": 0.8,
            "T3": 0.8,
            "T4": 0.8,
        },
        "global_mean": 0.8,
        "eligible_subjects": 1,
        "bonus_points": 2,
        "worst_subjects": [],
    }
    # Isole le contrôle de complétude du chargement des artefacts.
    monkeypatch.setattr(
        aggregate_experience_scores,
        "aggregate_experience_scores",
        lambda *_args, **_kwargs: report,
    )

    # Lance le mode officiel sans autoriser un rapport partiel.
    exit_code = aggregate_experience_scores.main([])

    # Une preuve incomplète doit faire échouer la commande globale.
    assert exit_code == 1
    # Le diagnostic doit rendre visible le nombre de sujets manquants.
    assert "1/109" in capsys.readouterr().out


# Protège le chemin de succès, l'affichage des pires sujets et l'export CSV.
def test_main_reports_worst_subjects_and_writes_requested_csv(
    tmp_path, monkeypatch, capsys
) -> None:
    """Couvre les sorties utiles à la défense lorsque le seuil est atteint."""

    # Prépare un rapport complet dont un sujet possède une moyenne exploitable.
    report = {
        "subjects": [],
        "global_experience_means": {
            "T1": 0.8,
            "T2": 0.8,
            "T3": 0.8,
            "T4": 0.8,
        },
        "global_mean": 0.8,
        "eligible_subjects": 1,
        "bonus_points": 2,
        "worst_subjects": [
            {"subject": "S001", "mean_of_means": None},
            {"subject": "S002", "mean_of_means": 0.76},
        ],
    }
    # Évite toute découverte de dataset pendant ce test de rendu.
    monkeypatch.setattr(
        aggregate_experience_scores,
        "aggregate_experience_scores",
        lambda *_args, **_kwargs: report,
    )
    # Capture l'appel d'export sans dupliquer ses tests de format détaillés.
    written_paths = []
    # Remplace l'écriture réelle par l'enregistrement du chemin demandé.
    monkeypatch.setattr(
        aggregate_experience_scores,
        "write_csv",
        lambda _report, path: written_paths.append(path),
    )
    # Choisit un chemin temporaire représentatif de la commande documentée.
    csv_path = tmp_path / "scores.csv"

    # Exécute le moteur canonique avec l'export explicitement demandé.
    exit_code = aggregate_experience_scores.main(
        ["--csv-output", str(csv_path), "--allow-partial"]
    )

    # Un score supérieur au seuil doit produire un succès CLI.
    assert exit_code == 0
    # Le moteur doit transmettre exactement le chemin choisi à l'exporteur.
    assert written_paths == [csv_path]
    # Seul le sujet disposant d'une moyenne doit apparaître dans le classement.
    assert "S002: 0.760" in capsys.readouterr().out


# Vérifie le formatage des helpers de rendu des lignes
def test_format_helpers_render_subject_and_global_rows() -> None:
    # Prépare une entrée sujet complète
    subject_entry = {
        "subject": "S010",
        "means": {"T1": 0.8, "T2": 0.7, "T3": 0.9, "T4": 0.6},
        "eligible": True,
        "meets_threshold": True,
        "mean_of_means": 0.75,
    }
    # Prépare un rapport global minimal
    report = {
        "subjects": [subject_entry],
        "global_experience_means": {
            "T1": 0.8,
            "T2": 0.7,
            "T3": 0.9,
            "T4": 0.6,
        },
        "global_mean": 0.75,
        "eligible_subjects": 1,
        "bonus_points": 0,
    }
    # Construit la ligne sujet via le helper dédié
    subject_row = aggregate_experience_scores._format_subject_row(subject_entry)
    # Vérifie que la ligne sujet contient le seuil et l'éligibilité
    assert "yes" in subject_row
    # Construit la ligne globale via le helper dédié
    global_row = aggregate_experience_scores._format_global_row(report)
    # Vérifie que la ligne globale commence par Global
    assert global_row.startswith("Global")


# Vérifie les branches du libellé de seuil global
def test_format_global_threshold_label_handles_none_and_low_value() -> None:
    # Vérifie le libellé pour une moyenne absente
    assert aggregate_experience_scores._format_global_threshold_label(None) == "n/a"
    # Vérifie le libellé pour une moyenne sous le seuil
    assert aggregate_experience_scores._format_global_threshold_label(0.5) == "no"


# Vérifie le formatage d'une moyenne absente
def test_format_mean_value_handles_none() -> None:
    # Vérifie que la valeur absente est rendue en n/a
    assert aggregate_experience_scores._format_mean_value(None) == "n/a"


# Vérifie le formatage d'une moyenne numérique
def test_format_mean_value_handles_float() -> None:
    # Vérifie que la valeur numérique est formatée
    assert aggregate_experience_scores._format_mean_value(0.8) == "0.800"


# Vérifie le libellé de seuil global pour une moyenne suffisante
def test_format_global_threshold_label_handles_high_value() -> None:
    # Vérifie que le seuil est marqué comme atteint
    assert aggregate_experience_scores._format_global_threshold_label(0.8) == "yes"


# Vérifie la ligne globale quand la moyenne globale est absente
def test_format_global_row_handles_missing_global_mean() -> None:
    # Prépare un rapport global sans moyenne calculée
    report = {
        "global_experience_means": {
            "T1": None,
            "T2": None,
            "T3": None,
            "T4": None,
        },
        "global_mean": None,
        "eligible_subjects": 0,
        "bonus_points": 0,
    }
    # Construit la ligne globale via le helper dédié
    global_row = aggregate_experience_scores._format_global_row(report)
    # Vérifie que la ligne contient n/a pour la moyenne globale
    assert "n/a" in global_row


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


# Vérifie que l'accuracy est extraite du manifeste de validation
def test_load_cached_accuracy_returns_value(tmp_path) -> None:
    # Définit un sujet et un run pour le cache
    subject = "S001"
    # Définit un run pour le cache
    run = "R03"
    # Construit le dossier d'artefacts cible
    target_dir = tmp_path / subject / run
    # Crée l'arborescence des artefacts
    target_dir.mkdir(parents=True, exist_ok=True)
    # Prépare un manifeste contenant les folds et leur moyenne officielle.
    (target_dir / "manifest.json").write_text(
        '{"scores": {"cv_scores": [0.8, 0.9], "cv_mean": 0.85}}',
        encoding="utf-8",
    )
    # Charge l'accuracy via la fonction utilitaire
    accuracy = aggregate_experience_scores._load_cached_accuracy(tmp_path, subject, run)
    # Vérifie que l'accuracy est bien lue et convertie
    assert accuracy == 0.85


# Empêche la réutilisation des anciens scores calculés sur le fit final.
def test_load_cached_accuracy_ignores_legacy_report(tmp_path) -> None:
    """Un report.json isolé ne doit plus alimenter l'agrégation officielle."""

    # Prépare un ancien rapport optimiste sans manifeste de validation.
    target_dir = tmp_path / "S001" / "R03"
    # Crée l'arborescence représentative d'un artefact historique.
    target_dir.mkdir(parents=True)
    # Simule le score train-on-train que le refactoring doit neutraliser.
    (target_dir / "report.json").write_text('{"accuracy": 1.0}', encoding="utf-8")

    # Tente de charger le cache via le moteur officiel.
    accuracy = aggregate_experience_scores._load_cached_accuracy(
        tmp_path,
        "S001",
        "R03",
    )

    # Sans preuve CV, aucune accuracy ne peut être réutilisée.
    assert accuracy is None


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


# Vérifie que le rapport global conserve la preuve de chaque fold.
def test_build_run_evidence_and_write_json(tmp_path) -> None:
    """Rassemble commit, configuration et scores dans un JSON relisible."""

    # Prépare le manifeste d'un run moteur évalué.
    manifest_dir = tmp_path / "artifacts" / "S001" / "R03"
    # Crée le dossier selon la convention publique des artefacts.
    manifest_dir.mkdir(parents=True)
    # Stocke une preuve minimale représentative du vrai train.py.
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": {"epoch_window": [1.0, 3.0]},
                "hyperparams": {"dim_method": "csp"},
                "scores": {"cv_scores": [0.7, 0.9], "cv_mean": 0.8},
                "git_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )

    # Construit la preuve en incluant aussi un run baseline à ignorer.
    evidence = aggregate_experience_scores._build_run_evidence(
        [("S001", "R03"), ("S001", "R01")],
        tmp_path / "artifacts",
    )

    # Seul le run appartenant à T1 doit être documenté.
    assert len(evidence) == 1
    # Les folds et le commit doivent rester intacts pour la reproductibilité.
    assert evidence[0]["cv_scores"] == [0.7, 0.9]
    # Le mapping officiel doit accompagner le score du run.
    assert evidence[0]["experience"] == "T1"
    # Le commit source permet d'identifier le code ayant produit le modèle.
    assert evidence[0]["git_commit"] == "abc123"

    # Prépare une preuve globale compacte pour tester sa sérialisation.
    report = {"schema_version": 1, "evaluated_runs": evidence}
    # Choisit un sous-dossier absent comme le ferait la commande documentée.
    json_path = tmp_path / "evaluation" / "report.json"
    # Écrit la preuve via l'unique helper public de reporting JSON.
    aggregate_experience_scores.write_json(report, json_path)

    # Recharge le JSON pour vérifier sa validité et sa précision.
    persisted_report = json.loads(json_path.read_text(encoding="utf-8"))
    # La moyenne CV doit survivre exactement à l'aller-retour JSON.
    assert persisted_report["evaluated_runs"][0]["cv_mean"] == 0.8


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
