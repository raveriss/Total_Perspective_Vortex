"""Tests pour scripts.experiment_type_scores."""

# Préserve pathlib pour typer les chemins temporaires
from pathlib import Path

# Préserve pytest pour les assertions explicites
import pytest

# Importe les utilitaires de scoring par type d'expérience
from scripts import experiment_type_scores


# Vérifie le calcul des moyennes par type et du seuil global
def test_compute_experiment_type_report_meets_target() -> None:
    """La moyenne globale doit être >= 0.75 quand les scores le permettent."""

    # Construit la cartographie par défaut des types d'expériences
    experiment_types = experiment_type_scores.build_experiment_types()
    # Définit deux sujets pour réduire le périmètre de test
    subjects = ["S001", "S002"]
    # Prépare un mapping de scores déterministes par run
    score_map: dict[tuple[str, str], float] = {}
    # Définit les runs associés à chaque type pour construire le mapping
    for runs in experiment_types.values():
        # Parcourt chaque run associé
        for run in runs:
            # Attribue un score stable pour le sujet S001
            score_map[("S001", run)] = 0.8
            # Attribue un score stable pour le sujet S002
            score_map[("S002", run)] = 0.7

    # Définit une fonction de lookup qui lit dans le mapping
    def score_lookup(subject: str, run: str) -> float:
        """Retourne l'accuracy associée au couple sujet/run."""

        # Retourne la valeur précalculée pour ce couple
        return score_map[(subject, run)]

    # Calcule le rapport avec la fonction de lookup déterministe
    report = experiment_type_scores.compute_experiment_type_report(
        # Fournit la liste des sujets à agréger
        subjects=subjects,
        # Fournit la cartographie des types d'expériences
        experiment_types=experiment_types,
        # Fournit la fonction de lookup déterministe
        score_lookup=score_lookup,
        # Ferme l'appel pour construire le rapport
    )
    # Vérifie que chaque type obtient bien la moyenne attendue
    for accuracy in report["by_type"].values():
        # Vérifie la moyenne exacte à 0.75
        assert accuracy == pytest.approx(0.75)
    # Vérifie la moyenne globale attendue
    assert report["overall_mean"] == pytest.approx(0.75)
    # Vérifie que le seuil cible est atteint
    assert report["meets_target"] is True
    # Vérifie que le bonus est nul au seuil
    assert report["bonus_points"] == 0


# Vérifie le parsing d'une liste explicite de sujets
def test_parse_subjects_accepts_csv_list() -> None:
    """Le parseur doit accepter des sujets au format CSV mixte."""

    # Parse une liste contenant un identifiant préfixé et un index nu
    subjects = experiment_type_scores.parse_subjects("S001,2", 1, 109)
    # Vérifie que les identifiants sont formatés correctement
    assert subjects == ["S001", "S002"]


# Vérifie que parse_subjects rejette un identifiant invalide
def test_parse_subjects_rejects_invalid_subject() -> None:
    """Le parseur doit lever une erreur sur un sujet mal formé."""

    # Vérifie que la valeur invalide déclenche ValueError
    with pytest.raises(ValueError, match="Sujet invalide"):
        # Appelle le parseur avec un token non numérique
        experiment_type_scores.parse_subjects("S001,ZZ", 1, 109)


# Vérifie que build_pipeline_config propage les options de base
def test_build_pipeline_config_maps_args() -> None:
    """Les options CLI doivent être correctement traduites en config."""

    # Définit un conteneur d'arguments minimal pour la config
    class _Args:
        # Définit la fréquence d'échantillonnage choisie
        sfreq = 42.0
        # Définit la stratégie de features utilisée
        feature_strategy = "fft"
        # Définit la méthode de réduction dimensionnelle
        dim_method = "pca"
        # Définit le classifieur final
        classifier = "lda"
        # Définit le scaler optionnel demandé
        scaler = "none"
        # Définit le flag de normalisation désactivée
        no_normalize_features = True

    # Construit la config depuis les arguments factices
    config = experiment_type_scores.build_pipeline_config(_Args)
    # Vérifie la fréquence utilisée
    assert config.sfreq == 42.0
    # Vérifie la stratégie de features
    assert config.feature_strategy == "fft"
    # Vérifie la méthode de réduction
    assert config.dim_method == "pca"
    # Vérifie le classifieur final
    assert config.classifier == "lda"
    # Vérifie que le scaler "none" est converti en None
    assert config.scaler is None
    # Vérifie que la normalisation est désactivée
    assert config.normalize_features is False


# Vérifie que write_csv et write_json sérialisent le rapport
def test_write_reports(tmp_path: Path) -> None:
    """Les rapports doivent être écrits en CSV et JSON."""

    # Définit un rapport minimal conforme au TypedDict
    report = {
        # Fournit les moyennes par sujet
        "subjects": {"S001": {"imagery_left_right": 0.8}},
        # Fournit les moyennes par type
        "by_type": {"imagery_left_right": 0.8},
        # Fournit la moyenne globale
        "overall_mean": 0.8,
        # Indique que le seuil est atteint
        "meets_target": True,
        # Ferme le dictionnaire du rapport minimal
    }
    # Construit un sous-dossier temporaire pour le CSV
    csv_path = tmp_path / "reports" / "report.csv"
    # Construit un sous-dossier temporaire pour le JSON
    json_path = tmp_path / "reports_json" / "report.json"
    # Écrit le CSV via la fonction testée
    experiment_type_scores.write_csv(report, csv_path)
    # Écrit le JSON via la fonction testée
    experiment_type_scores.write_json(report, json_path)
    # Vérifie que le CSV existe
    assert csv_path.exists()
    # Vérifie que le JSON existe
    assert json_path.exists()


# Vérifie que train_and_score_run délègue à run_training/evaluate_run
def test_train_and_score_run_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """La fonction doit entraîner puis scorer le run demandé."""

    # Définit une configuration d'entraînement minimale
    config = experiment_type_scores.ExperimentTrainingConfig(
        # Fournit la configuration de pipeline
        pipeline_config=experiment_type_scores.PipelineConfig(
            # Fixe la fréquence d'échantillonnage de référence
            sfreq=50.0,
            # Fixe la stratégie de features utilisée
            feature_strategy="fft",
            # Active la normalisation des features
            normalize_features=True,
            # Fixe la méthode de réduction dimensionnelle
            dim_method="pca",
            # Laisse le nombre de composantes par défaut
            n_components=None,
            # Fixe le classifieur par défaut
            classifier="lda",
            # Désactive le scaler optionnel
            scaler=None,
            # Ferme la configuration de pipeline
        ),
        # Fournit le répertoire de données par défaut
        data_dir=experiment_type_scores.DEFAULT_DATA_DIR,
        # Fournit le répertoire d'artefacts par défaut
        artifacts_dir=experiment_type_scores.DEFAULT_ARTIFACTS_DIR,
        # Fournit le répertoire EDF par défaut
        raw_dir=experiment_type_scores.DEFAULT_RAW_DIR,
        # Ferme la configuration d'entraînement minimale
    )
    # Prépare un conteneur de suivi des appels
    seen: dict[str, object] = {}

    # Définit un stub run_training pour capturer la requête
    def _fake_run_training(request: object) -> None:
        """Capture la requête d'entraînement transmise."""

        # Stocke la requête pour assertion
        seen["request"] = request

    # Définit un stub evaluate_run qui retourne une accuracy fixe
    def _fake_evaluate_run(*_: object, **__: object) -> dict:
        """Retourne une accuracy factice."""

        # Retourne une structure minimale attendue
        return {"accuracy": 0.42}

    # Remplace run_training par le stub
    monkeypatch.setattr(experiment_type_scores, "run_training", _fake_run_training)
    # Remplace evaluate_run par le stub
    monkeypatch.setattr(experiment_type_scores, "evaluate_run", _fake_evaluate_run)
    # Exécute la fonction de scoring
    accuracy = experiment_type_scores.train_and_score_run("S001", "R03", config)
    # Vérifie que l'accuracy propagée est correcte
    assert accuracy == pytest.approx(0.42)
    # Vérifie que la requête d'entraînement a bien été créée
    assert "request" in seen


# Vérifie que main renvoie 0 lorsque le seuil est atteint
def test_main_returns_success_on_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le main doit renvoyer 0 si la moyenne atteint 0.75."""

    # Force la liste de sujets pour limiter le périmètre
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la fonction de parsing
        "parse_subjects",
        # Retourne une liste de sujets déterministe
        lambda *_: ["S001"],
        # Ferme le patch pour parse_subjects
    )
    # Force les types d'expériences à un set minimal
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la construction des types d'expériences
        "build_experiment_types",
        # Retourne un mapping minimal
        lambda: {"imagery_left_right": ("R03",)},
        # Ferme le patch pour build_experiment_types
    )
    # Force train_and_score_run pour retourner une accuracy cible
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la fonction de scoring
        "train_and_score_run",
        # Retourne une accuracy cible
        lambda *_: 0.8,
        # Ferme le patch pour train_and_score_run
    )
    # Exécute main avec des args neutres
    result = experiment_type_scores.main(["--subjects", "S001"])
    # Vérifie que le code de sortie est 0
    assert result == 0


# Vérifie que safe_mean retourne 0.0 sur séquence vide
def test_safe_mean_returns_zero_on_empty() -> None:
    """Le helper doit sécuriser les séquences vides."""

    # Calcule la moyenne sur une séquence vide
    result = experiment_type_scores.safe_mean([])
    # Vérifie que la moyenne vaut 0.0
    assert result == 0.0


# Vérifie que main écrit les rapports et utilise le cache
def test_main_writes_reports_and_uses_cache(
    # Reçoit le monkeypatch pytest pour les stubs
    monkeypatch: pytest.MonkeyPatch,
    # Reçoit le chemin temporaire pour les fichiers
    tmp_path: Path,
) -> None:
    """Le main doit écrire les sorties et éviter les appels doublons."""

    # Prépare un compteur d'appels pour train_and_score_run
    call_count = {"count": 0}

    # Définit un stub train_and_score_run avec compteur
    def _fake_train_and_score_run(*_: object, **__: object) -> float:
        """Retourne une accuracy fixe en incrémentant le compteur."""

        # Incrémente le compteur d'appels
        call_count["count"] += 1
        # Retourne une accuracy stable
        return 0.8

    # Force une liste de sujets unique
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la fonction de parsing
        "parse_subjects",
        # Retourne un seul sujet pour limiter le périmètre
        lambda *_: ["S001"],
        # Ferme le patch pour parse_subjects
    )
    # Force des types avec un run partagé pour tester le cache
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la construction des types d'expériences
        "build_experiment_types",
        # Retourne un mapping avec un run dupliqué
        lambda: {"type_a": ("R03",), "type_b": ("R03",)},
        # Ferme le patch pour build_experiment_types
    )
    # Remplace train_and_score_run par le stub compteur
    monkeypatch.setattr(
        # Cible le module à patcher
        experiment_type_scores,
        # Cible la fonction de scoring
        "train_and_score_run",
        # Retourne l'accuracy stable et incrémente
        _fake_train_and_score_run,
        # Ferme le patch pour train_and_score_run
    )
    # Définit le chemin CSV attendu
    csv_path = tmp_path / "report.csv"
    # Définit le chemin JSON attendu
    json_path = tmp_path / "report.json"
    # Définit la liste d'arguments pour main
    args = [
        # Ajoute le flag subjects
        "--subjects",
        # Ajoute l'identifiant du sujet
        "S001",
        # Ajoute l'option CSV
        "--csv-output",
        # Ajoute le chemin CSV
        str(csv_path),
        # Ajoute l'option JSON
        "--json-output",
        # Ajoute le chemin JSON
        str(json_path),
        # Ferme la liste d'arguments
    ]
    # Exécute main avec les sorties demandées
    result = experiment_type_scores.main(args)
    # Vérifie que main indique un succès
    assert result == 0
    # Vérifie que le cache a évité un double appel
    assert call_count["count"] == 1
    # Vérifie que le CSV a été écrit
    assert csv_path.exists()
    # Vérifie que le JSON a été écrit
    assert json_path.exists()


# Vérifie le calcul des points bonus
def test_compute_bonus_points_counts_steps() -> None:
    """Le bonus doit compter chaque tranche complète de 3 %."""

    # Calcule le bonus juste sous le seuil
    assert experiment_type_scores.compute_bonus_points(0.74) == 0
    # Calcule le bonus à un incrément complet
    assert experiment_type_scores.compute_bonus_points(0.78) == 1
    # Calcule le bonus à deux incréments complets
    assert experiment_type_scores.compute_bonus_points(0.81) == 2
