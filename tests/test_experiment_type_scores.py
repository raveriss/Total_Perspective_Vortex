"""Tests pour scripts.experiment_type_scores."""

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
        subjects=subjects,
        experiment_types=experiment_types,
        score_lookup=score_lookup,
    )
    # Vérifie que chaque type obtient bien la moyenne attendue
    for accuracy in report["by_type"].values():
        # Vérifie la moyenne exacte à 0.75
        assert accuracy == pytest.approx(0.75)
    # Vérifie la moyenne globale attendue
    assert report["overall_mean"] == pytest.approx(0.75)
    # Vérifie que le seuil cible est atteint
    assert report["meets_target"] is True


# Vérifie le parsing d'une liste explicite de sujets
def test_parse_subjects_accepts_csv_list() -> None:
    """Le parseur doit accepter des sujets au format CSV mixte."""

    # Parse une liste contenant un identifiant préfixé et un index nu
    subjects = experiment_type_scores.parse_subjects("S001,2", 1, 109)
    # Vérifie que les identifiants sont formatés correctement
    assert subjects == ["S001", "S002"]
