import csv
import io
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import aggregate_scores


def test_build_parser_builds_parser_with_stable_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stocke les paramètres capturés pour inspecter description=...
    init_kwargs: dict[str, object] = {}
    # Stocke les signatures d'arguments pour verrouiller help/default/type
    declared: list[tuple[tuple[str, ...], dict[str, object]]] = []

    # Espionne ArgumentParser pour valider description et add_argument(...)
    class SpyArgumentParser:
        # Capture les kwargs d'initialisation pour stabiliser la description
        def __init__(self, *args: object, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

        # Capture les appels add_argument pour vérifier le contrat des options
        def add_argument(self, *args: object, **kwargs: object) -> None:
            flags = tuple(str(value) for value in args)
            declared.append((flags, dict(kwargs)))

    # Remplace ArgumentParser pour observer build_parser sans exécuter argparse réel
    monkeypatch.setattr(
        aggregate_scores.argparse,
        "ArgumentParser",
        SpyArgumentParser,
    )

    # Construit le parser via la fonction à verrouiller
    parser = aggregate_scores.build_parser()
    # Valide que build_parser retourne bien le parser construit
    assert isinstance(parser, SpyArgumentParser)

    # Verrouille la description exacte pour tuer description=None et altérations
    assert init_kwargs["description"] == (
        "Agrège les accuracies par run, sujet et global à partir des "
        "artefacts et écrit CSV/JSON"
    )

    # Récupère les kwargs déclarés pour un flag donné
    def kwargs_for(flag: str) -> dict[str, object]:
        for flags, kwargs in declared:
            if flag in flags:
                return kwargs
        raise AssertionError(f"Argument manquant: {flag}")

    # Verrouille --data-dir (type, default, help) pour tuer help=None/altérations
    data_dir = kwargs_for("--data-dir")
    assert data_dir["type"] is Path
    assert data_dir["default"] == aggregate_scores.DEFAULT_DATA_DIR
    assert data_dir["help"] == (
        "Répertoire racine contenant les matrices numpy utilisées pour le scoring"
    )

    # Verrouille --artifacts-dir (type, default, help) pour tuer help=None/altérations
    artifacts_dir = kwargs_for("--artifacts-dir")
    assert artifacts_dir["type"] is Path
    assert artifacts_dir["default"] == aggregate_scores.DEFAULT_ARTIFACTS_DIR
    assert artifacts_dir["help"] == (
        "Répertoire racine où sont stockés les modèles et matrices W"
    )

    # Verrouille --csv-output et impose la présence explicite de default=None
    csv_output = kwargs_for("--csv-output")
    assert csv_output["type"] is Path
    assert "default" in csv_output
    assert csv_output["default"] is None
    assert csv_output["help"] == (
        "Chemin de sortie du rapport CSV (type,subject,run,accuracy,thresholds)"
    )

    # Verrouille --json-output et impose la présence explicite de default=None
    json_output = kwargs_for("--json-output")
    assert json_output["type"] is Path
    assert "default" in json_output
    assert json_output["default"] is None
    assert json_output["help"] == "Chemin de sortie du rapport JSON aligné CI"


def test_score_run_minimum_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Force une accuracy exactement au seuil minimum pour tuer "> MINIMUM"
    def fake_evaluate_run(
        subject: str,
        run: str,
        data_dir: Path,
        artifacts_dir: Path,
    ) -> dict:
        return {"accuracy": aggregate_scores.MINIMUM_ACCURACY}

    # Remplace l'évaluation réelle pour isoler le contrat de _score_run
    monkeypatch.setattr(aggregate_scores.predict_cli, "evaluate_run", fake_evaluate_run)

    # Exécute le scoring avec des chemins arbitraires
    entry = aggregate_scores._score_run("S001", "R01", tmp_path, tmp_path)

    # Verrouille l'inclusivité du seuil minimum (>=) au point d'égalité
    assert entry["meets_minimum"] is True
    # Vérifie la cohérence du drapeau cible via le seuil actuel
    assert entry["meets_target"] is (
        aggregate_scores.MINIMUM_ACCURACY >= aggregate_scores.TARGET_ACCURACY
    )


def test_score_run_target_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Force une accuracy exactement au seuil cible pour tuer "> TARGET"
    def fake_evaluate_run(
        subject: str,
        run: str,
        data_dir: Path,
        artifacts_dir: Path,
    ) -> dict:
        return {"accuracy": aggregate_scores.TARGET_ACCURACY}

    # Remplace l'évaluation réelle pour isoler le contrat de _score_run
    monkeypatch.setattr(aggregate_scores.predict_cli, "evaluate_run", fake_evaluate_run)

    # Exécute le scoring avec des chemins arbitraires
    entry = aggregate_scores._score_run("S001", "R01", tmp_path, tmp_path)

    # Verrouille l'inclusivité du seuil cible (>=) au point d'égalité
    assert entry["meets_target"] is True
    # Le seuil minimum doit aussi être satisfait si la cible l'est
    assert entry["meets_minimum"] is True


def test_aggregate_scores_subject_target_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Force un seul run pour avoir une moyenne sujet == TARGET exactement
    monkeypatch.setattr(
        aggregate_scores, "_discover_runs", lambda *_: [("S001", "R01")]
    )

    # Injecte une entrée run dont l'accuracy est exactement la cible
    def fake_score_run(
        subject: str,
        run: str,
        data_dir: Path,
        artifacts_dir: Path,
    ) -> dict:
        acc = aggregate_scores.TARGET_ACCURACY
        return {
            "subject": subject,
            "run": run,
            "accuracy": acc,
            "meets_minimum": acc >= aggregate_scores.MINIMUM_ACCURACY,
            "meets_target": acc >= aggregate_scores.TARGET_ACCURACY,
        }

    # Remplace _score_run pour isoler l'agrégation (moyennes + seuils)
    monkeypatch.setattr(aggregate_scores, "_score_run", fake_score_run)

    # Exécute l'agrégation complète
    report = aggregate_scores.aggregate_scores(tmp_path, tmp_path)

    # Récupère l'entrée sujet unique
    subject_entry = next(e for e in report["subjects"] if e["subject"] == "S001")

    # Verrouille la moyenne et l'inclusivité du seuil cible au point d'égalité
    assert subject_entry["accuracy"] == pytest.approx(aggregate_scores.TARGET_ACCURACY)
    assert subject_entry["meets_target"] is True


def test_aggregate_scores_global_minimum_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Force un seul run pour avoir une moyenne globale == MINIMUM exactement
    monkeypatch.setattr(
        aggregate_scores, "_discover_runs", lambda *_: [("S001", "R01")]
    )

    # Injecte une entrée run dont l'accuracy est exactement le minimum
    def fake_score_run(
        subject: str,
        run: str,
        data_dir: Path,
        artifacts_dir: Path,
    ) -> dict:
        acc = aggregate_scores.MINIMUM_ACCURACY
        return {
            "subject": subject,
            "run": run,
            "accuracy": acc,
            "meets_minimum": acc >= aggregate_scores.MINIMUM_ACCURACY,
            "meets_target": acc >= aggregate_scores.TARGET_ACCURACY,
        }

    # Remplace _score_run pour isoler l'agrégation (moyennes + seuils)
    monkeypatch.setattr(aggregate_scores, "_score_run", fake_score_run)

    # Exécute l'agrégation complète
    report = aggregate_scores.aggregate_scores(tmp_path, tmp_path)

    # Verrouille l'inclusivité du seuil minimum au point d'égalité
    assert report["global"]["accuracy"] == pytest.approx(
        aggregate_scores.MINIMUM_ACCURACY
    )
    assert report["global"]["meets_minimum"] is True


def test_aggregate_scores_global_target_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Force un seul run pour avoir une moyenne globale == TARGET exactement
    monkeypatch.setattr(
        aggregate_scores, "_discover_runs", lambda *_: [("S001", "R01")]
    )

    # Injecte une entrée run dont l'accuracy est exactement la cible
    def fake_score_run(
        subject: str,
        run: str,
        data_dir: Path,
        artifacts_dir: Path,
    ) -> dict:
        acc = aggregate_scores.TARGET_ACCURACY
        return {
            "subject": subject,
            "run": run,
            "accuracy": acc,
            "meets_minimum": acc >= aggregate_scores.MINIMUM_ACCURACY,
            "meets_target": acc >= aggregate_scores.TARGET_ACCURACY,
        }

    # Remplace _score_run pour isoler l'agrégation (moyennes + seuils)
    monkeypatch.setattr(aggregate_scores, "_score_run", fake_score_run)

    # Exécute l'agrégation complète
    report = aggregate_scores.aggregate_scores(tmp_path, tmp_path)

    # Verrouille l'inclusivité du seuil cible au point d'égalité
    assert report["global"]["accuracy"] == pytest.approx(
        aggregate_scores.TARGET_ACCURACY
    )
    assert report["global"]["meets_target"] is True
    assert report["global"]["meets_minimum"] is True


def test_write_csv_uses_stable_io_contract_and_row_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Prépare un report minimal mais représentatif pour générer 3 types de lignes
    report = {
        "runs": [
            {
                "subject": "S001",
                "run": "R01",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "subjects": [
            {
                "subject": "S001",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "global": {
            "accuracy": 0.8,
            "meets_minimum": True,
            "meets_target": True,
        },
    }

    # Cible un chemin imbriqué pour forcer mkdir() sur un parent non trivial
    csv_path = tmp_path / "out" / "reports" / "scores.csv"

    # Capture les paramètres mkdir(...) pour tuer les mutants parents/exist_ok
    mkdir_calls: list[dict[str, object]] = []
    real_mkdir: Callable[..., None] = Path.mkdir

    def mkdir_spy(self: Path, *args: Any, **kwargs: Any) -> None:
        mkdir_calls.append({"self": self, "args": args, "kwargs": dict(kwargs)})
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_spy)

    # Capture les paramètres open(...) et le contenu écrit pour tuer encoding/newline
    open_calls: list[dict[str, object]] = []
    buffer = io.StringIO()

    class Handle:
        # Expose un handle texte contrôlé pour capter le CSV sans toucher le disque
        def __enter__(self) -> io.StringIO:
            return buffer

        # Ferme logiquement le contexte sans fermer le buffer pour lecture après
        def __exit__(self, *exc: object) -> None:
            return None

    def open_spy(self: Path, *args: object, **kwargs: object) -> Handle:
        open_calls.append({"self": self, "args": args, "kwargs": dict(kwargs)})
        return Handle()

    monkeypatch.setattr(Path, "open", open_spy)

    # Exécute l'écriture CSV
    aggregate_scores.write_csv(report, csv_path)

    # Verrouille mkdir(...) exactement pour tuer:
    # parents=None / parents=False / suppression parents / exist_ok=None / exist_ok=False
    assert len(mkdir_calls) >= 1
    assert mkdir_calls[0]["self"] == csv_path.parent
    assert mkdir_calls[0]["args"] == ()
    assert mkdir_calls[0]["kwargs"] == {"parents": True, "exist_ok": True}

    # Verrouille open(...) exactement pour tuer:
    # encoding=None / newline=None / suppression encoding / suppression newline / encoding="UTF-8"
    assert len(open_calls) == 1
    assert open_calls[0]["self"] == csv_path
    assert open_calls[0]["args"] == ("w",)
    assert open_calls[0]["kwargs"] == {"encoding": "utf-8", "newline": ""}

    # Parse le CSV capturé pour tuer les mutants sur les valeurs "type"/vides
    buffer.seek(0)
    reader = csv.DictReader(buffer)
    rows = list(reader)

    # Verrouille l'en-tête (présence/ordre) pour stabiliser le schéma
    assert reader.fieldnames == [
        "type",
        "subject",
        "run",
        "accuracy",
        "meets_minimum",
        "meets_target",
    ]

    # Attend exactement 3 lignes: run + subject + global
    assert [r["type"] for r in rows] == ["run", "subject", "global"]

    # Verrouille la ligne run (tue "XXrunXX", "RUN", etc.)
    assert rows[0]["subject"] == "S001"
    assert rows[0]["run"] == "R01"

    # Verrouille la ligne subject: run doit être vide (tue run="XXXX")
    assert rows[1]["subject"] == "S001"
    assert rows[1]["run"] == ""

    # Verrouille la ligne global: subject et run doivent être vides (tue "XXXX")
    assert rows[2]["subject"] == ""
    assert rows[2]["run"] == ""


def test_write_json_uses_stable_io_contract_and_pretty_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Prépare un report minimal mais représentatif pour valider le JSON
    report = {
        "runs": [
            {
                "subject": "S001",
                "run": "R01",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "subjects": [
            {
                "subject": "S001",
                "accuracy": 0.8,
                "meets_minimum": True,
                "meets_target": True,
            }
        ],
        "global": {
            "accuracy": 0.8,
            "meets_minimum": True,
            "meets_target": True,
        },
    }

    # Cible un chemin imbriqué pour forcer mkdir() sur un parent non trivial
    json_path = tmp_path / "out" / "reports" / "scores.json"

    # Capture les paramètres mkdir(...) pour tuer parents=None/False/absent
    mkdir_calls: list[dict[str, object]] = []
    real_mkdir: Callable[..., None] = Path.mkdir

    def mkdir_spy(self: Path, *args: Any, **kwargs: Any) -> None:
        mkdir_calls.append({"self": self, "args": args, "kwargs": dict(kwargs)})
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_spy)

    # Capture les paramètres open(...) et le contenu écrit pour tuer encoding=None/absent
    open_calls: list[dict[str, object]] = []
    buffer = io.StringIO()

    class Handle:
        # Expose un handle texte contrôlé pour capter le JSON sans toucher le disque
        def __enter__(self) -> io.StringIO:
            return buffer

        # Ferme logiquement le contexte sans fermer le buffer pour lecture après
        def __exit__(self, *exc: object) -> None:
            return None

    def open_spy(self: Path, *args: Any, **kwargs: Any) -> Handle:
        open_calls.append({"self": self, "args": args, "kwargs": dict(kwargs)})
        return Handle()

    monkeypatch.setattr(Path, "open", open_spy)

    # Capture json.dump(...) pour tuer indent=None/absent/3
    dump_calls: list[dict[str, object]] = []
    real_dump: Callable[..., None] = aggregate_scores.json.dump

    def dump_spy(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> None:
        dump_calls.append({"obj": obj, "fp": fp, "args": args, "kwargs": dict(kwargs)})
        return real_dump(obj, fp, *args, **kwargs)

    monkeypatch.setattr(aggregate_scores.json, "dump", dump_spy)

    # Exécute l'écriture JSON
    aggregate_scores.write_json(report, json_path)

    # Verrouille mkdir(...) exactement pour tuer:
    # parents=None / parents=False / suppression parents / exist_ok=None / exist_ok=False
    assert len(mkdir_calls) >= 1
    assert mkdir_calls[0]["self"] == json_path.parent
    assert mkdir_calls[0]["args"] == ()
    assert mkdir_calls[0]["kwargs"] == {"parents": True, "exist_ok": True}

    # Verrouille open(...) exactement pour tuer:
    # encoding=None / suppression encoding / encoding="UTF-8"
    assert len(open_calls) == 1
    assert open_calls[0]["self"] == json_path
    assert open_calls[0]["args"] == ("w",)
    assert open_calls[0]["kwargs"] == {"encoding": "utf-8"}

    # Verrouille json.dump(..., indent=2) exactement pour tuer:
    # indent=None / suppression indent / indent=3
    assert len(dump_calls) == 1
    assert dump_calls[0]["obj"] == report
    assert dump_calls[0]["fp"] is buffer
    assert dump_calls[0]["args"] == ()
    assert dump_calls[0]["kwargs"] == {"indent": 2}

    # Verrouille la sortie texte pour stabiliser le format (indentation)
    assert buffer.getvalue() == json.dumps(report, indent=2)
