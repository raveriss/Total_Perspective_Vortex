"""Tests des points d'entrée tpv.train et tpv.predict."""

# Préserve pytest pour les assertions et monkeypatch
import pytest

# Importe le module train pour vérifier la délégation
from tpv import train as tpv_train

# Importe le module predict pour vérifier la délégation
from tpv import predict as tpv_predict


# Vérifie que tpv.train.main délègue au script CLI
def test_train_main_delegates_to_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assure la délégation de main vers scripts.train.main."""

    # Prépare un conteneur pour capturer les arguments
    seen: dict[str, object] = {}

    # Définit un stub de main pour intercepter l'appel
    def _fake_main(argv: list[str] | None) -> int:
        """Retourne un code stable et capture argv."""

        # Capture les arguments transmis pour vérification
        seen["argv"] = argv
        # Retourne un code de succès arbitraire
        return 0

    # Remplace la fonction main réelle par le stub
    monkeypatch.setattr(tpv_train.script_train, "main", _fake_main)
    # Définit des arguments factices pour vérifier le relais
    expected_args = ["S001", "R03", "train"]
    # Exécute le main proxy
    result = tpv_train.main(expected_args)
    # Vérifie que le code de retour est propagé
    assert result == 0
    # Vérifie que les arguments sont transmis tels quels
    assert seen["argv"] == expected_args


# Vérifie que run_training reste un alias explicite de scripts.train.run_training
def test_train_run_training_alias() -> None:
    """Valide l'alias direct vers la fonction d'entraînement."""

    # Vérifie que l'alias pointe vers la même fonction
    assert tpv_train.run_training is tpv_train.script_train.run_training


# Vérifie que tpv.predict.main délègue au script CLI
def test_predict_main_delegates_to_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assure la délégation de main vers scripts.predict.main."""

    # Prépare un conteneur pour capturer les arguments
    seen: dict[str, object] = {}

    # Définit un stub de main pour intercepter l'appel
    def _fake_main(argv: list[str] | None) -> int:
        """Retourne un code stable et capture argv."""

        # Capture les arguments transmis pour vérification
        seen["argv"] = argv
        # Retourne un code de succès arbitraire
        return 0

    # Remplace la fonction main réelle par le stub
    monkeypatch.setattr(tpv_predict.script_predict, "main", _fake_main)
    # Définit des arguments factices pour vérifier le relais
    expected_args = ["S001", "R03", "predict"]
    # Exécute le main proxy
    result = tpv_predict.main(expected_args)
    # Vérifie que le code de retour est propagé
    assert result == 0
    # Vérifie que les arguments sont transmis tels quels
    assert seen["argv"] == expected_args


# Vérifie que evaluate_run reste un alias explicite
def test_predict_evaluate_run_alias() -> None:
    """Valide l'alias direct vers scripts.predict.evaluate_run."""

    # Vérifie que l'alias pointe vers la même fonction
    assert tpv_predict.evaluate_run is tpv_predict.script_predict.evaluate_run


# Vérifie que build_report reste un alias explicite
def test_predict_build_report_alias() -> None:
    """Valide l'alias direct vers scripts.predict.build_report."""

    # Vérifie que l'alias pointe vers la même fonction
    assert tpv_predict.build_report is tpv_predict.script_predict.build_report
