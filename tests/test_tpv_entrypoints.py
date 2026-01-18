# Importe importlib pour charger les proxies tpv à la demande
import importlib

# Charge le module tpv.predict pour les tests d'entrée
tpv_predict = importlib.import_module("tpv.predict")
# Charge le module tpv.train pour les tests d'entrée
tpv_train = importlib.import_module("tpv.train")


# Vérifie que tpv.train.main délègue au main de scripts.train
def test_tpv_train_main_delegates_to_scripts_train(monkeypatch):
    # Prépare un conteneur pour capturer les arguments
    captured: dict[str, object] = {}

    # Définit un main factice pour simuler scripts.train.main
    def fake_main(argv):
        # Mémorise les arguments reçus pour assertion
        captured["argv"] = argv
        # Retourne un code de sortie non nul pour le test
        return 5

    # Remplace le main du module scripts.train importé dans tpv.train
    monkeypatch.setattr(tpv_train.script_train, "main", fake_main)
    # Appelle le proxy tpv.train.main avec un argv de test
    result = tpv_train.main(["--help"])

    # Vérifie que le code de retour est bien propagé
    assert result == 5
    # Vérifie que les arguments ont été transmis sans modification
    assert captured["argv"] == ["--help"]


# Vérifie que tpv.predict.main délègue au main de scripts.predict
def test_tpv_predict_main_delegates_to_scripts_predict(monkeypatch):
    # Prépare un conteneur pour capturer les arguments
    captured: dict[str, object] = {}

    # Définit un main factice pour simuler scripts.predict.main
    def fake_main(argv):
        # Mémorise les arguments reçus pour assertion
        captured["argv"] = argv
        # Retourne un code de sortie non nul pour le test
        return 3

    # Remplace le main du module scripts.predict importé dans tpv.predict
    monkeypatch.setattr(tpv_predict.script_predict, "main", fake_main)
    # Appelle le proxy tpv.predict.main avec un argv de test
    result = tpv_predict.main(["--version"])

    # Vérifie que le code de retour est bien propagé
    assert result == 3
    # Vérifie que les arguments ont été transmis sans modification
    assert captured["argv"] == ["--version"]
