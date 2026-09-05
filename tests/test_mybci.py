"""Contrats du routeur mybci volontairement minimal."""

# Fournit Namespace pour tester la traduction des options sans parser.
import argparse

# Importe le point d'entrée public depuis la racine du rendu.
import mybci


# Protège la normalisation des identifiants attendue pendant la soutenance.
def test_parser_normalizes_subject_run() -> None:
    """Accepte les formes courtes utilisées dans les commandes Make."""

    # Parse une invocation minimale équivalente à « make train 1 3 ».
    args = mybci.parse_args(["1", "3", "train"])
    # Le sujet doit correspondre au nom de dossier PhysioNet.
    assert args.subject == "S001"
    # Le run doit correspondre au nom de fichier PhysioNet.
    assert args.run == "R03"


# Protège la délégation train sans réintroduire de logique métier dans mybci.
def test_main_routes_train_to_specialized_module(monkeypatch) -> None:
    """Transmet sujet, run et options au module tpv.train."""

    # Évite que le test dépende des paquets installés sur la machine.
    monkeypatch.setattr(mybci, "_ensure_ml_dependencies", lambda: None)
    # Capture l'appel produit par le routeur.
    captured = {}

    # Remplace le sous-processus par une fonction d'observation locale.
    def fake_call(module_name, config):
        # Enregistre le module choisi pour vérifier le routage.
        captured["module"] = module_name
        # Enregistre la configuration sans exécuter d'entraînement.
        captured["config"] = config
        # Simule une commande métier terminée avec succès.
        return 0

    # Injecte l'observateur à la place du lancement réel.
    monkeypatch.setattr(mybci, "_call_module", fake_call)
    # Exécute le chemin public avec un bonus de features.
    exit_code = mybci.main(["1", "3", "train", "--feature-strategy", "wavelet"])

    # Le routeur doit propager le succès du module spécialisé.
    assert exit_code == 0
    # Aucun autre module ne doit être choisi pour le mode train.
    assert captured["module"] == "tpv.train"
    # Les options restent lisibles et séparées dans la configuration.
    assert captured["config"].module_args == ["--feature-strategy", "wavelet"]


# Protège l'usage d'un seul moteur pour le score global.
def test_main_delegates_global_evaluation_to_aggregator(monkeypatch) -> None:
    """Envoie les options globales à aggregate_experience_scores.main."""

    # Capture les arguments reçus par le moteur canonique.
    received = []

    # Remplace l'agrégation réelle par un retour déterministe.
    def fake_aggregate(args):
        # Conserve l'invocation pour vérifier la traduction de l'alias.
        received.append(args)
        # Utilise un code distinct afin de vérifier sa propagation.
        return 7

    # Injecte le moteur factice sans effet de bord hors du test.
    monkeypatch.setattr(
        mybci.aggregate_experience_scores,
        "main",
        fake_aggregate,
    )

    # Utilise l'ancien alias PCA pour vérifier sa compatibilité.
    exit_code = mybci.main(["--feature-strategy", "pca"])

    # Le code du moteur unique doit être propagé sans interprétation.
    assert exit_code == 7
    # L'alias devient une option de réduction comprise par l'agrégateur.
    assert received == [["--dim-method", "pca"]]


# Protège la traduction de l'ancien alias CSP en option de réduction.
def test_build_module_args_translates_dimensionality_alias() -> None:
    """Ne transmet jamais CSP comme extracteur fréquentiel."""

    # Construit le namespace minimal produit par argparse.
    args = argparse.Namespace(feature_strategy="csp")
    # Traduit l'alias avant l'appel du module spécialisé.
    module_args = mybci._build_module_args(args)
    # CSP doit rester une réduction et préserver l'algorithme évalué.
    assert module_args == ["--dim-method", "csp"]
