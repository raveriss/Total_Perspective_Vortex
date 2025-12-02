"""Point d'entrée module pour la prédiction via mybci."""

# Centralise le dispatch vers la logique CLI définie dans scripts/predict
from scripts import predict as script_predict


# Relaye l'exécution CLI pour rester compatible avec python -m tpv.predict
def main(argv: list[str] | None = None) -> int:
    """Proxy vers scripts.predict.main pour lancer la prédiction."""

    # Délègue l'ensemble du traitement au module scripts.predict
    return script_predict.main(argv)


# Expose explicitement l'évaluation de run pour les usages directs
evaluate_run = script_predict.evaluate_run


# Expose explicitement la construction de rapports agrégés
build_report = script_predict.build_report


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
