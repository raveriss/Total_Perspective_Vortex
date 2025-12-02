"""Point d'entrée module pour l'entraînement via mybci."""

# Centralise le dispatch vers la logique CLI définie dans scripts/train
from scripts import train as script_train


# Relaye l'exécution CLI pour rester compatible avec python -m tpv.train
def main(argv: list[str] | None = None) -> int:
    """Proxy vers scripts.train.main pour lancer l'entraînement."""

    # Délègue l'ensemble du traitement au module scripts.train
    return script_train.main(argv)


# Expose explicitement la fonction d'entraînement pour les imports directs
run_training = script_train.run_training


# Protège l'exécution directe pour exposer un exit code explicite
if __name__ == "__main__":  # pragma: no cover - exécution CLI directe
    # Retourne l'issue du main comme code de sortie du processus
    raise SystemExit(main())
