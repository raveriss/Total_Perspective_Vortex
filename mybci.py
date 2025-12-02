"""Interface CLI pour piloter les workflows d'entraînement et de prédiction."""

# Préserve argparse pour exposer les paramètres utilisateurs
import argparse

# Préserve subprocess pour lancer les modules en sous-processus isolés
import subprocess

# Préserve sys pour identifier l'interpréteur courant
import sys

# Préserve dataclass pour regrouper les paramètres du pipeline
from dataclasses import dataclass

# Garantit l'accès aux séquences typées pour mypy
from typing import Sequence


# Centralise les options nécessaires pour invoquer un module TPV
@dataclass
class ModuleCallConfig:
    """Conteneur des paramètres transmis aux modules train/predict."""

    # Identifie le sujet cible pour charger les données correspondantes
    subject: str
    # Identifie le run cible pour charger la bonne session
    run: str
    # Sélectionne le classifieur pour harmoniser train et predict
    classifier: str
    # Sélectionne le scaler optionnel pour stabiliser les features
    scaler: str | None
    # Harmonise la stratégie d'extraction des features
    feature_strategy: str
    # Choisit la méthode de réduction de dimension
    dim_method: str
    # Spécifie le nombre de composantes projetées
    n_components: int | None
    # Indique si les features doivent être normalisées
    normalize_features: bool


# Construit la ligne de commande pour invoquer un module TPV
def _call_module(module_name: str, config: ModuleCallConfig) -> int:
    """Invoke un module TPV en ajoutant les options du pipeline."""

    # Initialise la commande avec l'interpréteur courant et le module ciblé
    command: list[str] = [
        sys.executable,
        "-m",
        module_name,
        config.subject,
        config.run,
    ]
    # Ajoute le classifieur choisi pour transmettre la préférence utilisateur
    command.extend(["--classifier", config.classifier])
    # Ajoute la stratégie de scaler uniquement lorsqu'elle est définie
    if config.scaler is not None:
        # Propulse le scaler choisi vers le module appelé
        command.extend(["--scaler", config.scaler])
    # Ajoute la stratégie de features pour harmoniser train et predict
    command.extend(["--feature-strategy", config.feature_strategy])
    # Ajoute la méthode de réduction de dimension pour la cohérence
    command.extend(["--dim-method", config.dim_method])
    # Ajoute le nombre de composantes si fourni pour contrôler la compression
    if config.n_components is not None:
        # Passe n_components sous forme de chaîne pour argparse descendant
        command.extend(["--n-components", str(config.n_components)])
    # Ajoute un indicateur pour désactiver la normalisation si demandé
    if not config.normalize_features:
        # Utilise un flag booléen pour inverser la valeur par défaut
        command.append("--no-normalize-features")
    # Exécute la commande en capturant le code retour sans lever d'exception
    completed = subprocess.run(command, check=False)
    # Retourne le code retour pour propagation à l'appelant principal
    return completed.returncode


# Construit le parser CLI avec toutes les options du pipeline
def build_parser() -> argparse.ArgumentParser:
    """Construit l'argument parser pour mybci."""

    # Instancie le parser avec description et usage explicite
    parser = argparse.ArgumentParser(
        description="Pilote un workflow d'entraînement ou de prédiction TPV",
        usage="python mybci.py <subject> <run> {train,predict}",
    )
    # Ajoute l'identifiant du sujet pour cibler les données
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    # Ajoute l'identifiant du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    # Ajoute le mode pour distinguer entraînement et prédiction
    parser.add_argument(
        "mode",
        choices=("train", "predict"),
        help="Choix du pipeline à lancer",
    )
    # Ajoute le choix du classifieur pour composer le pipeline
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm"),
        default="lda",
        help="Choix du classifieur final",
    )
    # Ajoute le scaler optionnel pour stabiliser les features
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust", "none"),
        default="none",
        help="Scaler optionnel appliqué après les features",
    )
    # Ajoute la stratégie d'extraction de features pour harmoniser train/predict
    parser.add_argument(
        "--feature-strategy",
        choices=("fft", "wavelet"),
        default="fft",
        help="Méthode d'extraction des features",
    )
    # Ajoute la méthode de réduction de dimension pour ajuster la compression
    parser.add_argument(
        "--dim-method",
        choices=("pca", "csp"),
        default="pca",
        help="Technique de réduction de dimension",
    )
    # Ajoute le nombre de composantes pour contrôler la taille projetée
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Nombre de composantes à conserver",
    )
    # Ajoute un flag pour désactiver la normalisation des features
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Désactive la normalisation des features",
    )
    # Retourne le parser configuré
    return parser


# Parse les arguments fournis à la CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse les arguments passés à mybci."""

    # Construit le parser pour traiter argv
    parser = build_parser()
    # Retourne l'espace de noms après parsing
    return parser.parse_args(argv)


# Point d'entrée principal de la CLI
def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée exécutable de mybci."""

    # Parse les arguments fournis par l'utilisateur
    args = parse_args(argv)
    # Interprète le choix du scaler pour convertir "none" en None
    scaler = None if args.scaler == "none" else args.scaler
    # Applique la normalisation en inversant le flag d'opt-out
    normalize_features = not args.no_normalize_features
    # Construit la configuration partagée entre les modules train et predict
    config = ModuleCallConfig(
        subject=args.subject,
        run=args.run,
        classifier=args.classifier,
        scaler=scaler,
        feature_strategy=args.feature_strategy,
        dim_method=args.dim_method,
        n_components=args.n_components,
        normalize_features=normalize_features,
    )
    # Appelle le module train si le mode le demande
    if args.mode == "train":
        # Retourne le code retour du module train avec la configuration
        return _call_module(
            "tpv.train",
            config,
        )
    # Appelle le module predict pour le mode prédiction
    return _call_module(
        "tpv.predict",
        config,
    )


# Protège l'exécution directe pour déléguer au main
if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    # Expose le code retour comme exit code du processus
    raise SystemExit(main())
