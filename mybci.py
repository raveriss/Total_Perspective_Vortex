"""Interface CLI pour piloter les workflows d'entraînement et de prédiction."""

# Préserve argparse pour parser les options CLI avec validation
import argparse

# Préserve subprocess pour lancer les modules en sous-processus isolés
import subprocess

# Préserve sys pour identifier l'interpréteur courant
import sys

# Préserve dataclass pour regrouper les paramètres du pipeline
from dataclasses import dataclass

# Garantit l'accès aux séquences typées pour mypy
from typing import Sequence


# Centralise les options nécessaires pour invoquer le mode realtime
@dataclass
class RealtimeCallConfig:
    """Conteneur des paramètres transmis au module realtime."""

    # Identifie le sujet cible pour cibler les artefacts du modèle
    subject: str
    # Identifie le run cible pour sélectionner la session
    run: str
    # Fixe la taille de fenêtre glissante en échantillons
    window_size: int
    # Fixe le pas entre deux fenêtres successives
    step_size: int
    # Fixe la taille du buffer utilisé pour lisser les prédictions
    buffer_size: int
    # Renseigne la fréquence d'échantillonnage pour calculer les offsets
    sfreq: float
    # Spécifie la latence maximale autorisée pour chaque fenêtre
    max_latency: float
    # Spécifie le répertoire contenant les fichiers numpy streamés
    data_dir: str
    # Spécifie le répertoire racine où lire les artefacts entraînés
    artifacts_dir: str


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


# Construit la ligne de commande pour invoquer le mode realtime
def _call_realtime(config: RealtimeCallConfig) -> int:
    """Invoke le module tpv.realtime avec les paramètres streaming."""

    # Initialise la commande avec l'interpréteur courant et le module ciblé
    command: list[str] = [
        sys.executable,
        "-m",
        "tpv.realtime",
        config.subject,
        config.run,
    ]
    # Ajoute la taille de fenêtre demandée pour le streaming
    command.extend(["--window-size", str(config.window_size)])
    # Ajoute le pas de glissement entre fenêtres successives
    command.extend(["--step-size", str(config.step_size)])
    # Ajoute la taille du buffer utilisé pour lisser les prédictions
    command.extend(["--buffer-size", str(config.buffer_size)])
    # Ajoute la latence maximale autorisée pour surveiller le SLA
    command.extend(["--max-latency", str(config.max_latency)])
    # Ajoute la fréquence d'échantillonnage pour calculer les offsets
    command.extend(["--sfreq", str(config.sfreq)])
    # Ajoute le répertoire des données streamées
    command.extend(["--data-dir", config.data_dir])
    # Ajoute le répertoire d'artefacts contenant le modèle entraîné
    command.extend(["--artifacts-dir", config.artifacts_dir])
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
        usage="python mybci.py <subject> <run> {train,predict,realtime}",
    )
    # Ajoute l'identifiant du sujet pour cibler les données
    parser.add_argument("subject", help="Identifiant du sujet (ex: S01)")
    # Ajoute l'identifiant du run pour cibler la session
    parser.add_argument("run", help="Identifiant du run (ex: R01)")
    # Ajoute le mode pour distinguer entraînement et prédiction
    parser.add_argument(
        "mode",
        choices=("train", "predict", "realtime"),
        help="Choix du pipeline à lancer",
    )
    # Ajoute le choix du classifieur pour composer le pipeline
    parser.add_argument(
        "--classifier",
        choices=("lda", "logistic", "svm", "centroid"),
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
        default=argparse.SUPPRESS,
        help="Nombre de composantes à conserver",
    )
    # Ajoute un flag pour désactiver la normalisation des features
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Désactive la normalisation des features",
    )
    # Ajoute la taille de fenêtre pour la lecture streaming en realtime
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Taille de fenêtre glissante pour le mode realtime",
    )
    # Ajoute le pas entre deux fenêtres successives
    parser.add_argument(
        "--step-size",
        type=int,
        default=25,
        help="Pas entre deux fenêtres en streaming realtime",
    )
    # Ajoute la taille du buffer pour lisser les prédictions instantanées
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=3,
        help="Taille du buffer de lissage pour le mode realtime",
    )
    # Ajoute la fréquence d'échantillonnage utilisée pour les offsets
    parser.add_argument(
        "--sfreq",
        type=float,
        default=50.0,
        help="Fréquence d'échantillonnage appliquée au flux realtime",
    )
    # Ajoute la latence maximale tolérée pour surveiller la boucle
    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
        help="Latence maximale autorisée par fenêtre realtime",
    )
    # Ajoute le répertoire de données nécessaire au streaming
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Répertoire racine contenant les fichiers numpy",
    )
    # Ajoute le répertoire d'artefacts où lire le modèle entraîné
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Répertoire racine où récupérer le modèle entraîné",
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
    n_components = getattr(args, "n_components", None)

    config = ModuleCallConfig(
        subject=args.subject,
        run=args.run,
        classifier=args.classifier,
        scaler=scaler,
        feature_strategy=args.feature_strategy,
        dim_method=args.dim_method,
        n_components=n_components,
        normalize_features=normalize_features,
    )
    # Appelle le module train si le mode le demande
    if args.mode == "train":
        # Retourne le code retour du module train avec la configuration
        return _call_module(
            "tpv.train",
            config,
        )
    # Bascule vers le module realtime pour le streaming fenêtré
    if args.mode == "realtime":
        # Construit la configuration spécifique au lissage et fenêtrage
        realtime_config = RealtimeCallConfig(
            subject=args.subject,
            run=args.run,
            window_size=args.window_size,
            step_size=args.step_size,
            buffer_size=args.buffer_size,
            sfreq=args.sfreq,
            max_latency=args.max_latency,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
        )
        # Retourne le code retour du module realtime avec la configuration
        return _call_realtime(realtime_config)
    # Appelle le module predict pour le mode prédiction
    return _call_module(
        "tpv.predict",
        config,
    )


# Protège l'exécution directe pour déléguer au main
if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    # Expose le code retour comme exit code du processus
    raise SystemExit(main())
