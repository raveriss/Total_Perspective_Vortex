# Offre argparse pour exposer une CLI de benchmark reproductible
import argparse

# Ajoute src/ au PYTHONPATH pour importer tpv lors de l'exécution directe
import sys

# Importe Path pour définir proprement les chemins de sortie
from pathlib import Path

# Importe time pour mesurer la durée d'entraînement et de prédiction
from time import perf_counter

# Importe numpy pour générer les jeux de données synthétiques
import numpy as np

# Importe pandas pour agréger et sauvegarder les résultats de benchmark
import pandas as pd

# Importe train_test_split pour séparer entraînement et validation
from sklearn.model_selection import train_test_split

# Calcule la racine du projet pour construire le chemin src
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ajoute le chemin src au sys.path pour sécuriser les imports locaux
sys.path.append(str(PROJECT_ROOT / "src"))

# Importe la configuration et la construction de pipeline existantes
from tpv.pipeline import PipelineConfig, build_pipeline  # noqa: E402


def _generate_dataset(n_samples: int, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Génère un dataset synthétique linéairement séparable pour benchmarking."""

    # Calibre l'axe temporel sur une seconde pour simplifier les fréquences
    times = np.arange(0.0, 1.0, 1.0 / sfreq)
    # Génère un signal dominant beta pour la classe 0
    beta_signal = np.sin(2 * np.pi * 20.0 * times)
    # Génère un signal dominant alpha pour la classe 1
    alpha_signal = np.sin(2 * np.pi * 10.0 * times)
    # Prépare un générateur pour ajouter un bruit identique entre les classes
    rng = np.random.default_rng(seed=42)
    # Construit les essais de classe 0 avec énergie concentrée sur le premier canal
    class_zero = np.stack(
        [
            beta_signal + 0.05 * rng.standard_normal(times.size),
            rng.standard_normal(times.size),
        ]
    )
    # Construit les essais de classe 1 avec énergie concentrée sur le second canal
    class_one = np.stack(
        [
            rng.standard_normal(times.size),
            alpha_signal + 0.05 * rng.standard_normal(times.size),
        ]
    )
    # Réplique les essais pour atteindre le volume demandé
    samples = []
    # Prépare les labels correspondants pour chaque essai généré
    labels = []
    # Parcourt les échantillons pour alterner les classes et stabiliser la balance
    for sample_index in range(n_samples):
        # Sélectionne la classe en alternant pour équilibrer le dataset
        if sample_index % 2 == 0:
            # Ajoute un essai de classe 0 avec bruit léger pour la variabilité
            samples.append(class_zero + 0.02 * rng.standard_normal(class_zero.shape))
            # Ajoute le label associé pour suivre la classe
            labels.append(0)
        else:
            # Ajoute un essai de classe 1 avec bruit léger pour la variabilité
            samples.append(class_one + 0.02 * rng.standard_normal(class_one.shape))
            # Ajoute le label associé pour suivre la classe
            labels.append(1)
    # Convertit la liste en tenseur (n_samples, n_channels, n_times)
    X = np.stack(samples)
    # Convertit les labels en tableau numpy pour compatibilité scikit-learn
    y = np.array(labels)
    # Retourne les données synthétiques prêtes pour le benchmark
    return X, y


def _evaluate_combination(
    feature_strategy: str, classifier: str, sfreq: float, X: np.ndarray, y: np.ndarray
) -> dict[str, float | str]:
    """Évalue une combinaison features/classifieur sur un split train/test."""

    # Construit la configuration de pipeline alignée avec la combinaison testée
    config = PipelineConfig(
        sfreq=sfreq,
        feature_strategy=feature_strategy,
        normalize_features=True,
        dim_method="pca",
        n_components=4,
        classifier=classifier,
        scaler="standard",
    )
    # Sépare les données en train/test pour obtenir une mesure d'accuracy robuste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    # Construit le pipeline complet pour la combinaison en cours
    pipeline = build_pipeline(config)
    # Mesure la durée d'entraînement pour comparer la complexité des modèles
    start_train = perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = perf_counter() - start_train
    # Mesure la latence moyenne de prédiction pour simuler un usage realtime
    start_predict = perf_counter()
    predictions = pipeline.predict(X_test)
    predict_duration = (perf_counter() - start_predict) / len(X_test)
    # Calcule l'accuracy pour quantifier la performance
    accuracy = float((predictions == y_test).mean())
    # Retourne les métriques collectées pour cette combinaison
    return {
        "feature_strategy": feature_strategy,
        "classifier": classifier,
        "accuracy": accuracy,
        "train_seconds": train_duration,
        "predict_seconds": predict_duration,
    }


def run_synthetic_benchmark(n_samples: int = 120, sfreq: float = 128.0) -> pd.DataFrame:
    """Construit un benchmark synthétique pour comparer features et classifieurs."""

    # Génère le dataset unique pour toutes les combinaisons afin d'assurer l'équité
    X, y = _generate_dataset(n_samples=n_samples, sfreq=sfreq)
    # Définit les combinaisons à explorer pour couvrir FFT/wavelet et classifieurs
    combinations = [
        ("fft", "lda"),
        ("fft", "logistic"),
        ("fft", "centroid"),
        ("wavelet", "lda"),
        ("wavelet", "logistic"),
        ("wavelet", "centroid"),
    ]
    # Accumule les résultats pour chaque combinaison évaluée
    results = []
    # Parcourt les combinaisons pour mesurer accuracy et latence
    for feature_strategy, classifier in combinations:
        # Évalue la combinaison courante sur le dataset généré
        results.append(
            _evaluate_combination(feature_strategy, classifier, sfreq=sfreq, X=X, y=y)
        )
    # Convertit les résultats en DataFrame pour faciliter l'export
    return pd.DataFrame(results)


def _save_reports(df: pd.DataFrame, output_dir: Path) -> None:
    """Sauvegarde les résultats de benchmark en JSON et Markdown."""

    # Garantit l'existence du répertoire de sortie demandé
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sérialise les résultats en JSON pour un usage machine
    json_path = output_dir / "benchmark_results.json"
    # Sauvegarde les données tabulaires en JSON pour conservation
    json_path.write_text(df.to_json(orient="records", indent=2))
    # Construit un tableau Markdown lisible pour la documentation versionnée
    markdown_lines = [
        "| features | classifier | accuracy | train_s | predict_s |",
        "|---|---|---|---|---|",
    ]
    # Parcourt les lignes pour formatter chaque résultat
    for _, row in df.iterrows():
        # Ajoute la ligne formattée avec des arrondis stables
        markdown_lines.append(
            "| {feature} | {classifier} | {accuracy:.3f} | {train:.4f} | "
            "{predict:.6f} |".format(
                feature=row["feature_strategy"],
                classifier=row["classifier"],
                accuracy=row["accuracy"],
                train=row["train_seconds"],
                predict=row["predict_seconds"],
            )
        )
    # Écrit le tableau dans un fichier Markdown pour la section docs/project
    markdown_path = output_dir / "benchmark_results.md"
    # Sauvegarde le contenu Markdown pour consultation humaine
    markdown_path.write_text("\n".join(markdown_lines))


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour exécuter le benchmark synthétique."""

    # Crée le parser avec description pour cadrer l'usage
    parser = argparse.ArgumentParser(
        description=(
            "Exécute un benchmark synthétique pour comparer FFT/wavelet "
            "et LDA/Logistic/centroid."
        )
    )
    # Ajoute une option pour personnaliser le nombre d'échantillons générés
    parser.add_argument(
        "--n-samples",
        type=int,
        default=120,
        help="Nombre d'échantillons synthétiques générés",
    )
    # Ajoute une option pour personnaliser la fréquence d'échantillonnage des signaux
    parser.add_argument(
        "--sfreq",
        type=float,
        default=128.0,
        help="Fréquence d'échantillonnage du dataset synthétique",
    )
    # Ajoute une option pour spécifier le répertoire de sortie des rapports
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/project"),
        help="Répertoire où sauvegarder les rapports JSON et Markdown",
    )
    # Retourne le parser configuré pour la fonction main
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI pour générer et sauvegarder le benchmark."""

    # Construit le parser avec les options disponibles
    parser = build_parser()
    # Parse les arguments fournis par l'utilisateur
    args = parser.parse_args(argv)
    # Exécute le benchmark synthétique selon les paramètres demandés
    df = run_synthetic_benchmark(n_samples=args.n_samples, sfreq=args.sfreq)
    # Sauvegarde les rapports dans le répertoire cible
    _save_reports(df, args.output_dir)
    # Retourne 0 pour signaler un succès standard
    return 0


if __name__ == "__main__":
    # Lance la fonction principale lorsque le script est exécuté directement
    raise SystemExit(main())
