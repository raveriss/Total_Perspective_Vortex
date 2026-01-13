"""Visualisation WBS 3.3 : brut vs filtré Physionet."""

# Force un backend sans affichage pour les environnements CI et serveurs
import matplotlib

# Active Agg pour générer les PNG sans dépendre d'un écran
matplotlib.use("Agg")

# Importe argparse pour exposer les options CLI sujet/run/canaux
import argparse

# Importe json pour sérialiser la configuration accompagnant les figures
import json

# Importe dataclass pour encapsuler les paramètres de visualisation
from dataclasses import dataclass

# Importe pathlib pour gérer les chemins dataset et de sortie
from pathlib import Path

# Importe typing pour typer explicitement les séquences et tuples
from typing import Sequence, Tuple

# Importe pyplot pour tracer les figures comparatives
import matplotlib.pyplot as plt

# Importe numpy pour agréger les canaux par région
import numpy as np

# Importe Line2D pour construire une légende compacte par région
from matplotlib.lines import Line2D

# Importe BaseRaw pour typer précisément les enregistrements EEG
from mne.io import BaseRaw

# Importe le filtrage validé pour rester aligné avec preprocessing
from tpv.preprocessing import apply_bandpass_filter, load_physionet_raw

MAX_SUBJECTS_PREVIEW = 5

# Fixe l'ordre d'affichage des régions EEG principales
REGION_ORDER = ("Frontal", "Central", "Parietal", "Occipital", "Temporal", "Autres")

# Associe une couleur stable à chaque région pour guider la lecture
REGION_COLORS = {
    "Frontal": "#1f77b4",
    "Central": "#2ca02c",
    "Parietal": "#ff7f0e",
    "Occipital": "#9467bd",
    "Temporal": "#d62728",
    "Autres": "#7f7f7f",
}

# Limite le nombre de canaux détaillés pour éviter la surcharge visuelle
MAX_DETAIL_CHANNELS = 7

# Fixe la largeur de figure pour garantir la lisibilité des subplots
FIGURE_WIDTH = 12.0

# Fixe la hauteur par ligne pour garder des tracés équilibrés
FIGURE_ROW_HEIGHT = 2.4


# Normalise un nom de canal pour faciliter l'affectation régionale
def _normalize_channel_name(channel: str) -> str:
    """Retourne une version normalisée du nom de canal EEG."""

    # Supprime les espaces parasites pour éviter les faux négatifs
    normalized = channel.strip()
    # Retourne le nom en majuscules pour simplifier les préfixes
    return normalized.upper()


# Détermine la région EEG attendue à partir du nom de canal
def _infer_region_from_channel(channel: str) -> str:
    """Retourne une région EEG basée sur les préfixes 10-10."""

    # Normalise le nom de canal pour comparer les préfixes
    normalized = _normalize_channel_name(channel)
    # Capture les canaux fronto-centraux avant les frontaux purs
    if normalized.startswith(("FC", "C", "CP")):
        return "Central"
    # Capture les canaux frontaux sans ambiguité centrale
    if normalized.startswith(("FP", "AF", "F")):
        return "Frontal"
    # Capture les canaux pariétaux (PO/P)
    if normalized.startswith(("PO", "P")):
        return "Parietal"
    # Capture les canaux occipitaux (O)
    if normalized.startswith(("O",)):
        return "Occipital"
    # Capture les canaux temporaux (FT/TP/T)
    if normalized.startswith(("FT", "TP", "T")):
        return "Temporal"
    # Retourne Autres pour ne pas perdre d'informations
    return "Autres"


# Construit un mapping région -> indices de canaux pour l'agrégation
def _group_channels_by_region(ch_names: Sequence[str]) -> dict[str, list[int]]:
    """Retourne un dictionnaire des indices de canaux par région."""

    # Initialise le mapping avec l'ordre des régions pour stabilité
    grouped: dict[str, list[int]] = {region: [] for region in REGION_ORDER}
    # Parcourt chaque canal pour l'affecter à une région
    for idx, channel in enumerate(ch_names):
        # Détermine la région du canal courant
        region = _infer_region_from_channel(channel)
        # Ajoute l'index au groupe correspondant
        grouped.setdefault(region, []).append(idx)
    # Prépare un mapping filtré pour ignorer les régions vides
    non_empty: dict[str, list[int]] = {}
    # Conserve uniquement les régions qui contiennent des canaux
    for region, indices in grouped.items():
        # Ignore les régions sans canaux pour limiter les subplots
        if not indices:
            continue
        # Ajoute la région non vide au mapping final
        non_empty[region] = indices
    # Retourne les régions qui contiennent au moins un canal
    return non_empty


# Calcule la moyenne et l'écart-type d'un groupe de canaux
def _summarize_region_data(
    data: np.ndarray,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne la moyenne et l'écart-type par région."""

    # Sélectionne les canaux concernés pour l'agrégation
    region_data = data[np.asarray(indices)]
    # Calcule la moyenne temporelle pour réduire la charge visuelle
    mean_trace = region_data.mean(axis=0)
    # Calcule l'écart-type pour matérialiser la dispersion
    std_trace = region_data.std(axis=0)
    # Retourne les deux séries pour le tracé mean ± std
    return mean_trace, std_trace


# Fournit un récap rapide des sujets déjà présents sous data_root
def _describe_subjects(data_root: Path) -> str:
    """Retourne une ligne décrivant les sujets détectés sous data_root."""

    if not data_root.exists():
        return f"- Racine data absente : {data_root}"
    available_subjects = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if not available_subjects:
        return f"- Aucun sujet détecté dans {data_root}"
    preview = ", ".join(available_subjects[:MAX_SUBJECTS_PREVIEW])
    suffix = " …" if len(available_subjects) > MAX_SUBJECTS_PREVIEW else ""
    return f"- Sujets présents dans {data_root}: {preview}{suffix}"


# Fournit un récap rapide des runs disponibles pour le sujet demandé
def _describe_runs(subject_root: Path) -> str:
    """Retourne une ligne listant les runs EDF trouvés pour le sujet."""

    if not subject_root.exists():
        return f"- Répertoire sujet absent : {subject_root}"
    available_runs = sorted(p.stem for p in subject_root.glob("*.edf"))
    normalized_runs = []
    for run in available_runs:
        if run.startswith(subject_root.name):
            normalized_runs.append(run[len(subject_root.name) :])
        else:
            normalized_runs.append(run)
    if normalized_runs:
        return (
            f"- Runs disponibles pour {subject_root.name}: "
            f"{', '.join(normalized_runs)}"
        )
    if available_runs:
        return (
            f"- Runs disponibles pour {subject_root.name}: {', '.join(available_runs)}"
        )
    return f"- Aucun run EDF trouvé dans {subject_root}"


# Construit un message clair pour guider l'utilisateur lorsque le fichier est absent
def _format_missing_recording_message(recording_path: Path) -> str:
    """Assemble un message multi-lignes décrivant l'arborescence attendue."""

    data_root = recording_path.parent.parent
    subject_root = recording_path.parent
    subject = subject_root.name
    run_stem = recording_path.stem
    normalized_run = (
        run_stem[len(subject) :] if run_stem.startswith(subject) else run_stem
    )
    primary_candidate = subject_root / f"{subject}{normalized_run}.edf"
    secondary_candidate = subject_root / f"{normalized_run}.edf"
    candidates = {primary_candidate, secondary_candidate}
    return "\n".join(
        [
            f"Recording not found: {recording_path}",
            "- Chemins tentés : "
            + ", ".join(str(candidate) for candidate in sorted(candidates)),
            (
                f"- Structure attendue : {data_root}/<subject>/<subject><run>.edf "
                "(ex: S001/S001R03.edf) ou <subject>/<run>.edf si déjà préfixé"
            ),
            _describe_subjects(data_root),
            _describe_runs(subject_root),
            (
                "Téléchargez le dataset Physionet EEG Motor Movement/Imagery "
                "et pointez --data-root vers sa racine."
            ),
        ]
    )


# Regroupe les paramètres de visualisation pour limiter les arguments
@dataclass
class VisualizationConfig:
    """Conteneur des options CLI pour la visualisation brut/filtré."""

    # Définit la sélection facultative de canaux à tracer
    channels: Sequence[str] | None
    # Définit le répertoire de sortie des figures générées
    output_dir: Path
    # Choisit la méthode de filtrage alignée avec preprocessing
    filter_method: str
    # Spécifie la bande fréquentielle appliquée au signal
    freq_band: Tuple[float, float]
    # Spécifie la durée de padding pour réduire les artefacts de bord
    pad_duration: float
    # Définit un titre optionnel pour annoter la figure
    title: str | None


# Centralise la construction du parseur pour harmoniser l'interface CLI
def build_parser() -> argparse.ArgumentParser:
    """Crée le parseur d'arguments pour la visualisation brut/filtré."""

    # Initialise le parseur avec une description alignée WBS 3.3
    parser = argparse.ArgumentParser(
        description=(
            "Charge un run Physionet, applique le filtre 8-40 Hz, "
            "et enregistre un plot brut vs filtré."
        )
    )
    # Ajoute l'argument sujet pour cibler un répertoire data/<subject>
    parser.add_argument(
        "subject",
        help="Identifiant du sujet ex: S001",
    )
    # Ajoute l'argument run pour choisir le fichier EDF au sein du sujet
    parser.add_argument(
        "run",
        help="Identifiant du run ex: R01",
    )
    # Ajoute la racine dataset pour autoriser les chemins personnalisés
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine locale des données Physionet",
    )
    # Ajoute le répertoire de sortie pour ranger les figures générées
    parser.add_argument(
        "--output-dir",
        default="docs/viz",
        help="Répertoire de sauvegarde PNG",
    )
    # Ajoute la liste de canaux facultative pour filtrer l'affichage
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Sous-ensemble de canaux à tracer (ex: C3 C4 Cz)",
    )
    # Ajoute le choix du filtre pour comparer FIR/IIR sans modifier preprocessing
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir"],
        default="fir",
        help="Famille de filtre à appliquer",
    )
    # Ajoute la bande de fréquences basse/haute pour l'exploration rapide
    parser.add_argument(
        "--freq-band",
        nargs=2,
        type=float,
        default=(8.0, 40.0),
        metavar=("LOW", "HIGH"),
        help="Bande passe-bas/passe-haut en Hz",
    )
    # Ajoute la durée de padding pour contrôler les effets de bord
    parser.add_argument(
        "--pad-duration",
        type=float,
        default=0.5,
        help="Durée de padding réfléchissant en secondes",
    )
    # Ajoute le titre optionnel pour annoter le graphique sauvegardé
    parser.add_argument(
        "--title",
        default=None,
        help="Titre personnalisé pour la figure",
    )
    # Retourne le parseur configuré pour réutilisation en test
    return parser


# Construit le chemin EDF attendu pour un sujet et un run donnés
def build_recording_path(
    data_root: Path,
    subject: str,
    run: str,
) -> Path:
    """Retourne le chemin EDF data/<subject>/<subject><run>.edf si présent."""

    # Normalise la racine pour éviter les surprises sur les chemins relatifs
    normalized_root = Path(data_root).expanduser().resolve()
    subject_root = normalized_root / subject
    run_stem = Path(run).stem
    candidates: list[Path] = []
    if run_stem.startswith(subject):
        candidates.append(subject_root / f"{run_stem}.edf")
    else:
        candidates.append(subject_root / f"{subject}{run_stem}.edf")
        candidates.append(subject_root / f"{run_stem}.edf")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Retourne la préférence avec préfixe sujet pour matcher Physionet
    return candidates[0]


# Charge un enregistrement Physionet complet avec métadonnées associées
def load_recording(
    recording_path: Path,
) -> Tuple[BaseRaw, dict]:
    """Charge un Raw EDF et retourne le Raw plus ses métadonnées."""

    # Vérifie l'existence du fichier pour fournir un message clair au CLI
    if not recording_path.exists():
        # Lève une erreur explicite pour guider l'utilisateur sur data
        raise FileNotFoundError(_format_missing_recording_message(recording_path))
    # Appelle le loader validé pour conserver les contraintes 2.2.x
    raw, metadata = load_physionet_raw(recording_path)
    # Ajoute les clés sujet/run pour faciliter la sérialisation jointe au plot
    subject = recording_path.parent.name
    run_label = recording_path.stem
    if run_label.startswith(subject):
        run_label = run_label[len(subject) :]
    metadata.update(
        {
            "subject": subject,
            "run": run_label,
        }
    )
    # Retourne le Raw et les métadonnées enrichies pour la visualisation
    return raw, metadata


# Sélectionne éventuellement les canaux avant filtrage pour accélérer les plots
def pick_channels(
    raw: BaseRaw,
    channels: Sequence[str] | None,
) -> BaseRaw:
    """Retourne un Raw limité aux canaux demandés si fournis."""

    # Court-circuite si aucun filtre de canaux n'est demandé
    if channels is None:
        # Retourne directement le Raw pour limiter les copies inutiles
        return raw
    # Vérifie que tous les canaux demandés existent dans l'enregistrement
    missing = [ch for ch in channels if ch not in raw.ch_names]
    # Refuse silencieux pour préserver la traçabilité des erreurs utilisateur
    if missing:
        # Lève un message explicite listant les canaux absents
        raise ValueError(f"Unknown channels: {', '.join(missing)}")
    # Copie le Raw pour éviter de modifier l'objet d'origine
    picked = raw.copy().pick(channels)
    # Retourne le Raw restreint pour le filtrage et le tracé
    return picked


# Applique le filtre passe-bande avec les paramètres CLI
def filter_recording(
    raw: BaseRaw,
    method: str,
    freq_band: Tuple[float, float],
    pad_duration: float,
) -> BaseRaw:
    """Applique le filtre bande-passante 8–40 Hz sur le Raw fourni."""

    # Délègue au helper preprocessing pour rester cohérent avec WBS 3.1
    filtered = apply_bandpass_filter(
        raw,
        method=method,
        freq_band=freq_band,
        pad_duration=pad_duration,
    )
    # Retourne le Raw filtré pour la visualisation
    return filtered


# Trace le signal brut et filtré sur deux sous-graphiques alignés
def plot_raw_vs_filtered(
    raw: BaseRaw,
    filtered: BaseRaw,
    output_path: Path,
    config: VisualizationConfig,
    metadata: dict,
) -> Path:
    """Enregistre une figure structurée par régions (brut vs filtré)."""

    # Crée le répertoire parent pour éviter les erreurs d'écriture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Récupère les données temporelles pour la synchronisation des deux tracés
    times = raw.times
    # Extrait les données brutes pour afficher chaque canal en amplitude microV
    raw_data = raw.get_data()
    # Extrait les données filtrées pour comparaison directe
    filtered_data = filtered.get_data()
    # Regroupe les canaux par région pour structurer la lecture
    region_map = _group_channels_by_region(raw.ch_names)
    # Liste les régions présentes dans l'enregistrement
    regions = list(region_map.keys())
    # Garantit au moins une région pour éviter subplots(0, 2)
    if not regions:
        regions = ["Autres"]
        region_map = {"Autres": list(range(len(raw.ch_names)))}
    # Calcule le nombre de lignes de subplots (une par région)
    nrows = len(regions)
    # Calcule la taille de figure proportionnelle au nombre de régions
    fig_height = FIGURE_ROW_HEIGHT * nrows
    # Crée une grille 2 colonnes pour comparer brut vs filtré
    fig, axes = plt.subplots(
        nrows,
        2,
        sharex=True,
        sharey="row",
        figsize=(FIGURE_WIDTH, fig_height),
    )
    # Force axes en matrice 2D pour simplifier l'indexation
    axes = np.atleast_2d(axes)
    # Extrait la bande de fréquence depuis la configuration
    freq_band = config.freq_band
    # Prépare le titre principal pour guider l'utilisateur
    title_text = config.title or (
        f"EEG – Comparaison brut vs filtré ({freq_band[0]:g}-{freq_band[1]:g} Hz)"
    )
    # Applique le titre principal pour expliciter la comparaison
    fig.suptitle(title_text)
    # Prépare un sous-titre informatif avec contexte sujet/run
    subtitle_text = (
        f"Sujet {metadata['subject']} • Run {metadata['run']} • "
        f"{len(raw.ch_names)} canaux"
    )
    # Ajoute le sous-titre au-dessus des subplots
    fig.text(0.5, 0.94, subtitle_text, ha="center", fontsize="small")

    # Parcourt les régions pour tracer les comparaisons alignées
    for row_index, region in enumerate(regions):
        # Récupère les indices de canaux pour la région courante
        indices = region_map[region]
        # Récupère la couleur dédiée pour l'identité régionale
        color = REGION_COLORS.get(region, REGION_COLORS["Autres"])
        # Sélectionne l'axe brut pour la région
        raw_axis = axes[row_index, 0]
        # Sélectionne l'axe filtré pour la région
        filtered_axis = axes[row_index, 1]
        # Calcule la moyenne et l'écart-type du signal brut
        raw_mean, raw_std = _summarize_region_data(raw_data, indices)
        # Calcule la moyenne et l'écart-type du signal filtré
        filtered_mean, filtered_std = _summarize_region_data(filtered_data, indices)
        # Trace les canaux bruts détaillés uniquement si peu nombreux
        if len(indices) <= MAX_DETAIL_CHANNELS:
            # Parcourt les canaux pour un contexte local discret
            for idx in indices:
                # Trace chaque canal brut avec faible opacité
                raw_axis.plot(
                    times,
                    raw_data[idx],
                    color=color,
                    alpha=0.2,
                    linewidth=0.6,
                )
        # Trace la moyenne brute pour la lisibilité immédiate
        raw_axis.plot(times, raw_mean, color=color, linewidth=1.6)
        # Ajoute l'enveloppe brute pour matérialiser la dispersion
        raw_axis.fill_between(
            times,
            raw_mean - raw_std,
            raw_mean + raw_std,
            color=color,
            alpha=0.2,
        )
        # Trace les canaux filtrés détaillés uniquement si peu nombreux
        if len(indices) <= MAX_DETAIL_CHANNELS:
            # Parcourt les canaux pour un contexte local discret
            for idx in indices:
                # Trace chaque canal filtré avec faible opacité
                filtered_axis.plot(
                    times,
                    filtered_data[idx],
                    color=color,
                    alpha=0.2,
                    linewidth=0.6,
                )
        # Trace la moyenne filtrée pour la lecture rapide
        filtered_axis.plot(times, filtered_mean, color=color, linewidth=1.6)
        # Ajoute l'enveloppe filtrée pour matérialiser la dispersion
        filtered_axis.fill_between(
            times,
            filtered_mean - filtered_std,
            filtered_mean + filtered_std,
            color=color,
            alpha=0.2,
        )
        # Ajoute un titre contextualisé à la ligne brute
        raw_axis.set_title(f"Brut — {region}")
        # Ajoute un titre contextualisé à la ligne filtrée
        filtered_axis.set_title(
            f"Filtré {freq_band[0]:g}-{freq_band[1]:g} Hz — {region}"
        )
        # Ajoute le label Y pour rappeler l'unité en brut
        raw_axis.set_ylabel("Amplitude (a.u.)")
        # Ajoute le label Y pour rappeler l'unité en filtré
        filtered_axis.set_ylabel("Amplitude filtrée (a.u.)")
        # Ajoute une grille verticale légère pour l'alignement visuel
        raw_axis.grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.6)
        # Ajoute une grille verticale légère pour l'alignement visuel
        filtered_axis.grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.6)
        # Ajoute le label X uniquement sur la dernière ligne
        if row_index == nrows - 1:
            # Ajoute le label temps pour l'axe brut
            raw_axis.set_xlabel("Temps (s)")
            # Ajoute le label temps pour l'axe filtré
            filtered_axis.set_xlabel("Temps (s)")

    # Prépare les handles de légende par région
    # Initialise la liste des handles pour la légende
    legend_handles: list[Line2D] = []
    # Construit un handle de couleur pour chaque région affichée
    for region in regions:
        # Déclare la couleur associée à la région
        region_color = REGION_COLORS.get(region, REGION_COLORS["Autres"])
        # Ajoute un handle de légende pour la région
        legend_handles.append(Line2D([0], [0], color=region_color))
    # Construit une légende compacte pour les régions visibles
    fig.legend(
        handles=legend_handles,
        labels=regions,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(regions), 3),
        fontsize="small",
        frameon=False,
    )

    # Ajuste le layout en gardant de la place pour titre et légende
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    # Sauvegarde la figure au chemin fourni
    fig.savefig(output_path)
    # Ferme la figure pour libérer la mémoire en batch
    plt.close(fig)
    # Écrit un sidecar JSON pour archiver la configuration de traçage
    output_path.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    # Retourne le chemin de la figure pour les tests et logs CLI
    return output_path


# Orchestration complète du chargement, filtrage et traçage
def visualize_run(
    data_root: Path,
    subject: str,
    run: str,
    config: VisualizationConfig,
) -> Path:
    """Pipeline complet de visualisation brut/filtré pour un run Physionet."""

    # Construit le chemin du fichier EDF cible
    recording_path = build_recording_path(data_root, subject, run)
    # Charge l'enregistrement et récupère ses métadonnées enrichies
    raw, metadata = load_recording(recording_path)
    # Limite éventuellement aux canaux sélectionnés pour focaliser la figure
    picked_raw = pick_channels(raw, config.channels)
    # Applique le filtre avec les paramètres demandés
    filtered = filter_recording(
        picked_raw,
        method=config.filter_method,
        freq_band=config.freq_band,
        pad_duration=config.pad_duration,
    )
    # Détermine le chemin de sortie en nommant selon sujet/run
    output_path = Path(config.output_dir) / f"raw_vs_filtered_{subject}_{run}.png"
    # Enregistre la figure et retourne le chemin généré
    return plot_raw_vs_filtered(
        picked_raw,
        filtered,
        output_path,
        config,
        metadata,
    )


# Point d'entrée CLI conforme aux instructions README
def main() -> None:
    """Parse les arguments et lance la visualisation du run demandé."""

    # Construit et parse les arguments fournis en CLI
    parser = build_parser()
    # Récupère la structure d'arguments utilisateur
    args = parser.parse_args()
    # Déclenche la visualisation avec les paramètres fournis
    try:
        # Instancie la configuration pour respecter la limite d'arguments
        config = VisualizationConfig(
            channels=args.channels,
            output_dir=Path(args.output_dir),
            filter_method=args.filter_method,
            freq_band=(float(args.freq_band[0]), float(args.freq_band[1])),
            pad_duration=float(args.pad_duration),
            title=args.title,
        )
        visualize_run(
            data_root=Path(args.data_root),
            subject=args.subject,
            run=args.run,
            config=config,
        )
    except Exception as error:  # noqa: BLE001
        # Affiche l'erreur pour que la CI capture la cause exacte
        print(error)
        # Termine avec un code non nul pour signaler l'échec
        raise SystemExit(1) from error


# Active l'exécution du main uniquement en invocation directe
if __name__ == "__main__":
    main()
