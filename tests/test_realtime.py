"""Tests de cohérence des artefacts pour un usage temps-réel."""

# Préserve builtins pour déléguer max sans modifier la sémantique
import builtins

# Préserve numpy pour construire des données EEG synthétiques
# Offre la lecture des artefacts sauvegardés par joblib
# Fournit time pour simuler une latence volontaire
import time

# Fournit deque pour construire un buffer identique à l'exécution
from collections import deque

# Importe Path pour configurer les répertoires temporaires
from pathlib import Path

import joblib
import numpy as np

# Importe pytest pour vérifier les exceptions attendues
import pytest

# Importe la logique de prédiction pour vérifier les matrices W
# Importe la logique d'entraînement pour orchestrer la sauvegarde
from scripts import predict as predict_cli
from scripts import train as train_cli

# Importe la boucle temps réel pour vérifier les métriques de streaming
from tpv import realtime

# Importe la configuration et l'entrée pour vérifier le routage
from tpv.realtime import RealtimeConfig, run_realtime_inference

# Définit une latence minimale attendue pour les tests de performance
MIN_EXPECTED_LATENCY = 0.009
# Définit la latence maximale autorisée pour les fenêtres streaming
MAX_ALLOWED_LATENCY = 2.0
# Fige la fenêtre par défaut pour empêcher les valeurs magiques
DEFAULT_WINDOW_SIZE = 50
# Fige le pas par défaut pour empêcher les valeurs magiques
DEFAULT_STEP_SIZE = 25
# Fige la taille de buffer par défaut pour empêcher les valeurs magiques
DEFAULT_BUFFER_SIZE = 3
# Fige la latence par défaut pour empêcher les valeurs magiques
DEFAULT_MAX_LATENCY = 2.0
# Fige la fréquence par défaut pour empêcher les valeurs magiques
DEFAULT_SFREQ = 50.0
# Fige le set de libellés par défaut pour la CLI
DEFAULT_LABEL_SET = "t1-t2"
# Fige l'étiquette par défaut pour la classe zéro
DEFAULT_LABEL_ZERO = "T1"
# Fige l'étiquette par défaut pour la classe un
DEFAULT_LABEL_ONE = "T2"
# Fige la fenêtre personnalisée pour documenter les attentes CLI
CUSTOM_WINDOW_SIZE = 128
# Fige le pas personnalisé pour documenter les attentes CLI
CUSTOM_STEP_SIZE = 32
# Fige la taille de buffer personnalisée pour documenter les attentes CLI
CUSTOM_BUFFER_SIZE = 5
# Fige la latence personnalisée pour documenter les attentes CLI
CUSTOM_MAX_LATENCY = 1.5
# Fige la fréquence personnalisée pour documenter les attentes CLI
CUSTOM_SFREQ = 250.0
# Fige le set de libellés personnalisé pour la CLI
CUSTOM_LABEL_SET = "left-right"
# Fige l'étiquette personnalisée pour la classe zéro
CUSTOM_LABEL_ZERO = "main gauche"
# Fige l'étiquette personnalisée pour la classe un
CUSTOM_LABEL_ONE = "main droite"

# Fige une latence exacte pour couvrir le cas frontière du SLA
EQUAL_LATENCY = 0.5
# Fige une latence dépassant le SLA avec une valeur représentable en binaire
OVER_LATENCY = 0.625
# Fige la latence maximale stricte pour tester l'égalité et le dépassement
STRICT_MAX_LATENCY = 0.5
# Fige un instant de base pour stabiliser les timestamps relatifs
PERF_BASE_TIME = 10.0
# Fige un instant de démarrage d'inférence pour contrôler inference_started_at
PERF_INFERENCE_START = 11.0


# Vérifie que le parser realtime expose toutes les options attendues
def test_realtime_build_parser_defines_cli_contract():
    # Construit le parser pour inspecter la configuration CLI
    parser = realtime.build_parser()
    # Indexe les actions par destination pour simplifier les vérifications
    actions = {action.dest: action for action in parser._actions}
    # Vérifie que la description reflète l'usage streaming documenté
    assert parser.description == "Applique un modèle entraîné sur un flux fenêtré"
    # Vérifie que l'argument subject reste positionnel et documenté
    assert actions["subject"].help == "Identifiant du sujet (ex: S001)"
    # Vérifie que l'argument run reste positionnel et documenté
    assert actions["run"].help == "Identifiant du run (ex: R01)"
    # Vérifie que le répertoire de données accepte des chemins Path
    assert actions["data_dir"].type is Path
    # Vérifie que le répertoire de données pointe vers data par défaut
    assert actions["data_dir"].default == Path("data")
    # Vérifie que l'option --data-dir est bien exposée dans les flags
    assert actions["data_dir"].option_strings == ["--data-dir"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["data_dir"].help == "Répertoire racine contenant les fichiers numpy"
    # Vérifie que le répertoire d'artefacts accepte des chemins Path
    assert actions["artifacts_dir"].type is Path
    # Vérifie que le répertoire d'artefacts cible artifacts par défaut
    assert actions["artifacts_dir"].default == Path("artifacts")
    # Vérifie que l'option --artifacts-dir est bien exposée dans les flags
    assert actions["artifacts_dir"].option_strings == ["--artifacts-dir"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["artifacts_dir"].help == "Répertoire racine où lire le modèle"
    # Vérifie que la taille de fenêtre reste un entier avec valeur 50
    assert actions["window_size"].type is int
    assert actions["window_size"].default == DEFAULT_WINDOW_SIZE
    # Vérifie que l'option --window-size est présente dans le parser
    assert actions["window_size"].option_strings == ["--window-size"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["window_size"].help == "Taille de fenêtre glissante en échantillons"
    # Vérifie que le pas de glissement reste un entier avec valeur 25
    assert actions["step_size"].type is int
    assert actions["step_size"].default == DEFAULT_STEP_SIZE
    # Vérifie que l'option --step-size est présente dans le parser
    assert actions["step_size"].option_strings == ["--step-size"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["step_size"].help == "Pas entre deux fenêtres successives"
    # Vérifie que la taille du buffer reste un entier avec valeur 3
    assert actions["buffer_size"].type is int
    assert actions["buffer_size"].default == DEFAULT_BUFFER_SIZE
    # Vérifie que l'option --buffer-size est présente dans le parser
    assert actions["buffer_size"].option_strings == ["--buffer-size"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["buffer_size"].help == "Taille du buffer pour lisser les prédictions"
    # Vérifie que la latence maximale reste un flottant avec valeur 2.0
    assert actions["max_latency"].type is float
    assert actions["max_latency"].default == DEFAULT_MAX_LATENCY
    # Vérifie que l'option --max-latency est présente dans le parser
    assert actions["max_latency"].option_strings == ["--max-latency"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["max_latency"].help == "Latence maximale tolérée en secondes"
    # Vérifie que la fréquence d'échantillonnage reste un flottant 50.0
    assert actions["sfreq"].type is float
    assert actions["sfreq"].default == DEFAULT_SFREQ
    # Vérifie que l'option --sfreq est présente dans le parser
    assert actions["sfreq"].option_strings == ["--sfreq"]
    # Verrouille le texte d'aide pour éviter une rupture de contrat CLI
    assert actions["sfreq"].help == "Fréquence d'échantillonnage utilisée pour l'offset"
    # Vérifie que le set de libellés est exposé par défaut
    assert actions["label_set"].default == DEFAULT_LABEL_SET
    # Vérifie que l'option --label-set est exposée par le parser
    assert actions["label_set"].option_strings == ["--label-set"]
    # Verrouille le texte d'aide pour le set de libellés
    assert actions["label_set"].help == "Type de libellés à afficher (T1/T2, A/B, etc.)"
    # Normalise la liste de choix pour éviter les None du type hints
    label_set_choices = actions["label_set"].choices or []
    # Vérifie que les choix proposés incluent le mapping t1-t2
    assert "t1-t2" in label_set_choices
    # Vérifie que les choix proposés incluent le mapping left-right
    assert "left-right" in label_set_choices
    # Vérifie que les choix proposés incluent le mapping fists-feet
    assert "fists-feet" in label_set_choices
    # Vérifie que l'étiquette zéro est paramétrable via la CLI
    assert actions["label_zero"].default is None
    # Vérifie que l'option --label-zero est exposée par le parser
    assert actions["label_zero"].option_strings == ["--label-zero"]
    # Verrouille le texte d'aide pour la classe zéro
    assert actions["label_zero"].help == "Étiquette affichée pour la classe 0"
    # Vérifie que l'étiquette un est paramétrable via la CLI
    assert actions["label_one"].default is None
    # Vérifie que l'option --label-one est exposée par le parser
    assert actions["label_one"].option_strings == ["--label-one"]
    # Verrouille le texte d'aide pour la classe un
    assert actions["label_one"].help == "Étiquette affichée pour la classe 1"


# Vérifie que le chargement des données retourne exactement les tableaux écrits
def test_load_data_reads_numpy_files(tmp_path):
    # Construit un tableau de features identifiable pour l'appel
    features = np.array([[1, 2], [3, 4]])
    # Construit un tableau de labels pour vérifier la seconde sortie
    labels = np.array([1, 0])
    # Sauvegarde les features au format numpy dans un fichier temporaire
    features_path = tmp_path / "features.npy"
    np.save(features_path, features)
    # Sauvegarde les labels au format numpy dans un fichier temporaire
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    # Charge les fichiers via la fonction dédiée pour vérifier la lecture
    loaded_features, loaded_labels = realtime._load_data(features_path, labels_path)
    # Vérifie que les features rechargées conservent la structure écrite
    assert np.array_equal(loaded_features, features)
    # Vérifie que les labels rechargés conservent la structure écrite
    assert np.array_equal(loaded_labels, labels)


# Vérifie que le chargement signale clairement les fichiers manquants
def test_load_data_raises_with_missing_files(tmp_path):
    # Définit un chemin de features inexistant pour le scénario d'erreur
    features_path = tmp_path / "features.npy"
    # Définit un chemin de labels inexistant pour le scénario d'erreur
    labels_path = tmp_path / "labels.npy"
    # Capture l'exception attendue pour inspecter le message utilisateur
    with pytest.raises(FileNotFoundError) as exc_info:
        # Appelle le loader avec des chemins absents pour déclencher l'erreur
        realtime._load_data(features_path, labels_path)
    # Convertit l'exception en texte pour vérifier le contenu UX
    message = str(exc_info.value)
    # Vérifie que le code d'erreur couvre l'absence totale de fichiers
    assert "ERROR[TPV-RT-001]" in message
    # Vérifie que le message signale l'absence de fichiers numpy attendus
    assert "fichiers features et labels manquants" in message
    # Vérifie que le chemin des features est indiqué dans le message
    assert str(features_path) in message
    # Vérifie que le chemin des labels est indiqué dans le message
    assert str(labels_path) in message
    # Vérifie que la commande de train suggérée est explicitement mentionnée
    assert "python mybci.py" in message
    # Vérifie que l'action data-dir est proposée pour un chemin alternatif
    assert "--data-dir" in message


# Vérifie que le chargement signale un code dédié aux features manquants
def test_load_data_raises_with_missing_features_only(tmp_path):
    # Définit un chemin de features inexistant pour le scénario d'erreur
    features_path = tmp_path / "features.npy"
    # Définit un chemin de labels existant pour isoler le cas features manquants
    labels_path = tmp_path / "labels.npy"
    # Sauvegarde un tableau de labels pour déclencher l'erreur ciblée
    np.save(labels_path, np.array([0, 1]))
    # Capture l'exception attendue pour inspecter le message utilisateur
    with pytest.raises(FileNotFoundError) as exc_info:
        # Appelle le loader avec un seul fichier manquant pour l'erreur
        realtime._load_data(features_path, labels_path)
    # Convertit l'exception en texte pour vérifier le contenu UX
    message = str(exc_info.value)
    # Vérifie que le code d'erreur cible l'absence de features
    assert "ERROR[TPV-RT-002]" in message
    # Vérifie que le message mentionne explicitement les features manquants
    assert "fichiers features manquants" in message


# Vérifie que le chargement signale un code dédié aux labels manquants
def test_load_data_raises_with_missing_labels_only(tmp_path):
    # Définit un chemin de features existant pour isoler le cas labels manquants
    features_path = tmp_path / "features.npy"
    # Définit un chemin de labels inexistant pour le scénario d'erreur
    labels_path = tmp_path / "labels.npy"
    # Sauvegarde un tableau de features pour déclencher l'erreur ciblée
    np.save(features_path, np.array([[1, 2], [3, 4]]))
    # Capture l'exception attendue pour inspecter le message utilisateur
    with pytest.raises(FileNotFoundError) as exc_info:
        # Appelle le loader avec un seul fichier manquant pour l'erreur
        realtime._load_data(features_path, labels_path)
    # Convertit l'exception en texte pour vérifier le contenu UX
    message = str(exc_info.value)
    # Vérifie que le code d'erreur cible l'absence de labels
    assert "ERROR[TPV-RT-003]" in message
    # Vérifie que le message mentionne explicitement les labels manquants
    assert "fichiers labels manquants" in message


# Vérifie que le parser interprète correctement des arguments explicites
def test_realtime_parser_parses_custom_cli_values():
    # Construit le parser pour interpréter des valeurs non par défaut
    parser = realtime.build_parser()
    # Parse une ligne de commande complète pour inspecter les conversions
    args = parser.parse_args(
        [
            "S42",
            "R99",
            "--data-dir",
            "custom_data",
            "--artifacts-dir",
            "custom_artifacts",
            "--window-size",
            "128",
            "--step-size",
            "32",
            "--buffer-size",
            "5",
            "--max-latency",
            "1.5",
            "--sfreq",
            "250.0",
            "--label-set",
            "left-right",
        ]
    )
    # Vérifie la conversion automatique en Path pour data_dir
    assert args.data_dir == Path("custom_data")
    # Vérifie la conversion automatique en Path pour artifacts_dir
    assert args.artifacts_dir == Path("custom_artifacts")
    # Vérifie la prise en compte de la fenêtre personnalisée
    assert args.window_size == CUSTOM_WINDOW_SIZE
    # Vérifie la prise en compte du pas personnalisé
    assert args.step_size == CUSTOM_STEP_SIZE
    # Vérifie la prise en compte de la taille de buffer personnalisée
    assert args.buffer_size == CUSTOM_BUFFER_SIZE
    # Vérifie la prise en compte de la latence maximale personnalisée
    assert args.max_latency == CUSTOM_MAX_LATENCY
    # Vérifie la prise en compte de la fréquence d'échantillonnage personnalisée
    assert args.sfreq == CUSTOM_SFREQ
    # Vérifie la prise en compte du set de libellés personnalisé
    assert args.label_set == CUSTOM_LABEL_SET
    # Vérifie que l'override est absent lorsque non fourni
    assert args.label_zero is None
    # Vérifie que l'override est absent lorsque non fourni
    assert args.label_one is None


# Vérifie que la matrice W sauvegardée est cohérente avec la pipeline
def test_w_matrix_matches_pipeline(tmp_path):
    # Fige la fréquence d'échantillonnage pour aligner les features FFT
    sfreq = 120.0
    # Génère un signal simple pour deux classes opposées
    t = np.arange(0, 1, 1 / sfreq)
    # Crée un essai pour la classe 0 avec énergie sur le premier canal
    class_a = np.stack([np.sin(2 * np.pi * 6 * t), np.zeros_like(t)])
    # Crée un essai pour la classe 1 avec énergie sur le second canal
    class_b = np.stack([np.zeros_like(t), np.sin(2 * np.pi * 10 * t)])
    # Assemble les essais dans un tenseur (essai, canal, temps)
    X = np.stack([class_a, class_b, class_a * 1.1, class_b * 1.1])
    # Construit les labels associés pour chaque essai
    y = np.array([0, 1, 0, 1])
    # Construit le répertoire des données pour le sujet S03
    data_dir = tmp_path / "data" / "S03"
    # Assure la création du répertoire cible avant sauvegarde
    data_dir.mkdir(parents=True)
    # Sauvegarde les features au format attendu par la CLI
    np.save(data_dir / "R03_X.npy", X)
    # Sauvegarde les labels au format attendu par la CLI
    np.save(data_dir / "R03_y.npy", y)
    # Construit le répertoire d'artefacts isolé pour le test
    artifacts_dir = tmp_path / "artifacts"
    # Construit la configuration alignée sur la CLI pour l'entraînement
    config = train_cli.PipelineConfig(
        sfreq=sfreq,
        feature_strategy="fft",
        normalize_features=False,
        dim_method="pca",
        n_components=2,
        classifier="lda",
        scaler=None,
    )
    # Regroupe les paramètres d'entraînement dans une requête dédiée
    request = train_cli.TrainingRequest(
        subject="S03",
        run="R03",
        pipeline_config=config,
        data_dir=tmp_path / "data",
        artifacts_dir=artifacts_dir,
    )
    # Entraîne une pipeline pour alimenter la prédiction
    train_cli.run_training(request)
    # Charge la pipeline complète pour comparer la matrice interne
    pipeline = joblib.load(artifacts_dir / "S03" / "R03" / "model.joblib")
    # Charge la matrice W sauvegardée par le réducteur
    stored_matrix = joblib.load(artifacts_dir / "S03" / "R03" / "w_matrix.joblib")
    # Recharge le réducteur pour simuler une utilisation temps-réel
    loaded_reducer = predict_cli._load_w_matrix(
        artifacts_dir / "S03" / "R03" / "w_matrix.joblib"
    )
    # Vérifie que les matrices W sont bien présentes avant comparaison
    assert loaded_reducer.w_matrix is not None
    assert pipeline.named_steps["dimensionality"].w_matrix is not None
    # Vérifie que la matrice W rechargée correspond à celle de la pipeline
    assert np.allclose(
        loaded_reducer.w_matrix, pipeline.named_steps["dimensionality"].w_matrix
    )
    # Vérifie que la matrice W stockée contient la même structure
    assert np.allclose(stored_matrix["w_matrix"], loaded_reducer.w_matrix)


# Simule une pipeline déterministe pour mesurer la latence
class _FakePipeline:
    """Pipeline synthétique pour contrôler la latence et les sorties."""

    # Enregistre la latence artificielle à appliquer sur predict
    def __init__(self, delay: float, outputs: list[int]):
        # Conserve le délai simulé pour représenter le temps de calcul
        self.delay = delay
        # Conserve la séquence de prédictions à jouer pendant le test
        self.outputs = outputs

    # Simule l'appel predict en ajoutant une latence volontaire
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
        # Ajoute une pause contrôlée pour mesurer la latence rapportée
        time.sleep(self.delay)
        # Sélectionne la prochaine prédiction ou recycle la dernière valeur
        if self.outputs:
            # Retire la première valeur pour suivre l'ordre d'appel
            value = self.outputs.pop(0)
        else:
            # Réutilise la dernière prédiction pour compléter la séquence
            value = 0
        # Retourne un tableau numpy compatible avec les attentes sklearn
        return np.array([value])


# Vérifie que la latence enregistrée reflète le délai imposé
def test_realtime_latency_metrics():
    # Instancie une pipeline factice avec une latence de 10 ms
    pipeline = _FakePipeline(delay=0.01, outputs=[0] * 9)
    # Construit un flux synthétique pour générer neuf fenêtres
    stream = np.zeros((2, 100))
    # Exécute la boucle temps réel avec une fenêtre de 20 échantillons
    result = run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=RealtimeConfig(
            window_size=20,
            step_size=10,
            buffer_size=2,
            # Fige la latence max pour garantir le respect du SLA de 2 s
            max_latency=2.0,
            sfreq=50.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )
    # Vérifie que chaque latence dépasse le délai simulé
    assert all(event.latency >= MIN_EXPECTED_LATENCY for event in result["events"])
    # Vérifie que la moyenne et le maximum sont cohérents avec les mesures
    assert result["latency_max"] >= result["latency_mean"] > 0.0


# Vérifie que la boucle s'arrête si la latence dépasse le SLA
def test_realtime_latency_threshold_enforced():
    # Instancie une pipeline factice qui dépasse le délai autorisé
    pipeline = _FakePipeline(delay=2.5, outputs=[0])
    # Construit un flux minimal pour déclencher une seule fenêtre
    stream = np.zeros((1, 4))
    # Vérifie que le SLA déclenche une exception explicite
    with pytest.raises(TimeoutError):
        # Lance la boucle realtime avec une latence maximale de 2 s
        run_realtime_inference(
            pipeline=pipeline,
            stream=stream,
            config=RealtimeConfig(
                window_size=4,
                step_size=2,
                buffer_size=1,
                # Fige la latence maximale à la valeur autorisée
                max_latency=MAX_ALLOWED_LATENCY,
                sfreq=5.0,
                label_zero=DEFAULT_LABEL_ZERO,
                label_one=DEFAULT_LABEL_ONE,
            ),
        )


# Vérifie que l'égalité au SLA ne déclenche pas de TimeoutError
def test_realtime_latency_threshold_allows_equal_boundary_and_relative_start(
    monkeypatch,
):
    # Définit une pipeline minimale pour éviter les dépendances temporelles réelles
    class _StubPipeline:
        """Pipeline déterministe pour contrôler les valeurs de perf_counter."""

        # Retourne une prédiction fixe pour déclencher une seule itération
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
            return np.array([0])

    # Prépare une séquence perf_counter couvrant base, start, end
    times = iter(
        [
            PERF_BASE_TIME,
            PERF_INFERENCE_START,
            PERF_INFERENCE_START + EQUAL_LATENCY,
        ]
    )
    # Remplace perf_counter pour rendre la latence exactement égale au SLA
    monkeypatch.setattr(realtime.time, "perf_counter", lambda: next(times))

    # Construit un flux minimal générant exactement une fenêtre
    stream = np.zeros((1, 4))
    # Exécute la boucle en fixant max_latency au cas frontière
    result = run_realtime_inference(
        pipeline=_StubPipeline(),
        stream=stream,
        config=RealtimeConfig(
            window_size=4,
            step_size=4,
            buffer_size=1,
            # Fige le SLA pour couvrir le cas latency == max_latency
            max_latency=STRICT_MAX_LATENCY,
            sfreq=4.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )

    # Vérifie qu'un événement a bien été produit sans exception
    assert len(result["events"]) == 1
    # Vérifie que la latence correspond exactement au cas frontière
    assert result["events"][0].latency == EQUAL_LATENCY
    # Vérifie que le timestamp est relatif à base_time et non absolu
    assert result["events"][0].inference_started_at == (
        PERF_INFERENCE_START - PERF_BASE_TIME
    )


# Vérifie que le TimeoutError conserve un message explicite et stable
def test_realtime_timeout_error_includes_latency_message(monkeypatch):
    # Définit une pipeline minimale pour isoler le formatage de l'exception
    class _StubPipeline:
        """Pipeline déterministe pour déclencher un dépassement de SLA."""

        # Retourne une prédiction fixe pour atteindre le contrôle de latence
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
            return np.array([0])

    # Prépare une séquence perf_counter couvrant base, start, end
    times = iter(
        [
            PERF_BASE_TIME,
            PERF_INFERENCE_START,
            PERF_INFERENCE_START + OVER_LATENCY,
        ]
    )
    # Remplace perf_counter pour rendre la latence strictement supérieure au SLA
    monkeypatch.setattr(realtime.time, "perf_counter", lambda: next(times))

    # Construit un flux minimal générant exactement une fenêtre
    stream = np.zeros((1, 4))
    # Valide que l'exception expose un message utile pour le diagnostic
    with pytest.raises(TimeoutError) as excinfo:
        # Exécute la boucle avec un SLA inférieur à la latence simulée
        run_realtime_inference(
            pipeline=_StubPipeline(),
            stream=stream,
            config=RealtimeConfig(
                window_size=4,
                step_size=4,
                buffer_size=1,
                # Fixe le SLA pour déclencher la branche d'erreur
                max_latency=STRICT_MAX_LATENCY,
                sfreq=4.0,
                label_zero=DEFAULT_LABEL_ZERO,
                label_one=DEFAULT_LABEL_ONE,
            ),
        )

    # Verrouille le message formaté pour éviter les régressions silencieuses
    assert str(excinfo.value) == "Latence 0.625s dépasse 0.500s"


# Vérifie que les agrégats valent zéro lorsque le flux ne produit aucun événement
def test_realtime_empty_stream_returns_zero_latencies():
    # Définit une pipeline factice qui ne doit jamais être appelée
    class _NeverCalledPipeline:
        """Pipeline de garde pour vérifier l'absence d'appels predict."""

        # Signale un bug si une prédiction est demandée sans fenêtre
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
            raise AssertionError("predict must not be called for empty event stream")

    # Construit un flux trop court pour générer une fenêtre complète
    stream = np.zeros((1, 3))
    # Exécute la boucle avec une fenêtre supérieure à la longueur du flux
    result = run_realtime_inference(
        pipeline=_NeverCalledPipeline(),
        stream=stream,
        config=RealtimeConfig(
            window_size=4,
            step_size=2,
            buffer_size=1,
            # Définit un SLA standard pour éviter des effets de bord
            max_latency=MAX_ALLOWED_LATENCY,
            sfreq=4.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )

    # Vérifie qu'aucun événement n'a été produit
    assert result["events"] == []
    # Vérifie que la moyenne vaut zéro en absence de fenêtres
    assert result["latency_mean"] == 0.0
    # Vérifie que le maximum vaut zéro en absence de fenêtres
    assert result["latency_max"] == 0.0


# Vérifie que les fenêtres sont traitées dans l'ordre chronologique
def test_realtime_time_ordering():
    # Instancie une pipeline factice sans latence pour isoler l'ordre
    pipeline = _FakePipeline(delay=0.0, outputs=[1] * 5)
    # Construit un flux synthétique correspondant à cinq fenêtres
    stream = np.zeros((1, 40))
    # Exécute la boucle temps réel avec un pas constant
    result = run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=RealtimeConfig(
            window_size=20,
            step_size=5,
            buffer_size=2,
            # Définit la latence maximale pour sécuriser la boucle
            max_latency=2.0,
            sfreq=20.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )
    # Extrait les offsets temporels pour vérifier la progression
    offsets = [event.window_offset for event in result["events"]]
    # Vérifie que chaque offset est strictement croissant
    assert offsets == sorted(offsets)
    # Extrait les timestamps relatifs des appels predict
    starts = [event.inference_started_at for event in result["events"]]
    # Vérifie que les timestamps respectent l'ordre de traitement
    assert starts == sorted(starts)


# Vérifie que le fenêtrage n'inclut pas d'échantillon futur
def test_realtime_windowing_avoids_future_leakage():
    # Conserve les fenêtres reçues pour contrôler la découpe temporelle
    captured: list[np.ndarray] = []

    # Définit une pipeline qui enregistre les fenêtres pour inspection
    class _RecordingPipeline:
        """Pipeline minimale enregistrant chaque entrée reçue."""

        # Enregistre la fenêtre et retourne le dernier échantillon
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
            # Clone la fenêtre pour éviter les effets de bord
            captured.append(np.copy(X[0, 0]))
            # Retourne la valeur finale pour vérifier la découpe
            return np.array([int(X[0, 0, -1])])

    # Construit un flux monotone pour tracer l'indexation temporelle
    stream = np.arange(12, dtype=float).reshape(1, 12)
    # Exécute la boucle realtime avec un pas inférieur à la fenêtre
    result = run_realtime_inference(
        pipeline=_RecordingPipeline(),
        stream=stream,
        config=RealtimeConfig(
            window_size=4,
            step_size=3,
            buffer_size=2,
            # Fige la latence maximale pour respecter le SLA
            max_latency=MAX_ALLOWED_LATENCY,
            sfreq=6.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )
    # Décrit les fenêtres attendues pour exclure tout échantillon futur
    expected_windows = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 5.0, 6.0]),
        np.array([6.0, 7.0, 8.0, 9.0]),
    ]
    # Vérifie que chaque fenêtre observée correspond à la découpe prévue
    assert all(
        np.array_equal(window, expected)
        # Force strict=True pour éviter toute fuite temporelle silencieuse
        for window, expected in zip(captured, expected_windows, strict=True)
    )
    # Vérifie que les prédictions reflètent le dernier point de chaque fenêtre
    assert [event.raw_prediction for event in result["events"]] == [3, 6, 9]


# Vérifie que le buffer de lissage stabilise les prédictions
def test_realtime_smoothed_predictions():
    # Instancie une pipeline factice avec une séquence oscillante
    pipeline = _FakePipeline(delay=0.0, outputs=[0, 1, 1, 0, 1])
    # Construit un flux synthétique pour générer cinq fenêtres
    stream = np.zeros((1, 12))
    # Exécute la boucle temps réel avec un buffer de taille trois
    result = run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=RealtimeConfig(
            window_size=4,
            step_size=2,
            buffer_size=3,
            # Définit la latence maximale attendue pour les prédictions
            max_latency=2.0,
            sfreq=10.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )
    # Extrait les prédictions lissées pour les comparer à l'attendu
    smoothed = [event.smoothed_prediction for event in result["events"]]
    # Vérifie que le lissage maintient la majorité récente
    assert smoothed == [0, 0, 1, 1, 1]


# Vérifie que le lissage incrémente bien de une par occurrence observée
def test_smooth_prediction_counts_increment_by_one(monkeypatch):
    # Construit un buffer contrôlé pour rendre le comptage observable
    buffer = deque([1, 1, 0], maxlen=3)
    # Fige les comptes attendus pour bloquer les offsets et les increments doubles
    expected_counts = {1: 2, 0: 1}

    # Capture max builtin pour déléguer sans récursion via le patch
    builtin_max = builtins.max

    # Définit un espion pour observer les valeurs produites par key(k)
    def spy_max(iterable, *args, **kwargs):
        # Récupère la fonction key afin de lire counts[k] sans accéder au dict
        key = kwargs.get("key")
        # Vérifie que la key est présente pour éviter un faux positif
        assert key is not None
        # Reconstruit les comptes observés via la key pour chaque classe
        observed_counts = {item: key(item) for item in iterable}
        # Vérifie que les comptes correspondent exactement aux occurrences
        assert observed_counts == expected_counts
        # Délègue à max builtin pour conserver le comportement nominal
        return builtin_max(iterable, *args, **kwargs)

    # Patch max builtin car _smooth_prediction appelle max(...) directement
    monkeypatch.setattr(realtime, "max", spy_max, raising=False)

    # Exécute le lissage pour déclencher max(key=...) avec le comptage calculé
    majority = realtime._smooth_prediction(buffer)
    # Vérifie la classe majoritaire attendue pour garder le test cohérent
    assert majority == 1


# Vérifie que le libellé retourne les classes attendues pour l'UX
def test_label_prediction_returns_configured_labels_and_fallback():
    # Construit une configuration simple pour piloter le mapping des labels
    config = RealtimeConfig(
        # Fixe la taille de fenêtre pour le test de mapping
        window_size=4,
        # Fixe le pas de fenêtre pour un cas minimal
        step_size=2,
        # Fixe la taille du buffer pour respecter le constructeur
        buffer_size=2,
        # Fixe la latence maximale pour satisfaire le constructeur
        max_latency=2.0,
        # Fixe la fréquence d'échantillonnage pour le constructeur
        sfreq=10.0,
        # Fournit l'étiquette explicite pour la classe zéro
        label_zero=DEFAULT_LABEL_ZERO,
        # Fournit l'étiquette explicite pour la classe un
        label_one=DEFAULT_LABEL_ONE,
    )

    # Vérifie le libellé explicite pour la classe zéro
    assert realtime._label_prediction(0, config) == DEFAULT_LABEL_ZERO
    # Vérifie le libellé explicite pour la classe un
    assert realtime._label_prediction(1, config) == DEFAULT_LABEL_ONE
    # Vérifie le libellé de repli pour une classe inconnue
    assert realtime._label_prediction(2, config) == "classe 2"


# Vérifie que les libellés sont résolus selon le set choisi
def test_resolve_label_pair_uses_label_set_defaults_and_overrides():
    # Résout les libellés par défaut pour le set left-right
    left_zero, left_one = realtime._resolve_label_pair(
        # Spécifie le set de libellés pour main gauche/droite
        "left-right",
        # Ne fournit pas d'override pour la classe zéro
        None,
        # Ne fournit pas d'override pour la classe un
        None,
    )
    # Vérifie que la classe zéro suit le set choisi
    assert left_zero == "main gauche"
    # Vérifie que la classe un suit le set choisi
    assert left_one == "main droite"

    # Résout les libellés avec overrides explicites
    override_zero, override_one = realtime._resolve_label_pair(
        # Conserve le set t1-t2 pour établir un contexte
        "t1-t2",
        # Force l'override explicite pour la classe zéro
        "main gauche",
        # Force l'override explicite pour la classe un
        "main droite",
    )
    # Vérifie que l'override écrase le set par défaut
    assert override_zero == "main gauche"
    # Vérifie que l'override écrase le set par défaut
    assert override_one == "main droite"


# Vérifie que la pipeline factice retourne une valeur de secours sans outputs
def test_realtime_pipeline_default_prediction():
    # Instancie une pipeline avec une seule prédiction fournie
    pipeline = _FakePipeline(delay=0.0, outputs=[1])
    # Construit un flux minimal générant deux fenêtres
    stream = np.zeros((1, 4))
    # Exécute la boucle temps réel pour épuiser les outputs prévus
    result = run_realtime_inference(
        pipeline=pipeline,
        stream=stream,
        config=RealtimeConfig(
            window_size=2,
            step_size=2,
            buffer_size=2,
            # Définit la latence maximale pour respecter le SLA
            max_latency=2.0,
            sfreq=10.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )

    # Vérifie que la prédiction de repli est bien utilisée après épuisement
    assert [event.raw_prediction for event in result["events"]] == [1, 0]


# Vérifie que la session realtime charge les données et invoque le modèle
def test_run_realtime_session_streams_saved_data(monkeypatch, tmp_path):
    # Conserve le chemin transmis à load_pipeline pour validation
    captured: dict[str, str] = {}

    # Définit une pipeline factice pour contrôler les prédictions
    class _StubPipeline:
        """Pipeline minimale pour simuler une prédiction constante."""

        # Retourne systématiquement la classe zéro pour chaque fenêtre
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - API sklearn
            return np.array([0])

    # Conserve le chemin du modèle demandé avant d'invoquer la pipeline
    def fake_load(path: str) -> _StubPipeline:
        captured["model_path"] = path
        return _StubPipeline()

    # Remplace load_pipeline pour éviter des accès disque inutiles
    monkeypatch.setattr(realtime, "load_pipeline", fake_load)

    # Construit un flux synthétique avec deux essais concaténables
    X = np.stack([np.ones((1, 4)), np.full((1, 4), 2.0)])
    # Crée des labels alignés avec la forme attendue
    y = np.array([0, 1])
    # Prépare l'arborescence data/subject pour respecter la CLI
    data_dir = tmp_path / "data"
    # Prépare l'arborescence artifacts/subject pour simuler le modèle
    artifacts_dir = tmp_path / "artifacts"
    # Crée les répertoires nécessaires pour les sauvegardes
    (data_dir / "S55").mkdir(parents=True)
    (artifacts_dir / "S55" / "R02").mkdir(parents=True)
    # Sauvegarde les features et labels au format numpy attendu
    np.save(data_dir / "S55" / "R02_X.npy", X)
    np.save(data_dir / "S55" / "R02_y.npy", y)

    # Lance la session realtime pour parcourir le flux sauvegardé
    result = realtime.run_realtime_session(
        subject="S55",
        run="R02",
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        config=RealtimeConfig(
            window_size=4,
            step_size=4,
            buffer_size=1,
            # Définit la latence maximale pour sécuriser la simulation
            max_latency=2.0,
            sfreq=10.0,
            label_zero=DEFAULT_LABEL_ZERO,
            label_one=DEFAULT_LABEL_ONE,
        ),
    )

    # Vérifie que la pipeline chargée provient du chemin attendu
    assert captured["model_path"].endswith(
        str(Path("artifacts") / "S55" / "R02" / "model.joblib")
    )
    # Vérifie que le flux a généré deux événements séquentiels
    assert [event.window_index for event in result["events"]] == [0, 1]
    # Vérifie que les offsets reflètent la concaténation des essais
    assert [event.window_offset for event in result["events"]] == [0.0, 0.4]


# Vérifie que l'exécutable realtime parse les options et délègue la session
def test_realtime_main_invokes_session(monkeypatch, tmp_path):
    # Conserve les arguments reçus par la session simulée
    captured: dict[str, object] = {}

    # Simule run_realtime_session pour éviter un chargement réel
    def fake_session(subject, run, data_dir, artifacts_dir, config):
        captured["subject"] = subject
        captured["run"] = run
        captured["data_dir"] = data_dir
        captured["artifacts_dir"] = artifacts_dir
        captured["config"] = config
        return {"events": [], "latency_mean": 0.0, "latency_max": 0.0}

    # Remplace la fonction pour capturer les paramètres construits
    monkeypatch.setattr(realtime, "run_realtime_session", fake_session)

    # Construit des chemins temporaires pour éviter l'usage du repo local
    data_dir = tmp_path / "input"
    artifacts_dir = tmp_path / "models"

    # Exécute le main avec des paramètres explicites
    exit_code = realtime.main(
        [
            "S77",
            "R03",
            "--data-dir",
            str(data_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--window-size",
            "6",
            "--step-size",
            "3",
            "--buffer-size",
            "2",
            "--sfreq",
            "25",
            "--label-set",
            "left-right",
        ]
    )

    # Vérifie que l'exécution renvoie un code de succès explicite
    assert exit_code == 0
    # Vérifie que la session simulée a reçu les bons arguments
    assert captured == {
        "subject": "S77",
        "run": "R03",
        "data_dir": data_dir,
        "artifacts_dir": artifacts_dir,
        "config": RealtimeConfig(
            window_size=6,
            step_size=3,
            buffer_size=2,
            # Spécifie la latence maximale attendue pour chaque fenêtre
            max_latency=2.0,
            sfreq=25.0,
            # Vérifie que le set left-right est résolu en labels explicites
            label_zero=CUSTOM_LABEL_ZERO,
            # Vérifie que le set left-right est résolu en labels explicites
            label_one=CUSTOM_LABEL_ONE,
        ),
    }
