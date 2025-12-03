"""Tests de cohérence des artefacts pour un usage temps-réel."""

# Préserve numpy pour construire des données EEG synthétiques
# Offre la lecture des artefacts sauvegardés par joblib
# Fournit time pour simuler une latence volontaire
import time

import joblib
import numpy as np

# Importe Path pour configurer les répertoires temporaires
from pathlib import Path

# Importe la logique de prédiction pour vérifier les matrices W
# Importe la logique d'entraînement pour orchestrer la sauvegarde
from scripts import predict as predict_cli
from scripts import train as train_cli

# Importe la boucle temps réel pour vérifier les métriques de streaming
import tpv.realtime as realtime
from tpv.realtime import RealtimeConfig, run_realtime_inference

# Définit une latence minimale attendue pour les tests de performance
MIN_EXPECTED_LATENCY = 0.009


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
            sfreq=50.0,
        ),
    )
    # Vérifie que chaque latence dépasse le délai simulé
    assert all(event.latency >= MIN_EXPECTED_LATENCY for event in result["events"])
    # Vérifie que la moyenne et le maximum sont cohérents avec les mesures
    assert result["latency_max"] >= result["latency_mean"] > 0.0


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
            sfreq=20.0,
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
            sfreq=10.0,
        ),
    )
    # Extrait les prédictions lissées pour les comparer à l'attendu
    smoothed = [event.smoothed_prediction for event in result["events"]]
    # Vérifie que le lissage maintient la majorité récente
    assert smoothed == [0, 0, 1, 1, 1]


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
            sfreq=10.0,
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
            sfreq=10.0,
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
            sfreq=25.0,
        ),
    }
