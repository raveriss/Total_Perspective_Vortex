# Matrice checklist → WBS → tests/commandes

| Item checklist TPV | WBS / livrable | Test ou commande reproductible |
| --- | --- | --- |
| Visualisation raw vs filtré ("Watch it for the plot") | 3.3.1–3.3.4 (visualisation filtrage) | `poetry run python scripts/visualize_raw_filtered.py data/S001` ; `poetry run pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Bande passante 8–40 Hz conservée | 3.1.1–3.1.3 (filtre) | `poetry run pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Extraction de fréquences pertinentes motor imagery | 4.1.2–4.2.3 (features spectrales) | `poetry run pytest tests/test_pipeline.py::test_pipeline_respects_input_and_output_shapes` |
| Réduction de dimension implémentée (PCA/CSP) | 5.2.1–5.2.4 (implémentation) | `poetry run pytest tests/test_dimensionality.py::test_csp_returns_log_variances_and_orthogonality` |
| Intégration sklearn (BaseEstimator/TransformerMixin) | 5.3.1–5.3.4 (intégration pipeline) | `poetry run pytest tests/test_pipeline.py::test_pipeline_pickling_roundtrip` |
| Mode train avec score affiché (validation croisée) | 6.3.1–6.3.4 (tests pipeline) & 7.1.x (entraînement global) | `poetry run pytest tests/test_classifier.py::test_training_cli_main_covers_parser_and_paths` |
| Mode predict qui retourne l’ID de classe | 1.2.1–1.2.5 (CLI) & 6.2.x (classifieur) | `poetry run pytest tests/test_classifier.py::test_predict_cli_main_covers_parser_and_report` |
| Temps réel < 2 s | 8.2.x–8.3.x (realtime) | `poetry run pytest tests/test_realtime.py::test_realtime_latency_threshold_enforced` |
| Script score ≥ 75 % (moyenne par run) | 7.2.x (agrégation des scores) | `poetry run pytest tests/test_classifier.py::test_aggregate_scores_exports_files_and_thresholds` |
| Datasets additionnels correctement gérés | 2.1.x–2.3.x (parsing + labels) | `poetry run pytest tests/test_preprocessing.py::test_verify_dataset_integrity_checks_hash_and_runs` |

Chaque ligne renvoie à la checklist officielle (`docs/total_perspective_vortex.en.checklist.pdf`), au WBS détaillé (`docs/project/wbs_tpv.md`) et à une preuve de vérification par test ou commande reproductible.
