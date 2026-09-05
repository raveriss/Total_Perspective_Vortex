# Matrice checklist → WBS → tests/commandes

| Item checklist TPV | WBS / livrable | Test ou commande reproductible |
| --- | --- | --- |
| Visualisation raw vs filtré ("Watch it for the plot") | 3.3.1–3.3.4 (visualisation filtrage) | `uv run --frozen python scripts/visualize_raw_filtered.py data/S001` ; `uv run --frozen pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Bande passante 8–40 Hz conservée | 3.1.1–3.1.3 (filtre) | `uv run --frozen pytest tests/test_preprocessing.py::test_apply_bandpass_filter_preserves_shape_and_stability` |
| Extraction de fréquences pertinentes motor imagery | 4.1.2–4.2.3 (features spectrales) | `uv run --frozen pytest tests/test_pipeline.py::test_pipeline_respects_input_and_output_shapes` |
| Réduction de dimension implémentée (PCA/CSP) | 5.2.1–5.2.4 (implémentation) | `uv run --frozen pytest tests/test_dimensionality.py::test_csp_returns_log_variances_and_orthogonality` |
| Intégration sklearn (BaseEstimator/TransformerMixin) | 5.3.1–5.3.4 (intégration pipeline) | `uv run --frozen pytest tests/test_pipeline.py::test_pipeline_pickling_roundtrip` |
| Mode train avec score affiché (validation croisée) | 6.3.1–6.3.4 (tests pipeline) & 7.1.x (entraînement global) | `uv run --frozen pytest tests/test_classifier.py::test_training_cli_main_covers_parser_and_paths` |
| Mode predict qui retourne l’ID de classe | 1.2.1–1.2.5 (CLI) & 6.2.x (classifieur) | `uv run --frozen pytest tests/test_classifier.py::test_predict_cli_main_covers_parser_and_report` |
| Temps réel < 2 s | 8.2.x–8.3.x (realtime) | `uv run --frozen pytest tests/test_realtime.py::test_realtime_latency_threshold_enforced` |
| Score imbriqué + run tenu hors train + paliers 75/81/87/90 % | 7.1.x, 7.4.2–7.4.4 ; TPV-036, TPV-043, TPV-045, TPV-759 | `uv run --frozen pytest tests/test_evaluation.py tests/test_experience_scores.py` |
| FBCSP causal + MIBIF 4–16 + ERD/CAR/Laplacien | 7.4.3 ; TPV-028, TPV-030, TPV-032, TPV-687, TPV-756, TPV-757 | `uv run --frozen pytest tests/test_dimensionality.py tests/test_mibif_spatial.py` |
| Campagnes figées 10 → 30 → 109 puis branche riemannienne | 7.4.3–7.4.4 ; TPV-043, TPV-048 | `make score-campaign-10`, `make score-campaign-30`, `make compute-mean-of-means`, `make score-riemannian-10` |
| Datasets additionnels correctement gérés | 2.1.x–2.3.x (parsing + labels) | `uv run --frozen pytest tests/test_preprocessing.py::test_verify_dataset_integrity_checks_hash_and_runs` |

Chaque ligne renvoie à la checklist officielle (`docs/total_perspective_vortex.en.checklist.pdf`), au WBS détaillé (`docs/project/wbs_tpv.md`) et à une preuve de vérification par test ou commande reproductible.
