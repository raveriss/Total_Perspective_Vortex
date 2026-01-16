# Backlog prioritaire aligné sur le WBS

## État de l'implémentation
- Parsing et préparation Physionet réalisés : scripts `fetch_physionet.py`, `prepare_physionet.py` et `sync_dataset.py` couvrent le Lot 2, avec tests de contrôle d'intégrité et de synchronisation.
- Pipeline offline fonctionnelle : filtrage 8–40 Hz, contrôle qualité des epochs, extraction PSD/bandes Welch, réduction PCA/CSP, classifieurs LDA/Logistic/SVM et scripts `train.py`/`predict.py` déjà raccordés à `mybci.py`.
- Visualisation et tests : script `visualize_raw_filtered.py` opérationnel, couverture de tests sur preprocessing, features, réduction dimensionnelle, pipeline, CLI et import dataset.

## Prochaines étapes (alignées sur le WBS)
1. **Stratégie Train/Val/Test et score global** (WBS 7.1, 7.4)
   - Documenter dans `docs/project/` la politique de splits sujet/run, seeds et métriques exigées (≥75 %).
   - Ajouter un script/commande pour agréger les accuracies par run, sujet et globale à partir des artefacts sauvegardés.
   - Étendre les tests `test_classifier.py`/`test_realtime.py` avec un scénario d'agrégation sur dataset jouet.

2. **Consolider l'API temps réel** (WBS 8.1–8.3)
   - Implémenter `src/tpv/realtime.py` pour lire un flux fenêtré, appliquer le pipeline existant et mesurer la latence < 2 s.
   - Ajouter un mode CLI (ex. `mybci.py realtime`) et un buffer/queue pour lisser les prédictions.
   - Créer des tests synthétiques validant l'absence de fuite temporelle et la stabilité des délais.

3. **Renforcer la traçabilité des artefacts et rapports** (WBS 7.2, 7.3, 10.3)
   - Persister aux côtés des modèles un manifeste (JSON/CSV) listant dataset, hyperparams, scores CV et version du code.
   - Générer automatiquement un rapport de prédiction par sujet/run (confusion matrix ou tableau d'accuracy) dans `artifacts/`.
   - Ajouter un test d'intégration qui vérifie la présence et la cohérence de ces rapports après `train` puis `predict` sur données jouets.

4. **Documentation de défense et traçabilité checklist** (WBS 10.1, 10.5)
   - Compléter `README.md` et `docs/project/` avec la matrice “Checklist ↔ tâches WBS” et les choix de features/filtrage.
   - Ajouter un index `docs/index.md` reliant WBS, Gantt, roadmap et Murphy map.
   - Vérifier que chaque item critique de la checklist TPV pointe vers un test ou un script reproductible.

5. **Bonus ciblés pour robustesse/science** (WBS 11.1, 11.2)
   - Implémenter l'option wavelets dans `features.py` et un classifieur maison (ex. réseau linéaire léger) sélectionnables via CLI.
   - Comparer FFT vs wavelets et LDA/Logistic vs classifieur bonus sur un benchmark synthétique documenté.
   - Reporter les gains/coûts dans un tableau de résultats versionné dans `docs/project/`.
