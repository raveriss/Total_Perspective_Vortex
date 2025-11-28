# Backlog prioritaire à implémenter

Les éléments ci-dessous convertissent les manques identifiés en tâches exécutables. Ils sont classés par ordre logique d'implémentation et doivent toutes être réalisées pour atteindre les exigences BCI définies dans le dépôt.

1. **Parsing EEG (Physionet) et structuration des events dans `preprocessing.py`**
   - Charger les fichiers bruts avec MNE, extraire les métadonnées des runs et mapper les labels en epochs prêts à traiter.
   - Valider la cohérence des dimensions channels × time et des marqueurs d'événements.

2. **Filtrage passe-bande 8–40 Hz et gestion des artefacts**
   - Implémenter le filtrage de base (band-pass, notch si besoin) et les routines de nettoyage avancé pour éliminer artefacts musculaires/ligne 50-60 Hz.
   - Ajouter des paramètres configurables pour l'ordre du filtre et les fréquences de coupure.

3. **Script de visualisation brut vs filtré `scripts/visualize_raw_filtered.py`**
   - Générer des tracés comparant les signaux d'entrée et ceux après prétraitement pour un run donné.
   - Sauvegarder les figures dans `docs/assets/` avec titres et légendes normalisées.

4. **Extraction de features fréquentielles dans `features.py`**
   - Concevoir une fonction `extract_features` produisant des descripteurs PSD/bandes fréquentielles adaptées à l'imagerie motrice.
   - Paramétrer le choix des bandes (theta/alpha/beta) et la fenêtre de calcul (Welch ou similaire).

5. **Réduction dimensionnelle maison (CSP/PCA/ICA) dans `dimensionality.py`**
   - Implémenter les décompositions sans utiliser les versions prêtes de sklearn/MNE, en construisant la matrice de projection W.
   - Documenter les hypothèses numériques (régularisation, tolérance) et ajouter des tests de stabilité.

6. **Pipeline sklearn complet dans `pipeline.py`**
   - Chaîner preprocessing, extraction de features, réduction dimensionnelle et classifieur via `sklearn.pipeline.Pipeline` avec des transformers maison.
   - Exposer une interface de configuration unique pour ajuster chaque étape.

7. **Implémentation des classifieurs (LDA/Logistic/SVM) dans `classifier.py`**
   - Fournir des wrappers cohérents avec le pipeline, avec réglage des hyperparamètres par défaut adaptés aux EEG.
   - Garantir la compatibilité avec la validation croisée et l'export des modèles entraînés.

8. **Script d’entraînement `scripts/train.py` avec cross-validation et sauvegarde des artefacts**
   - Mettre en place le chargement du dataset, la construction du pipeline, la cross-validation et la persistance des modèles/paramètres.
   - Reporter l’accuracy moyenne et enregistrer les métriques par run et par sujet.

9. **Script de prédiction `scripts/predict.py`**
   - Charger les artefacts entraînés, ingérer de nouvelles données et calculer l’accuracy sur un jeu de test distinct.
   - Exposer une CLI simple pour sélectionner le modèle et la source de données.

10. **Flux temps réel dans `realtime.py`**
    - Simuler la lecture progressive du flux EEG, appliquer le pipeline et publier chaque prédiction en moins de 2 secondes après réception du chunk.
    - Intégrer une queue/buffer pour lisser la latence et journaliser les délais.
