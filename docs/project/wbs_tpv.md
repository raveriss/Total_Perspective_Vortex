## 1. Lot 1 – Source & Repo (Source & Repo, CLI)

### 1.1 Initialisation du dépôt

* **1.1.1** Créer le dépôt Git `Total_Perspective_Vortex`
* **1.1.2** Mettre en place la structure `src/tpv`, `tests`, `docs`, `docs/risk`
* **1.1.3** Ajouter `pyproject.toml` (Poetry) avec dépendances MNE, sklearn, numpy, scipy, matplotlib
* **1.1.4** Ajouter `.gitignore` (datasets EEG, caches, artefacts modèles)
* **1.1.5** Créer `LICENSE` et squelette `README.md`

_Risque Murphy (section 1.1) : à lier à TPV-XXX… (Phase: Source & Repo)_

### 1.2 CLI & script principal

* **1.2.1** Définir les modes obligatoires `train` et `predict` (cf. sujet TPV)
* **1.2.2** Concevoir l’interface CLI `mybci.py` (arguments: sujet, run, mode)
* **1.2.3** Assurer la compatibilité avec `src/tpv/train.py` et `src/tpv/predict.py`
* **1.2.4** Ajouter commandes Makefile `make train`, `make predict`
* **1.2.5** Tester CLI sur un environnement propre (machine neuve)

_Risque Murphy (section 1.2) : à lier à TPV-XXX… (Phase: CLI / Source & Repo)_

### 1.3 Qualité du dépôt

* **1.3.1** Intégrer `.pre-commit-config.yaml` (ruff, black, isort, mypy, pytest)
* **1.3.2** Configurer GitHub Actions (CI: lint + tests + coverage)
* **1.3.3** Ajouter Codecov (upload de rapport coverage)
* **1.3.4** Vérifier qu’aucun dataset n’est versionné par erreur
* **1.3.5** Vérifier que le dépôt ne contient **que le programme Python** (scripts + modules) et **aucun dataset**, conformément aux règles de turn-in

* **1.3.6** Vérifier ✓ dans la checklist TPV les items liés à la structure du dépôt, au respect du format de rendu et à l’absence de données brutes

_Risque Murphy (section 1.3) : à lier à TPV-XXX… (Phase: Source & Repo / Robustesse)_

---

## 2. Lot 2 – Parsing EEG (Parsing EEG, Events & labels)

### 2.1 Gestion des datasets Physionet

* **2.1.1** Documenter les datasets EEG requis (lien Physionet, format)
* **2.1.2** Définir la structure locale `data/` (non versionnée)
* **2.1.3** Créer script de téléchargement / copie des données (si autorisé)
* **2.1.4** Vérifier l’intégrité (taille, hash, nb de sujets/runs attendus)

_Risque Murphy (section 2.1) : à lier à TPV-XXX… (Phase: Parsing EEG)_

### 2.2 Parsing avec MNE (`preprocessing.py`)

* **2.2.1** Charger les fichiers EEG avec MNE (format adapté: EDF/BDF/etc.)
* **2.2.2** Extraire les méta-données (sampling rate, channels, montage 10–20)
* **2.2.3** Gérer les canaux défectueux / manquants
* **2.2.4** Définir la structure interne des objets MNE (Raw, Epochs, Events)
* **2.2.5** Mapper les labels de tâches A/B sur les events (`Events & labels`)

* **2.2.6** Vérifier ✓ dans la checklist TPV les items liés à la cohérence des events, des labels et de la structure des données d’entrée

_Risque Murphy (section 2.2) : à lier à TPV-XXX… (Phase: Parsing EEG / Events & labels)_

### 2.3 Validation des events & labels

* **2.3.1** Vérifier la cohérence entre symboles expérimentaux et labels classes
* **2.3.2** Gérer les segments invalides (artefacts massifs, segments incomplets)
* **2.3.3** Générer un rapport de parsing (nb d’epochs par classe et par sujet)
* **2.3.4** Ajouter des tests `test_preprocessing.py` sur un petit dataset factice
* **2.3.5** Vérifier ✓ dans la checklist TPV les items relatifs à la qualité des labels, au déséquilibre de classes et à la traçabilité des sujets/runs

_Risque Murphy (section 2.3) : à lier à TPV-XXX… (Phase: Events & labels / Robustesse)_

---

## 3. Lot 3 – Filtrage & Nettoyage (Filtrage, Robustesse)

### 3.1 Filtrage de base

* **3.1.1** Implémenter le filtre passe-bande 8–40 Hz (motor imagery)
* **3.1.2** Tester filtre FIR vs IIR et choisir un compromis latence/stabilité
* **3.1.3** Gérer les effets de bord (edge effects) sur les segments fenêtrés
* **3.1.4** Documenter les paramètres (ordre filtre, fréquence coupure)

_Risque Murphy (section 3.1) : à lier à TPV-XXX… (Phase: Filtrage)_

### 3.2 Nettoyage avancé

* **3.2.1** Détecter et marquer les artefacts (clignements, muscles)
* **3.2.2** Décider des politiques de rejet / interpolation de canaux
* **3.2.3** Ajouter un module de normalisation par canal (z-score ou robust)
* **3.2.4** Créer tests unitaires pour les fonctions de filtrage

_Risque Murphy (section 3.2) : à lier à TPV-XXX… (Phase: Filtrage / Robustesse)_

### 3.3 Visualisation raw vs filtré

* **3.3.1** Implémenter `scripts/visualize_raw_filtered.py`
* **3.3.2** Générer les plots comparatifs raw/filtered pour un sujet/run type
* **3.3.3** Sauvegarder des figures dans `docs/viz/`
* **3.3.4** Ajouter un check dans la checklist de défense : “plots conformes à la vidéo de référence”
* **3.3.5** Vérifier ✓ dans la checklist TPV tous les items relatifs aux visualisations de signaux (brut vs filtré) et à la conformité avec les exemples fournis

_Risque Murphy (section 3.3) : à lier à TPV-XXX… (Phase: Viz / Filtrage)_

---

## 4. Lot 4 – Feature Engineering (Feature engineering)

### 4.1 Définition des features

* **4.1.1** Choisir les features spectrales (puissance par bande de fréquence)
* **4.1.2** Décider des bandes (theta/alpha/beta/gamma) pertinentes pour la tâche
* **4.1.3** Choisir la résolution fréquentielle (FFT, STFT, wavelets – bonus)
* **4.1.4** Fixer la taille de fenêtre temporelle et l’overlap (pour temps réel)

_Risque Murphy (section 4.1) : à lier à TPV-XXX… (Phase: Feature engineering)_

### 4.2 Extraction par canal

* **4.2.1** Implémenter la projection du signal en spectre (FFT / wavelets)
* **4.2.2** Calculer la puissance moyenne par bande et par canal
* **4.2.3** Concaténer les features (channels × bands) en vecteurs d’entrée
* **4.2.4** Normaliser les vecteurs de features (scaler sklearn)

_Risque Murphy (section 4.2) : à lier à TPV-XXX… (Phase: Feature engineering / Robustesse)_

### 4.3 Gestion des features dans `features.py`

* **4.3.1** Définir une API claire `extract_features(epochs) -> X, y`
* **4.3.2** Autoriser différentes stratégies de features (configurable dans `config.py`)
* **4.3.3** Ajouter tests unitaires `test_features.py`
* **4.3.4** Benchmarquer le coût de calcul (pour contrainte <2s)
* **4.3.5** Vérifier ✓ dans la checklist TPV les items relatifs à la définition des features et à leur cohérence avec la tâche BCI

_Risque Murphy (section 4.3) : à lier à TPV-XXX… (Phase: Feature engineering / Temps réel)_

---

## 5. Lot 5 – Réduction Dimensionnelle (Réduction dimension)

### 5.1 Choix de l’algorithme

* **5.1.1** Étudier PCA vs CSP vs autres (ICA/…)
* **5.1.2** Choisir l’algorithme principal pour TPV (ex: CSP pour EEG)
* **5.1.3** Définir le nombre de composantes (hyperparamètres)
* **5.1.4** Définir la stratégie d’entrainement (par sujet, global, par tâche)

_Risque Murphy (section 5.1) : à lier à TPV-XXX… (Phase: Réduction dimension)_

### 5.2 Implémentation dans `dimensionality.py`

* **5.2.1** Implémenter la construction de la matrice de covariance/from data
* **5.2.2** Calculer la décomposition (eigen/SVD via numpy/scipy)
* **5.2.3** Construire la matrice de projection W
* **5.2.4** Implémenter la transformation `transform(X)` vers l’espace réduit

_Risque Murphy (section 5.2) : à lier à TPV-XXX… (Phase: Réduction dimension / Robustesse num.)_

### 5.3 Intégration sklearn

* **5.3.1** Créer une classe `TPVDimReducer` héritant de `BaseEstimator`, `TransformerMixin`
* **5.3.2** Implémenter `fit(X, y)` et `transform(X)` conformes à sklearn
* **5.3.3** Gérer la sérialisation (pickle/joblib) de la matrice W
* **5.3.4** Ajouter tests unitaires `test_dimensionality.py` (cas simples, matrice jouet)
* **5.3.5** Vérifier ✓ dans la checklist TPV les items relatifs à l’intégration sklearn, à la compatibilité pipeline et à la reproductibilité des résultats

_Risque Murphy (section 5.3) : à lier à TPV-XXX… (Phase: Pipeline sklearn / Réduction dimension)_

---

## 6. Lot 6 – Pipeline sklearn & Classification (Pipeline sklearn, Classification)

### 6.1 Construction du pipeline

* **6.1.1** Définir le pipeline sklearn complet : preprocessing → features → dim-reduction → classifier
* **6.1.2** Créer `pipeline.py` avec fonction `build_pipeline(config)`
* **6.1.3** Prévoir l’injection de scalers (StandardScaler/RobustScaler)
* **6.1.4** Paramétrer le choix du classifieur (LDA, Logistic, SVM…)

_Risque Murphy (section 6.1) : à lier à TPV-XXX… (Phase: Pipeline sklearn)_

### 6.2 Implémentation de la classification (`classifier.py`)

* **6.2.1** Implémenter une LDA baseline
* **6.2.2** Ajouter une option Ridge/Logistic regression
* **6.2.3** (Option) Ajouter un SVM linéaire / RBF pour benchmark
* **6.2.4** Gérer les probabilités de classe et la décision finale

_Risque Murphy (section 6.2) : à lier à TPV-XXX… (Phase: Classification)_

### 6.3 Tests pipeline

* **6.3.1** Écrire `test_pipeline.py` (pipelines jouets, X synthétique)
* **6.3.2** Tester la compatibilité avec `cross_val_score` (pas de fuite de données)
* **6.3.3** Vérifier que le pipeline est picklable
* **6.3.4** Mesurer le temps d’inférence par batch d’epochs
* **6.3.5** Vérifier ✓ dans la checklist TPV les items liés au pipeline global, à la non-fuite de labels et à la stabilité des scores

_Risque Murphy (section 6.3) : à lier à TPV-XXX… (Phase: Pipeline sklearn / Robustesse)_

---

## 7. Lot 7 – Train / Validation / Test (Train/Val/Test, Score global)

### 7.1 Stratégie de split

* **7.1.1** Définir la stratégie Train/Val/Test (par sujet, par runs)
* **7.1.2** Gérer le non-overfitting (splits figés, random_state)
* **7.1.3** Documenter la stratégie dans `docs/project/` (pour la défense)

_Risque Murphy (section 7.1) : à lier à TPV-XXX… (Phase: Train/Val/Test)_

### 7.2 Script d’entraînement (`train.py`)

* **7.2.1** Implémenter la CLI `mybci.py subject run train`
* **7.2.2** Charger les données, features, labels, pipeline
* **7.2.3** Appeler `cross_val_score` sur pipeline complet
* **7.2.4** Sauvegarder modèle + scaler + dim-reducer dans `artifacts/`

_Risque Murphy (section 7.2) : à lier à TPV-XXX… (Phase: Train/Val/Test / Robustesse)_

### 7.3 Script de prédiction (`predict.py`)

* **7.3.1** Implémenter la CLI `mybci.py subject run predict`
* **7.3.2** Charger artefacts (pipeline entraîné)
* **7.3.3** Prédire sur données jamais vues (Test set)
* **7.3.4** Calculer l’accuracy par run, par sujet, et globale (6 runs)

_Risque Murphy (section 7.3) : à lier à TPV-XXX… (Phase: Score global / Classification)_

### 7.4 Score global & contraintes de réussite

* **7.4.1** Implémenter le script global “tous sujets / toutes expériences”
* **7.4.2** Calculer la mean accuracy par expérience, puis moyenne des 6 expériences sur données jamais vues, et **garantir ≥ 60 %** (exigence minimale du sujet)
* **7.4.3** Définir une tâche d’optimisation dédiée pour viser **≥ 75 % de mean accuracy** (exigence du scale pour notation max + bonus)
* **7.4.4** Ajouter un rapport texte/CSV avec ces scores (par sujet, par expérience, global)
* **7.4.5** Ajouter tests `test_realtime.py` / `test_classifier.py` pour la stabilité du score sur dataset jouet
* **7.4.6** Vérifier ✓ dans la checklist TPV les items relatifs au score global, aux contraintes d’accuracy et à l’interprétation des résultats

_Risque Murphy (section 7.4) : à lier à TPV-XXX… (Phase: Score global / Robustesse)_

---

## 8. Lot 8 – Temps Réel & Playback (Temps réel)

### 8.1 Lecture en flux

* **8.1.1** Définir le format du stream (lecture fichier simulant le temps réel)
* **8.1.2** Implémenter le playback avec latence contrôlée (< 2 s)
* **8.1.3** Gérer la taille de fenêtre et l’overlap (stride)
* **8.1.4** Synchroniser les événements (labels) avec les fenêtres

_Risque Murphy (section 8.1) : à lier à TPV-XXX… (Phase: Temps réel / Robustesse)_

### 8.2 Intégration temps réel (`realtime.py`)

* **8.2.1** Connecter le flux aux mêmes étapes de pipeline (features + dim-reduction + classifier)
* **8.2.2** Mesurer la latence bout-en-bout par événement
* **8.2.3** Gérer la queue de prédictions et l’affichage (CLI ou log)

_Risque Murphy (section 8.2) : à lier à TPV-XXX… (Phase: Temps réel / Classification)_

### 8.3 Tests temps réel

* **8.3.1** Créer un dataset simulé pour test de latence
* **8.3.2** Ajouter asserts sur latence moyenne et maximale (< 2 s)
* **8.3.3** Vérifier que le modèle ne fuit pas les labels du futur
* **8.3.4** Ajouter `test_realtime.py` (cas nominal & cas avec retard d’E/S)
* **8.3.5** Vérifier ✓ dans la checklist TPV les items relatifs au temps réel, à la latence < 2 s et à la non-fuite d’information future

_Risque Murphy (section 8.3) : à lier à TPV-XXX… (Phase: Temps réel / Robustesse)_

---

## 9. Lot 9 – Robustesse, MLOps & Qualité (Robustesse, Score global, Viz)

### 9.1 Robustesse du code

* **9.1.1** Ajouter gestion d’erreurs pour chemins de données manquants
* **9.1.2** Valider le comportement en absence de certains channels
* **9.1.3** Tester la résistance aux NaN / Inf dans les features
* **9.1.4** Assurer la compatibilité Python 3.10+ (ou version cible)

_Risque Murphy (section 9.1) : à lier à TPV-XXX… (Phase: Robustesse)_

### 9.2 Qualité logicielle

* **9.2.1** Couverture de tests ≥ X% (objectif 100% pour ton standard)
* **9.2.2** Ajouter mutation testing (mutmut) sur modules clés
* **9.2.3** Analyser complexité (radon/xenon) et refactor si nécessaire
* **9.2.4** Configurer Bandit/Safety pour sécurité des dépendances

_Risque Murphy (section 9.2) : à lier à TPV-XXX… (Phase: Robustesse / Source & Repo)_

### 9.3 CI/CD

* **9.3.1** Workflow GitHub Actions pour lint + tests + coverage
* **9.3.2** Job dédié “TPV-full-eval” (éventuellement manuel) pour lancer le score global
* **9.3.3** Publication des artefacts de modèles (si taille raisonnable)
* **9.3.4** Failure modes CI liés à la Murphy Map (si tel test échoue, pointer un risque précis)
* **9.3.5** Vérifier ✓ dans la checklist TPV les items relatifs à l’automatisation, aux tests et à la reproductibilité

_Risque Murphy (section 9.3) : à lier à TPV-XXX… (Phase: Robustesse / Score global)_

---

## 10. Lot 10 – Documentation, Visualisation & Défense (Documentation & défense, Viz)

### 10.1 Documentation technique

* **10.1.1** Compléter `README.md` (objectif, architecture, prérequis, usage)
* **10.1.2** Documenter les modules `preprocessing.py`, `features.py`, `dimensionality.py`, `pipeline.py`, `classifier.py`, `realtime.py`
* **10.1.3** Ajouter une section “Risques & Murphy Map” avec lien vers `docs/risk/tpv_murphy_map_v8.csv`
* **10.1.4** Documenter la stratégie Train/Val/Test et les choix de features

_Risque Murphy (section 10.1) : à lier à TPV-XXX… (Phase: Documentation & défense)_

### 10.2 Documentation projet & WBS

* **10.2.1** Rédiger `docs/project/wbs_tpv_v1.md` (basé sur ce WBS)
* **10.2.2** Créer un Gantt simplifié `gantt_tpv.png`
* **10.2.3** Définir la roadmap `roadmap.md` (v0.1 → v1.0 → v2.0 temps réel)
* **10.2.4** Lier WBS, Gantt, roadmap dans un index `docs/index.md`

_Risque Murphy (section 10.2) : à lier à TPV-XXX… (Phase: Documentation & défense / Source & Repo)_

### 10.3 Visualisations et résultats

* **10.3.1** Générer figures de scores (par sujet, par expérience, global)
* **10.3.2** Créer des plots de spectre et de features pour 1–2 exemples
* **10.3.3** Préparer un notebook ou script de démonstration
* **10.3.4** Sauvegarder toutes les figures utiles à la défense

_Risque Murphy (section 10.3) : à lier à TPV-XXX… (Phase: Viz / Score global)_

### 10.4 Préparation défense 42

* **10.4.1** Préparer le pitch de 5–10 minutes (objectifs, pipeline, résultats)
* **10.4.2** Préparer la démonstration live (train rapide + predict + temps réel)
* **10.4.3** Lister les questions attendues (filtrage, CSP, overfitting, bonus)
* **10.4.4** Relire la checklist officielle pour valider chaque point avant soutenance

_Risque Murphy (section 10.4) : à lier à TPV-XXX… (Phase: Documentation & défense)_

### 10.5 Vérification systématique de la checklist TPV

* **10.5.1** Construire une matrice de traçabilité “Checklist item ↔ Tâche WBS”
* **10.5.2** Vérifier que chaque item de `total_perspective_vortex.en.checklist.pdf` est couvert par au moins une tâche
* **10.5.3** Marquer ✓ dans la checklist après validation de chaque item en conditions réelles (exécution end-to-end)
* **10.5.4** Ajouter cette matrice au dossier de défense (docs/project/checklist_traceability.md)

_Risque Murphy (section 10.5) : à lier à TPV-XXX… (Phase: Documentation & défense / Robustesse)_

---

## 11. Lot 11 – Bonus TPV (Bonus, Implémentations avancées)

### 11.1 Bonus preprocessing / spectre

* **11.1.1** Explorer l’usage de **wavelet transform** sur le signal EEG (variation spectrale fine)
* **11.1.2** Comparer FFT vs wavelets sur un jeu d’epochs (qualité des features / coût)
* **11.1.3** Intégrer une option “wavelets” dans `features.py` (configurable)
* **11.1.4** Documenter l’impact sur le score et la robustesse (rapport bonus)
* **11.1.5** Vérifier ✓ dans la checklist TPV les items relatifs aux bonus sur les features avancées

_Risque Murphy (section 11.1) : à lier à TPV-XXX… (Phase: Feature engineering / Bonus)_

### 11.2 Bonus classifier

* **11.2.1** Concevoir un **classifieur maison** (ex: réseau simple, modèle custom)
* **11.2.2** L’intégrer dans `classifier.py` avec la même API que les classifieurs sklearn
* **11.2.3** Comparer la performance vs LDA / Logistic / SVM (tableau de scores)
* **11.2.4** Ajouter une option CLI pour sélectionner ce classifier bonus
* **11.2.5** Vérifier ✓ dans la checklist TPV les items relatifs aux bonus sur les classifieurs

_Risque Murphy (section 11.2) : à lier à TPV-XXX… (Phase: Classification / Bonus)_

### 11.3 Bonus datasets

* **11.3.1** Identifier au moins un **autre dataset EEG** compatible (BCI competition / autre Physionet)
* **11.3.2** Adapter `preprocessing.py` pour supporter ce dataset additionnel
* **11.3.3** Lancer le pipeline complet sur ce dataset et mesurer le score
* **11.3.4** Documenter différences de qualité de signal et d’accuracy (section Bonus)

_Risque Murphy (section 11.3) : à lier à TPV-XXX… (Phase: Train/Val/Test / Bonus datasets)_

### 11.4 Bonus implémentations bas niveau

* **11.4.1** Implémenter (optionnellement) une fonction maison pour **covariance matrix estimation** sur signaux bruités
* **11.4.2** Implémenter (optionnellement) une SVD ou eigendecomposition custom et la comparer à numpy/scipy (stabilité numérique)
* **11.4.3** Brancher cette implémentation dans `dimensionality.py` via un flag de config
* **11.4.4** Évaluer l’impact sur le score, le temps de calcul et la stabilité
* **11.4.5** Vérifier ✓ dans la checklist TPV les items relatifs aux bonus “from scratch” math/numérique

_Risque Murphy (section 11.4) : à lier à TPV-XXX… (Phase: Réduction dimension / Bonus)_
