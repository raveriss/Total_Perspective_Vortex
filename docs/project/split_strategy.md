# Stratégie de split et seuils de métriques (WBS 7.1, 7.4)

## Frontières de split
- **Séparation par sujet** : chaque sujet possède ses propres répertoires `data/<subject>/` et `artifacts/<subject>/` afin d'empêcher toute fuite entre participants.
- **Isolement des runs** : les runs restent disjoints et sont entraînés et évalués individuellement (`artifacts/<subject>/<run>/`). Les métriques consolidées s'appuient uniquement sur ces artefacts.
- **Seeds reproductibles** : les entrées déterministes des datasets jouets fixent les seeds NumPy, tandis que la pipeline scikit-learn repose sur les seeds implicites pour assurer la stabilité des résultats.

## Seuils et objectifs
- **Accuracy minimale** : un run est considéré valide au-delà de **0,60** afin de respecter les exigences de robustesse.
- **Cible ambitieuse** : la trajectoire produit un objectif de **0,75** en accuracy, utilisé pour le suivi des livrables WBS.
- **Consolidation** : les accuracies sont calculées par run, moyennées par sujet, puis moyennées globalement pour piloter les livrables WBS 7.1 (mesure par run) et 7.4 (reporting consolidé).

## Outil d'agrégation
- **Script** : `scripts/aggregate_scores.py` agrège les accuracies générées par `train.py` et `predict.py` pour produire un reporting CSV/JSON aligné sur WBS 7.1/7.4.
- **Sorties** :
  - CSV : lignes typées (`run`, `subject`, `global`) contenant l'accuracy et les drapeaux `meets_minimum` (≥0,60) et `meets_target` (≥0,75).
  - JSON : structure sérialisée reprenant les mêmes informations pour un usage CI ou dataops.
