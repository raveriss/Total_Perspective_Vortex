# Règles de splits et métriques (WBS 7.1, 7.4)

## Politique de splits
- **Sujet comme frontière** : aucun mélange d'epochs entre sujets pour éviter les fuites. Les répertoires `data/<subject>/` restent disjoints.
- **Runs indépendants** : un modèle par couple `(sujet, run)` avec artefacts dans `artifacts/<subject>/<run>/`. Les scores sont calculés run par run.
- **Validation croisée** : `StratifiedKFold` avec `n_splits=min(3, min_class_count)` pour respecter l'équilibre. Si une classe possède moins de 3 échantillons, la CV est ignorée et seuls les scores d'entraînement/éval locale sont produits.
- **Seeds** : la reproductibilité repose sur les seeds scikit-learn implicites. Les datasets jouets utilisés dans les tests sont déterministes via NumPy.

## Mécanique des métriques
- **Accuracy par run** : `pipeline.score(X, y)` exécuté sur le run ciblé, sans mélange inter-run. La matrice W est rechargée pour valider la complétude des artefacts.
- **Accuracy par sujet** : moyenne arithmétique des accuracies des runs d'un même sujet.
- **Accuracy globale** : moyenne des accuracies de tous les runs découverts.
- **Cible** : score attendu ≥ 0,75 sur les données réelles. Les tests synthétiques vérifient ≥ 0,90 pour garantir la cohérence de la pipeline.

## Commande d'agrégation
- **Script** : `scripts/aggregate_accuracy.py` (WBS 7.1/7.4).
- **Usage** :
  ```bash
  poetry run python scripts/aggregate_accuracy.py --data-dir data --artifacts-dir artifacts
  ```
- **Sortie** : tableau texte listant l'accuracy par run (`run`, `subject`, `accuracy`), la moyenne par sujet, puis l'accuracy globale.

## Alignement WBS
- **7.1** : définition explicite des splits train/validation/test et de l'accuracy par run.
- **7.4** : reporting consolidé par sujet et global pour le suivi des livrables.
