# Stratégie de split et seuils de métriques (WBS 7.1, 7.4)

## Frontières de split
- **Test externe immuable** : la moyenne principale utilise une CV stratifiée externe. Toute sélection MIBIF et tout réglage d'hyperparamètre ont lieu dans une CV interne distincte.
- **Preuve stricte par run** : un second rapport applique `LeaveOneGroupOut` aux trois runs d'un type ; le run testé ne contribue ni au CSP, ni à MIBIF, ni au classifieur.
- **Aucun mélange de sujets** : chaque sujet est évalué séparément. Les campagnes 10, 30 et 109 utilisent toujours les préfixes figés S001…S010, S001…S030 et S001…S109.
- **Seeds reproductibles** : les deux niveaux stratifiés utilisent `random_state=42`. Le test final 109 n'est jamais redéfini après observation d'un score.

## Seuils et objectifs
- **Accuracy minimale** : un run est considéré valide au-delà de **0,75** afin de respecter les exigences de robustesse.
- **Paliers** : la trajectoire figée vise successivement **0,75**, **0,81**, **0,87**, puis **0,90**. Un point de bonus exige chaque tranche complète de 3 % au-dessus de 75 % : 78/81/84/87/90 % donnent 1/2/3/4/5 points.
- **Consolidation** : les accuracies sont calculées sur quatre types d'expérience, trois runs par type, puis moyennées par sujet et globalement.

## Outil d'agrégation
- **Script** : `scripts/aggregate_experience_scores.py` est l'unique moteur qui agrège les scores de validation par sujet et expérience, aligné sur WBS 7.1/7.4.
- **Sorties** :
  - CSV : synthèse par sujet et par type d'expérience.
  - JSON : plis externes, runs tenus à l'écart, configuration, résultats stricts, jalons atteints et fingerprint Git.

## Pipeline évalué

Le chemin principal est `SpatialReference -> FilterBankCSP -> MIBIF -> LDA`.
Les tâches gauche/droite (T1/T2) et mains/pieds (T3/T4) ont des ROI et fenêtres
distinctes. L'ablation compare ERD/ERS, CAR, Laplacien et filtrage causal
uniquement dans la CV interne. La covariance tangentielle est une branche secondaire lancée après
l'ablation FBCSP, jamais une substitution choisie à partir du test final.
