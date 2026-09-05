# WBS 7.4.3 — Optimisation du score sans fuite

## Traçabilité

- WBS principal : 7.4.3 ; supports : 3.1, 4.2, 5.2, 7.1, 7.4.2–7.4.4.
- Risques : TPV-028, TPV-030, TPV-032, TPV-036, TPV-043, TPV-045,
  TPV-048, TPV-687, TPV-690, TPV-756, TPV-757 et TPV-759.
- Issues existantes associées aux risques : #42, #44, #46, #50, #57, #59,
  #62, #65 et #94.
- Issue WBS à créer : **`WBS 7.4.3 — CV imbriquée, MIBIF et campagne
  10→30→109`**, avec liens vers cette page, `wbs_tpv.md` §7.4 et les lignes
  Murphy ci-dessus. Champs Project proposés : Status `In progress`, Phase
  `5 – Évaluation / Validation`, Type `Feature`, Priority `Must`, Risk score
  `20` (maximum des risques directement traités).

## Ordre expérimental figé

1. Valider les tests synthétiques et un sujet en diagnostic.
2. Lancer `make score-ablation-10`, puis conserver la configuration FBCSP selon
   la moyenne des plis externes, jamais selon le test final 109.
3. Lancer `make score-campaign-10`, puis `make score-campaign-30`.
4. Si FBCSP reste stable, lancer `make score-riemannian-10` comme comparaison.
5. Figer la configuration, puis lancer `make compute-mean-of-means` une seule
   fois sur S001…S109.

Les jalons sont 75 %, 81 %, 87 % et 90 %. Les paliers de bonus exigent des
tranches complètes : 78/81/84/87/90 % correspondent à 1/2/3/4/5 points.

## Contrats techniques

- Epochs -1…4 s ; la baseline ERD/ERS -1…0 s est optionnelle dans l'ablation.
- Le témoin reprend les six bandes mu/bêta historiques ; l'ablation compare les
  neuf bandes de 4 Hz et le filtrage causal avant toute promotion en production.
- ROI latérale pour T1/T2 ; ROI incluant la ligne médiane pour T3/T4.
- MIBIF est dans la `Pipeline` et retient 4–16 features.
- CV interne pour MIBIF/hyperparamètres ; CV stratifiée externe pour le score.
- Rapport distinct `LeaveOneGroupOut` pour les trois runs de chaque type.
- CAR et Laplacien sont des transformeurs non mutants dans la pipeline.
- La covariance SPD tangentielle est une branche d'ablation secondaire.

## Résultats locaux du 3 septembre 2026

La cohorte de développement est figée à S001…S010. Les mesures suivantes ont
été obtenues sur les EDF locaux, sans modifier les plis après observation :

| Variante | CV externe | Leave-one-run-out | Décision |
| --- | ---: | ---: | --- |
| ERD + CAR + causal, paramètres fixes | 61,39 % | 60,33 % | rejetée |
| Témoin v2 + MIBIF(12), paramètres fixes | 74,50 % | 73,17 % | à comparer |
| Témoin v2 + MIBIF(4/8/12/16), CV imbriquée | 73,72 % | 73,78 % | arrêt avant 30 |

Sur S001 seul, la dernière variante atteint 82,22 % en CV externe et 80,56 %
en leave-one-run-out. Ce résultat individuel ne remplace pas la moyenne de
cohorte. Le seuil de passage à 30 sujets (75 %) n'étant pas atteint, les
campagnes 30/109 et la branche riemannienne ne sont pas lancées dans cette
itération. L'ancien rapport v2 complet reste donc le résultat global courant :
73,302 % sur 109 sujets.
