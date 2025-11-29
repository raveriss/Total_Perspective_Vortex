# Physionet EEG Motor Imagery — récupération des données

## Lien officiel
- Portail Physionet : https://physionet.org/content/eegmmidb/1.0.0/

## Format des fichiers
- Enregistrements EDF issus de BCI2000 (64 canaux, 160 Hz) structurés en matrices channels × time.
- Sessions séparées par sujet/run avec fichiers nommés `SXXXRYY.edf`.
- Les téléchargements s’effectuent sans ré-emballage : la structure Physionet doit être conservée dans `data/raw/`.

## Prérequis d’accès
- Compte Physionet validé (accès restreint et acceptation de la licence des données EEG Motor Movement/Imagery).
- Espace disque disponible ≥ 1.5 Go pour l’ensemble des EDF bruts.
- Outils réseau standard (`curl` ou `wget`) opérationnels si vous utilisez les URLs HTTP(S).

## Script de récupération
- Le script `scripts/fetch_physionet.py` automatise le téléchargement ou la copie locale vers `data/raw/`.
- Il repose sur un manifeste JSON listant les fichiers attendus avec leurs métadonnées :
  ```json
  {
    "files": [
      {
        "path": "S001/S001R01.edf",
        "size": 1513488,
        "sha256": "<hash officiel de S001R01.edf>"
      }
    ]
  }
  ```
- Commande type pour copier depuis un répertoire local :
  ```bash
  python scripts/fetch_physionet.py --source /chemin/vers/physionet/eegmmidb --manifest manifest.json
  ```
- Commande type pour télécharger depuis Physionet :
  ```bash
  python scripts/fetch_physionet.py --source https://physionet.org/static/published-projects/eegmmidb --manifest manifest.json
  ```
- Le script vérifie la présence, la taille et le hash SHA-256 de chaque fichier ; un échec explicite est déclenché en cas d’écart ou d’absence.
