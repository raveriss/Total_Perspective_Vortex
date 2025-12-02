# Physionet EEG Motor Imagery — récupération des données

## Jeu de données attendu
- Portail officiel : https://physionet.org/content/eegmmidb/1.0.0/
- Périmètre : 109 sujets, 14 runs par sujet (`R01` à `R14`).
- Volume : ~1.5 Go en EDF bruts, prévoir 2 Go libres pour les copies temporaires.
- Licence d'accès : inscription Physionet requise (ensemble "EEG Motor Movement/Imagery").

## Format des fichiers
- Enregistrements EDF issus de BCI2000 (64 canaux, 160 Hz) structurés en matrices channels × time.
- Sessions séparées par sujet/run avec fichiers nommés `SXXXRYY.edf`.
- Les téléchargements s’effectuent sans ré-emballage : la structure Physionet doit être conservée dans `data/raw/`.

## Arborescence locale `data/raw/`
```
data/
└── raw/
    ├── manifest.json
    ├── S001/
    │   ├── S001R01.edf
    │   ├── ...
    │   └── S001R14.edf
    └── S109/
        ├── S109R01.edf
        └── S109R14.edf
```
Le fichier `manifest.json` suit la structure :
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

## Points de vigilance sur le stockage
- Les disques formatés FAT limitent la longueur des chemins : privilégier ext4 ou APFS.
- Vérifier la présence des 64 canaux et de l'échantillonnage 160 Hz avant tout pré-traitement.

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
