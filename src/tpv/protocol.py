"""Constantes du protocole EEGMMIDB évalué par Total Perspective Vortex."""

# Le dataset officiel EEG Motor Movement/Imagery contient 109 sujets.
EXPECTED_SUBJECT_COUNT = 109

# Chaque sujet possède quatorze enregistrements EDF dans la distribution complète.
EXPECTED_RUN_COUNT = 14

# Les quatre expériences motrices évaluées couvrent les runs R03 à R14.
EXPERIENCE_RUNS: dict[str, tuple[str, ...]] = {
    # T1 : mouvement réel de la main gauche ou droite.
    "T1": ("R03", "R07", "R11"),
    # T2 : imagination du mouvement de la main gauche ou droite.
    "T2": ("R04", "R08", "R12"),
    # T3 : mouvement réel des deux poings ou des deux pieds.
    "T3": ("R05", "R09", "R13"),
    # T4 : imagination du mouvement des deux poings ou des deux pieds.
    "T4": ("R06", "R10", "R14"),
}

# Cet ordre stable est utilisé dans les rapports et les moyennes finales.
EXPERIENCE_ORDER = tuple(EXPERIENCE_RUNS)

# L'ordre lexical préserve l'ancien parcours R03, R04, ..., R14 de train-all.
MOTOR_RUNS = tuple(sorted(run for runs in EXPERIENCE_RUNS.values() for run in runs))
