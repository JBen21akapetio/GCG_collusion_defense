#!/bin/bash
# Lance les trois études de robustesse en parallèle, chacune détachée.
#
#     bash scripts/launch_robustness.sh            # affiche le plan seulement
#     bash scripts/launch_robustness.sh --go       # lance pour de vrai
#
# Les trois sont independants : trois conteneurs, trois GPU, en meme temps.
# Duree totale = la plus longue (~30 min) au lieu de la somme (~54 min).
set -u

GPU="${GPU:-A100}"
REF_RUN="${REF_RUN:-20261024-133052_default_seed0}"
GO=""
[ "${1:-}" = "--go" ] && GO=1

run() {
    label="$1"; shift
    echo ""
    echo "── $label"
    if [ -n "$GO" ]; then
        "$@" &
    else
        echo "   (plan) $*"
    fi
}

echo "GPU=$GPU   run de reference=$REF_RUN"

# 1. Modele 3B : relance complete, batch plafonne (35.2 GiB sur 39.5 au step 0)
run "exp 1 — Qwen 3B  (~30 min, ~1.10 \$)" \
    modal run --detach scripts/modal_app.py::episode \
        --gpu "$GPU" --s-per-step 10.6 --n-steps 150 --skip-sanity \
        --overrides 'model.name=Qwen/Qwen2.5-3B-Instruct;gcg.n_steps=150;model.max_batch_size=256;run.tag=m3b'

# 2. Perturbations du suffixe de reference : aucun GCG, scoring seul
run "exp 2 — perturbations  (~6 min, ~0.20 \$)" \
    modal run --detach scripts/modal_app.py::perturb \
        --run-id "$REF_RUN" --gpu "$GPU"

# 3. Autre jeu d'instructions d'attaque
run "exp 3 — attaquant alternatif  (~18 min, ~0.70 \$)" \
    modal run --detach scripts/modal_app.py::episode \
        --gpu "$GPU" --s-per-step 6.5 --n-steps 150 --skip-sanity \
        --overrides 'prompts.train_instruction_ids=[8,9,10,11,12];gcg.n_steps=150;run.tag=attack2'

# 4. Hors boucle : decodage (gratuit) + generation libre sur le run de reference
run "hors boucle — decodage + generation  (~12 min, ~0.45 \$)" \
    modal run --detach scripts/modal_app.py::out_of_loop \
        --run-id "$REF_RUN" --gpu "$GPU"

if [ -n "$GO" ]; then
    wait
    echo ""
    echo "Les trois apps sont lancees et detachees."
else
    echo ""
    echo "Plan seulement. Relancez avec --go pour lancer."
fi
echo ""
echo "Suivre :   modal app list"
echo "           modal app logs <app-id> -f"
echo "Recuperer: modal volume ls gcg-defense-results"
echo "           modal volume get gcg-defense-results <run_id> results/"
