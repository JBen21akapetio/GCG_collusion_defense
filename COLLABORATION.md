# Travailler à deux sur ce dépôt

Dépôt : `JBen21akapetio/gcg-collusion-defense` (privé) · branche principale : `main`

---

## 1. Installation (une seule fois)

```bash
git clone https://github.com/JBen21akapetio/gcg-collusion-defense.git
cd gcg-collusion-defense

# environnement local : seul `modal` est nécessaire, le reste tourne sur le GPU distant
python3 -m venv venv && source venv/bin/activate
pip install modal && modal setup

# éviter les commits de fusion parasites
git config pull.rebase true
```

`requirements.txt` ne sert que si vous exécutez le code sur votre propre GPU.
Sur Modal, l'image est construite par `scripts/modal_app.py`.

**Accès GPU** : le dépôt ne donne pas accès à Modal. Il faut soit son propre
compte (palier gratuit : 30 $/mois de crédits, suffisant pour l'étape 1 à
~1,30 $), soit être ajouté au workspace Modal de Benjamin pour accéder au
volume `gcg-defense-results`.

---

## 2. Le cycle de travail

```bash
# 1. partir de la dernière version de main
git switch main && git pull

# 2. une branche par tâche
git switch -c eval-holdout

# 3. travailler, puis
git add -A
git commit -m "Evaluate the reference run on held-out instructions"
git push -u origin eval-holdout

# 4. ouvrir la pull request
gh pr create --fill

# 5. une fois relue, fusionner et nettoyer
gh pr merge --squash --delete-branch
git switch main && git pull
```

Si `main` a avancé pendant votre travail :

```bash
git fetch origin && git rebase origin/main
```

Règle unique : **on ne pousse pas directement sur `main`**. Tout passe par une
branche et une PR, même une correction d'une ligne. À deux, la relecture prend
deux minutes et évite les écrasements.

---

## 3. Qui touche quoi (pour éviter les conflits)

Les conflits viendront des fichiers de prose, pas du code. Répartition proposée :

| Zone | Fichiers | Remarque |
|---|---|---|
| Évaluation / figures | `src/gcg_defense/{evaluate,plots,report}.py` | |
| Optimiseur / coûts | `src/gcg_defense/{gcg,cost,sweep}.py`, `scripts/modal_app.py` | |
| Rédaction | `README.md`, `ETAT_DES_LIEUX.md` | **une seule personne à la fois** |
| Résultats | `results/<run_id>/` | pas de conflit possible, voir ci-dessous |

Les `run_id` contiennent l'horodatage **et** le seed
(`20261024-133052_default_seed0`), donc deux runs simultanés écrivent dans des
dossiers différents : aucun risque d'écrasement, même en lançant en même temps.

Par défaut `results/` est ignoré par git. Pour partager un run précis :

```bash
git add -f results/<run_id>
```

---

## 4. Lancer des jobs GPU

Toute commande qui démarre un conteneur accepte `--dry-run`, qui affiche le
plan (config, seed, GPU, timeout, coût estimé) **sans rien lancer**. Prenez
l'habitude de la passer d'abord.

```bash
# plan
modal run scripts/modal_app.py::evaluate --run-id <run_id> --dry-run

# lancement, détaché (fermer le portable ne tue pas le job)
modal run --detach scripts/modal_app.py::evaluate --run-id <run_id>
```

Avant tout sweep, chiffrer la facture :

```bash
python3 scripts/estimate_cost.py plan --gpu A100 --s-per-step 6.50 \
    --seeds 0,1,2 --lambda-util 0.1,1,10 --max-concurrent 4
```

**Dites-vous ce que vous lancez.** Les jobs sont facturés à la seconde et deux
sweeps simultanés coûtent le double. Un message avant de lancer suffit.

---

## 5. Par où commencer

`ETAT_DES_LIEUX.md` contient l'état exact du projet et les quatre étapes
restantes, avec les commandes, les durées et les coûts. C'est le point d'entrée.

`README.md` contient la justification scientifique, les écarts au brief et la
section usage dual.
