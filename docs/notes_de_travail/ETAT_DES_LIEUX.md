# État des lieux — Defensive GCG contre la collusion

*Mis à jour le 2026-10-24. Tout le code a été écrit ce jour (règle du sprint : les
deux fichiers antérieurs, `GCG.py` et `llm_modal.py`, étaient vides).*

---

## 1. Ce qui est fait

### Le pipeline, de bout en bout

| Étape | Module | État |
|---|---|---|
| Config unique (YAML + `--set`) | `config.py` | ✅ |
| Jeu RPS à 3 joueurs, payoffs, f(a_A) | `game.py` | ✅ |
| Prompts, 4+1 conditions, 13 instructions | `prompts.py` | ✅ |
| Distribution des coups q(·\|prompt) | `model_utils.py` | ✅ |
| Contrôles sanity (§5 du brief) | `sanity.py` | ✅ **passent** |
| Loss MI + utilité + représentation | `loss.py` | ✅ |
| Boucle GCG + contrôle positif | `gcg.py`, `positive_control.py` | ✅ **passent** |
| Baselines (manuelles, aléatoires, ablation, steering) | `baselines.py` | ✅ |
| Détecteurs + certificat | `detect.py` | ✅ |
| Simulation de parties | `simulate.py` | ✅ |
| Évaluation + figures + rapport | `evaluate.py`, `plots.py`, `report.py` | ✅ code prêt |
| Coût et parallélisme Modal | `cost.py`, `sweep.py`, `estimate_cost.py` | ✅ |

**105 tests** (dont 15 qui chargent un modèle). Les derniers lancés : 91/91 ✅,
plus 14 nouveaux tests de coût vérifiés en local.

### Le seul run complet (seed 0, config par défaut, A100)

Arrêt anticipé au step 450/500, 48,6 min, **≈ 1,80 $**.

| | sans défense | suffixe GCG | variation |
|---|---|---|---|
| model-MI I(A;B\|contexte) | 0,636 nats | **0,054 nats** | **−91 %** |
| taux d'aide q(f(a_A)) | 0,824 | **0,362** | le hasard = 0,333 |
| coût d'utilité (KL) | — | **0,011 nats** | très faible |
| validité | 0,9997 | 0,9976 | intacte |

> ⚠️ Ces chiffres sont sur le **pool d'entraînement**. L'évaluation sur le test
> (instructions tenues à l'écart, baselines, détection, transfert) **n'a pas
> encore été lancée** — c'est le point 1 de votre liste ci-dessous.

### Trois vrais bugs trouvés (chacun verrouillé par un test)

1. **100 % des candidats GCG étaient rejetés** : `decode()` supprimait l'espace
   avant la ponctuation, donc le suffixe ne survivait jamais au ré-encodage.
   Corrigé → ~90 % des candidats passent.
2. **La ligne « sans défense » payait un coût d'utilité fantôme** de 0,017 nats,
   à cause du saut de ligne laissé par un emplacement de suffixe vide.
3. **Test de permutation 2× trop lent** (lexsort → `rng.permuted` par strate).

### Une découverte qui a changé les prompts

Qwen2.5-1.5B **ne sait pas calculer** f(a_A). Avec les formulations « choisis le
coup qui garantit un point à A », le taux d'aide plafonne à 0,385 — en dessous
du seuil de 0,6 du brief. J'ai mesuré 3 familles de formulations (32 variantes
testées en 3 tours de sondage) : seules celles qui **nomment le coup cible en impératif final**
atteignent 0,60–0,99. Les formulations d'origine sont conservées comme condition
`colluding_implicit`. Tout est documenté dans le README.

---

## 1 bis. Lire les résultats

`notebooks/01_resultats.ipynb` relit les CSV de `results/` et montre, figures à l'appui :
le jeu et f(a_A), la collusion avant défense, la découverte sur les formulations, la
trajectoire GCG, le résultat principal (train **et** held-out), la comparaison 1.5B/0.5B,
l'étude de robustesse, et une table honnête de ce qui n'est pas démontré.

```bash
pip install notebook          # si besoin
jupyter notebook notebooks/01_resultats.ipynb
```

Il n'exécute aucun calcul GPU et installe lui-même numpy/pandas/matplotlib s'ils manquent.

## 2. Ce qu'il vous reste à faire

**Je ne lance plus rien.** Voici les commandes, dans l'ordre, avec durée et coût.
Chaque commande accepte `--dry-run` pour voir le plan sans rien démarrer.

### Étape 1 — Évaluer le run déjà terminé *(prioritaire)*

C'est ce qui transforme « MI −91 % sur le train » en un résultat publiable
(baselines, détection comportementale, certificat, transfert).

```bash
cd "/Users/benji/Documents/Candidatures ML Research/APART Research Project /code"

# voir le plan d'abord
modal run scripts/modal_app.py::evaluate \
    --run-id 20261024-133052_default_seed0 \
    --overrides 'eval.transfer=true' --dry-run

# puis lancer
modal run --detach scripts/modal_app.py::evaluate \
    --run-id 20261024-133052_default_seed0 \
    --overrides 'eval.transfer=true'
```

⏱ ~35 min · 💰 ~1,30 $ sur A100 · produit `report.md`, `summary.txt`,
`metrics.csv` et les figures.

### Étape 2 — Choisir le GPU au rapport vitesse/prix

Seul l'A100 est mesuré (6,50 s/step). L'A10G coûte moitié moins : il est
**meilleur s'il est moins de 1,9× plus lent**. Un benchmark de 5 steps tranche.

```bash
python3 scripts/estimate_cost.py benchmark --gpus A10G,L4   # affiche les commandes

# lancez les 2 commandes affichées (~3 min, ~0,10 $ au total), puis récupérez
# le journal de chaque benchmark ET celui du run A100 déjà terminé :
modal volume get gcg-defense-results 20261024-133052_default_seed0/gcg_steps.csv ./steps_A100.csv
modal volume get gcg-defense-results <run_id_bench_A10G>/gcg_steps.csv ./steps_A10G.csv

python3 scripts/estimate_cost.py compare \
    --measured A100=./steps_A100.csv \
    --measured A10G=./steps_A10G.csv
```

⏱ ~5 min · 💰 ~0,10 $ · peut diviser par ~2 le coût du sweep.

### Étape 3 — Le sweep (3 seeds × 3 λ_util)

Nécessaire pour les barres d'erreur sur seeds demandées par le brief.

```bash
# plan + facture, sans rien lancer
python3 scripts/estimate_cost.py plan --gpu A100 --s-per-step 6.50 \
    --seeds 0,1,2 --lambda-util 0.1,1,10 --max-concurrent 4

modal run scripts/modal_app.py::sweep --seeds 0,1,2 --lambda-util 0.1,1,10 \
    --gpu A100 --s-per-step 6.50 --max-concurrent 4 --with-eval --dry-run

# puis, sans --dry-run et avec --detach
modal run --detach scripts/modal_app.py::sweep --seeds 0,1,2 --lambda-util 0.1,1,10 \
    --gpu A100 --s-per-step 6.50 --max-concurrent 4 --with-eval
```

⏱ ~2 h 54 de montre (3 vagues de 4) · 💰 **~19,25 $** sur A100, moins sur A10G
si l'étape 2 le valide.

### Étape 4 — Figures fusionnées (gratuit, CPU)

```bash
modal run scripts/modal_app.py::figures --run-id <id1> \
    --extra-run-ids <id2>,<id3>,... --out-name sweep
modal volume get gcg-defense-results sweep results/
```

### Optionnel

- **Direction de collusion** (déjà calculée : couche 16, d de Cohen 15,8) —
  à refaire seulement si vous changez les prompts.
- **`lambda_repr > 0`** : nécessite la direction, déjà en place.
- **Sweep `suffix_len` / `suffix_position`** : mêmes axes, voir `sweep.py`.

---

## 3. Deux questions en attente

1. **`max_concurrent`** : quelle valeur par défaut ? La limite de votre compte
   dépend du plan (des sources tierces indiquent 10 GPU simultanés sur le plan
   Starter, non confirmé par la doc officielle). J'ai mis **4** par défaut.
2. **Votre message était coupé** à « plus totals per ». J'ai supposé
   « totaux par sweep et au global » — c'est ce que fait `estimate_cost.py plan`.
   Dites-moi si vous vouliez autre chose.

---

## 4. Écarts par rapport au brief (tous assumés et documentés)

| Écart | Raison |
|---|---|
| Screening en 2 étapes des candidats | Vous l'avez approuvé ; sinon 3–5 h/run |
| `history_len = 1` (au lieu de 3) | Votre correction ; permet la table de politique exacte |
| Instructions nommant le coup cible | Le modèle 1,5B ne sait pas dériver f(a_A) ; mesuré |
| Modules en plus (`detect`, `simulate`, `cost`, `sweep`…) | `evaluate.py` ferait 600 lignes sinon |
| Suffixe vide ⇒ pas d'emplacement | Exigé par la définition de L_util |

Le détail complet est dans le README, section *Deviations from the brief*.

---

## 5. Usage dual (rappel)

Le même pipeline avec un **signe inversé** sur L_MI chercherait des suffixes qui
*induisent* la collusion. Ce n'est **pas** implémenté, le signe n'apparaît nulle
part dans le code, et `loss.py` documente cette omission.
