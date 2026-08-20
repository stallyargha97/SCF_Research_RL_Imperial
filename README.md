# SCF Research — RL for a Solar Collector Field (Imperial)

This is my project on reinforcement-learning control of the outlet temperature of a
solar thermal collector field, benchmarked against an anti-windup PI controller.
The idea is an offline-to-online pipeline: clone the expert with behavioural cloning,
learn a conservative value function offline with CQL, then fine-tune online with
DDPG/TD3 — and check how well each stage actually transfers to new field data
("Juan's" June-2026 days).

Everything shares one library, [`main_script/`](main_script), and each experiment
folder just carries its own `config.py`.

```
For GitHub/
├── main_script/                 shared library (env, actors, critics, agents, rollouts)
├── Behavioral Cloning Actor/    BC of the anti-windup expert          (Regular + CIRL)
├── CQL Offline Actor/           offline Conservative Q-Learning        (Regular + CIRL)
├── Online ActorCritic Finetune/ online DDPG/TD3, BC actor + CQL critic (2 approaches)
└── BC vs CQL Comparison/        BC-vs-CQL quick-refinement comparison
```

## How `main_script` + `config.py` fit together

`main_script` has all the actual behaviour — the plant model, `FlowActor`/`GainActor`,
`SingleCritic`/`TwinCritic`, `DDPG`/`TD3`/`CQL`, the replay buffer, rollouts and metrics.
Each folder's `config.py` only holds the constants that change between experiments:
gain box, observation box, state/action dims, hyperparameters, which datasets to use.

A notebook or script wires the two together like this:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))  # reach main_script
from main_script import *
import config as cfg
from config import *
configure(cfg)          # inject this folder's constants into the shared classes
```

Once `configure(cfg)` runs, things like `Actor()`, `Critic()`, `SolarFieldEnv`, `DDPG`,
`CQL` build themselves off the active config. That's how the same code handles both the
direct-flow variant (`ACTOR_KIND='flow'`, 10-D state) and the gain variant
(`ACTOR_KIND='gain'`, 9-D state) without duplicating anything.

## The control problem

Positional PI with back-calculation anti-windup (`Kw = Ki/Kp`), expert gains
`Kp=-0.5`, `Ti=300 s`. Setpoints are 80 °C sunny / 65 °C cloudy (picked up
automatically from the filename). Flow is bounded to `q ∈ [0, 40] L/min`.

Two actor parameterisations show up everywhere in this repo:
- **Regular** — the actor just outputs the flow `q` directly (10-D state, includes `q_prev`).
- **CIRL** — the actor outputs PI gains `[Kp,Ki,Kw]` instead (9-D state). BC keeps `Kw`
  pinned to the expert's `Ki/Kp`; CQL learns `Kw` itself within a wide gain box (after
  the CQL-CIRL retrain — see `CQL Offline Actor/`); online fine-tuning always learns it.

## The four experiments

| Folder | What it does |
|---|---|
| **Behavioral Cloning Actor** | Clones the anti-windup expert with supervised learning; trains 15 dataset-combo policies per variant (Regular / CIRL) |
| **CQL Offline Actor** | Offline CQL-H from an expert-PI + noise replay buffer, again 15 combos per variant |
| **Online ActorCritic Finetune** | Online DDPG/TD3 on Juan's days, starting from the BC actor + CQL critic; two approaches — gain search then critic-exploit, or purely critic-driven |
| **BC vs CQL Comparison** | Quick head-to-head: BC-init vs CQL-init, each given a light DDPG refine on Juan's days |

Each folder has its own README with the full method and per-day numbers. Treat those
notebooks as the source of truth for results, not this file.

## How the pipeline actually flows, phase by phase

1. **Phase 1 (Behavioural Cloning).** Train both actor variants on every non-empty
   combination of the five original dataset days, just by imitating the expert PI.
   Compare closed-loop MAE across all the combos and keep the best-generalising one —
   that becomes the actor warm start for Phase 2, and later Phase 4.
2. **Phase 2 (Offline CQL).** Retrain both variants from scratch, this time with CQL-H
   on a replay buffer built from the expert logs plus some injected action noise. CQL's
   conservative penalty keeps the policy close to the data while still letting it
   optimise against the reward instead of just copying the expert. This phase is also
   where the twin critics get trained — those become the critic warm start for Phase 4.
3. **Phase 3 (BC vs CQL comparison).** Take the Phase 1 and Phase 2 policies and run
   them zero-shot (no extra training) on four days they've never seen. This is really
   just checking which offline approach actually generalises, and which actor/critic
   combination is worth carrying into Phase 4.
4. **Phase 4 (online fine-tuning).** Combine the BC actor with the CQL critic into one
   actor-critic pair and fine-tune it online with DDPG and TD3, under two strategies —
   search-then-exploit, and pure critic-driven. This is the step that takes the policy
   from "just imitates a safe controller" to something that can actually adapt online.

All four phases run on the same `main_script` code and only swap out `config.py`, so
the actor/critic classes, environment, and rollout/metric functions are identical
end to end — the only thing that changes between phases is the constants.

## Datasets

Closed-loop `.xlsx` logs with columns `T_sc, Tin, Ta, I, theta, q` (and sometimes
`T_ref`). The original data is 4 sunny days (Oct 21–24 2025) + 1 cloudy day
(Oct 20 2025); online experiments are tested against 4 new "Juan" sunny days
(Jun 16–19 2026). Each folder ships whatever data it needs under `data/`.

## Usage

```bash
git clone https://github.com/xylinum97/SCF_Research_RL_Imperial.git
cd SCF_Research_RL_Imperial
pip install -r requirements.txt
```

- **Train**: run the `train/*.py` scripts (BC, CQL) or the training notebooks (online).
- **Evaluate**: open the `evaluate/*.ipynb` notebooks and Run All.
- **Regenerate the seed-13 generalisation figure**: run
  `Online ActorCritic Finetune/generate_seed13_tout_grid.py`. It rebuilds the T_out-only,
  3-box grid straight from the checked-in policy checkpoints — no manual image
  assembly needed.

Notebooks find `main_script` and their own `config.py` through a relative path, so
they just run from inside their folder, no install needed. You'll need Python 3.10+,
PyTorch (CPU is fine), NumPy, pandas, matplotlib, openpyxl.

## Requirements

```
pip install -r requirements.txt
```
