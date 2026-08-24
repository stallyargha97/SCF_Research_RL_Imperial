# SCF Research — Actor-Critic Reinforcement Learning for SCF Control System

Solar collector fields are difficult to regulate using a fixed-gain PI loop, since
transient disturbances such as passing cloud cover are precisely the class of event a
fixed-gain controller cannot anticipate. This project investigates whether a learned
control policy can improve upon such a baseline, subject to the constraint that an
untrained agent must not be permitted to explore unsafely on the real
outlet-temperature loop during training.

The approach adopted is an offline-to-online pipeline: training an agent directly on
the plant from initialisation was not a viable option under this constraint. The
actor is therefore first trained to imitate the existing anti-windup PI controller
via behavioral cloning; a separate offline stage then learns a conservative value
function from the same logged data using Conservative Q-Learning (CQL); only once
both stages are complete does the policy undergo online fine-tuning, using DDPG and
TD3. Every stage is evaluated on its ability to transfer to four days of field data
withheld from training throughout, the New Unseen (16–19 June 2026) Dataset, since
generalization to unseen operating conditions is the primary criterion of interest.

All experiments share a single library, [`main_script/`](main_script); each
experiment folder supplies its own `config.py`.

```
For GitHub/
├── main_script/                 shared library (env, actors, critics, agents, rollouts)
├── Behavioral Cloning Actor/    BC of the anti-windup expert          (Regular + CIRL)
├── CQL Offline Actor/           offline Conservative Q-Learning        (Regular + CIRL)
├── Online ActorCritic Finetune/ online DDPG/TD3, BC actor + CQL critic (2 approaches)
└── BC vs CQL Comparison/        BC-vs-CQL quick-refinement comparison
```

## How `main_script` and `config.py` fit together

`main_script` contains the shared implementation: the plant model, `FlowActor`/
`GainActor`, `SingleCritic`/`TwinCritic`, `DDPG`/`TD3`/`CQL`, the replay buffer, and
the rollout and metric functions. Each folder's `config.py` defines only the
constants that vary between experiments: gain box, observation box, state and action
dimensions, hyperparameters, and dataset selection. A notebook or script wires the
two together as follows:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))  # reach main_script
from main_script import *
import config as cfg
from config import *
configure(cfg)          # inject this folder's constants into the shared classes
```

Once `configure(cfg)` executes, `Actor()`, `Critic()`, `SolarFieldEnv`, `DDPG`, and
`CQL` are instantiated from the active configuration. This mechanism allows a single
implementation to support both the direct-flow parameterization (`ACTOR_KIND='flow'`,
10-D state) and the gain parameterization (`ACTOR_KIND='gain'`, 9-D state) without
code duplication.

## The control problem

The expert policy being cloned is a positional PI controller with back-calculation
anti-windup (`Kw = Ki/Kp`), tuned with `Kp=-0.5`, `Ti=300 s`. The setpoint (80 °C
under sunny conditions, 65 °C under cloudy conditions) is determined automatically
from the dataset filename, and the flow rate is constrained to `q ∈ [0, 40] L/min`
throughout.

Two actor parameterizations are used across this repository, reflecting a design
question addressed empirically rather than assumed in advance. The **Regular** actor
outputs the flow rate `q` directly, from a 10-D state that includes the previous flow
`q_prev`. The **CIRL** actor instead outputs the PI gains `[Kp,Ki,Kw]`, from a 9-D
state. Under behavioral cloning, `Kw` is fixed to the expert's `Ki/Kp` ratio, since
no learning signal is available to inform it otherwise; under CQL, `Kw` is learned
once the policy is retrained within a widened gain box (see `CQL Offline Actor/`);
under online fine-tuning, `Kw` is learned throughout.

## How the pipeline is structured, phase by phase

Each phase corresponds to one folder in this repository, and each folder maintains
its own local `policies/` directory: checkpoints are produced there by that folder's
training scripts; where a later phase depends on an earlier one, the relevant
`*_best.pt` checkpoint is copied into the downstream folder's `policies/` as a
warm-start input. This keeps every phase independently reproducible from its own
folder, without requiring cross-folder path resolution at runtime.

**Phase 1** (`Behavioral Cloning Actor/`): running `train/BC_Regular_Anti_Windup.py`
and `train/BC_CIRL_Setpoint_Anti_Windup.py` trains both actor variants by imitation of
the expert PI controller, across every non-empty combination of the five original
dataset days: fifteen combinations per variant, since the subset of days most
conducive to generalization was not known a priori. Each combination is saved as its
own checkpoint; the combination achieving the lowest closed-loop MAE is additionally
saved as `*_best.pt`. The `evaluate/*.ipynb` notebooks then roll every checkpoint out
closed-loop to produce the per-day metrics reported in this folder's README.

**Phase 2** (`CQL Offline Actor/`): `train/CQL_Regular_Anti_Windup.py` and
`train/CQL_CIRL_Setpoint_Anti_Windup.py` retrain both variants from initialisation
using CQL-H, on a replay buffer constructed from the expert's logged trajectories with
injected action noise for state-action coverage, again across the same fifteen
combinations, again keeping a `*_best.pt`. The conservative penalty permits this stage
to optimize directly against the reward rather than merely imitating the expert, while
constraining the policy to remain close to the observed data distribution. This phase
also trains the twin critics later copied forward as the critic warm start for Phase 4.
Evaluation follows the same pattern as Phase 1, via this folder's own `evaluate/*.ipynb`.

**Phase 3** (`BC vs CQL Comparison/`): the `*_best.pt` checkpoints from Phase 1 and
Phase 2 are copied into this folder's `policies/`, and `Four_Way_Offline_Policy_Comparison.ipynb`
(together with `BC_vs_CQL_Offline_Policy_Comparison.ipynb` and `BC_vs_CQL_Online_Tuning.ipynb`)
evaluates them zero-shot, without further training, on the New Unseen
(16–19 June 2026) Dataset. This
comparison establishes which offline approach generalizes more effectively, and which
actor/critic combination is carried forward into Phase 4.

**Phase 4** (`Online ActorCritic Finetune/`): the winning BC actor and CQL critic
checkpoints are copied into this folder's `policies/` and used as the warm start for
`DDPG_Approach1_SearchThenCriticExploit.ipynb`, `DDPG_Approach2_CriticDriven.ipynb`,
`TD3_Approach1_SearchThenCriticExploit.ipynb`, and `TD3_Approach2_CriticDriven.ipynb`.
These are the two strategies referenced above: search followed by critic
exploitation, and purely critic-driven optimization throughout. The curated, final per-box checkpoints
produced by these runs are collected under `main/policies/`, and are what
`Evaluate_TunedBox_Policies.ipynb` and `Multiseed_RandomStream_DDPG_TD3.ipynb` load
for the reported results. This phase constitutes the transition from a policy that
reproduces safe baseline behaviour to one capable of online adaptation.

All four phases execute identical `main_script` code; only `config.py` and the
contents of each folder's `policies/` differ between them. Each folder contains its
own README with the complete methodology and per-day results; those notebooks, not
this document, should be treated as the authoritative source for reported results.

## Datasets

Closed-loop `.xlsx` logs contain the columns `T_sc, Tin, Ta, I, theta, q` (and, in
some cases, `T_ref`). The original dataset comprises four sunny days (21–24 October
2025) and one cloudy day (20 October 2025); the online experiments are subsequently
evaluated on the New Unseen (16–19 June 2026) Dataset, four additional sunny days that
are withheld from all offline training. Each experiment folder includes the data it
requires under `data/`.

## Usage

```bash
git clone https://github.com/stallyargha97/SCF_Research_RL_Imperial.git
cd SCF_Research_RL_Imperial
pip install -r requirements.txt
```

Training is performed by running the `train/*.py` scripts (BC, CQL) or the online
training notebooks. Evaluation is performed by executing the `evaluate/*.ipynb`
notebooks in full. The seed-13 generalization figure can be regenerated by running
`Online ActorCritic Finetune/generate_seed13_tout_grid.py`, which reconstructs the
T_out-only, three-box grid directly from the checked-in policy checkpoints.

Notebooks resolve `main_script` and their own `config.py` via relative paths and
therefore require no separate installation, beyond Python 3.10+, PyTorch (CPU
execution is sufficient), NumPy, pandas, matplotlib, and openpyxl.
