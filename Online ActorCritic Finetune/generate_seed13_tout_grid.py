"""Regenerate the report's Figure 4 (T_out-only, seed-13, 3-box grid).

Report figure `randstream_seed13_tout_grid.png` (used in
sourcecode/04_results_discussion.tex and in main/randstream/ as
Best_{Narrow,Mid,Wide}Box_Randstream_Seed13_State.png) had no generating
script in the repo -- it was assembled by hand outside version control. This
script closes that gap: it re-runs the same random-stream rollout used by
Multiseed_RandomStream_DDPG_TD3.ipynb (seed 13, all four best policies per
box, deterministic zero-shot rollout) for the Narrow/Mid/Wide gain boxes and
plots only the T_out panel per box, stacked into one 3-row figure.

Run from this folder: python generate_seed13_tout_grid.py
Takes a few minutes (3 boxes x 5 rollouts x ~72k steps).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from main_script import *
import config as cfg
from config import *

configure(cfg)

SEED = 13
N_SEGMENTS = 16
SEG_MIN, SEG_MAX = 3500, 5500

BOX_GAINS = {
    'Narrow': (np.array([-3.5, -0.060, -0.35], dtype=np.float32), np.array([-0.1, -0.0002, 0.05], dtype=np.float32)),
    'Mid':    (np.array([-6.0, -0.100, -0.60], dtype=np.float32), np.array([-0.05, -0.0001, 0.10], dtype=np.float32)),
    'Wide':   (np.array([-15.0, -1.5625, -0.525], dtype=np.float32), np.array([-0.05, -0.0001, 0.10], dtype=np.float32)),
}
BOX_LABELS = {'Narrow': 'narrow', 'Mid': 'mid', 'Wide': 'wide'}
COLORS = {'DDPG-A1': '#d62728', 'DDPG-A2': '#ff7f0e', 'TD3-A1': '#1f77b4', 'TD3-A2': '#9467bd'}
POLICY_ORDER = ['DDPG-A1', 'DDPG-A2', 'TD3-A1', 'TD3-A2']

ORIG_DATA = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'CQL Offline Actor', 'data'))
d_cloudy = load_dataset(os.path.join(ORIG_DATA, '20_10_2025__Cloudy_Closed_Loop.xlsx'))
sunny_pool = ([load_dataset(f) for f in JUAN_FILES] +
              [load_dataset(os.path.join(ORIG_DATA, '21_10_2025__Sunny_Closed_Loop.xlsx'))])


def build_random_stream(seed, n_segments=N_SEGMENTS, seg_min=SEG_MIN, seg_max=SEG_MAX):
    rng = np.random.default_rng(seed)
    keys = ['T_sc', 'Tin', 'Ta', 'I_sol', 'theta', 'q']
    cat = {k: [] for k in keys}
    tref = []
    for _ in range(n_segments):
        cloudy = bool(rng.random() < 0.5)
        src = d_cloudy if cloudy else sunny_pool[int(rng.integers(len(sunny_pool)))]
        trv = TREF_CLOUDY if cloudy else TREF_SUNNY
        L = min(int(rng.integers(seg_min, seg_max + 1)), src['N'])
        st = int(rng.integers(0, max(1, src['N'] - L)))
        for k in keys:
            cat[k].append(src[k][st:st + L])
        tref.append(np.full(L, trv))
    data = {k: np.concatenate(cat[k]) for k in keys}
    data['N'] = len(data['T_sc'])
    data['name'] = f'RANDOM_stream_seed{seed}'
    data['tref_seq'] = np.concatenate(tref)
    return data


fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)

for row, (box, ax) in enumerate(zip(['Narrow', 'Mid', 'Wide'], axes)):
    cfg.GAIN_LOW, cfg.GAIN_HIGH = BOX_GAINS[box]
    main_pol = os.path.join(BASE_DIR, 'main', 'policies')
    policies = {
        'DDPG-A1': f'Best_DDPG_1_{box}BoxPolicy.pt',
        'DDPG-A2': f'Best_DDPG_2_{box}BoxPolicy.pt',
        'TD3-A1':  f'Best_TD3_1_{box}BoxPolicy.pt',
        'TD3-A2':  f'Best_TD3_2_{box}BoxPolicy.pt',
    }
    actors = {tag: load_actor_raw(os.path.join(main_pol, f)) for tag, f in policies.items()}
    print(f"[{box}] loaded {len(actors)} policies | gain box LOW={list(cfg.GAIN_LOW)} HIGH={list(cfg.GAIN_HIGH)}")

    d_rand = build_random_stream(SEED)
    Te, _ = rollout_expert(d_rand)
    Ts = {}
    for tag, actor in actors.items():
        Ts[tag], _, _ = rollout_policy(actor, d_rand)
    n = min([len(Te)] + [len(v) for v in Ts.values()])
    tref_r = np.asarray(d_rand['tref_seq'][:n], float)
    t = np.arange(n) * TS

    mae_e, _ = mae_rmse(Te[:n], tref_r)
    ax.plot(t, tref_r, color='green', ls='--', lw=1.3, label='T_ref')
    ax.plot(t, Te[:n], color='#888888', ls='-.', lw=0.9, label=f'Expert  MAE={mae_e:.3f}')
    for tag in POLICY_ORDER:
        mae_p, _ = mae_rmse(Ts[tag][:n], tref_r)
        ax.plot(t, Ts[tag][:n], color=COLORS[tag], lw=1.0, label=f'{tag}  MAE={mae_p:.3f}')
    ax.set_ylim(45, 95)
    ax.set_ylabel('T_out (C)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=11, ncol=2)
    ax.set_title(f'{box} box (seed={SEED})', fontsize=16, fontweight='bold')
    print(f"[{box}] done: n={n} steps")

fig.suptitle(f'Zero-shot generalisation, stream seed {SEED}: T_out tracking', fontsize=16)
fig.tight_layout()

out_dir = os.path.join(CHART_DIR, 'randstream_multiseed')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'randstream_seed13_tout_grid.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"saved -> {out_path}")
plt.close(fig)
