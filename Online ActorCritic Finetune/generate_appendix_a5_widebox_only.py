"""Regenerate the report's Appendix A.5 per-box, per-seed random-stream state
and gains figures (12 images: oldbox/kp6box/wide x seed13/seed27 x state/gains).

Like randstream_seed13_tout_grid.png before generate_seed13_tout_grid.py was
written, these had no generating script in the repo -- the per-box PNGs in
main/randstream/ (Best_{Box}Box_Randstream_Seed{13,27}_{State,Gains}.png)
exist but nothing reproduces them. This script re-runs the same random-stream
rollout as generate_seed13_tout_grid.py / Multiseed_RandomStream_DDPG_TD3.ipynb
for both seeds and all three boxes, saving separate state (T_out) and gains
(Kp/Ki/Kw) figures per box/seed, with larger label/legend/tick text and no
redundant overall title.

Run from this folder: python generate_appendix_a5_randstream.py
Takes a while (3 boxes x 2 seeds x 4 policies x ~72k steps).
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

SEEDS = [13, 27]
N_SEGMENTS = 16
SEG_MIN, SEG_MAX = 3500, 5500

# box name -> (gain box bounds, appendix filename suffix)
BOX_GAINS = {
    'Wide':   (np.array([-15.0, -1.5625, -0.525], dtype=np.float32), np.array([-0.05, -0.0001, 0.10], dtype=np.float32)),
}
BOX_SUFFIX = {'Narrow': '_oldbox', 'Mid': '_kp6box', 'Wide': ''}  # matches app A.5 filenames
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


main_pol = os.path.join(BASE_DIR, 'main', 'policies')
out_dir = os.path.join(CHART_DIR, 'randstream_multiseed')
os.makedirs(out_dir, exist_ok=True)

for box, (glow, ghigh) in BOX_GAINS.items():
    cfg.GAIN_LOW, cfg.GAIN_HIGH = glow, ghigh
    policies = {
        'DDPG-A1': f'Best_DDPG_1_{box}BoxPolicy.pt',
        'DDPG-A2': f'Best_DDPG_2_{box}BoxPolicy.pt',
        'TD3-A1':  f'Best_TD3_1_{box}BoxPolicy.pt',
        'TD3-A2':  f'Best_TD3_2_{box}BoxPolicy.pt',
    }
    actors = {tag: load_actor_raw(os.path.join(main_pol, f)) for tag, f in policies.items()}
    print(f"[{box}] loaded {len(actors)} policies")

    for seed in SEEDS:
        d_rand = build_random_stream(seed)
        Te, _ = rollout_expert(d_rand)
        Ts = {}; Gs = {}
        for tag, actor in actors.items():
            T, Q, G = rollout_full(actor, d_rand)
            Ts[tag] = T; Gs[tag] = G
        n = min([len(Te)] + [len(v) for v in Ts.values()])
        tref_r = np.asarray(d_rand['tref_seq'][:n], float)
        t = np.arange(n) * TS

        suffix = BOX_SUFFIX[box]

        # ── state (T_out) figure ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(15, 5))
        mae_e, _ = mae_rmse(Te[:n], tref_r)
        ax.plot(t, tref_r, color='green', ls='--', lw=1.3, label='T_ref')
        ax.plot(t, Te[:n], color='#888888', ls='-.', lw=0.9, label=f'Expert  MAE={mae_e:.3f}')
        for tag in POLICY_ORDER:
            mae_p, _ = mae_rmse(Ts[tag][:n], tref_r)
            ax.plot(t, Ts[tag][:n], color=COLORS[tag], lw=1.0, label=f'{tag}  MAE={mae_p:.3f}')
        ax.set_ylim(45, 95)
        ax.set_ylabel('T_out (C)', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11, ncol=3)
        fig.tight_layout()
        state_out = os.path.join(out_dir, f"randstream{suffix}_seed{seed}.png")
        fig.savefig(state_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[{box}] seed={seed} saved -> {state_out}")

        # ── gains (Kp, Ki, Kw) figure ────────────────────────────────────────
        fig, axg = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        gain_labels = ['Kp', 'Ki', 'Kw']
        for gi, gname in enumerate(gain_labels):
            for tag in POLICY_ORDER:
                axg[gi].plot(t, Gs[tag][:n, gi], color=COLORS[tag], lw=0.9, label=tag)
            axg[gi].set_ylabel(gname, fontsize=14)
            axg[gi].tick_params(axis='both', labelsize=12)
            axg[gi].ticklabel_format(axis='y', useOffset=False, style='plain')
            axg[gi].grid(alpha=0.3)
            if gi == 0:
                axg[gi].legend(fontsize=11, ncol=4)
        axg[-1].set_xlabel('Time (s)', fontsize=14)
        fig.tight_layout()
        gains_out = os.path.join(out_dir, f"randstream_gains{suffix}_seed{seed}.png")
        fig.savefig(gains_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[{box}] seed={seed} saved -> {gains_out}")
