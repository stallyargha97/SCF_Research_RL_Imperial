"""Regenerate the report's Appendix A.4 per-controller training-curve figures
(12 images: ddpg/td3 x approach1/2 x oldbox/kp6box/kp15_search).

These filenames are historical names for the box variants now called
Narrow/Mid/Wide in the main report (oldbox=Narrow, kp6box=Mid,
kp15_search=Wide -- confirmed against sourcecode/09_supplementary_information.tex).
The underlying data is the same saved per-episode history CSVs already used
by _reward_grid_figure.py for the main-body Figure 3, so no retraining or
rollout re-run is needed -- only re-plotting with larger text and no
redundant overall title (matches the original notebook cell's 3-panel
layout: Reward / MAE / mean gains).

Run from this folder: python generate_appendix_a4_curves.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

BOX_MAP = {"oldbox": "Narrow", "kp6box": "Mid", "kp15_search": "Wide"}
CONTROLLERS = [("ddpg", "DDPG", "1"), ("ddpg", "DDPG", "2"),
                ("td3", "TD3", "1"), ("td3", "TD3", "2")]

for algo_lc, algo, approach in CONTROLLERS:
    for suffix, box in BOX_MAP.items():
        tag = f"{algo_lc}_online_ac_approach{approach}_{suffix}"
        csv_path = f"charts/Best_{algo}_{approach}_{box}BoxPolicy_history.csv"
        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        ax[0].plot(df.ep, df.train_reward, label='train')
        ax[0].plot(df.ep, df.val_reward, label='val')
        ax[0].set_title('Reward', fontsize=15)
        ax[0].set_xlabel('episode', fontsize=13)
        ax[0].tick_params(axis='both', labelsize=11)
        ax[0].legend(fontsize=12)
        ax[0].grid(alpha=.3)

        ax[1].plot(df.ep, df.train_mae, label='train')
        ax[1].plot(df.ep, df.val_mae, label='val')
        ax[1].set_title('MAE [C]', fontsize=15)
        ax[1].set_xlabel('episode', fontsize=13)
        ax[1].tick_params(axis='both', labelsize=11)
        ax[1].legend(fontsize=12)
        ax[1].grid(alpha=.3)

        ax[2].plot(df.ep, df.Kp, label='Kp')
        ax[2].plot(df.ep, df.Ki, label='Ki')
        ax[2].plot(df.ep, df.Kw, label='Kw')
        ax[2].set_title('mean gains', fontsize=15)
        ax[2].set_xlabel('episode', fontsize=13)
        ax[2].tick_params(axis='both', labelsize=11)
        ax[2].legend(fontsize=12)
        ax[2].grid(alpha=.3)

        fig.tight_layout()
        out = os.path.join("charts", f"{tag}_curves.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved -> {out}")
