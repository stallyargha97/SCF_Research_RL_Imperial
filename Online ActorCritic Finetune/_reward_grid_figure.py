import pandas as pd
import matplotlib.pyplot as plt

BOXES = ['Narrow', 'Mid', 'Wide']
BOX_TITLES = {'Narrow': 'Narrow box', 'Mid': 'Mid box', 'Wide': 'Wide box'}
CONTROLLERS = [('DDPG', '1', 'DDPG-1'), ('TD3', '1', 'TD3-1'), ('DDPG', '2', 'DDPG-2'), ('TD3', '2', 'TD3-2')]

YLIM = (-3.0, 9.0)  # shared across every panel

fig, ax = plt.subplots(4, 3, figsize=(15, 14), sharex=False, sharey=True)

for j, box in enumerate(BOXES):
    for i, (algo, approach, label) in enumerate(CONTROLLERS):
        a = ax[i, j]
        df = pd.read_csv(f'charts/Best_{algo}_{approach}_{box}BoxPolicy_history.csv')
        a.plot(df.ep, df.train_reward, label='train')
        a.plot(df.ep, df.val_reward, label='val')
        if i == 0:
            a.set_title(BOX_TITLES[box], fontsize=20, fontweight='bold')
        a.set_xlabel('episode', fontsize=12)
        a.set_ylim(*YLIM)
        a.grid(alpha=.3)
        a.tick_params(axis='both', labelsize=11)
        if j == 0:
            a.set_ylabel(label, fontsize=22, fontweight='bold')
        if i == 0 and j == 0:
            a.legend(fontsize=12)

fig.tight_layout()
fig.savefig('charts/gainbox_reward_grid.png', dpi=150)
print('saved charts/gainbox_reward_grid.png')
