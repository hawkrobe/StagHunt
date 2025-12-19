#!/usr/bin/env python3
"""
Visualize the key difference between IW and standard belief models.

The insight: IW updates on BOTH players' movements, so when both move
consistently toward stag, beliefs rise faster than the standard model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_trial, find_trial_files, get_outcome
from models.belief_model_iw import add_iw_beliefs_batch
from models.belief_model_jax import add_beliefs_batch_fast


def main():
    print("Loading trials...")
    files = find_trial_files(task_type='main')
    trials = [load_trial(f) for f in files]

    print("Computing beliefs...")
    trials = add_iw_beliefs_batch(trials)
    trials = add_beliefs_batch_fast(trials)

    # Find trials where IW and standard beliefs diverge most
    divergence = []
    for i, trial in enumerate(trials):
        iw = trial['joint_goal_stag'].values
        std_avg = (trial['p1_belief_p2_stag'].values + trial['p2_belief_p1_stag'].values) / 2
        # Mean absolute difference
        diff = np.mean(np.abs(iw - std_avg))
        outcome = get_outcome(trial)['outcome']
        divergence.append((i, diff, outcome))

    # Sort by divergence
    divergence.sort(key=lambda x: -x[1])

    # Pick example cooperation and defection trials with high divergence
    coop_examples = [(i, d, o) for i, d, o in divergence if o == 'cooperation'][:3]
    defect_examples = [(i, d, o) for i, d, o in divergence if o == 'mutual_defection'][:3]

    # Create figure: 2 rows (coop, defect) x 3 examples
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (idx, div, outcome) in enumerate(coop_examples):
        ax = axes[0, col]
        trial = trials[idx]
        t = np.arange(len(trial))

        # Plot IW belief
        ax.plot(t, trial['joint_goal_stag'], 'g-', linewidth=2, label='IW (joint goal)')

        # Plot standard beliefs (both P1 and P2)
        ax.plot(t, trial['p1_belief_p2_stag'], 'b--', alpha=0.7, label='P1→P2 (std)')
        ax.plot(t, trial['p2_belief_p1_stag'], 'r--', alpha=0.7, label='P2→P1 (std)')

        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Cooperation (div={div:.2f})')
        if col == 0:
            ax.set_ylabel('Belief')
            ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    for col, (idx, div, outcome) in enumerate(defect_examples):
        ax = axes[1, col]
        trial = trials[idx]
        t = np.arange(len(trial))

        ax.plot(t, trial['joint_goal_stag'], 'g-', linewidth=2, label='IW')
        ax.plot(t, trial['p1_belief_p2_stag'], 'b--', alpha=0.7, label='P1→P2')
        ax.plot(t, trial['p2_belief_p1_stag'], 'r--', alpha=0.7, label='P2→P1')

        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Defection (div={div:.2f})')
        ax.set_xlabel('Timestep')
        if col == 0:
            ax.set_ylabel('Belief')
        ax.grid(alpha=0.3)

    plt.suptitle('IW vs Standard Beliefs: Example Trials with High Divergence\n'
                 'Green = IW joint goal | Blue/Red = Standard partner intentions',
                 fontsize=12)
    plt.tight_layout()

    output_path = Path(__file__).parent / 'output' / 'iw_vs_standard_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")

    # Also create summary comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Average belief trajectories normalized by trial time
    ax = axes2[0]
    n_points = 100

    for outcome, color, label in [('cooperation', '#2ecc71', 'Cooperation'),
                                   ('mutual_defection', '#e74c3c', 'Defection')]:
        outcome_trials = [t for t in trials if get_outcome(t)['outcome'] == outcome]

        # IW beliefs
        iw_norm = np.array([np.interp(np.linspace(0, 1, n_points),
                                       np.linspace(0, 1, len(t)),
                                       t['joint_goal_stag'].values)
                           for t in outcome_trials])
        # Standard avg beliefs
        std_norm = np.array([np.interp(np.linspace(0, 1, n_points),
                                        np.linspace(0, 1, len(t)),
                                        (t['p1_belief_p2_stag'].values +
                                         t['p2_belief_p1_stag'].values) / 2)
                            for t in outcome_trials])

        x = np.linspace(0, 100, n_points)
        ax.plot(x, iw_norm.mean(axis=0), color=color, linewidth=2, label=f'{label} (IW)')
        ax.plot(x, std_norm.mean(axis=0), color=color, linewidth=2, linestyle='--',
                alpha=0.6, label=f'{label} (Std)')

    ax.set_xlabel('Trial Progress (%)')
    ax.set_ylabel('Belief')
    ax.set_title('Average Belief Trajectories')
    ax.legend(fontsize=9)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Right: Model comparison
    ax = axes2[1]
    models = ['Null', 'Distance', 'Belief', 'Coord.', 'IW']
    lls = [-676923, -556271, -554177, -553375, -485022]
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22', '#2ecc71']

    # Normalize to Null model improvement
    improvements = [(lls[0] - ll) / 1000 for ll in lls]

    bars = ax.bar(models, improvements, color=colors)
    ax.set_ylabel('LL Improvement over Null (thousands)')
    ax.set_title('Model Comparison')

    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{imp:.0f}k', ha='center', fontsize=9)

    plt.tight_layout()

    output_path2 = Path(__file__).parent / 'output' / 'iw_model_summary.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path2}")

    plt.show()


if __name__ == '__main__':
    main()
