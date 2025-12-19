#!/usr/bin/env python3
"""
Improved gallery visualizations - larger panels, clearer comparisons.

Focus on readability over completeness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from stag_hunt import load_trial, find_trial_files, get_trial_info, get_outcome, RAW_DATA_DIR
from models.belief_model_iw import add_iw_beliefs_batch

# Colors
PLAYER1_COLOR = '#E63946'
PLAYER2_COLOR = '#F4A261'
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
COOP_COLOR = '#2A9D8F'
DEFECT_COLOR = '#E63946'


def load_trials_with_beliefs(subject=None, opponent=None, max_trials=None):
    """Load trials and compute beliefs."""
    files = find_trial_files(RAW_DATA_DIR, subject=subject, opponent=opponent, task_type='main')

    if max_trials:
        files = files[:max_trials]

    # Load all trials
    trial_dfs = []
    infos = []
    for f in files:
        try:
            trial_dfs.append(load_trial(f))
            infos.append(get_trial_info(f))
        except:
            continue

    # Compute beliefs in batch
    trials_with_beliefs = add_iw_beliefs_batch(trial_dfs)

    # Build result list
    trials = []
    for df_beliefs, info in zip(trials_with_beliefs, infos):
        outcome = get_outcome(df_beliefs)
        trials.append({
            'data': df_beliefs,
            'info': info,
            'outcome': outcome
        })

    return trials


def plot_single_trial(ax, trial, show_title=True):
    """Plot a single trial with trajectory and belief inset."""
    df = trial['data']
    outcome = trial['outcome']
    info = trial['info']

    # Main trajectory plot
    ax.set_facecolor('#FAFAFA')

    # Trajectories with gradient alpha
    n = len(df)
    for i in range(1, n):
        alpha = 0.3 + 0.7 * (i / n)
        ax.plot(df['player1_x'].iloc[i-1:i+1], df['player1_y'].iloc[i-1:i+1],
                '-', color=PLAYER1_COLOR, alpha=alpha, linewidth=2)
        ax.plot(df['player2_x'].iloc[i-1:i+1], df['player2_y'].iloc[i-1:i+1],
                '-', color=PLAYER2_COLOR, alpha=alpha, linewidth=2)

    # Start markers
    ax.plot(df['player1_x'].iloc[0], df['player1_y'].iloc[0], 'o',
            color=PLAYER1_COLOR, markersize=8, alpha=0.5)
    ax.plot(df['player2_x'].iloc[0], df['player2_y'].iloc[0], 'o',
            color=PLAYER2_COLOR, markersize=8, alpha=0.5)

    # End markers
    ax.plot(df['player1_x'].iloc[-1], df['player1_y'].iloc[-1], 'o',
            color=PLAYER1_COLOR, markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(df['player2_x'].iloc[-1], df['player2_y'].iloc[-1], 'o',
            color=PLAYER2_COLOR, markersize=12, markeredgecolor='white', markeredgewidth=2)

    # Targets
    ax.plot(df['stag_x'].iloc[-1], df['stag_y'].iloc[-1], 's',
            color=STAG_COLOR, markersize=15, markeredgecolor='white', markeredgewidth=2,
            label='Stag')
    ax.plot(df['rabbit_x'].iloc[-1], df['rabbit_y'].iloc[-1], '^',
            color=RABBIT_COLOR, markersize=14, markeredgecolor='white', markeredgewidth=2,
            label='Rabbit')

    # Set equal aspect and bounds
    all_x = pd.concat([df['player1_x'], df['player2_x'], df['stag_x'], df['rabbit_x']])
    all_y = pd.concat([df['player1_y'], df['player2_y'], df['stag_y'], df['rabbit_y']])
    padding = 50
    x_min, x_max = all_x.min() - padding, all_x.max() + padding
    y_min, y_max = all_y.min() - padding, all_y.max() + padding

    max_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Border color based on outcome
    border_color = COOP_COLOR if outcome['outcome'] == 'cooperation' else DEFECT_COLOR
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(3)

    if show_title:
        opp = info.get('opponent', '?')
        sub = info.get('subject', '?')
        outcome_str = 'COOP' if outcome['outcome'] == 'cooperation' else 'DEFECT'
        duration = df['time_point'].iloc[-1] - df['time_point'].iloc[0]
        ax.set_title(f"sub-{sub} | {opp} | {outcome_str}\n{duration:.1f}s, {len(df)} frames",
                    fontsize=10, fontweight='bold')


def plot_trial_with_beliefs(ax_traj, ax_belief, trial):
    """Plot trajectory and belief panel side by side."""
    df = trial['data']
    outcome = trial['outcome']
    info = trial['info']

    # Trajectory panel
    plot_single_trial(ax_traj, trial, show_title=False)

    # Belief panel (IW joint goal belief)
    time = np.arange(len(df))
    belief = df['joint_goal_stag']
    ax_belief.fill_between(time, 0.5, belief,
                           where=belief >= 0.5,
                           color=COOP_COLOR, alpha=0.3)
    ax_belief.fill_between(time, 0.5, belief,
                           where=belief < 0.5,
                           color=DEFECT_COLOR, alpha=0.3)
    ax_belief.plot(time, belief, '-', color='#264653', linewidth=2)

    ax_belief.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax_belief.set_ylim(0, 1)
    ax_belief.set_xlim(0, len(df))
    ax_belief.set_ylabel('P(Joint Goal\n= Stag)', fontsize=9)
    ax_belief.set_xlabel('Frame', fontsize=9)

    # Title spanning both
    opp = info.get('opponent', '?')
    sub = info.get('subject', '?')
    outcome_str = 'COOPERATION' if outcome['outcome'] == 'cooperation' else 'DEFECTION'
    duration = df['time_point'].iloc[-1] - df['time_point'].iloc[0]
    ax_traj.set_title(f"sub-{sub} | {opp} | {outcome_str} | {duration:.1f}s",
                     fontsize=11, fontweight='bold')


def make_comparison_figure(trials, output_file='comparison_coop_vs_defect.png'):
    """Show representative cooperation vs defection trials side by side."""

    # Find good examples
    coop_trials = [t for t in trials if t['outcome']['outcome'] == 'cooperation']
    defect_trials = [t for t in trials if t['outcome']['outcome'] != 'cooperation']

    # Pick diverse examples (different opponents)
    def pick_diverse(trial_list, n=3):
        by_opponent = defaultdict(list)
        for t in trial_list:
            opp = t['info'].get('opponent', 'unknown')
            by_opponent[opp].append(t)

        selected = []
        for opp in ['ieeg', 'same', 'diff', 'computer']:
            if by_opponent[opp] and len(selected) < n:
                selected.append(by_opponent[opp][0])

        # Fill remaining
        while len(selected) < n and trial_list:
            for t in trial_list:
                if t not in selected:
                    selected.append(t)
                    break
            else:
                break

        return selected[:n]

    coop_examples = pick_diverse(coop_trials, 3)
    defect_examples = pick_diverse(defect_trials, 3)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Layout: 3 rows × 4 columns (trajectory | beliefs for coop, then trajectory | beliefs for defect)
    gs = GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4,
                  left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Column headers
    fig.text(0.27, 0.95, 'COOPERATION', fontsize=14, fontweight='bold',
             ha='center', color=COOP_COLOR)
    fig.text(0.73, 0.95, 'DEFECTION', fontsize=14, fontweight='bold',
             ha='center', color=DEFECT_COLOR)

    for row in range(3):
        # Cooperation example (left side)
        if row < len(coop_examples):
            ax_traj = fig.add_subplot(gs[row, 0])
            ax_belief = fig.add_subplot(gs[row, 1])
            plot_trial_with_beliefs(ax_traj, ax_belief, coop_examples[row])

        # Defection example (right side)
        if row < len(defect_examples):
            ax_traj = fig.add_subplot(gs[row, 2])
            ax_belief = fig.add_subplot(gs[row, 3])
            plot_trial_with_beliefs(ax_traj, ax_belief, defect_examples[row])

    fig.suptitle('Representative Trials: Cooperation vs Defection',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def make_opponent_comparison(trials, output_file='comparison_by_opponent.png'):
    """Show one example per opponent type."""

    by_opponent = defaultdict(list)
    for t in trials:
        opp = t['info'].get('opponent', 'unknown')
        by_opponent[opp].append(t)

    opponents = ['computer', 'same', 'diff', 'ieeg']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.35,
                  left=0.05, right=0.95, top=0.90, bottom=0.08)

    for col, opp in enumerate(opponents):
        opp_trials = by_opponent.get(opp, [])
        if not opp_trials:
            continue

        # Pick one coop and one defect example
        coop = next((t for t in opp_trials if t['outcome']['outcome'] == 'cooperation'), None)
        defect = next((t for t in opp_trials if t['outcome']['outcome'] != 'cooperation'), None)

        # Stats
        coop_rate = sum(1 for t in opp_trials if t['outcome']['outcome'] == 'cooperation') / len(opp_trials)

        # Column header
        fig.text(0.05 + col * 0.23 + 0.115, 0.93,
                f"{opp.upper()}\nn={len(opp_trials)}, coop={coop_rate:.0%}",
                fontsize=11, fontweight='bold', ha='center')

        # Cooperation example (top row)
        ax = fig.add_subplot(gs[0, col])
        if coop:
            plot_single_trial(ax, coop, show_title=False)
            ax.set_title('Cooperation', fontsize=10, color=COOP_COLOR)
        else:
            ax.text(0.5, 0.5, 'No coop\nexamples', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        # Defection example (bottom row)
        ax = fig.add_subplot(gs[1, col])
        if defect:
            plot_single_trial(ax, defect, show_title=False)
            ax.set_title('Defection', fontsize=10, color=DEFECT_COLOR)
        else:
            ax.text(0.5, 0.5, 'No defect\nexamples', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PLAYER1_COLOR, linewidth=2, label='Player 1'),
        Line2D([0], [0], color=PLAYER2_COLOR, linewidth=2, label='Player 2'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=STAG_COLOR,
               markersize=10, label='Stag'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=RABBIT_COLOR,
               markersize=10, label='Rabbit'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('Trajectory Examples by Opponent Type', fontsize=14, fontweight='bold')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def make_belief_summary(trials, output_file='belief_summary.png'):
    """Show average belief trajectories with confidence bands."""

    by_opponent = defaultdict(lambda: {'coop': [], 'defect': []})

    for t in trials:
        opp = t['info'].get('opponent', 'unknown')
        df = t['data']

        # Normalize time to 0-100 steps
        n_bins = 50
        time_norm = np.linspace(0, 1, len(df))
        time_bins = np.linspace(0, 1, n_bins)

        belief_interp = np.interp(time_bins, time_norm, df['joint_goal_stag'])

        if t['outcome']['outcome'] == 'cooperation':
            by_opponent[opp]['coop'].append(belief_interp)
        else:
            by_opponent[opp]['defect'].append(belief_interp)

    opponents = ['computer', 'same', 'diff', 'ieeg']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, opp in zip(axes, opponents):
        coop_beliefs = np.array(by_opponent[opp]['coop']) if by_opponent[opp]['coop'] else None
        defect_beliefs = np.array(by_opponent[opp]['defect']) if by_opponent[opp]['defect'] else None

        time = np.linspace(0, 100, 50)

        if coop_beliefs is not None and len(coop_beliefs) > 0:
            mean = coop_beliefs.mean(axis=0)
            std = coop_beliefs.std(axis=0)
            ax.fill_between(time, mean - std, mean + std, color=COOP_COLOR, alpha=0.2)
            ax.plot(time, mean, '-', color=COOP_COLOR, linewidth=2,
                   label=f'Coop (n={len(coop_beliefs)})')

        if defect_beliefs is not None and len(defect_beliefs) > 0:
            mean = defect_beliefs.mean(axis=0)
            std = defect_beliefs.std(axis=0)
            ax.fill_between(time, mean - std, mean + std, color=DEFECT_COLOR, alpha=0.2)
            ax.plot(time, mean, '-', color=DEFECT_COLOR, linewidth=2,
                   label=f'Defect (n={len(defect_beliefs)})')

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Normalized Time (%)')
        ax.set_title(f'{opp.upper()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Average Belief\n(partner → stag)', fontsize=11)

    fig.suptitle('Belief Dynamics: Cooperation vs Defection by Opponent Type',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='visualization/output')
    parser.add_argument('--max-trials', type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trials with belief computation...")
    trials = load_trials_with_beliefs(max_trials=args.max_trials)
    print(f"Loaded {len(trials)} trials")

    coop_rate = sum(1 for t in trials if t['outcome']['outcome'] == 'cooperation') / len(trials)
    print(f"Cooperation rate: {coop_rate:.1%}\n")

    # Generate all figures
    print("Generating comparison figures...")
    make_comparison_figure(trials, output_dir / 'v2_coop_vs_defect.png')
    make_opponent_comparison(trials, output_dir / 'v2_by_opponent.png')
    make_belief_summary(trials, output_dir / 'v2_belief_summary.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
