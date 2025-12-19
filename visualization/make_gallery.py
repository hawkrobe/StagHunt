#!/usr/bin/env python3
"""
Create gallery visualizations of Stag Hunt trials across conditions.

Generates static images showing trajectory patterns and belief dynamics
organized by opponent type, reward condition, and outcome.

Usage:
------
# Create main gallery (trajectory thumbnails by condition)
python visualization/make_gallery.py

# Focus on specific conditions
python visualization/make_gallery.py --opponent ieeg
python visualization/make_gallery.py --subject 120

# Create summary statistics figure
python visualization/make_gallery.py --summary

# Create belief dynamics overview
python visualization/make_gallery.py --beliefs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import argparse
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stag_hunt import load_trial, find_trial_files, get_trial_info, get_outcome, RAW_DATA_DIR
from models.belief import add_iw_beliefs

# Colors
PLAYER1_COLOR = '#E63946'
PLAYER2_COLOR = '#F4A261'
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
COOP_COLOR = '#2A9D8F'
DEFECT_COLOR = '#E63946'


def load_all_trial_data(subject=None, opponent=None, reward=None, max_trials=None, verbose=True):
    """Load trial data with metadata."""
    files = find_trial_files(
        RAW_DATA_DIR,
        subject=subject,
        opponent=opponent,
        reward=reward,
        task_type='main'
    )

    if max_trials and len(files) > max_trials:
        files = files[:max_trials]

    if verbose:
        print(f"Loading {len(files)} trials...")

    trials = []
    for f in files:
        try:
            df = load_trial(f)
            info = get_trial_info(f)
            outcome = get_outcome(df)
            trials.append({
                'data': df,
                'info': info,
                'outcome': outcome,
                'filepath': f
            })
        except Exception as e:
            if verbose:
                print(f"  Error loading {f}: {e}")

    if verbose:
        print(f"Successfully loaded {len(trials)} trials")

    return trials


def plot_trajectory_thumbnail(ax, trial_data, show_labels=False):
    """Plot a single trial trajectory as a small thumbnail."""
    df = trial_data['data']
    outcome = trial_data['outcome']

    # Get coordinate bounds
    x_min = min(df['player1_x'].min(), df['player2_x'].min(),
                df['stag_x'].min(), df['rabbit_x'].min()) - 20
    x_max = max(df['player1_x'].max(), df['player2_x'].max(),
                df['stag_x'].max(), df['rabbit_x'].max()) + 20
    y_min = min(df['player1_y'].min(), df['player2_y'].min(),
                df['stag_y'].min(), df['rabbit_y'].min()) - 20
    y_max = max(df['player1_y'].max(), df['player2_y'].max(),
                df['stag_y'].max(), df['rabbit_y'].max()) + 20

    # Make square
    max_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Background color based on outcome
    if outcome['outcome'] == 'cooperation':
        ax.set_facecolor('#E8F5E9')  # Light green
    elif outcome['outcome'] == 'mutual_defection':
        ax.set_facecolor('#FFEBEE')  # Light red
    else:
        ax.set_facecolor('#FFF8E1')  # Light yellow

    # Plot trajectories
    ax.plot(df['player1_x'], df['player1_y'], '-', color=PLAYER1_COLOR,
            alpha=0.7, linewidth=1)
    ax.plot(df['player2_x'], df['player2_y'], '-', color=PLAYER2_COLOR,
            alpha=0.7, linewidth=1)

    # Start positions (small dots)
    ax.plot(df['player1_x'].iloc[0], df['player1_y'].iloc[0], 'o',
            color=PLAYER1_COLOR, markersize=3)
    ax.plot(df['player2_x'].iloc[0], df['player2_y'].iloc[0], 'o',
            color=PLAYER2_COLOR, markersize=3)

    # End positions (larger dots)
    ax.plot(df['player1_x'].iloc[-1], df['player1_y'].iloc[-1], 'o',
            color=PLAYER1_COLOR, markersize=6, markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(df['player2_x'].iloc[-1], df['player2_y'].iloc[-1], 'o',
            color=PLAYER2_COLOR, markersize=6, markeredgecolor='white', markeredgewidth=0.5)

    # Targets
    ax.plot(df['stag_x'].iloc[-1], df['stag_y'].iloc[-1], 's',
            color=STAG_COLOR, markersize=8, markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(df['rabbit_x'].iloc[-1], df['rabbit_y'].iloc[-1], '^',
            color=RABBIT_COLOR, markersize=7, markeredgecolor='white', markeredgewidth=0.5)


def make_condition_gallery(trials, output_file='gallery_by_condition.png'):
    """Create gallery organized by opponent type and reward condition."""

    # Organize trials by condition
    by_condition = defaultdict(list)
    for t in trials:
        opponent = t['info'].get('opponent', 'unknown')
        reward = t['info'].get('reward', 'unknown')
        by_condition[(opponent, reward)].append(t)

    # Define grid
    opponents = ['computer', 'same', 'diff', 'ieeg']
    rewards = ['rabbitincrease', 'stagdecrease']

    n_cols_per_cell = 6  # trials per condition cell
    n_rows_per_cell = 2

    fig = plt.figure(figsize=(20, 12))

    # Create outer grid for conditions
    outer_grid = GridSpec(len(rewards), len(opponents),
                          wspace=0.1, hspace=0.15,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    for row, reward in enumerate(rewards):
        for col, opponent in enumerate(opponents):
            condition_trials = by_condition.get((opponent, reward), [])

            # Create inner grid for thumbnails
            inner_grid = outer_grid[row, col].subgridspec(
                n_rows_per_cell, n_cols_per_cell, wspace=0.05, hspace=0.05
            )

            # Calculate cooperation rate
            if condition_trials:
                coop_rate = sum(1 for t in condition_trials
                               if t['outcome']['outcome'] == 'cooperation') / len(condition_trials)
            else:
                coop_rate = 0

            # Add condition label
            ax_label = fig.add_subplot(outer_grid[row, col])
            ax_label.set_xticks([])
            ax_label.set_yticks([])
            ax_label.patch.set_alpha(0)
            for spine in ax_label.spines.values():
                spine.set_visible(False)

            # Title with cooperation rate
            title = f"{opponent.upper()}\n{reward}\nn={len(condition_trials)}, coop={coop_rate:.0%}"
            ax_label.set_title(title, fontsize=9, fontweight='bold', pad=2)

            # Plot thumbnails
            for idx in range(n_rows_per_cell * n_cols_per_cell):
                i_row = idx // n_cols_per_cell
                i_col = idx % n_cols_per_cell

                ax = fig.add_subplot(inner_grid[i_row, i_col])

                if idx < len(condition_trials):
                    plot_trajectory_thumbnail(ax, condition_trials[idx])
                else:
                    ax.set_visible(False)

    fig.suptitle('Stag Hunt Trial Gallery by Condition', fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=PLAYER1_COLOR, linewidth=2, label='Player 1'),
        plt.Line2D([0], [0], color=PLAYER2_COLOR, linewidth=2, label='Player 2'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=STAG_COLOR,
                   markersize=10, label='Stag'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=RABBIT_COLOR,
                   markersize=10, label='Rabbit'),
        patches.Patch(facecolor='#E8F5E9', edgecolor='gray', label='Cooperation'),
        patches.Patch(facecolor='#FFEBEE', edgecolor='gray', label='Defection'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def make_subject_gallery(trials, output_file='gallery_by_subject.png'):
    """Create gallery organized by subject."""

    # Organize by subject
    by_subject = defaultdict(list)
    for t in trials:
        subject = t['info'].get('subject', 'unknown')
        by_subject[subject].append(t)

    subjects = sorted([s for s in by_subject.keys() if s])
    n_subjects = len(subjects)

    if n_subjects == 0:
        print("No subjects found")
        return

    # Layout
    n_cols = 8  # trials per row
    n_rows_per_subject = 2

    fig = plt.figure(figsize=(18, 3 * n_subjects))

    outer_grid = GridSpec(n_subjects, 1, hspace=0.2,
                          left=0.08, right=0.98, top=0.95, bottom=0.02)

    for sub_idx, subject in enumerate(subjects):
        subject_trials = by_subject[subject][:n_cols * n_rows_per_subject]

        # Cooperation rate
        coop_rate = sum(1 for t in by_subject[subject]
                       if t['outcome']['outcome'] == 'cooperation') / len(by_subject[subject])

        inner_grid = outer_grid[sub_idx].subgridspec(
            n_rows_per_subject, n_cols, wspace=0.03, hspace=0.03
        )

        for idx, trial in enumerate(subject_trials):
            i_row = idx // n_cols
            i_col = idx % n_cols

            ax = fig.add_subplot(inner_grid[i_row, i_col])
            plot_trajectory_thumbnail(ax, trial)

            # Add opponent label on first row
            if i_row == 0:
                opp = trial['info'].get('opponent', '?')[:4]
                ax.set_title(opp, fontsize=7, pad=1)

        # Subject label
        ax_label = fig.add_subplot(outer_grid[sub_idx])
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        ax_label.patch.set_alpha(0)
        for spine in ax_label.spines.values():
            spine.set_visible(False)
        ax_label.set_ylabel(f"sub-{subject}\nn={len(by_subject[subject])}\ncoop={coop_rate:.0%}",
                           fontsize=10, fontweight='bold', rotation=0, ha='right', va='center',
                           labelpad=50)

    fig.suptitle('Stag Hunt Trial Gallery by Subject', fontsize=14, fontweight='bold')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def make_summary_figure(trials, output_file='summary_statistics.png'):
    """Create summary statistics figure."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Cooperation rate by opponent
    ax = axes[0, 0]
    opponents = ['computer', 'same', 'diff', 'ieeg']
    coop_rates = []
    n_trials = []
    for opp in opponents:
        opp_trials = [t for t in trials if t['info'].get('opponent') == opp]
        if opp_trials:
            rate = sum(1 for t in opp_trials if t['outcome']['outcome'] == 'cooperation') / len(opp_trials)
            coop_rates.append(rate)
            n_trials.append(len(opp_trials))
        else:
            coop_rates.append(0)
            n_trials.append(0)

    bars = ax.bar(opponents, coop_rates, color=[COOP_COLOR if r > 0.3 else DEFECT_COLOR for r in coop_rates])
    ax.set_ylabel('Cooperation Rate')
    ax.set_xlabel('Opponent Type')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cooperation by Opponent Type')
    for i, (rate, n) in enumerate(zip(coop_rates, n_trials)):
        ax.annotate(f'{rate:.0%}\n(n={n})', xy=(i, rate), ha='center', va='bottom', fontsize=9)

    # 2. Cooperation rate by reward
    ax = axes[0, 1]
    rewards = ['rabbitincrease', 'stagdecrease']
    coop_rates = []
    for rew in rewards:
        rew_trials = [t for t in trials if t['info'].get('reward') == rew]
        if rew_trials:
            rate = sum(1 for t in rew_trials if t['outcome']['outcome'] == 'cooperation') / len(rew_trials)
            coop_rates.append(rate)
        else:
            coop_rates.append(0)

    ax.bar(['Rabbit↑', 'Stag↓'], coop_rates, color=[COOP_COLOR, COOP_COLOR])
    ax.set_ylabel('Cooperation Rate')
    ax.set_xlabel('Reward Condition')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cooperation by Reward Condition')
    for i, rate in enumerate(coop_rates):
        ax.annotate(f'{rate:.0%}', xy=(i, rate), ha='center', va='bottom', fontsize=10)

    # 3. Cooperation rate by subject
    ax = axes[0, 2]
    by_subject = defaultdict(list)
    for t in trials:
        sub = t['info'].get('subject')
        if sub:
            by_subject[sub].append(t)

    subjects = sorted(by_subject.keys())
    coop_rates = []
    for sub in subjects:
        rate = sum(1 for t in by_subject[sub] if t['outcome']['outcome'] == 'cooperation') / len(by_subject[sub])
        coop_rates.append(rate)

    ax.bar([f'{s}' for s in subjects], coop_rates,
           color=[COOP_COLOR if r > 0.3 else DEFECT_COLOR for r in coop_rates])
    ax.set_ylabel('Cooperation Rate')
    ax.set_xlabel('Subject')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cooperation by Subject')
    ax.tick_params(axis='x', rotation=45)

    # 4. Trial duration distribution
    ax = axes[1, 0]
    coop_durations = []
    defect_durations = []
    for t in trials:
        df = t['data']
        duration = df['time_point'].iloc[-1] - df['time_point'].iloc[0]
        if t['outcome']['outcome'] == 'cooperation':
            coop_durations.append(duration)
        else:
            defect_durations.append(duration)

    ax.hist([coop_durations, defect_durations], bins=30,
            label=['Cooperation', 'Defection'], color=[COOP_COLOR, DEFECT_COLOR], alpha=0.7)
    ax.set_xlabel('Trial Duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Trial Duration by Outcome')
    ax.legend()

    # 5. Outcome breakdown
    ax = axes[1, 1]
    outcomes = defaultdict(int)
    for t in trials:
        outcomes[t['outcome']['outcome']] += 1

    labels = ['Cooperation', 'Mutual\nDefection', 'P1 Stag\nP2 Rabbit', 'P1 Rabbit\nP2 Stag']
    keys = ['cooperation', 'mutual_defection', 'p1_stag_p2_rabbit', 'p1_rabbit_p2_stag']
    values = [outcomes.get(k, 0) for k in keys]
    colors = [COOP_COLOR, DEFECT_COLOR, '#FFA726', '#FFB74D']

    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Outcome Distribution')

    # 6. Opponent × Reward heatmap
    ax = axes[1, 2]
    matrix = np.zeros((len(rewards), len(opponents)))
    for i, rew in enumerate(rewards):
        for j, opp in enumerate(opponents):
            cond_trials = [t for t in trials
                          if t['info'].get('opponent') == opp and t['info'].get('reward') == rew]
            if cond_trials:
                matrix[i, j] = sum(1 for t in cond_trials
                                  if t['outcome']['outcome'] == 'cooperation') / len(cond_trials)

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(opponents)))
    ax.set_xticklabels(opponents)
    ax.set_yticks(range(len(rewards)))
    ax.set_yticklabels(['Rabbit↑', 'Stag↓'])
    ax.set_title('Cooperation Rate: Opponent × Reward')

    # Add values
    for i in range(len(rewards)):
        for j in range(len(opponents)):
            ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center',
                   color='white' if matrix[i,j] < 0.5 else 'black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Cooperation Rate')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def make_belief_overview(trials, output_file='belief_dynamics.png', n_samples=50):
    """Create overview of belief dynamics across conditions."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    opponents = ['computer', 'same', 'diff', 'ieeg']

    for col, opponent in enumerate(opponents):
        opp_trials = [t for t in trials if t['info'].get('opponent') == opponent]

        if not opp_trials:
            continue

        # Sample trials
        sample_trials = opp_trials[:n_samples]

        # Run belief model on all samples at once (batched)
        trial_dfs = [t['data'] for t in sample_trials]
        trials_with_beliefs = add_iw_beliefs(trial_dfs)

        coop_beliefs = []
        defect_beliefs = []

        for t, df_with_beliefs in zip(sample_trials, trials_with_beliefs):
            # Normalize time to 0-1
            n_steps = len(df_with_beliefs)
            time_norm = np.linspace(0, 1, n_steps)

            # IW model has joint_goal_stag column
            beliefs = df_with_beliefs['joint_goal_stag'].values

            if t['outcome']['outcome'] == 'cooperation':
                coop_beliefs.append((time_norm, beliefs))
            else:
                defect_beliefs.append((time_norm, beliefs))

        # Plot joint goal beliefs (top row = coop, bottom row = defect)
        ax = axes[0, col]
        for time_norm, beliefs in coop_beliefs:
            ax.plot(time_norm, beliefs, color=COOP_COLOR, alpha=0.3, linewidth=0.5)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(f'{opponent.upper()}\n(n={len(opp_trials)})', fontweight='bold')
        if col == 0:
            ax.set_ylabel('Cooperation Trials\nP(Joint Goal = Stag)')
        ax.set_xticks([])

        ax = axes[1, col]
        for time_norm, beliefs in defect_beliefs:
            ax.plot(time_norm, beliefs, color=DEFECT_COLOR, alpha=0.3, linewidth=0.5)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Time')
        if col == 0:
            ax.set_ylabel('Defection Trials\nP(Joint Goal = Stag)')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=COOP_COLOR, linewidth=2, label='Cooperation trials'),
        plt.Line2D([0], [0], color=DEFECT_COLOR, linewidth=2, label='Defection trials'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Belief Dynamics by Opponent Type', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create Stag Hunt gallery visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--subject', type=str, help='Filter by subject ID')
    parser.add_argument('--opponent', type=str, help='Filter by opponent type')
    parser.add_argument('--reward', type=str, help='Filter by reward condition')
    parser.add_argument('--summary', action='store_true', help='Create summary statistics figure')
    parser.add_argument('--beliefs', action='store_true', help='Create belief dynamics figure')
    parser.add_argument('--by-subject', action='store_true', help='Create gallery organized by subject')
    parser.add_argument('--output-dir', type=str, default='visualization/output',
                       help='Output directory for figures')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Maximum trials to load')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    trials = load_all_trial_data(
        subject=args.subject,
        opponent=args.opponent,
        reward=args.reward,
        max_trials=args.max_trials
    )

    if not trials:
        print("No trials found!")
        return 1

    print(f"\nLoaded {len(trials)} trials")
    coop_count = sum(1 for t in trials if t['outcome']['outcome'] == 'cooperation')
    print(f"Cooperation rate: {coop_count}/{len(trials)} ({100*coop_count/len(trials):.1f}%)")

    # Generate requested visualizations
    if args.summary:
        make_summary_figure(trials, output_dir / 'summary_statistics.png')
    elif args.beliefs:
        make_belief_overview(trials, output_dir / 'belief_dynamics.png')
    elif args.by_subject:
        make_subject_gallery(trials, output_dir / 'gallery_by_subject.png')
    else:
        # Default: create condition gallery
        make_condition_gallery(trials, output_dir / 'gallery_by_condition.png')

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
