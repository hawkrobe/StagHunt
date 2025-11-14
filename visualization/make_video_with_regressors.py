#!/usr/bin/env python3
"""
Enhanced video showing trajectories + model-based neural regressors.

Layout:
- Main panel: Spatial trajectories (as before)
- Right panels: Time series of key cognitive variables
  1. Cross-trial expectations + within-trial beliefs
  2. Expected values (EU_stag vs EU_rabbit)
  3. Choice probabilities (P_choose_stag)
  4. Coordination estimate (P_coord)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = 'enriched_trials'  # Directory with enriched trial files
OUTPUT_FILE = 'stag_hunt_with_regressors.mp4'

FPS = 15
TRANSITION_DURATION = 1.0

# Colors
PLAYER1_COLOR = '#E63946'  # Red (P1)
PLAYER2_COLOR = '#F4A261'  # Orange (P2)
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
BACKGROUND_COLOR = '#F8F9FA'

print("="*70)
print("CREATING VIDEO WITH NEURAL REGRESSORS")
print("="*70)
print()

# ============================================================================
# LOAD ENRICHED DATA
# ============================================================================

trial_files = sorted(glob.glob(f'{DATA_DIR}/trial_*_with_regressors.csv'))

if not trial_files:
    print(f"ERROR: No enriched trial files found in {DATA_DIR}/")
    print("Run extract_neural_regressors.py first!")
    exit(1)

print(f"✓ Found {len(trial_files)} enriched trial files\n")

# Load all trials
trials_data = []
trial_info = []

for trial_file in trial_files:
    df = pd.read_csv(trial_file)

    # Drop NaN rows
    df = df.dropna(subset=['player1_x', 'player1_y', 'player2_x', 'player2_y'])

    if len(df) == 0:
        continue

    trial_num = int(Path(trial_file).stem.split('_')[1])

    # Get outcome
    outcome = "Ongoing"
    if df['event'].max() == 5:
        outcome = "COOPERATION ✓"
    elif 3 in df['event'].values:
        outcome = "P1 caught rabbit"
    elif 4 in df['event'].values:
        outcome = "P2 caught rabbit"

    # Get cross-trial expectations (constant per trial)
    p1_expectation = df['p1_cross_trial_expectation'].iloc[0]
    p2_expectation = df['p2_cross_trial_expectation'].iloc[0]

    trial_info.append({
        'trial_num': trial_num,
        'n_frames': len(df),
        'duration_s': len(df) * 0.04,  # Assuming ~25Hz
        'outcome': outcome,
        'p1_expectation': p1_expectation,
        'p2_expectation': p2_expectation
    })

    trials_data.append(df)

print("Trial information:")
for info in trial_info:
    print(f"  Trial {info['trial_num']:2d}: {info['n_frames']:4d} frames ({info['duration_s']:5.2f}s), "
          f"{info['outcome']:20s} | Expectations: P1={info['p1_expectation']:.3f}, P2={info['p2_expectation']:.3f}")

print()

# ============================================================================
# PREPARE ANIMATION DATA
# ============================================================================

print("="*70)
print("PREPARING ANIMATION DATA")
print("="*70)
print()

# Compute viewing window from all data
all_x = []
all_y = []

for df in trials_data:
    all_x.extend(df['player1_x'].values)
    all_x.extend(df['player2_x'].values)
    all_x.extend(df['stag_x'].values)
    all_x.extend(df['rabbit_x'].values)

    all_y.extend(df['player1_y'].values)
    all_y.extend(df['player2_y'].values)
    all_y.extend(df['stag_y'].values)
    all_y.extend(df['rabbit_y'].values)

x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)

x_range = x_max - x_min
y_range = y_max - y_min
margin = 0.1

xlim = (x_min - margin * x_range, x_max + margin * x_range)
ylim = (y_min - margin * y_range, y_max + margin * y_range)

print(f"Viewing window: X=[{xlim[0]:.1f}, {xlim[1]:.1f}], Y=[{ylim[0]:.1f}, {ylim[1]:.1f}]\n")

# Create frame schedule
frame_schedule = []
cumulative_time = 0.0

for trial_idx, (df, info) in enumerate(zip(trials_data, trial_info)):
    n_frames = len(df)

    for frame_idx in range(n_frames):
        frame_schedule.append({
            'trial_idx': trial_idx,
            'frame_idx': frame_idx,
            'time': cumulative_time + frame_idx / FPS,
            'trial_num': info['trial_num']
        })

    cumulative_time += n_frames / FPS + TRANSITION_DURATION

total_frames = len(frame_schedule)
total_duration = cumulative_time - TRANSITION_DURATION

print(f"Total frames: {total_frames} ({total_duration:.1f} seconds)\n")

# ============================================================================
# CREATE FIGURE WITH MULTIPLE PANELS
# ============================================================================

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(4, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.3)

# Main panel: Spatial trajectories
ax_main = fig.add_subplot(gs[:, 0])
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_aspect('equal')
ax_main.set_facecolor(BACKGROUND_COLOR)
ax_main.set_xlabel('X Position', fontsize=12)
ax_main.set_ylabel('Y Position', fontsize=12)

# Right panels: Regressors
ax_beliefs = fig.add_subplot(gs[0, 1])  # Beliefs
ax_values = fig.add_subplot(gs[1, 1])   # Expected values
ax_choice = fig.add_subplot(gs[2, 1])   # Choice probabilities
ax_coord = fig.add_subplot(gs[3, 1])    # Coordination

# ============================================================================
# ANIMATION UPDATE FUNCTION
# ============================================================================

# Initialize plot elements
player1_marker, = ax_main.plot([], [], 'o', color=PLAYER1_COLOR, markersize=15, label='Player 1')
player2_marker, = ax_main.plot([], [], 'o', color=PLAYER2_COLOR, markersize=15, label='Player 2')
stag_marker, = ax_main.plot([], [], 's', color=STAG_COLOR, markersize=12, label='Stag')
rabbit_marker, = ax_main.plot([], [], '^', color=RABBIT_COLOR, markersize=12, label='Rabbit')

player1_trail, = ax_main.plot([], [], '-', color=PLAYER1_COLOR, alpha=0.3, linewidth=2)
player2_trail, = ax_main.plot([], [], '-', color=PLAYER2_COLOR, alpha=0.3, linewidth=2)

title_text = ax_main.text(0.5, 1.05, '', transform=ax_main.transAxes,
                          fontsize=14, ha='center', weight='bold')

ax_main.legend(loc='upper right', fontsize=10)

# Regressor plots
belief_p1_line, = ax_beliefs.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, label='P1 belief')
belief_p2_line, = ax_beliefs.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, label='P2 belief')
expect_p1_line, = ax_beliefs.plot([], [], '--', color=PLAYER1_COLOR, linewidth=1.5, alpha=0.5, label='P1 expectation')
expect_p2_line, = ax_beliefs.plot([], [], '--', color=PLAYER2_COLOR, linewidth=1.5, alpha=0.5, label='P2 expectation')

eu_stag_p1, = ax_values.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, label='P1 EU(stag)')
eu_rabbit_p1, = ax_values.plot([], [], ':', color=PLAYER1_COLOR, linewidth=2, label='P1 EU(rabbit)')
eu_stag_p2, = ax_values.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, label='P2 EU(stag)')
eu_rabbit_p2, = ax_values.plot([], [], ':', color=PLAYER2_COLOR, linewidth=2, label='P2 EU(rabbit)')

choice_p1, = ax_choice.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, label='P1 P(choose stag)')
choice_p2, = ax_choice.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, label='P2 P(choose stag)')

coord_p1, = ax_coord.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, label='P1 P_coord')
coord_p2, = ax_coord.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, label='P2 P_coord')

# Configure regressor axes
for ax, ylabel in [(ax_beliefs, 'Belief/Expectation'),
                   (ax_values, 'Expected Utility'),
                   (ax_choice, 'P(choose stag)'),
                   (ax_coord, 'P_coord')]:
    ax.set_xlim(0, 10)  # Will update dynamically
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

ax_coord.set_xlabel('Time (s)', fontsize=10)

def init():
    """Initialize animation."""
    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line, expect_p1_line, expect_p2_line,
            eu_stag_p1, eu_rabbit_p1, eu_stag_p2, eu_rabbit_p2,
            choice_p1, choice_p2, coord_p1, coord_p2)

def update(frame_num):
    """Update animation for given frame."""
    if frame_num >= len(frame_schedule):
        return init()

    frame_info = frame_schedule[frame_num]
    trial_idx = frame_info['trial_idx']
    frame_idx = frame_info['frame_idx']
    trial_num = frame_info['trial_num']

    df = trials_data[trial_idx]
    info = trial_info[trial_idx]
    row = df.iloc[frame_idx]

    # Update spatial positions
    player1_marker.set_data([row['player1_x']], [row['player1_y']])
    player2_marker.set_data([row['player2_x']], [row['player2_y']])
    stag_marker.set_data([row['stag_x']], [row['stag_y']])
    rabbit_marker.set_data([row['rabbit_x']], [row['rabbit_y']])

    # Update trails
    trail_start = max(0, frame_idx - 30)
    player1_trail.set_data(df['player1_x'].iloc[trail_start:frame_idx+1],
                          df['player1_y'].iloc[trail_start:frame_idx+1])
    player2_trail.set_data(df['player2_x'].iloc[trail_start:frame_idx+1],
                          df['player2_y'].iloc[trail_start:frame_idx+1])

    # Update title
    title_text.set_text(f"Trial {trial_num} | {info['outcome']} | "
                       f"Frame {frame_idx+1}/{info['n_frames']}")

    # Update regressor time series (show current trial only)
    times = df['time_point'].iloc[:frame_idx+1].values - df['time_point'].iloc[0]

    # Beliefs
    belief_p1_line.set_data(times, df['p1_belief_p2_stag'].iloc[:frame_idx+1])
    belief_p2_line.set_data(times, df['p2_belief_p1_stag'].iloc[:frame_idx+1])
    expect_p1_line.set_data([0, times[-1]], [info['p1_expectation'], info['p1_expectation']])
    expect_p2_line.set_data([0, times[-1]], [info['p2_expectation'], info['p2_expectation']])

    # Expected values
    eu_stag_p1.set_data(times, df['p1_EU_stag'].iloc[:frame_idx+1])
    eu_rabbit_p1.set_data(times, df['p1_EU_rabbit'].iloc[:frame_idx+1])
    eu_stag_p2.set_data(times, df['p2_EU_stag'].iloc[:frame_idx+1])
    eu_rabbit_p2.set_data(times, df['p2_EU_rabbit'].iloc[:frame_idx+1])

    # Choice probabilities
    choice_p1.set_data(times, df['p1_P_choose_stag'].iloc[:frame_idx+1])
    choice_p2.set_data(times, df['p2_P_choose_stag'].iloc[:frame_idx+1])

    # Coordination
    coord_p1.set_data(times, df['p1_P_coord'].iloc[:frame_idx+1])
    coord_p2.set_data(times, df['p2_P_coord'].iloc[:frame_idx+1])

    # Update x-axis limits for regressor plots
    if times[-1] > 0:
        for ax in [ax_beliefs, ax_values, ax_choice, ax_coord]:
            ax.set_xlim(0, times[-1] * 1.1)

    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line, expect_p1_line, expect_p2_line,
            eu_stag_p1, eu_rabbit_p1, eu_stag_p2, eu_rabbit_p2,
            choice_p1, choice_p2, coord_p1, coord_p2)

# ============================================================================
# GENERATE ANIMATION
# ============================================================================

print("="*70)
print("CREATING ANIMATION WITH REGRESSOR VISUALIZATION")
print("="*70)
print()

anim = FuncAnimation(fig, update, init_func=init,
                    frames=total_frames, interval=1000/FPS,
                    blit=True, repeat=False)

print(f"Saving to {OUTPUT_FILE}...")
print("(This may take several minutes)")
print()

writer = FFMpegWriter(fps=FPS, metadata={'artist': 'Stag Hunt Analysis'},
                     bitrate=3000)

anim.save(OUTPUT_FILE, writer=writer)

print("="*70)
print("✓ VIDEO WITH REGRESSORS SAVED SUCCESSFULLY!")
print("="*70)
print(f"File: {OUTPUT_FILE}")
print(f"Duration: {total_duration:.1f} seconds")
print(f"Frames: {total_frames}")
