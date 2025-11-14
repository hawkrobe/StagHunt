#!/usr/bin/env python3
"""
Enhanced video focusing on COORDINATION ESTIMATES as markers of cooperation.

Layout:
- Main panel: Spatial trajectories
- Large coordination panel: P_coord time series (key cooperation marker)
- Supporting panels:
  - Beliefs × timing alignment = P_coord breakdown
  - Cross-trial expectations (evolving priors)
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
# Use integrated model output
DATA_DIR = '.'
DATA_FILE = 'integrated_model_lr0.3.csv'  # Use learning rate 0.3
OUTPUT_FILE = 'stag_hunt_coordination_focus.mp4'

FPS = 15
TRANSITION_DURATION = 1.0

# Colors
PLAYER1_COLOR = '#E63946'  # Red (P1)
PLAYER2_COLOR = '#F4A261'  # Orange (P2)
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
BACKGROUND_COLOR = '#F8F9FA'
COORD_COLOR = '#457B9D'  # Blue for coordination

print("="*70)
print("CREATING VIDEO: COORDINATION AS COOPERATION MARKER")
print("="*70)
print()

# ============================================================================
# LOAD INTEGRATED MODEL DATA
# ============================================================================

print(f"Loading integrated model output: {DATA_FILE}")
all_data = pd.read_csv(DATA_FILE)

# Split by trial
trial_nums = sorted(all_data['trial_num'].unique())
trials_data = []
trial_info = []

for trial_num in trial_nums:
    df = all_data[all_data['trial_num'] == trial_num].copy()
    df = df.reset_index(drop=True)

    # Drop NaN rows
    df = df.dropna(subset=['player1_x', 'player1_y', 'player2_x', 'player2_y'])

    if len(df) == 0:
        continue

    # Get outcome
    outcome = "Ongoing"
    if df['event'].max() == 5:
        outcome = "COOPERATION ✓"
    elif 3 in df['event'].values:
        outcome = "P1 caught rabbit"
    elif 4 in df['event'].values:
        outcome = "P2 caught rabbit"

    # Get cross-trial expectations
    p1_expectation = df['p1_cross_trial_expectation'].iloc[0]
    p2_expectation = df['p2_cross_trial_expectation'].iloc[0]

    # Get coordination statistics
    p1_coord_mean = df['p1_P_coord'].mean()
    p2_coord_mean = df['p2_P_coord'].mean()
    p1_coord_max = df['p1_P_coord'].max()
    p2_coord_max = df['p2_P_coord'].max()

    trial_info.append({
        'trial_num': trial_num,
        'n_frames': len(df),
        'duration_s': len(df) * 0.04,
        'outcome': outcome,
        'p1_expectation': p1_expectation,
        'p2_expectation': p2_expectation,
        'p1_coord_mean': p1_coord_mean,
        'p2_coord_mean': p2_coord_mean,
        'p1_coord_max': p1_coord_max,
        'p2_coord_max': p2_coord_max,
    })

    trials_data.append(df)

print(f"✓ Loaded {len(trials_data)} trials\n")

print("Trial coordination statistics:")
for info in trial_info:
    print(f"  Trial {info['trial_num']:2d}: {info['outcome']:20s} | "
          f"Mean P_coord: P1={info['p1_coord_mean']:.3f}, P2={info['p2_coord_mean']:.3f} | "
          f"Max: P1={info['p1_coord_max']:.3f}, P2={info['p2_coord_max']:.3f}")

print()

# ============================================================================
# PREPARE ANIMATION DATA
# ============================================================================

# Compute viewing window
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
# CREATE FIGURE WITH COORDINATION EMPHASIS
# ============================================================================

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(5, 2, figure=fig, width_ratios=[2, 1], hspace=0.4, wspace=0.3)

# Main panel: Spatial trajectories (left, full height)
ax_main = fig.add_subplot(gs[:, 0])
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_aspect('equal')
ax_main.set_facecolor(BACKGROUND_COLOR)
ax_main.set_xlabel('X Position', fontsize=12)
ax_main.set_ylabel('Y Position', fontsize=12)

# Right panels: Coordination-focused
ax_coord = fig.add_subplot(gs[0:2, 1])     # LARGE: P_coord (KEY MARKER)
ax_breakdown = fig.add_subplot(gs[2:4, 1]) # Belief × timing breakdown
ax_expectations = fig.add_subplot(gs[4, 1]) # Cross-trial expectations

# ============================================================================
# ANIMATION ELEMENTS
# ============================================================================

# Spatial markers
player1_marker, = ax_main.plot([], [], 'o', color=PLAYER1_COLOR, markersize=15, label='Player 1')
player2_marker, = ax_main.plot([], [], 'o', color=PLAYER2_COLOR, markersize=15, label='Player 2')
stag_marker, = ax_main.plot([], [], 's', color=STAG_COLOR, markersize=12, label='Stag')
rabbit_marker, = ax_main.plot([], [], '^', color=RABBIT_COLOR, markersize=12, label='Rabbit')

player1_trail, = ax_main.plot([], [], '-', color=PLAYER1_COLOR, alpha=0.3, linewidth=2)
player2_trail, = ax_main.plot([], [], '-', color=PLAYER2_COLOR, alpha=0.3, linewidth=2)

title_text = ax_main.text(0.5, 1.05, '', transform=ax_main.transAxes,
                          fontsize=14, ha='center', weight='bold')

ax_main.legend(loc='upper right', fontsize=10)

# === COORDINATION PANEL (KEY MARKER) ===
coord_p1_line, = ax_coord.plot([], [], '-', color=PLAYER1_COLOR, linewidth=3, label='P1 P_coord')
coord_p2_line, = ax_coord.plot([], [], '-', color=PLAYER2_COLOR, linewidth=3, label='P2 P_coord')

# Highlight cooperation threshold
ax_coord.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')

ax_coord.set_xlim(0, 10)
ax_coord.set_ylim(0, 1.0)
ax_coord.set_ylabel('Coordination Probability\n(KEY MARKER)', fontsize=11, weight='bold')
ax_coord.set_title('P_coord = belief × timing_alignment', fontsize=10, style='italic')
ax_coord.legend(loc='upper right', fontsize=9)
ax_coord.grid(alpha=0.3)

# === BREAKDOWN PANEL ===
belief_p1, = ax_breakdown.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, alpha=0.7, label='P1 belief')
belief_p2, = ax_breakdown.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, alpha=0.7, label='P2 belief')
timing_p1, = ax_breakdown.plot([], [], ':', color=PLAYER1_COLOR, linewidth=2, alpha=0.7, label='P1 timing')
timing_p2, = ax_breakdown.plot([], [], ':', color=PLAYER2_COLOR, linewidth=2, alpha=0.7, label='P2 timing')

ax_breakdown.set_xlim(0, 10)
ax_breakdown.set_ylim(0, 1.0)
ax_breakdown.set_ylabel('Components', fontsize=10)
ax_breakdown.legend(loc='upper right', fontsize=8, ncol=2)
ax_breakdown.grid(alpha=0.3)

# === CROSS-TRIAL EXPECTATIONS ===
expect_p1, = ax_expectations.plot([], [], '-', color=PLAYER1_COLOR, linewidth=2, label='P1 expectation')
expect_p2, = ax_expectations.plot([], [], '-', color=PLAYER2_COLOR, linewidth=2, label='P2 expectation')

ax_expectations.set_xlim(0, 10)
ax_expectations.set_ylim(0, 1.0)
ax_expectations.set_xlabel('Time (s)', fontsize=10)
ax_expectations.set_ylabel('Cross-trial\nExpectations', fontsize=10)
ax_expectations.legend(loc='upper right', fontsize=8)
ax_expectations.grid(alpha=0.3)

def init():
    """Initialize animation."""
    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            coord_p1_line, coord_p2_line,
            belief_p1, belief_p2, timing_p1, timing_p2,
            expect_p1, expect_p2)

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

    # Update title with coordination statistics
    p1_coord_current = row['p1_P_coord']
    p2_coord_current = row['p2_P_coord']

    title_text.set_text(f"Trial {trial_num} | {info['outcome']} | "
                       f"P_coord: P1={p1_coord_current:.2f}, P2={p2_coord_current:.2f}")

    # Update time series (show current trial only)
    times = df['time_point'].iloc[:frame_idx+1].values - df['time_point'].iloc[0]

    # === COORDINATION (KEY) ===
    coord_p1_line.set_data(times, df['p1_P_coord'].iloc[:frame_idx+1])
    coord_p2_line.set_data(times, df['p2_P_coord'].iloc[:frame_idx+1])

    # === BREAKDOWN ===
    belief_p1.set_data(times, df['p1_belief_p2_stag'].iloc[:frame_idx+1])
    belief_p2.set_data(times, df['p2_belief_p1_stag'].iloc[:frame_idx+1])
    timing_p1.set_data(times, df['p1_timing_alignment'].iloc[:frame_idx+1])
    timing_p2.set_data(times, df['p2_timing_alignment'].iloc[:frame_idx+1])

    # === CROSS-TRIAL (constant per trial) ===
    expect_p1.set_data([0, times[-1]], [info['p1_expectation'], info['p1_expectation']])
    expect_p2.set_data([0, times[-1]], [info['p2_expectation'], info['p2_expectation']])

    # Update x-axis limits
    if times[-1] > 0:
        for ax in [ax_coord, ax_breakdown, ax_expectations]:
            ax.set_xlim(0, times[-1] * 1.1)

    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            coord_p1_line, coord_p2_line,
            belief_p1, belief_p2, timing_p1, timing_p2,
            expect_p1, expect_p2)

# ============================================================================
# GENERATE ANIMATION
# ============================================================================

print("="*70)
print("CREATING COORDINATION-FOCUSED VIDEO")
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
print("✓ COORDINATION-FOCUSED VIDEO SAVED SUCCESSFULLY!")
print("="*70)
print(f"File: {OUTPUT_FILE}")
print(f"Duration: {total_duration:.1f} seconds")
print(f"Frames: {total_frames}")
print()
print("Key features:")
print("  - Large P_coord panel (primary cooperation marker)")
print("  - Breakdown showing belief × timing components")
print("  - Cross-trial expectations (evolving priors)")
