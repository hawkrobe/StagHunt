#!/usr/bin/env python3
"""
Video emphasizing BELIEF as the primary social coordination regressor.

Clean, focused layout:
- Main panel: Spatial trajectories
- Large belief panel: p1_belief_p2_stag, p2_belief_p1_stag (PRIMARY SIGNAL)
- Small cross-trial panel: Evolving expectations across trials

This is the pure social prediction signal, unconfounded by spatial/motor factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '.'
DATA_FILE = 'integrated_model_lr0.3.csv'
OUTPUT_FILE = 'stag_hunt_belief_primary.mp4'

FPS = 15
TRANSITION_DURATION = 1.0

# Colors
PLAYER1_COLOR = '#E63946'  # Red (P1)
PLAYER2_COLOR = '#F4A261'  # Orange (P2)
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
BACKGROUND_COLOR = '#F8F9FA'
BELIEF_HIGH_COLOR = '#2A9D8F'  # Green for high belief (cooperation)
BELIEF_LOW_COLOR = '#E63946'   # Red for low belief (defection)

print("="*70)
print("CREATING VIDEO: BELIEF AS PRIMARY SOCIAL REGRESSOR")
print("="*70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"Loading data: {DATA_FILE}")
all_data = pd.read_csv(DATA_FILE)

trial_nums = sorted(all_data['trial_num'].unique())
trials_data = []
trial_info = []

for trial_num in trial_nums:
    df = all_data[all_data['trial_num'] == trial_num].copy()
    df = df.reset_index(drop=True)
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

    # Belief statistics
    p1_belief_mean = df['p1_belief_p2_stag'].mean()
    p2_belief_mean = df['p2_belief_p1_stag'].mean()
    p1_belief_max = df['p1_belief_p2_stag'].max()
    p2_belief_max = df['p2_belief_p1_stag'].max()

    # Cross-trial expectations
    p1_expectation = df['p1_cross_trial_expectation'].iloc[0]
    p2_expectation = df['p2_cross_trial_expectation'].iloc[0]

    trial_info.append({
        'trial_num': trial_num,
        'n_frames': len(df),
        'duration_s': len(df) * 0.04,
        'outcome': outcome,
        'p1_belief_mean': p1_belief_mean,
        'p2_belief_mean': p2_belief_mean,
        'p1_belief_max': p1_belief_max,
        'p2_belief_max': p2_belief_max,
        'p1_expectation': p1_expectation,
        'p2_expectation': p2_expectation,
    })

    trials_data.append(df)

print(f"✓ Loaded {len(trials_data)} trials\n")

print("Belief statistics by trial:")
for info in trial_info:
    print(f"  Trial {info['trial_num']:2d}: {info['outcome']:20s} | "
          f"Mean belief: P1={info['p1_belief_mean']:.3f}, P2={info['p2_belief_mean']:.3f}")
print()

# ============================================================================
# PREPARE ANIMATION
# ============================================================================

# Viewing window
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

# Frame schedule
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
# CREATE FIGURE - BELIEF EMPHASIZED
# ============================================================================

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1],
              height_ratios=[2, 1, 0.5], hspace=0.35, wspace=0.3)

# Main panel: Spatial trajectories (left, top 2 rows)
ax_main = fig.add_subplot(gs[0:2, 0])
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_aspect('equal')
ax_main.set_facecolor(BACKGROUND_COLOR)
ax_main.set_xlabel('X Position', fontsize=12)
ax_main.set_ylabel('Y Position', fontsize=12)

# RIGHT PANELS: BELIEF-FOCUSED

# LARGE BELIEF PANEL (PRIMARY REGRESSOR)
ax_belief = fig.add_subplot(gs[0:2, 1])
ax_belief.set_xlim(0, 10)
ax_belief.set_ylim(0, 1.0)
ax_belief.set_ylabel('Belief P(partner → stag)\n[PRIMARY REGRESSOR]',
                      fontsize=12, weight='bold')
ax_belief.set_title('Social Prediction: What will partner do?',
                     fontsize=11, style='italic', pad=10)
ax_belief.grid(alpha=0.3)

# Add interpretive zones
ax_belief.axhspan(0.0, 0.3, alpha=0.1, color='red', label='Defection expected')
ax_belief.axhspan(0.3, 0.7, alpha=0.1, color='gray', label='Uncertain')
ax_belief.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Cooperation expected')
ax_belief.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Cross-trial expectations (small panel) - x-axis is TRIAL NUMBER
ax_expectations = fig.add_subplot(gs[2, 1])
ax_expectations.set_xlim(0, 13)  # 12 trials
ax_expectations.set_ylim(0, 1.0)
ax_expectations.set_xlabel('Trial Number', fontsize=10)
ax_expectations.set_ylabel('Prior\nExpectations', fontsize=9)
ax_expectations.grid(alpha=0.3)

# Info text panel (bottom left)
ax_info = fig.add_subplot(gs[2, 0])
ax_info.axis('off')

# ============================================================================
# ANIMATION ELEMENTS
# ============================================================================

# Spatial markers
player1_marker, = ax_main.plot([], [], 'o', color=PLAYER1_COLOR,
                               markersize=15, label='Player 1', zorder=5)
player2_marker, = ax_main.plot([], [], 'o', color=PLAYER2_COLOR,
                               markersize=15, label='Player 2', zorder=5)
stag_marker, = ax_main.plot([], [], 's', color=STAG_COLOR,
                            markersize=12, label='Stag', zorder=4)
rabbit_marker, = ax_main.plot([], [], '^', color=RABBIT_COLOR,
                              markersize=12, label='Rabbit', zorder=4)

player1_trail, = ax_main.plot([], [], '-', color=PLAYER1_COLOR,
                              alpha=0.3, linewidth=2, zorder=3)
player2_trail, = ax_main.plot([], [], '-', color=PLAYER2_COLOR,
                              alpha=0.3, linewidth=2, zorder=3)

title_text = ax_main.text(0.5, 1.05, '', transform=ax_main.transAxes,
                          fontsize=14, ha='center', weight='bold')

ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)

# BELIEF LINES (PRIMARY)
belief_p1_line, = ax_belief.plot([], [], '-', color=PLAYER1_COLOR,
                                 linewidth=3, label='P1 belief about P2', zorder=5)
belief_p2_line, = ax_belief.plot([], [], '-', color=PLAYER2_COLOR,
                                 linewidth=3, label='P2 belief about P1', zorder=5)

# Current belief markers (large dots at current time)
belief_p1_marker, = ax_belief.plot([], [], 'o', color=PLAYER1_COLOR,
                                   markersize=12, zorder=6)
belief_p2_marker, = ax_belief.plot([], [], 'o', color=PLAYER2_COLOR,
                                   markersize=12, zorder=6)

ax_belief.legend(loc='upper left', fontsize=9, framealpha=0.9)

# CROSS-TRIAL EXPECTATIONS (one point per trial)
expect_p1_line, = ax_expectations.plot([], [], '-o', color=PLAYER1_COLOR,
                                       linewidth=2, markersize=8, label='P1 prior')
expect_p2_line, = ax_expectations.plot([], [], '-o', color=PLAYER2_COLOR,
                                       linewidth=2, markersize=8, label='P2 prior')
ax_expectations.legend(loc='upper right', fontsize=8)
ax_expectations.set_title('Learning from trial outcomes', fontsize=9, style='italic')

# Info text
info_text = ax_info.text(0.5, 0.5, '', transform=ax_info.transAxes,
                        fontsize=11, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

def init():
    """Initialize animation."""
    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line,
            belief_p1_marker, belief_p2_marker,
            expect_p1_line, expect_p2_line, info_text)

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

    # === UPDATE SPATIAL POSITIONS ===
    player1_marker.set_data([row['player1_x']], [row['player1_y']])
    player2_marker.set_data([row['player2_x']], [row['player2_y']])
    stag_marker.set_data([row['stag_x']], [row['stag_y']])
    rabbit_marker.set_data([row['rabbit_x']], [row['rabbit_y']])

    # Trails
    trail_start = max(0, frame_idx - 30)
    player1_trail.set_data(df['player1_x'].iloc[trail_start:frame_idx+1],
                          df['player1_y'].iloc[trail_start:frame_idx+1])
    player2_trail.set_data(df['player2_x'].iloc[trail_start:frame_idx+1],
                          df['player2_y'].iloc[trail_start:frame_idx+1])

    # Title
    title_text.set_text(f"Trial {trial_num} | {info['outcome']}")

    # === UPDATE BELIEFS (PRIMARY) ===
    times = df['time_point'].iloc[:frame_idx+1].values - df['time_point'].iloc[0]

    belief_p1_line.set_data(times, df['p1_belief_p2_stag'].iloc[:frame_idx+1])
    belief_p2_line.set_data(times, df['p2_belief_p1_stag'].iloc[:frame_idx+1])

    # Current belief markers
    current_p1_belief = row['p1_belief_p2_stag']
    current_p2_belief = row['p2_belief_p1_stag']

    if len(times) > 0:
        belief_p1_marker.set_data([times[-1]], [current_p1_belief])
        belief_p2_marker.set_data([times[-1]], [current_p2_belief])

    # === UPDATE CROSS-TRIAL EXPECTATIONS (accumulated across trials) ===
    # Show all trials up to and including current trial
    trials_so_far = []
    p1_expect_so_far = []
    p2_expect_so_far = []

    for t_info in trial_info:
        if t_info['trial_num'] <= trial_num:
            trials_so_far.append(t_info['trial_num'])
            p1_expect_so_far.append(t_info['p1_expectation'])
            p2_expect_so_far.append(t_info['p2_expectation'])

    expect_p1_line.set_data(trials_so_far, p1_expect_so_far)
    expect_p2_line.set_data(trials_so_far, p2_expect_so_far)

    # === UPDATE INFO TEXT ===
    info_str = (f"Current beliefs:\n"
                f"P1 believes P2 will cooperate: {current_p1_belief:.2%}\n"
                f"P2 believes P1 will cooperate: {current_p2_belief:.2%}\n\n"
                f"Prior expectations (from past trials):\n"
                f"P1: {info['p1_expectation']:.2%} | P2: {info['p2_expectation']:.2%}")
    info_text.set_text(info_str)

    # Update x-axis limits (only for belief panel)
    if len(times) > 0 and times[-1] > 0:
        ax_belief.set_xlim(0, times[-1] * 1.1)
        # Expectations panel keeps fixed x-axis (trial numbers)

    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line,
            belief_p1_marker, belief_p2_marker,
            expect_p1_line, expect_p2_line, info_text)

# ============================================================================
# GENERATE ANIMATION
# ============================================================================

print("="*70)
print("CREATING BELIEF-FOCUSED VIDEO")
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
print("✓ BELIEF-FOCUSED VIDEO SAVED SUCCESSFULLY!")
print("="*70)
print(f"File: {OUTPUT_FILE}")
print(f"Duration: {total_duration:.1f} seconds")
print(f"Frames: {total_frames}")
print()
print("Key features:")
print("  - Large belief panel (PRIMARY social regressor)")
print("  - Interpretive zones (defection/uncertain/cooperation)")
print("  - Current belief markers (large dots)")
print("  - Cross-trial expectations (evolving priors)")
print("  - Info text with belief percentages")
