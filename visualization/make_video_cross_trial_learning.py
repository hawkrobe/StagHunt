#!/usr/bin/env python3
"""
Video showing CROSS-TRIAL LEARNING across the full trial sequence.

Key feature: Continuous timeline across ALL trials showing:
- Within-trial belief dynamics (continuous updates)
- Cross-trial expectations (step changes between trials)
- Trial boundaries clearly marked

This reveals the slow-timescale learning process.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '.'
DATA_FILE = 'integrated_model_lr0.3.csv'
OUTPUT_FILE = 'stag_hunt_cross_trial_learning.mp4'

FPS = 15
TRANSITION_DURATION = 1.0

# Colors
PLAYER1_COLOR = '#E63946'  # Red
PLAYER2_COLOR = '#F4A261'  # Orange
STAG_COLOR = '#2A9D8F'
RABBIT_COLOR = '#264653'
BACKGROUND_COLOR = '#F8F9FA'

print("="*70)
print("CREATING VIDEO: CROSS-TRIAL LEARNING DYNAMICS")
print("="*70)
print()

# ============================================================================
# LOAD DATA WITH CONTINUOUS TIMELINE
# ============================================================================

print(f"Loading data: {DATA_FILE}")
all_data = pd.read_csv(DATA_FILE)

# Sort by trial number to get chronological order
trial_nums_sorted = sorted(all_data['trial_num'].unique())

print(f"Trial order: {trial_nums_sorted}")
print()

# Build continuous timeline across all trials
trials_data = []
trial_info = []
cumulative_time = 0.0

for trial_num in trial_nums_sorted:
    df = all_data[all_data['trial_num'] == trial_num].copy()
    df = df.reset_index(drop=True)
    df = df.dropna(subset=['player1_x', 'player1_y', 'player2_x', 'player2_y'])

    if len(df) == 0:
        continue

    # Add continuous time offset
    df['continuous_time'] = (df.index / FPS) + cumulative_time

    # Get outcome
    outcome = "Ongoing"
    if df['event'].max() == 5:
        outcome = "COOPERATION"
    elif 3 in df['event'].values:
        outcome = "P1 defect"
    elif 4 in df['event'].values:
        outcome = "P2 defect"

    # Cross-trial expectations (at start of trial)
    p1_expectation = df['p1_cross_trial_expectation'].iloc[0]
    p2_expectation = df['p2_cross_trial_expectation'].iloc[0]

    trial_start_time = cumulative_time
    trial_end_time = cumulative_time + (len(df) / FPS)

    trial_info.append({
        'trial_num': trial_num,
        'n_frames': len(df),
        'start_time': trial_start_time,
        'end_time': trial_end_time,
        'outcome': outcome,
        'p1_expectation': p1_expectation,
        'p2_expectation': p2_expectation,
    })

    cumulative_time = trial_end_time + TRANSITION_DURATION
    trials_data.append(df)

total_time = cumulative_time - TRANSITION_DURATION

print(f"Total timeline: {total_time:.1f} seconds across {len(trials_data)} trials\n")

print("Trial timeline:")
for info in trial_info:
    print(f"  Trial {info['trial_num']:2d}: t={info['start_time']:5.1f}-{info['end_time']:5.1f}s | "
          f"{info['outcome']:15s} | Prior: P1={info['p1_expectation']:.3f}, P2={info['p2_expectation']:.3f}")
print()

# Combine all data
all_combined = pd.concat(trials_data, ignore_index=True)

# ============================================================================
# PREPARE ANIMATION
# ============================================================================

# Viewing window for spatial plot
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
for trial_idx, (df, info) in enumerate(zip(trials_data, trial_info)):
    for frame_idx in range(len(df)):
        frame_schedule.append({
            'trial_idx': trial_idx,
            'frame_idx': frame_idx,
            'global_time': df.iloc[frame_idx]['continuous_time'],
            'trial_num': info['trial_num']
        })

total_frames = len(frame_schedule)

print(f"Animation: {total_frames} frames\n")

# ============================================================================
# CREATE FIGURE - CROSS-TRIAL LEARNING EMPHASIZED
# ============================================================================

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1],
              height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.3)

# Spatial trajectories (left, top)
ax_main = fig.add_subplot(gs[0, 0])
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_aspect('equal')
ax_main.set_facecolor(BACKGROUND_COLOR)
ax_main.set_xlabel('X Position', fontsize=12)
ax_main.set_ylabel('Y Position', fontsize=12)

# Timeline info (left, bottom)
ax_timeline = fig.add_subplot(gs[1:, 0])
ax_timeline.axis('off')

# RIGHT PANELS: CONTINUOUS ACROSS TRIALS

# Belief dynamics (continuous)
ax_belief = fig.add_subplot(gs[0, 1])
ax_belief.set_xlim(0, total_time)
ax_belief.set_ylim(0, 1.0)
ax_belief.set_ylabel('Within-Trial Beliefs', fontsize=11, weight='bold')
ax_belief.set_title('Moment-to-moment updates within each trial', fontsize=10, style='italic')
ax_belief.grid(alpha=0.3)

# Cross-trial expectations (MAIN FOCUS)
ax_expectations = fig.add_subplot(gs[1, 1])
ax_expectations.set_xlim(0, total_time)
ax_expectations.set_ylim(0, 1.0)
ax_expectations.set_ylabel('Cross-Trial Expectations\n[LEARNING]', fontsize=11, weight='bold')
ax_expectations.set_title('Trial-by-trial learning from outcomes', fontsize=10, style='italic')
ax_expectations.grid(alpha=0.3)
ax_expectations.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Neutral')

# Outcome markers
ax_outcomes = fig.add_subplot(gs[2, 1])
ax_outcomes.set_xlim(0, total_time)
ax_outcomes.set_ylim(-0.5, 1.5)
ax_outcomes.set_ylabel('Trial Outcomes', fontsize=11)
ax_outcomes.set_xlabel('Time (s)', fontsize=10)
ax_outcomes.set_yticks([0, 1])
ax_outcomes.set_yticklabels(['Defection', 'Cooperation'])
ax_outcomes.grid(alpha=0.3, axis='x')

# Add trial boundaries to all time-series plots
for info in trial_info:
    for ax in [ax_belief, ax_expectations, ax_outcomes]:
        ax.axvline(info['start_time'], color='gray', linestyle=':', alpha=0.3, linewidth=1)
        # Label trial number at top
        if ax == ax_belief:
            ax.text(info['start_time'], 1.05, f"T{info['trial_num']}",
                   fontsize=8, ha='left', transform=ax.get_xaxis_transform())

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

# Belief lines (continuous across all trials)
belief_p1_line, = ax_belief.plot([], [], '-', color=PLAYER1_COLOR,
                                 linewidth=2, alpha=0.7, label='P1 belief')
belief_p2_line, = ax_belief.plot([], [], '-', color=PLAYER2_COLOR,
                                 linewidth=2, alpha=0.7, label='P2 belief')
ax_belief.legend(loc='upper right', fontsize=9)

# Cross-trial expectations (STEP FUNCTION)
expect_p1_line, = ax_expectations.plot([], [], '-', color=PLAYER1_COLOR,
                                       linewidth=3, label='P1 expectation', drawstyle='steps-post')
expect_p2_line, = ax_expectations.plot([], [], '-', color=PLAYER2_COLOR,
                                       linewidth=3, label='P2 expectation', drawstyle='steps-post')
ax_expectations.legend(loc='upper right', fontsize=9)

# Outcome markers
outcome_markers = ax_outcomes.plot([], [], 'o', markersize=10, color='green', zorder=5)[0]

# Timeline text
timeline_text = ax_timeline.text(0.5, 0.5, '', transform=ax_timeline.transAxes,
                                fontsize=10, ha='center', va='center', family='monospace')

# Progress line
progress_line_belief = ax_belief.axvline(0, color='red', linewidth=2, alpha=0.7)
progress_line_expect = ax_expectations.axvline(0, color='red', linewidth=2, alpha=0.7)
progress_line_outcome = ax_outcomes.axvline(0, color='red', linewidth=2, alpha=0.7)

def init():
    """Initialize animation."""
    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line,
            expect_p1_line, expect_p2_line,
            outcome_markers, timeline_text)

def update(frame_num):
    """Update animation for given frame."""
    if frame_num >= len(frame_schedule):
        return init()

    frame_info = frame_schedule[frame_num]
    trial_idx = frame_info['trial_idx']
    frame_idx = frame_info['frame_idx']
    current_time = frame_info['global_time']
    trial_num = frame_info['trial_num']

    df = trials_data[trial_idx]
    info = trial_info[trial_idx]
    row = df.iloc[frame_idx]

    # === UPDATE SPATIAL POSITIONS ===
    player1_marker.set_data([row['player1_x']], [row['player1_y']])
    player2_marker.set_data([row['player2_x']], [row['player2_y']])
    stag_marker.set_data([row['stag_x']], [row['stag_y']])
    rabbit_marker.set_data([row['rabbit_x']], [row['rabbit_y']])

    # Trails (within current trial only)
    trail_start = max(0, frame_idx - 30)
    player1_trail.set_data(df['player1_x'].iloc[trail_start:frame_idx+1],
                          df['player1_y'].iloc[trail_start:frame_idx+1])
    player2_trail.set_data(df['player2_x'].iloc[trail_start:frame_idx+1],
                          df['player2_y'].iloc[trail_start:frame_idx+1])

    title_text.set_text(f"Trial {trial_num} | {info['outcome']}")

    # === UPDATE BELIEFS (continuous up to current time) ===
    belief_times = []
    belief_p1_vals = []
    belief_p2_vals = []

    for t_idx in range(trial_idx + 1):
        t_df = trials_data[t_idx]
        if t_idx < trial_idx:
            # Include full trial
            belief_times.extend(t_df['continuous_time'].values)
            belief_p1_vals.extend(t_df['p1_belief_p2_stag'].values)
            belief_p2_vals.extend(t_df['p2_belief_p1_stag'].values)
        else:
            # Include up to current frame
            belief_times.extend(t_df['continuous_time'].iloc[:frame_idx+1].values)
            belief_p1_vals.extend(t_df['p1_belief_p2_stag'].iloc[:frame_idx+1].values)
            belief_p2_vals.extend(t_df['p2_belief_p1_stag'].iloc[:frame_idx+1].values)

    belief_p1_line.set_data(belief_times, belief_p1_vals)
    belief_p2_line.set_data(belief_times, belief_p2_vals)

    # === UPDATE CROSS-TRIAL EXPECTATIONS (step function) ===
    expect_times = []
    expect_p1_vals = []
    expect_p2_vals = []

    for t_idx in range(trial_idx + 1):
        t_info = trial_info[t_idx]
        expect_times.append(t_info['start_time'])
        expect_p1_vals.append(t_info['p1_expectation'])
        expect_p2_vals.append(t_info['p2_expectation'])

    # Add current time point to extend line
    expect_times.append(current_time)
    expect_p1_vals.append(info['p1_expectation'])
    expect_p2_vals.append(info['p2_expectation'])

    expect_p1_line.set_data(expect_times, expect_p1_vals)
    expect_p2_line.set_data(expect_times, expect_p2_vals)

    # === UPDATE OUTCOMES ===
    outcome_times = []
    outcome_vals = []

    for t_idx in range(trial_idx):
        t_info = trial_info[t_idx]
        outcome_times.append(t_info['end_time'])
        if t_info['outcome'] == 'COOPERATION':
            outcome_vals.append(1.0)
        else:
            outcome_vals.append(0.0)

    outcome_markers.set_data(outcome_times, outcome_vals)

    # === UPDATE PROGRESS LINES ===
    progress_line_belief.set_xdata([current_time, current_time])
    progress_line_expect.set_xdata([current_time, current_time])
    progress_line_outcome.set_xdata([current_time, current_time])

    # === UPDATE TIMELINE TEXT ===
    timeline_str = "CROSS-TRIAL LEARNING TIMELINE\n\n"
    timeline_str += f"Current time: {current_time:.1f}s\n"
    timeline_str += f"Current trial: {trial_num}\n\n"

    timeline_str += "Trial history:\n"
    for t_idx in range(trial_idx + 1):
        t_info = trial_info[t_idx]
        marker = "→" if t_idx == trial_idx else " "
        timeline_str += f"{marker} T{t_info['trial_num']:2d}: {t_info['outcome']:12s} | "
        timeline_str += f"Prior: P1={t_info['p1_expectation']:.2f}, P2={t_info['p2_expectation']:.2f}\n"

    timeline_text.set_text(timeline_str)

    return (player1_marker, player2_marker, stag_marker, rabbit_marker,
            player1_trail, player2_trail, title_text,
            belief_p1_line, belief_p2_line,
            expect_p1_line, expect_p2_line,
            outcome_markers, timeline_text)

# ============================================================================
# GENERATE ANIMATION
# ============================================================================

print("="*70)
print("CREATING CROSS-TRIAL LEARNING VIDEO")
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
print("✓ CROSS-TRIAL LEARNING VIDEO SAVED!")
print("="*70)
print(f"File: {OUTPUT_FILE}")
print(f"Duration: {total_time:.1f} seconds")
print(f"Frames: {total_frames}")
print()
print("Key features:")
print("  - Continuous timeline across ALL trials")
print("  - Cross-trial expectations as STEP FUNCTION")
print("  - Shows trial-by-trial learning dynamics")
print("  - Trial boundaries marked with vertical lines")
print("  - Outcome markers show cooperation vs. defection")
