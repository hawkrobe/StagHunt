#!/usr/bin/env python3
"""
Create a polished video showing trajectory dynamics AND Bayesian belief updating
across all trials of the Stag Hunt task.

This enhanced version overlays each player's belief about their partner's intentions
(stag vs. rabbit) on top of the movement trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from pathlib import Path
import sys

# Import unified model API
from stag_hunt import BeliefModel

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR LOCAL SETUP
# ============================================================================
DATA_DIR = Path('inputs')  # Directory containing trial CSV files
OUTPUT_FILE = 'stag_hunt_trajectories_with_beliefs.mp4'

# Animation settings
FPS = 30
TRANSITION_DURATION = 1.0  # seconds between trials

# Bayesian model parameters (using fitted decision model)
PRIOR_STAG = 0.5  # Initial belief that partner is going for stag
BELIEF_BOUNDS = (0.01, 0.99)  # Prevent beliefs hitting ceiling/floor

# Fitted decision model parameters (from coordinated model)
DECISION_MODEL_PARAMS = {
    'temperature': 3.049,
    'timing_tolerance': 0.865,
    'action_noise': 10.0,
    'n_directions': 8
}

# Colors (professional academic palette)
PLAYER1_COLOR = '#E63946'  # Red (Chinese player)
PLAYER2_COLOR = '#F4A261'  # Orange/Yellow (US player)
STAG_COLOR = '#2A9D8F'     # Teal
RABBIT_COLOR = '#264653'   # Dark blue
ARENA_COLOR = '#888888'    # Gray for arena boundary
BACKGROUND_COLOR = '#F8F9FA'  # Light gray background

# Belief visualization colors
BELIEF_STAG_COLOR = STAG_COLOR  # Green for stag belief
BELIEF_RABBIT_COLOR = RABBIT_COLOR  # Red for rabbit belief

# ============================================================================
# LOAD AND VALIDATE DATA
# ============================================================================

print("=" * 70)
print("LOADING STAG HUNT DATA AND RUNNING BAYESIAN MODEL")
print("=" * 70)

trial_files = sorted(DATA_DIR.glob('stag_hunt_coop_trial*.csv'))

if not trial_files:
    print(f"❌ ERROR: No trial files found in {DATA_DIR.absolute()}")
    print("Looking for files matching: 'stag_hunt_coop_trial*.csv'")
    exit(1)

print(f"✓ Found {len(trial_files)} trial files\n")

# Outcome mapping
outcome_labels = {
    0: 'Ongoing',
    1: 'P1 touched stag alone',
    2: 'P2 touched stag alone', 
    3: 'P1 caught rabbit',
    4: 'P2 caught rabbit',
    5: 'Both caught stag (COOPERATION!)'
}

# Initialize Bayesian model WITH DECISION MODEL (using unified API)
print(f"Initializing Bayesian model with decision-based inference:")
print(f"  Prior belief (stag): {PRIOR_STAG}")
print(f"  Belief bounds: {BELIEF_BOUNDS}")
print(f"  Decision model parameters:")
print(f"    Temperature: {DECISION_MODEL_PARAMS['temperature']:.3f}")
print(f"    Timing tolerance: {DECISION_MODEL_PARAMS['timing_tolerance']:.3f}")
print(f"    Action noise: {DECISION_MODEL_PARAMS['action_noise']:.3f}")
model = BeliefModel(
    inference_type='decision',
    decision_model={'model_type': 'coordinated', 'params': DECISION_MODEL_PARAMS},
    prior_stag=PRIOR_STAG,
    belief_bounds=BELIEF_BOUNDS
)

# Load all trials and run model
trials_data = []
trial_outcomes = {}
trial_beliefs = {}  # Store belief trajectories

for trial_file in trial_files:
    trial_num = int(trial_file.stem.split('trial')[1].split('_')[0])
    
    try:
        df = pd.read_csv(trial_file)
        
        # Fix typo in column name if present
        if 'plater1_y' in df.columns:
            df = df.rename(columns={'plater1_y': 'player1_y'})
        
        # Drop rows with NaN values
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            print(f"  Trial {trial_num:2d}: Dropped {original_len - len(df)} NaN row(s)")
        
        if len(df) == 0:
            print(f"❌ ERROR: Trial {trial_num} has no valid data after dropping NaN")
            continue
        
        # Sort by time_point to ensure chronological order
        if not df['time_point'].is_monotonic_increasing:
            print(f"  Trial {trial_num:2d}: Sorting data by time (was out of order)")
            df = df.sort_values('time_point').reset_index(drop=True)
        
        # Validate required columns
        required_cols = ['player1_x', 'player1_y', 'player2_x', 'player2_y',
                        'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y', 'value', 'event', 'time_point']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ ERROR in {trial_file.name}: Missing columns: {missing_cols}")
            continue
        
        # RUN BAYESIAN MODEL ON THIS TRIAL
        df_with_beliefs = model.run_trial(df)
        
        # Add trial number
        df_with_beliefs['trial'] = trial_num
        
        # Determine outcome
        final_event = df_with_beliefs[df_with_beliefs['event'] > 0]['event'].iloc[-1] if (df_with_beliefs['event'] > 0).any() else 0
        trial_outcomes[trial_num] = final_event
        
        # Store belief data
        trial_beliefs[trial_num] = {
            'p1_belief_p2_stag': df_with_beliefs['p1_belief_p2_stag'].values,
            'p2_belief_p1_stag': df_with_beliefs['p2_belief_p1_stag'].values
        }
        
        trials_data.append(df_with_beliefs)
        
        duration = df_with_beliefs['time_point'].iloc[-1] - df_with_beliefs['time_point'].iloc[0]
        outcome_str = outcome_labels[final_event]
        coop_marker = " ✓" if final_event == 5 else ""
        
        # Show belief summary
        p1_final = df_with_beliefs['p1_belief_p2_stag'].iloc[-1]
        p2_final = df_with_beliefs['p2_belief_p1_stag'].iloc[-1]
        print(f"  Trial {trial_num:2d}: {len(df_with_beliefs):4d} rows, {duration:5.2f}s, {outcome_str}{coop_marker}")
        print(f"             Beliefs: P1→P2={p1_final:.2f}, P2→P1={p2_final:.2f}")
        
    except Exception as e:
        print(f"❌ ERROR loading {trial_file.name}: {e}")
        import traceback
        traceback.print_exc()
        continue

if not trials_data:
    print("\n❌ ERROR: No valid trial data loaded!")
    exit(1)

# Combine all trials
all_data = pd.concat(trials_data, ignore_index=True)
print(f"\n✓ Successfully loaded {len(trials_data)} trials")
print(f"✓ Total data points: {len(all_data)}")

coop_count = sum(1 for v in trial_outcomes.values() if v == 5)
print(f"✓ Cooperation: {coop_count}/{len(trials_data)} trials ({100*coop_count/len(trials_data):.1f}%)")

# ============================================================================
# CALCULATE COORDINATE BOUNDS
# ============================================================================

print("\n" + "=" * 70)
print("ANALYZING COORDINATE SPACE")
print("=" * 70)

x_coords = pd.concat([
    all_data['player1_x'], all_data['player2_x'], 
    all_data['stag_x'], all_data['rabbit_x']
])
y_coords = pd.concat([
    all_data['player1_y'], all_data['player2_y'],
    all_data['stag_y'], all_data['rabbit_y']
])

# Calculate bounds with padding
PADDING = 100
x_min, x_max = x_coords.min() - PADDING, x_coords.max() + PADDING
y_min, y_max = y_coords.min() - PADDING, y_coords.max() + PADDING

# Make axes equal (square viewing area)
max_range = max(x_max - x_min, y_max - y_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
x_min, x_max = x_center - max_range/2, x_center + max_range/2
y_min, y_max = y_center - max_range/2, y_center + max_range/2

print(f"Viewing window: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")

# ============================================================================
# PREPARE ANIMATION DATA
# ============================================================================

print("\n" + "=" * 70)
print("PREPARING ANIMATION DATA")
print("=" * 70)

def prepare_trial_animation_data(trial_num, target_fps=30):
    """Prepare smoothly interpolated data for a trial at target FPS"""
    trial_df = all_data[all_data['trial'] == trial_num].copy()
    
    # Calculate elapsed time from first timestamp
    trial_df['elapsed_time'] = trial_df['time_point'] - trial_df['time_point'].iloc[0]
    
    duration = trial_df['elapsed_time'].iloc[-1]
    
    # Sanity check
    if duration <= 0:
        print(f"  ❌ ERROR: Trial {trial_num} has non-positive duration: {duration:.3f}s")
        return None
    
    # Handle very short trials
    if duration < 0.1:
        print(f"  ⚠️  Trial {trial_num}: Very short duration ({duration:.3f}s)")
    
    # Create interpolation timestamps
    num_frames = max(2, int(duration * target_fps))
    target_times = np.linspace(0, duration, num_frames)
    
    # Interpolate all position columns AND belief columns
    interpolated_data = {'time': target_times}
    
    for col in ['player1_x', 'player1_y', 'player2_x', 'player2_y', 
                'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y', 'value',
                'p1_belief_p2_stag', 'p2_belief_p1_stag']:
        interpolated_data[col] = np.interp(target_times, trial_df['elapsed_time'], trial_df[col])
    
    return pd.DataFrame(interpolated_data)

trial_animations = {}
for trial_num in sorted(all_data['trial'].unique()):
    anim_data = prepare_trial_animation_data(trial_num, FPS)
    if anim_data is not None:
        trial_animations[trial_num] = anim_data
        n_frames = len(anim_data)
        duration = anim_data['time'].iloc[-1]
        print(f"  Trial {trial_num:2d}: {n_frames:4d} frames ({duration:.2f}s)")

if not trial_animations:
    print("\n❌ ERROR: No valid animation data prepared!")
    exit(1)

# ============================================================================
# CREATE FIGURE WITH BELIEF PANELS
# ============================================================================

print("\n" + "=" * 70)
print("CREATING ANIMATION WITH BELIEF VISUALIZATION")
print("=" * 70)

# Create figure with subplots: main arena + 2 belief panels
fig = plt.figure(figsize=(16, 12), facecolor=BACKGROUND_COLOR)

# Main arena (larger, left side)
ax_arena = plt.subplot2grid((2, 2), (0, 0), rowspan=2, fig=fig)
ax_arena.set_facecolor(BACKGROUND_COLOR)
ax_arena.set_xlim(x_min, x_max)
ax_arena.set_ylim(y_min, y_max)
ax_arena.set_aspect('equal')
ax_arena.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax_arena.set_xticks([])
ax_arena.set_yticks([])
ax_arena.set_title('Movement Arena', fontsize=12, fontweight='bold')

# Belief panels (right side)
ax_p1_belief = plt.subplot2grid((2, 2), (0, 1), fig=fig)
ax_p2_belief = plt.subplot2grid((2, 2), (1, 1), fig=fig)

# Configure belief panels
for ax, player_name, color in [(ax_p1_belief, "Player 1 (China)", PLAYER1_COLOR),
                                (ax_p2_belief, "Player 2 (US)", PLAYER2_COLOR)]:
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim(0, 100)  # Will update based on trial length
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('Belief (Partner → Stag)', fontsize=10)
    ax.set_title(f"{player_name}'s Beliefs", fontsize=11, fontweight='bold', color=color)
    
ax_p2_belief.set_xlabel('Time (frames)', fontsize=10)

# Initialize plot elements in arena
player1_trail, = ax_arena.plot([], [], '-', color=PLAYER1_COLOR, alpha=0.3, linewidth=2, label='Player 1 (China)')
player2_trail, = ax_arena.plot([], [], '-', color=PLAYER2_COLOR, alpha=0.3, linewidth=2, label='Player 2 (US)')
player1_marker, = ax_arena.plot([], [], 'o', color=PLAYER1_COLOR, markersize=15, markeredgecolor='white', markeredgewidth=2)
player2_marker, = ax_arena.plot([], [], 'o', color=PLAYER2_COLOR, markersize=15, markeredgecolor='white', markeredgewidth=2)
stag_marker, = ax_arena.plot([], [], 's', color=STAG_COLOR, markersize=20, markeredgecolor='white', 
                       markeredgewidth=2, label='Stag (cooperation)')
rabbit_marker, = ax_arena.plot([], [], '^', color=RABBIT_COLOR, markersize=18, markeredgecolor='white',
                         markeredgewidth=2, label='Rabbit (defection)')

# Initialize belief plot elements
p1_belief_line, = ax_p1_belief.plot([], [], '-', color=BELIEF_STAG_COLOR, linewidth=2, label='Belief: Partner → Stag')
p1_belief_marker, = ax_p1_belief.plot([], [], 'o', color=BELIEF_STAG_COLOR, markersize=8)

p2_belief_line, = ax_p2_belief.plot([], [], '-', color=BELIEF_STAG_COLOR, linewidth=2, label='Belief: Partner → Stag')
p2_belief_marker, = ax_p2_belief.plot([], [], 'o', color=BELIEF_STAG_COLOR, markersize=8)

# Text annotations
trial_text = ax_arena.text(0.02, 0.98, '', transform=ax_arena.transAxes, fontsize=14, 
                     verticalalignment='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
outcome_text = ax_arena.text(0.02, 0.91, '', transform=ax_arena.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
value_text = ax_arena.text(0.98, 0.98, '', transform=ax_arena.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Belief value displays
p1_belief_text = ax_p1_belief.text(0.98, 0.95, '', transform=ax_p1_belief.transAxes, 
                                   fontsize=10, ha='right', va='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
p2_belief_text = ax_p2_belief.text(0.98, 0.95, '', transform=ax_p2_belief.transAxes,
                                   fontsize=10, ha='right', va='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax_arena.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, 
          frameon=True, fancybox=True, shadow=True, fontsize=9)

fig.suptitle('Stag Hunt: Movement Trajectories + Bayesian Belief Updating', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ============================================================================
# ANIMATION LOGIC
# ============================================================================

transition_frames = int(FPS * TRANSITION_DURATION)

def get_frame_trial_and_index(frame_num):
    """Map global frame number to (trial_num, frame_index, is_transition)"""
    global_frame = frame_num
    
    for trial_num in sorted(trial_animations.keys()):
        trial_data = trial_animations[trial_num]
        trial_frames = len(trial_data)
        
        if global_frame < trial_frames:
            return trial_num, global_frame, False
        
        global_frame -= trial_frames
        
        # Add transition after each trial except the last
        if trial_num < max(trial_animations.keys()):
            if global_frame < transition_frames:
                return trial_num, trial_frames - 1, True
            global_frame -= transition_frames
    
    # Fallback: last trial's last frame
    last_trial = max(trial_animations.keys())
    return last_trial, len(trial_animations[last_trial]) - 1, False

# Calculate total frames
total_frames = sum(len(trial_animations[t]) for t in trial_animations.keys())
total_frames += (len(trial_animations) - 1) * transition_frames

print(f"\nTotal frames: {total_frames} ({total_frames/FPS:.1f} seconds)")

def animate(frame):
    """Animation update function"""
    trial_num, trial_frame_idx, is_transition = get_frame_trial_and_index(frame)
    
    trial_data = trial_animations[trial_num]
    
    # Bounds check
    if trial_frame_idx >= len(trial_data):
        trial_frame_idx = len(trial_data) - 1
    
    # Get trajectory up to current frame
    x1 = trial_data['player1_x'].iloc[:trial_frame_idx+1]
    y1 = trial_data['player1_y'].iloc[:trial_frame_idx+1]
    x2 = trial_data['player2_x'].iloc[:trial_frame_idx+1]
    y2 = trial_data['player2_y'].iloc[:trial_frame_idx+1]
    
    # Update trails
    player1_trail.set_data(x1, y1)
    player2_trail.set_data(x2, y2)
    
    # Update current positions
    player1_marker.set_data([x1.iloc[-1]], [y1.iloc[-1]])
    player2_marker.set_data([x2.iloc[-1]], [y2.iloc[-1]])
    stag_marker.set_data([trial_data['stag_x'].iloc[trial_frame_idx]], 
                         [trial_data['stag_y'].iloc[trial_frame_idx]])
    rabbit_marker.set_data([trial_data['rabbit_x'].iloc[trial_frame_idx]], 
                           [trial_data['rabbit_y'].iloc[trial_frame_idx]])
    
    # Update belief trajectories
    frames = np.arange(trial_frame_idx + 1)
    p1_beliefs = trial_data['p1_belief_p2_stag'].iloc[:trial_frame_idx+1]
    p2_beliefs = trial_data['p2_belief_p1_stag'].iloc[:trial_frame_idx+1]
    
    p1_belief_line.set_data(frames, p1_beliefs)
    p1_belief_marker.set_data([frames[-1]], [p1_beliefs.iloc[-1]])
    
    p2_belief_line.set_data(frames, p2_beliefs)
    p2_belief_marker.set_data([frames[-1]], [p2_beliefs.iloc[-1]])
    
    # Update belief panel x-limits
    max_frames = len(trial_data)
    ax_p1_belief.set_xlim(-1, max_frames + 1)
    ax_p2_belief.set_xlim(-1, max_frames + 1)
    
    # Update text
    trial_text.set_text(f'Trial {trial_num} of {len(trial_animations)}')
    
    outcome = trial_outcomes[trial_num]
    outcome_str = outcome_labels[outcome]
    if outcome == 5:
        outcome_text.set_text(f'Outcome: {outcome_str}')
        outcome_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    elif outcome in [3, 4]:
        outcome_text.set_text(f'Outcome: {outcome_str}')
        outcome_text.set_bbox(dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    else:
        outcome_text.set_text(f'Outcome: {outcome_str}')
        outcome_text.set_bbox(dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    current_value = trial_data['value'].iloc[trial_frame_idx]
    value_text.set_text(f'Dynamic value: {current_value:.1f}')
    
    # Update belief text displays
    p1_belief_val = p1_beliefs.iloc[-1]
    p2_belief_val = p2_beliefs.iloc[-1]
    
    p1_belief_text.set_text(f'P(P2→Stag) = {p1_belief_val:.3f}')
    p2_belief_text.set_text(f'P(P1→Stag) = {p2_belief_val:.3f}')
    
    return (player1_trail, player2_trail, player1_marker, player2_marker, 
            stag_marker, rabbit_marker, trial_text, outcome_text, value_text,
            p1_belief_line, p1_belief_marker, p2_belief_line, p2_belief_marker,
            p1_belief_text, p2_belief_text)

# Create animation
print("\nGenerating animation (this may take several minutes)...")
anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/FPS, blit=True)

# Save animation
print(f"Saving to {OUTPUT_FILE}...")
writer = FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264')
anim.save(OUTPUT_FILE, writer=writer, dpi=120)

print(f"\n{'='*70}")
print("✓ VIDEO WITH BELIEF TRACKING SAVED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"File: {OUTPUT_FILE}")
print(f"Duration: {total_frames/FPS:.1f} seconds")
print(f"Cooperation rate: {coop_count}/{len(trial_animations)} trials")
print(f"\nBayesian Model Parameters (Decision-Based Inference):")
print(f"  Prior: {PRIOR_STAG}")
print(f"  Bounds: {BELIEF_BOUNDS} (prevents ceiling/floor)")
print(f"  Decision Model:")
print(f"    Temperature: {DECISION_MODEL_PARAMS['temperature']:.3f}")
print(f"    Timing tolerance: {DECISION_MODEL_PARAMS['timing_tolerance']:.3f}")
print(f"    Action noise: {DECISION_MODEL_PARAMS['action_noise']:.3f}")

plt.close()
