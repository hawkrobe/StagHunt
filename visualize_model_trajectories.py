"""
Visualize model-generated trajectories vs actual behavior.

Sample from the fitted decision model and compare simulated paths
to observed player movements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from belief_model_distance import BayesianIntentionModel
from decision_model_basic import UtilityDecisionModel


def load_trial(filepath):
    """Load and clean a single trial CSV."""
    df = pd.read_csv(filepath)
    if 'plater1_y' in df.columns:
        df = df.rename(columns={'plater1_y': 'player1_y'})
    df = df.dropna()
    return df


def simulate_trajectory(decision_model, trial_data, player='player1',
                       belief_column='p1_belief_p2_stag', action_noise=0.73,
                       max_steps=None):
    """
    Simulate a trajectory by sampling actions from the model.

    Uses actual beliefs and game state from trial, but generates new actions.
    """
    if max_steps is None:
        max_steps = len(trial_data)

    # Initialize at actual starting position
    if player == 'player1':
        sim_x = [trial_data.iloc[0]['player1_x']]
        sim_y = [trial_data.iloc[0]['player1_y']]
    else:
        sim_x = [trial_data.iloc[0]['player2_x']]
        sim_y = [trial_data.iloc[0]['player2_y']]

    # Simulate forward
    for t in range(1, min(max_steps, len(trial_data))):
        # Current simulated position
        curr_x = sim_x[-1]
        curr_y = sim_y[-1]

        # Get game state from actual trial
        stag_x = trial_data.iloc[t-1]['stag_x']
        stag_y = trial_data.iloc[t-1]['stag_y']
        rabbit_x = trial_data.iloc[t-1]['rabbit_x']
        rabbit_y = trial_data.iloc[t-1]['rabbit_y']
        stag_value = trial_data.iloc[t-1]['value']
        rabbit_value = 1.0
        belief = trial_data.iloc[t-1][belief_column]

        # Sample action from model
        probs, utilities = decision_model.compute_action_probabilities(
            curr_x, curr_y,
            stag_x, stag_y, stag_value,
            rabbit_x, rabbit_y, rabbit_value,
            belief
        )

        # Sample discrete action
        action_idx = np.random.choice(decision_model.n_directions, p=probs)
        action_angle = decision_model.action_angles[action_idx]

        # Add motor noise using von Mises
        from scipy.stats import vonmises
        noisy_angle = vonmises.rvs(kappa=action_noise, loc=action_angle)

        # Update position (using actual speed from data)
        if player == 'player1':
            actual_dx = trial_data.iloc[t]['player1_x'] - trial_data.iloc[t-1]['player1_x']
            actual_dy = trial_data.iloc[t]['player1_y'] - trial_data.iloc[t-1]['player1_y']
        else:
            actual_dx = trial_data.iloc[t]['player2_x'] - trial_data.iloc[t-1]['player2_x']
            actual_dy = trial_data.iloc[t]['player2_y'] - trial_data.iloc[t-1]['player2_y']

        actual_speed = np.sqrt(actual_dx**2 + actual_dy**2)

        new_x = curr_x + actual_speed * np.cos(noisy_angle)
        new_y = curr_y + actual_speed * np.sin(noisy_angle)

        sim_x.append(new_x)
        sim_y.append(new_y)

    return np.array(sim_x), np.array(sim_y)


def plot_trajectory_comparison(trial_data, sim_p1_x, sim_p1_y,
                               sim_p2_x, sim_p2_y,
                               trial_name='Trial', save_path=None):
    """Plot actual vs simulated trajectories for both players and prey."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Get actual trajectories
    p1_x = trial_data['player1_x'].values
    p1_y = trial_data['player1_y'].values
    p2_x = trial_data['player2_x'].values
    p2_y = trial_data['player2_y'].values

    # Prey trajectories
    stag_x = trial_data['stag_x'].values
    stag_y = trial_data['stag_y'].values
    rabbit_x = trial_data['rabbit_x'].values
    rabbit_y = trial_data['rabbit_y'].values

    # Plot 1: Actual trajectories (all entities)
    ax1.plot(p1_x, p1_y, '-', color='red', alpha=0.7, linewidth=2, label='Player 1')
    ax1.plot(p1_x[0], p1_y[0], 'o', color='red', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(p1_x[-1], p1_y[-1], 's', color='red', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

    ax1.plot(p2_x, p2_y, '-', color='gold', alpha=0.7, linewidth=2, label='Player 2')
    ax1.plot(p2_x[0], p2_y[0], 'o', color='gold', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(p2_x[-1], p2_y[-1], 's', color='gold', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

    ax1.plot(stag_x, stag_y, '--', color='brown', alpha=0.5, linewidth=2, label='Stag')
    ax1.plot(stag_x[0], stag_y[0], 'D', color='brown', markersize=15,
             markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(stag_x[-1], stag_y[-1], 'D', color='brown', markersize=15,
             markeredgecolor='black', markeredgewidth=1.5, fillstyle='none')

    ax1.plot(rabbit_x, rabbit_y, '--', color='gray', alpha=0.5, linewidth=2, label='Rabbit')
    ax1.plot(rabbit_x[0], rabbit_y[0], 'D', color='gray', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(rabbit_x[-1], rabbit_y[-1], 'D', color='gray', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5, fillstyle='none')

    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Y Position', fontsize=11)
    ax1.set_title(f'{trial_name}: Actual Behavior', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Simulated trajectories (players only, prey same as actual)
    ax2.plot(sim_p1_x, sim_p1_y, '-', color='red', alpha=0.7, linewidth=2,
             label='Player 1 (simulated)', linestyle=':')
    ax2.plot(sim_p1_x[0], sim_p1_y[0], 'o', color='red', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(sim_p1_x[-1], sim_p1_y[-1], 's', color='red', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)

    ax2.plot(sim_p2_x, sim_p2_y, '-', color='gold', alpha=0.7, linewidth=2,
             label='Player 2 (simulated)', linestyle=':')
    ax2.plot(sim_p2_x[0], sim_p2_y[0], 'o', color='gold', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(sim_p2_x[-1], sim_p2_y[-1], 's', color='gold', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)

    # Same prey trajectories as actual
    ax2.plot(stag_x, stag_y, '--', color='brown', alpha=0.5, linewidth=2, label='Stag')
    ax2.plot(stag_x[0], stag_y[0], 'D', color='brown', markersize=15,
             markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(stag_x[-1], stag_y[-1], 'D', color='brown', markersize=15,
             markeredgecolor='black', markeredgewidth=1.5, fillstyle='none')

    ax2.plot(rabbit_x, rabbit_y, '--', color='gray', alpha=0.5, linewidth=2, label='Rabbit')
    ax2.plot(rabbit_x[0], rabbit_y[0], 'D', color='gray', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(rabbit_x[-1], rabbit_y[-1], 'D', color='gray', markersize=12,
             markeredgecolor='black', markeredgewidth=1.5, fillstyle='none')

    ax2.set_xlabel('X Position', fontsize=11)
    ax2.set_ylabel('Y Position', fontsize=11)
    ax2.set_title(f'{trial_name}: Model Simulation', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()


def main():
    # Load trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*.csv'))
    print(f"Found {len(trial_files)} trials\n")

    # Initialize belief model
    belief_model = BayesianIntentionModel(
        prior_stag=0.5,
        concentration=1.5,
        belief_bounds=(0.01, 0.99)
    )

    # Initialize decision model with FITTED parameters (from fair comparison)
    decision_model = UtilityDecisionModel(
        n_directions=8,
        temperature=9.3904,
        w_stag=4.9816,
        w_rabbit=8.7587,
        speed=1.0
    )

    action_noise = 0.7292

    print("="*70)
    print("MODEL TRAJECTORY SIMULATION")
    print("="*70)
    print(f"\nUsing fitted parameters:")
    print(f"  w_stag:       {decision_model.w_stag:.4f}")
    print(f"  w_rabbit:     {decision_model.w_rabbit:.4f}")
    print(f"  temperature:  {decision_model.temperature:.4f}")
    print(f"  action_noise: {action_noise:.4f}")
    print(f"\n{'='*70}\n")

    # Sample a few interesting trials
    # Trial 3 = cooperation, Trial 1 = defection
    trials_to_visualize = [0, 2, 5, 8]  # Indices

    np.random.seed(42)  # For reproducibility

    for trial_idx in trials_to_visualize:
        print(f"\n{'-'*70}")
        print(f"Trial {trial_idx + 1}")
        print(f"{'-'*70}")

        trial_data = load_trial(trial_files[trial_idx])
        trial_with_beliefs = belief_model.run_trial(trial_data)

        # Determine outcome
        if trial_data['event'].max() == 5:
            outcome = "COOPERATION"
        elif 3 in trial_data['event'].values:
            outcome = "P1 defected (rabbit)"
        elif 4 in trial_data['event'].values:
            outcome = "P2 defected (rabbit)"
        else:
            outcome = "Unknown"

        print(f"Outcome: {outcome}")
        print(f"Duration: {len(trial_data)} timesteps\n")

        # Simulate both players
        print(f"Simulating player1...")
        sim_p1_x, sim_p1_y = simulate_trajectory(
            decision_model,
            trial_with_beliefs,
            player='player1',
            belief_column='p1_belief_p2_stag',
            action_noise=action_noise
        )

        print(f"Simulating player2...")
        sim_p2_x, sim_p2_y = simulate_trajectory(
            decision_model,
            trial_with_beliefs,
            player='player2',
            belief_column='p2_belief_p1_stag',
            action_noise=action_noise
        )

        # Save visualization
        save_path = f'outputs/trajectory_trial{trial_idx+1}.png'
        plot_trajectory_comparison(
            trial_with_beliefs,
            sim_p1_x, sim_p1_y,
            sim_p2_x, sim_p2_y,
            trial_name=f'Trial {trial_idx+1} - {outcome}',
            save_path=save_path
        )

    print(f"\n{'='*70}")
    print("Trajectory visualizations saved to outputs/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
