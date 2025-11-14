#!/usr/bin/env python3
"""
Test whether alternation is deliberate or coincidental.

Deliberate turn-taking:
- Player yields/slows down when it's "partner's turn"
- Winner is not predicted by initial distance to rabbit
- Active avoidance of rabbit when partner is closer

Coincidental (greedy racing):
- Both players go straight for rabbit
- Closer player (at trial start) wins
- No evidence of yielding behavior
"""

import pandas as pd
import numpy as np
import glob


def analyze_deliberate_vs_coincidental():
    """Test if alternation is strategic or just spatial dynamics."""

    print("="*70)
    print("DELIBERATE TURN-TAKING vs. COINCIDENTAL ALTERNATION")
    print("="*70)
    print()

    # Load all trials
    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    results = []

    for trial_file in trial_files:
        trial_data = pd.read_csv(trial_file)
        if 'plater1_y' in trial_data.columns:
            trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

        trial_num = int(trial_file.split('trial')[1].split('_')[0])

        # Get outcome
        outcome = None
        if trial_data['event'].max() == 5:
            outcome = "cooperation"
        elif 3 in trial_data['event'].values:
            outcome = "p1"
        elif 4 in trial_data['event'].values:
            outcome = "p2"

        if outcome in ["p1", "p2"]:
            # Get initial positions (first row with valid positions)
            first_row = trial_data.iloc[0]

            p1_x, p1_y = first_row['player1_x'], first_row['player1_y']
            p2_x, p2_y = first_row['player2_x'], first_row['player2_y']
            rabbit_x, rabbit_y = first_row['rabbit_x'], first_row['rabbit_y']

            # Compute initial distances to rabbit
            dist_p1 = np.sqrt((rabbit_x - p1_x)**2 + (rabbit_y - p1_y)**2)
            dist_p2 = np.sqrt((rabbit_x - p2_x)**2 + (rabbit_y - p2_y)**2)

            # Who was closer?
            closer_player = "p1" if dist_p1 < dist_p2 else "p2"

            # Did closer player win?
            closer_won = (closer_player == outcome)

            # How much closer?
            distance_advantage = abs(dist_p1 - dist_p2)

            results.append({
                'trial': trial_num,
                'outcome': outcome,
                'dist_p1': dist_p1,
                'dist_p2': dist_p2,
                'closer_player': closer_player,
                'closer_won': closer_won,
                'distance_advantage': distance_advantage
            })

    df = pd.DataFrame(results).sort_values('trial')

    print("Analysis of each defection trial:")
    print()
    print(f"{'Trial':<8} {'Winner':<8} {'Closer':<8} {'Match?':<10} {'P1 dist':<10} {'P2 dist':<10} {'Advantage':<12}")
    print("-" * 70)

    for _, row in df.iterrows():
        match_symbol = "✓" if row['closer_won'] else "✗ YIELDED?"
        print(f"{row['trial']:<8} {row['outcome']:<8} {row['closer_player']:<8} "
              f"{match_symbol:<10} {row['dist_p1']:<10.1f} {row['dist_p2']:<10.1f} {row['distance_advantage']:<12.1f}")

    print()
    print("="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    print()

    # Count how often closer player wins
    total_trials = len(df)
    closer_wins = df['closer_won'].sum()
    closer_win_rate = closer_wins / total_trials

    print(f"Closer player wins: {closer_wins}/{total_trials} = {closer_win_rate*100:.1f}%")
    print()

    if closer_win_rate > 0.9:
        print("✓ STRONG EVIDENCE FOR GREEDY RACING (coincidental alternation)")
        print("  → Both players go for rabbit, closer one wins")
        print("  → Alternation is just due to spatial positioning")
    elif closer_win_rate > 0.7:
        print("✓ MODERATE EVIDENCE FOR GREEDY RACING")
        print("  → Mostly distance-driven, but some exceptions")
    elif closer_win_rate < 0.6:
        print("✓ EVIDENCE FOR DELIBERATE TURN-TAKING")
        print("  → Winner not predicted by distance")
        print("  → Players must be yielding strategically")
    else:
        print("? MIXED EVIDENCE")
        print("  → Some distance effect, but not deterministic")

    print()

    # Binomial test: is closer-wins rate significantly > 0.5?
    from scipy.stats import binomtest
    p_value = binomtest(closer_wins, total_trials, 0.5, alternative='greater').pvalue

    print(f"Binomial test (H0: random, p=0.5):")
    print(f"  p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  → Closer player wins significantly more than chance")
        print(f"     (supports greedy racing hypothesis)")
    else:
        print(f"  → Not significantly different from chance")

    print()

    # Analyze cases where closer player LOST
    print("="*70)
    print("CASES WHERE CLOSER PLAYER LOST (potential yielding)")
    print("="*70)
    print()

    yielded_trials = df[~df['closer_won']]

    if len(yielded_trials) == 0:
        print("✓ NO YIELDING DETECTED")
        print("  Closer player won every single trial!")
        print("  → Strong evidence for greedy racing")
    else:
        print(f"Found {len(yielded_trials)} cases where closer player lost:")
        print()

        for _, row in yielded_trials.iterrows():
            print(f"  Trial {row['trial']}: "
                  f"{row['closer_player']} was closer ({row['distance_advantage']:.1f} units advantage) "
                  f"but {row['outcome']} won")

            # Was the advantage small (could be random)?
            if row['distance_advantage'] < 50:
                print(f"    → Small advantage ({row['distance_advantage']:.1f}), could be race/randomness")
            else:
                print(f"    → Large advantage ({row['distance_advantage']:.1f}), possible deliberate yielding?")

        print()

    # Correlation: distance advantage vs. winning
    print("="*70)
    print("DISTANCE ADVANTAGE ANALYSIS")
    print("="*70)
    print()

    # For each trial, compute: advantage of winner over loser
    winner_advantages = []
    for _, row in df.iterrows():
        if row['outcome'] == 'p1':
            # P1 won, what was P1's distance advantage?
            advantage = row['dist_p2'] - row['dist_p1']  # Positive = P1 closer
        else:  # p2 won
            advantage = row['dist_p1'] - row['dist_p2']  # Positive = P2 closer

        winner_advantages.append(advantage)

    avg_winner_advantage = np.mean(winner_advantages)

    print(f"Average distance advantage of winner: {avg_winner_advantage:.1f} units")
    print()

    if avg_winner_advantage > 50:
        print("✓ Winners have substantial distance advantage")
        print("  → Supports greedy racing (spatial positioning determines outcome)")
    elif avg_winner_advantage < 20:
        print("✓ Winners do NOT have much distance advantage")
        print("  → Suggests deliberate turn-taking (not distance-driven)")
    else:
        print("? Moderate distance advantage")
        print("  → Mixed evidence")

    print()

    # Save results
    df.to_csv('distance_analysis.csv', index=False)
    print(f"✓ Saved detailed analysis to distance_analysis.csv")

    return df


if __name__ == '__main__':
    results = analyze_deliberate_vs_coincidental()
