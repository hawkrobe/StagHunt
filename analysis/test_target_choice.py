#!/usr/bin/env python3
"""
Test whether distance alone predicts target choice.

Key question: Do people ever go for rabbit when stag is closer?
If yes → distance-based model is insufficient
If no → maybe distance explains everything (but why always rabbit?)
"""

import pandas as pd
import numpy as np
import glob


def analyze_target_choice():
    """Check if initial distances predict target choice."""

    print("="*70)
    print("DOES DISTANCE PREDICT TARGET CHOICE?")
    print("="*70)
    print()

    trial_files = sorted(glob.glob('inputs/stag_hunt_coop_trial*_2024_08_24_0848.csv'))

    results = []

    for trial_file in trial_files:
        trial_data = pd.read_csv(trial_file)
        if 'plater1_y' in trial_data.columns:
            trial_data = trial_data.rename(columns={'plater1_y': 'player1_y'})

        trial_num = int(trial_file.split('trial')[1].split('_')[0])
        first_row = trial_data.iloc[0]

        # Get positions
        p1_x, p1_y = first_row['player1_x'], first_row['player1_y']
        p2_x, p2_y = first_row['player2_x'], first_row['player2_y']
        stag_x, stag_y = first_row['stag_x'], first_row['stag_y']
        rabbit_x, rabbit_y = first_row['rabbit_x'], first_row['rabbit_y']

        # Compute distances for each player
        p1_dist_stag = np.sqrt((stag_x - p1_x)**2 + (stag_y - p1_y)**2)
        p1_dist_rabbit = np.sqrt((rabbit_x - p1_x)**2 + (rabbit_y - p1_y)**2)
        p2_dist_stag = np.sqrt((stag_x - p2_x)**2 + (stag_y - p2_y)**2)
        p2_dist_rabbit = np.sqrt((rabbit_x - p2_x)**2 + (rabbit_y - p2_y)**2)

        # What does distance predict?
        p1_closer_to = "stag" if p1_dist_stag < p1_dist_rabbit else "rabbit"
        p2_closer_to = "stag" if p2_dist_stag < p2_dist_rabbit else "rabbit"

        # What actually happened?
        outcome = "unknown"
        if trial_data['event'].max() == 5:
            outcome = "cooperation"
        elif 3 in trial_data['event'].values:
            outcome = "p1_rabbit"
        elif 4 in trial_data['event'].values:
            outcome = "p2_rabbit"

        # Did distance predict correctly?
        if outcome == "cooperation":
            p1_chose = "stag"
            p2_chose = "stag"
        elif outcome == "p1_rabbit":
            p1_chose = "rabbit"
            p2_chose = "unknown"  # P2's choice unclear (may have tried stag)
        elif outcome == "p2_rabbit":
            p1_chose = "unknown"
            p2_chose = "rabbit"

        results.append({
            'trial': trial_num,
            'outcome': outcome,
            'p1_dist_stag': p1_dist_stag,
            'p1_dist_rabbit': p1_dist_rabbit,
            'p1_closer_to': p1_closer_to,
            'p1_chose': p1_chose,
            'p1_match': (p1_closer_to == p1_chose) if p1_chose != "unknown" else None,
            'p2_dist_stag': p2_dist_stag,
            'p2_dist_rabbit': p2_dist_rabbit,
            'p2_closer_to': p2_closer_to,
            'p2_chose': p2_chose,
            'p2_match': (p2_closer_to == p2_chose) if p2_chose != "unknown" else None,
        })

    df = pd.DataFrame(results).sort_values('trial')

    print("Initial distances and target choices:")
    print()
    print(f"{'Trial':<8} {'Outcome':<15} {'P1 closer':<12} {'P1 chose':<12} {'Match':<8} "
          f"{'P2 closer':<12} {'P2 chose':<12} {'Match':<8}")
    print("-" * 100)

    for _, row in df.iterrows():
        p1_symbol = "✓" if row['p1_match'] else ("✗" if row['p1_match'] is not None else "?")
        p2_symbol = "✓" if row['p2_match'] else ("✗" if row['p2_match'] is not None else "?")

        print(f"{row['trial']:<8} {row['outcome']:<15} {row['p1_closer_to']:<12} "
              f"{row['p1_chose']:<12} {p1_symbol:<8} "
              f"{row['p2_closer_to']:<12} {row['p2_chose']:<12} {p2_symbol:<8}")

    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Count matches
    p1_matches = df['p1_match'].dropna()
    p2_matches = df['p2_match'].dropna()

    if len(p1_matches) > 0:
        p1_correct = p1_matches.sum()
        p1_rate = p1_correct / len(p1_matches)
        print(f"P1: Distance predicted choice in {p1_correct}/{len(p1_matches)} cases ({p1_rate*100:.1f}%)")

    if len(p2_matches) > 0:
        p2_correct = p2_matches.sum()
        p2_rate = p2_correct / len(p2_matches)
        print(f"P2: Distance predicted choice in {p2_correct}/{len(p2_matches)} cases ({p2_rate*100:.1f}%)")

    print()

    # Key question: Any cases where they chose rabbit despite stag being closer?
    p1_violations = df[(df['p1_closer_to'] == 'stag') & (df['p1_chose'] == 'rabbit')]
    p2_violations = df[(df['p2_closer_to'] == 'stag') & (df['p2_chose'] == 'rabbit')]

    total_violations = len(p1_violations) + len(p2_violations)

    print(f"Cases where player chose RABBIT despite STAG being closer:")
    print(f"  P1: {len(p1_violations)} cases")
    print(f"  P2: {len(p2_violations)} cases")
    print(f"  Total: {total_violations} cases")
    print()

    if total_violations > 0:
        print("✓ DISTANCE ALONE CANNOT EXPLAIN TARGET CHOICE")
        print("  → Players sometimes prefer rabbit even when stag is closer")
        print("  → This requires additional mechanisms:")
        print("     - Risk aversion (rabbit is safe)")
        print("     - Low cooperation beliefs (stag requires partner)")
        print("     - Strategic reasoning")
    else:
        print("? Distance perfectly predicts target choice")
        print("  BUT: Why would stag ALWAYS be further?")
        print("  → May indicate task design bias toward rabbit")

    print()

    # Distance differences
    print("="*70)
    print("DISTANCE DIFFERENCES")
    print("="*70)
    print()

    for _, row in df.iterrows():
        p1_diff = row['p1_dist_rabbit'] - row['p1_dist_stag']  # Positive = stag closer
        p2_diff = row['p2_dist_rabbit'] - row['p2_dist_stag']

        print(f"Trial {row['trial']:2d}: "
              f"P1: stag {row['p1_dist_stag']:6.1f}, rabbit {row['p1_dist_rabbit']:6.1f} "
              f"(diff: {p1_diff:+6.1f}) | "
              f"P2: stag {row['p2_dist_stag']:6.1f}, rabbit {row['p2_dist_rabbit']:6.1f} "
              f"(diff: {p2_diff:+6.1f})")

    return df


if __name__ == '__main__':
    results = analyze_target_choice()
