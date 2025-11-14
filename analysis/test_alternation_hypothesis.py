#!/usr/bin/env python3
"""
Test the hypothesis that players alternate who gets the rabbit.

This could be evidence of:
1. Fairness norms ("take turns")
2. Coordination on a simpler convention after cooperation failed
3. Strategic reciprocity ("I'll let you win if you let me win")
"""

import pandas as pd
import numpy as np


def test_alternation():
    """Test if rabbit outcomes follow alternation pattern."""

    print("="*70)
    print("TESTING ALTERNATION / TURN-TAKING HYPOTHESIS")
    print("="*70)
    print()

    # Load cross-trial results
    df = pd.read_csv('cross_trial_beliefs.csv')

    # IMPORTANT: Sort by trial number (1, 2, 3, ..., 10, 11, 12)
    df = df.sort_values('trial').reset_index(drop=True)

    # Extract who got rabbit (ignore cooperation trial)
    sequence = []
    for _, row in df.iterrows():
        if row['outcome'] == 'P1 rabbit':
            sequence.append(1)
        elif row['outcome'] == 'P2 rabbit':
            sequence.append(2)
        else:  # cooperation
            sequence.append(0)

    print("Sequence of outcomes (1=P1 rabbit, 2=P2 rabbit, 0=cooperation):")
    print()

    for i, (trial_num, outcome) in enumerate(zip(df['trial'], sequence)):
        symbol = "🔴" if outcome == 1 else "🟡" if outcome == 2 else "🦌"
        label = f"P{outcome} rabbit" if outcome != 0 else "COOPERATION"
        print(f"  Trial {trial_num:2d}: {symbol} {label:15s}", end="")

        # Check if this alternated from previous
        if i > 0 and outcome != 0 and sequence[i-1] != 0:
            if outcome != sequence[i-1]:
                print("  ✓ Alternates")
            else:
                print("  ✗ Repeat")
        else:
            print()

    print()

    # Count alternations (excluding cooperation trial)
    defection_trials = [(i, outcome) for i, outcome in enumerate(sequence) if outcome != 0]

    alternations = 0
    repeats = 0

    for i in range(1, len(defection_trials)):
        prev_idx, prev_outcome = defection_trials[i-1]
        curr_idx, curr_outcome = defection_trials[i]

        if curr_outcome != prev_outcome:
            alternations += 1
        else:
            repeats += 1

    total = alternations + repeats
    alternation_rate = alternations / total if total > 0 else 0

    print("="*70)
    print("STATISTICS")
    print("="*70)
    print()
    print(f"Total defection trials: {len(defection_trials)}")
    print(f"Consecutive pairs: {total}")
    print(f"  Alternations: {alternations} ({alternation_rate*100:.1f}%)")
    print(f"  Repeats: {repeats} ({(1-alternation_rate)*100:.1f}%)")
    print()

    # Compare to chance
    # Under random choice, P(alternate) = 0.5
    # But is this sequence significantly different?

    from scipy.stats import binomtest
    p_value = binomtest(alternations, total, 0.5, alternative='greater').pvalue

    print(f"Binomial test (H0: random, p=0.5):")
    print(f"  p-value = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  → Significantly more alternation than chance! ✓")
    else:
        print(f"  → Not significantly different from chance")

    print()

    # Analyze timing of alternation onset
    print("="*70)
    print("WHEN DID ALTERNATION START?")
    print("="*70)
    print()

    # Check alternation rate in different epochs
    coop_idx = next(i for i, outcome in enumerate(sequence) if outcome == 0)
    coop_trial_num = df.iloc[coop_idx]['trial']

    print(f"Cooperation occurred at trial {coop_trial_num} (index {coop_idx})")
    print()

    # Before cooperation
    before_coop = [outcome for outcome in sequence[:coop_idx] if outcome != 0]
    if len(before_coop) > 1:
        before_alt = sum(1 for i in range(1, len(before_coop)) if before_coop[i] != before_coop[i-1])
        before_rate = before_alt / (len(before_coop) - 1)
        print(f"Before cooperation (trials 1-{coop_trial_num-1}):")
        print(f"  Alternation rate: {before_alt}/{len(before_coop)-1} = {before_rate*100:.1f}%")
        print()

    # After cooperation
    after_coop = [outcome for outcome in sequence[coop_idx+1:] if outcome != 0]
    if len(after_coop) > 1:
        after_alt = sum(1 for i in range(1, len(after_coop)) if after_coop[i] != after_coop[i-1])
        after_rate = after_alt / (len(after_coop) - 1)
        print(f"After cooperation (trials {coop_trial_num+1}-end):")
        print(f"  Alternation rate: {after_alt}/{len(after_coop)-1} = {after_rate*100:.1f}%")
        print()

        if after_rate > before_rate:
            print(f"  → Alternation INCREASED after cooperation failure")
            print(f"     (from {before_rate*100:.0f}% to {after_rate*100:.0f}%)")
        else:
            print(f"  → Alternation did not increase after cooperation")

    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()

    if alternation_rate > 0.7:
        print("✓ Strong evidence for turn-taking / fairness convention")
        print()
        print("Possible explanations:")
        print("  1. Explicit agreement: \"Let's take turns\"")
        print("  2. Implicit fairness norm: \"I got it last time, so you go now\"")
        print("  3. Reciprocal altruism: \"I'll let you win so you let me win\"")
        print()

        if after_rate > 0.8 and before_rate < 0.6:
            print("✓ Convention emerged AFTER cooperation failed:")
            print("  → \"We can't catch stag together, but at least we can share rabbits fairly\"")
    elif alternation_rate > 0.5:
        print("✓ Some evidence for turn-taking, but not conclusive")
    else:
        print("✗ No strong evidence for turn-taking")

    return alternation_rate


if __name__ == '__main__':
    rate = test_alternation()
