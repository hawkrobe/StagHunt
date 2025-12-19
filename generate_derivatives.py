#!/usr/bin/env python
"""
Generate Derivative Files with Model Regressors

This script processes raw trial data and saves derivatives with computed
regressors for neural encoding analysis.

Output columns added:
- joint_goal_stag: IW model belief (P(stag is joint goal))
- p1_belief_p2_stag, p2_belief_p1_stag: Standard per-player beliefs
- joint_goal_pe, p1_belief_pe, p2_belief_pe: Belief prediction errors (Δbelief)
- p1_dist_stag, p1_dist_rabbit, p2_dist_stag, p2_dist_rabbit: Distance to targets
- player_distance: Distance between players
- p1_speed, p2_speed: Player movement speeds
- outcome: Trial outcome (cooperation/mutual_defection/etc.)

Usage:
------
# Generate all derivatives (all regressors)
python generate_derivatives.py

# Filter by subject
python generate_derivatives.py --subject 120

# Filter by opponent type
python generate_derivatives.py --opponent ieeg

# Custom output directory
python generate_derivatives.py --output custom_derivatives/
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    save_all_derivatives,
    RAW_DATA_DIR,
    DERIVATIVES_DIR
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate derivative files with model regressors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_derivatives.py                    # All trials, distance model
  python generate_derivatives.py --model planning   # Planning-based model
  python generate_derivatives.py --subject 120      # Single subject
  python generate_derivatives.py --opponent ieeg    # Single opponent type
        """
    )

    parser.add_argument(
        '--subject', '-s',
        help='Filter by subject ID (e.g., 120, 255)'
    )
    parser.add_argument(
        '--opponent', '-o',
        choices=['computer', 'same', 'diff', 'ieeg'],
        help='Filter by opponent type'
    )
    parser.add_argument(
        '--input', '-i',
        default=RAW_DATA_DIR,
        help=f'Input directory (default: {RAW_DATA_DIR})'
    )
    parser.add_argument(
        '--output',
        default=DERIVATIVES_DIR,
        help=f'Output directory (default: {DERIVATIVES_DIR})'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Use add_all_regressors to include everything
    from models.belief import add_all_regressors
    belief_func = add_all_regressors

    if not args.quiet:
        print("=" * 60)
        print("Generating Derivative Files")
        print("=" * 60)
        print(f"\nRegressors: beliefs (IW + standard), distances, prediction errors")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        if args.subject:
            print(f"Subject: sub-{args.subject}")
        if args.opponent:
            print(f"Opponent: {args.opponent}")
        print()

    # Generate derivatives
    saved_paths = save_all_derivatives(
        belief_func=belief_func,
        data_dir=args.input,
        output_dir=args.output,
        subject=args.subject,
        opponent=args.opponent,
        verbose=not args.quiet
    )

    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"Complete! Generated {len(saved_paths)} derivative files.")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
