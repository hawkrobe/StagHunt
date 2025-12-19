#!/usr/bin/env python
"""
Generate Derivative Files with Model Regressors

This script processes raw trial data and saves derivatives with computed
belief regressors for neural analysis.

Output columns added:
- p1_belief_p2_stag: Player 1's belief that Player 2 is going for stag
- p2_belief_p1_stag: Player 2's belief that Player 1 is going for stag
- p1_movement_angle: Player 1's movement direction (radians)
- p2_movement_angle: Player 2's movement direction (radians)
- outcome: Trial outcome (cooperation/mutual_defection/etc.)

Usage:
------
# Generate all derivatives with distance-based belief model
python generate_derivatives.py

# Use planning-based model (recommended, but requires fitted params)
python generate_derivatives.py --model planning

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
        '--model', '-m',
        choices=['standard', 'iw'],
        default='iw',
        help='Belief model type: standard (per-player) or iw (joint goal, default)'
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

    # Initialize belief model
    if args.model == 'iw':
        from models.belief import add_iw_beliefs
        belief_func = add_iw_beliefs
        model_desc = "Imagined We (joint goal)"
    else:
        from models.belief import add_standard_beliefs
        belief_func = add_standard_beliefs
        model_desc = "Standard (per-player intentions)"

    if not args.quiet:
        print("=" * 60)
        print("Generating Derivative Files")
        print("=" * 60)
        print(f"\nModel:  {model_desc}")
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
