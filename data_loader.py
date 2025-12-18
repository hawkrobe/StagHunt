"""
Unified Data Loader for Stag Hunt Cooperation Task

Handles multiple file formats from different subjects and sessions:
- 16-column format (sub-120): with session, version columns
- 14-column format (sub-233): intermediate format
- 10-column format (most subjects): simple format with tp column

All formats are normalized to a standard schema with columns:
- time_point, player1_x, player1_y, player2_x, player2_y,
  stag_x, stag_y, rabbit_x, rabbit_y, event

Directory Structure:
-------------------
data/
  raw/           <- Original data files from Heejung
    sub-120/
    sub-231/
    ...
  derivatives/   <- Processed files with model regressors
    sub-120/
    ...

Usage:
------
```python
from data_loader import load_trial, load_all_trials, get_trial_info

# Load a single trial (raw)
trial = load_trial('data/raw/sub-120/ses-01/ieeg/sub-120_task-...tsv')

# Load all trials for a subject
trials = load_all_trials(subject='120')

# Load all trials with metadata
all_data = load_all_trials(include_metadata=True)

# Get info about a trial from filename
info = get_trial_info(filename)

# Save derivatives with model regressors
save_derivatives(trial_with_beliefs, info, output_dir='data/derivatives')
```
"""

# Default directories
RAW_DATA_DIR = 'data/raw'
DERIVATIVES_DIR = 'data/derivatives'

import pandas as pd
import numpy as np
import glob
import os
import re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path


# Standard column schema that models expect
STANDARD_COLUMNS = [
    'time_point', 'player1_x', 'player1_y', 'player2_x', 'player2_y',
    'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y', 'event'
]


def detect_format(filepath: str) -> str:
    """
    Detect the file format based on header structure.

    Returns:
        '16col' - sub-120 format with session/version
        '14col' - sub-233 format
        '10col' - simple format (most subjects)
        'legacy' - old CSV format (stag_hunt_coop_trial*.csv)
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()

    header_cols = header.split(',')
    n_cols = len(header_cols)

    # Check for legacy format (old CSV files)
    if header_cols[0] == 'time' or 'time' in header_cols:
        return 'legacy'

    if n_cols == 10:
        return '10col'
    elif n_cols == 14:
        return '14col'
    elif n_cols >= 16:
        return '16col'
    else:
        raise ValueError(f"Unknown format with {n_cols} columns in {filepath}")


def load_trial(filepath: str, normalize: bool = True) -> pd.DataFrame:
    """
    Load a single trial file, handling different formats.

    Parameters:
    -----------
    filepath : str
        Path to the trial TSV/CSV file
    normalize : bool
        If True, normalize to standard column schema

    Returns:
    --------
    pd.DataFrame with standardized columns
    """
    file_format = detect_format(filepath)

    if file_format == '10col':
        # Simple format: tp,player1_x,player1_y,...,event + extra unlabeled cols
        cols = ['time_point', 'player1_x', 'player1_y', 'player2_x', 'player2_y',
                'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y', 'event',
                '_e1', '_e2', '_e3', '_e4', '_e5', '_e6']
        df = pd.read_csv(filepath, skiprows=1, names=cols, index_col=False)

    elif file_format == '14col':
        # Intermediate format with value, delay columns
        cols = ['time_point', 'player1_x', 'player1_y', 'player2_x', 'player2_y',
                'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y', 'value', 'event',
                'delay', 'now_time', 'stime', '_e1', '_e2', '_e3', '_e4']
        df = pd.read_csv(filepath, skiprows=1, names=cols, index_col=False)
        # Fix typo if present
        if 'plater1_y' in df.columns:
            df = df.rename(columns={'plater1_y': 'player1_y'})

    elif file_format == '16col':
        # Full format with session, version columns
        cols = ['time_point', 'session', 'version', 'player1_x', 'player1_y',
                'player2_x', 'player2_y', 'stag_x', 'stag_y', 'rabbit_x', 'rabbit_y',
                'value', 'event', 'delay', 'now_time', 'stime', '_e1', '_e2']
        df = pd.read_csv(filepath, skiprows=1, names=cols, index_col=False)
        # Fix typo if present (plater1_y in header but we're using our own names)

    elif file_format == 'legacy':
        # Old CSV format from original data
        df = pd.read_csv(filepath)
        # Fix typo in old format
        if 'plater1_y' in df.columns:
            df = df.rename(columns={'plater1_y': 'player1_y'})
        # Rename 'time' to 'time_point' if needed
        if 'time' in df.columns and 'time_point' not in df.columns:
            df = df.rename(columns={'time': 'time_point'})
        # Add event column if missing
        if 'event' not in df.columns:
            df['event'] = 0

    else:
        raise ValueError(f"Unknown format: {file_format}")

    # Normalize to standard columns
    if normalize:
        # Keep only standard columns (plus any that exist)
        available = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[available].copy()

        # Ensure all standard columns exist
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

    # Clean up: remove rows with all NaN positions
    position_cols = ['player1_x', 'player1_y', 'player2_x', 'player2_y']
    df = df.dropna(subset=position_cols, how='all')

    return df


def get_trial_info(filepath: str) -> Dict:
    """
    Extract trial metadata from filename.

    Parameters:
    -----------
    filepath : str
        Path or filename of trial file

    Returns:
    --------
    dict with keys: subject, run, opponent, reward, trial, date, time, task_type
    """
    fname = os.path.basename(filepath)

    # Pattern for main task files
    main_pattern = r'sub-(\d+)_task-cooperation_run-(\d+[ab]?)_opponent-(\w+)_reward-(\w+)_trial-(\d+)_date-(\d+)_time-(\d+)'

    # Pattern for passive viewing
    passive_pattern = r'sub-(\d+)_task-cooperation_run-passive_trial-(\d+)_date-(\d+)_time-(\d+)'

    # Pattern for practice
    practice_pattern = r'sub-(\d+)_task-cooperation_run-practice_trial-(\d+)_date-(\d+)_time-(\d+)'

    # Pattern for legacy files
    legacy_pattern = r'stag_hunt_coop_trial(\d+)_(\d{4}_\d{2}_\d{2}_\d{4})'

    info = {
        'subject': None,
        'run': None,
        'opponent': None,
        'reward': None,
        'trial': None,
        'date': None,
        'time': None,
        'task_type': 'main',
        'filepath': filepath
    }

    match = re.search(main_pattern, fname)
    if match:
        info['subject'] = match.group(1)
        info['run'] = match.group(2)
        info['opponent'] = match.group(3)
        info['reward'] = match.group(4)
        info['trial'] = int(match.group(5))
        info['date'] = match.group(6)
        info['time'] = match.group(7)
        info['task_type'] = 'main'
        return info

    match = re.search(passive_pattern, fname)
    if match:
        info['subject'] = match.group(1)
        info['trial'] = int(match.group(2))
        info['date'] = match.group(3)
        info['time'] = match.group(4)
        info['task_type'] = 'passive'
        return info

    match = re.search(practice_pattern, fname)
    if match:
        info['subject'] = match.group(1)
        info['trial'] = int(match.group(2))
        info['date'] = match.group(3)
        info['time'] = match.group(4)
        info['task_type'] = 'practice'
        return info

    match = re.search(legacy_pattern, fname)
    if match:
        info['trial'] = int(match.group(1))
        info['date'] = match.group(2)
        info['task_type'] = 'legacy'
        return info

    return info


def find_trial_files(
    data_dir: str = None,
    subject: Optional[str] = None,
    opponent: Optional[str] = None,
    reward: Optional[str] = None,
    task_type: str = 'main'
) -> List[str]:
    """
    Find trial files matching criteria.

    Parameters:
    -----------
    data_dir : str
        Base data directory (default: data/raw)
    subject : str, optional
        Filter by subject ID (e.g., '120', '255')
    opponent : str, optional
        Filter by opponent type ('computer', 'same', 'diff', 'ieeg')
    reward : str, optional
        Filter by reward condition ('rabbitincrease', 'stagdecrease')
    task_type : str
        Type of trials to find ('main', 'passive', 'practice', 'legacy', 'all')

    Returns:
    --------
    List of file paths
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    files = []

    # Handle legacy files
    if task_type in ['legacy', 'all']:
        legacy_files = glob.glob(os.path.join(data_dir, 'stag_hunt_coop_trial*.csv'))
        files.extend(legacy_files)

    # Handle new format files
    if task_type != 'legacy':
        # Build pattern
        if subject:
            sub_pattern = f'sub-{subject}'
        else:
            sub_pattern = 'sub-*'

        # Find all TSV files in subject directories
        pattern = os.path.join(data_dir, sub_pattern, '**', '*.tsv')
        tsv_files = glob.glob(pattern, recursive=True)

        for f in tsv_files:
            info = get_trial_info(f)

            # Filter by task type
            if task_type != 'all' and info['task_type'] != task_type:
                continue

            # Filter by opponent
            if opponent and info['opponent'] != opponent:
                continue

            # Filter by reward
            if reward and info['reward'] != reward:
                continue

            files.append(f)

    return sorted(files)


def load_all_trials(
    data_dir: str = None,
    subject: Optional[str] = None,
    opponent: Optional[str] = None,
    reward: Optional[str] = None,
    task_type: str = 'main',
    include_metadata: bool = False
) -> Union[List[pd.DataFrame], List[Tuple[pd.DataFrame, Dict]]]:
    """
    Load all trials matching criteria.

    Parameters:
    -----------
    data_dir : str
        Base data directory
    subject : str, optional
        Filter by subject ID
    opponent : str, optional
        Filter by opponent type
    reward : str, optional
        Filter by reward condition
    task_type : str
        Type of trials ('main', 'passive', 'practice', 'legacy', 'all')
    include_metadata : bool
        If True, return (trial_data, metadata) tuples

    Returns:
    --------
    List of DataFrames, or list of (DataFrame, dict) tuples if include_metadata=True
    """
    files = find_trial_files(data_dir, subject, opponent, reward, task_type)

    results = []
    for f in files:
        try:
            df = load_trial(f)
            if include_metadata:
                info = get_trial_info(f)
                results.append((df, info))
            else:
                results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
            continue

    return results


def get_outcome(trial_data: pd.DataFrame) -> Dict:
    """
    Determine trial outcome based on final positions.

    Parameters:
    -----------
    trial_data : pd.DataFrame
        Trial data with player and prey positions

    Returns:
    --------
    dict with keys:
        - outcome: 'cooperation', 'mutual_defection', 'p1_stag_p2_rabbit', 'p1_rabbit_p2_stag'
        - p1_target: 'stag' or 'rabbit'
        - p2_target: 'stag' or 'rabbit'
        - p1_dist_stag, p1_dist_rabbit, p2_dist_stag, p2_dist_rabbit
    """
    last = trial_data.iloc[-1]

    p1 = np.array([last['player1_x'], last['player1_y']])
    p2 = np.array([last['player2_x'], last['player2_y']])
    stag = np.array([last['stag_x'], last['stag_y']])
    rabbit = np.array([last['rabbit_x'], last['rabbit_y']])

    dist_p1_stag = np.linalg.norm(p1 - stag)
    dist_p1_rabbit = np.linalg.norm(p1 - rabbit)
    dist_p2_stag = np.linalg.norm(p2 - stag)
    dist_p2_rabbit = np.linalg.norm(p2 - rabbit)

    p1_target = 'stag' if dist_p1_stag < dist_p1_rabbit else 'rabbit'
    p2_target = 'stag' if dist_p2_stag < dist_p2_rabbit else 'rabbit'

    if p1_target == 'stag' and p2_target == 'stag':
        outcome = 'cooperation'
    elif p1_target == 'rabbit' and p2_target == 'rabbit':
        outcome = 'mutual_defection'
    elif p1_target == 'stag':
        outcome = 'p1_stag_p2_rabbit'
    else:
        outcome = 'p1_rabbit_p2_stag'

    return {
        'outcome': outcome,
        'p1_target': p1_target,
        'p2_target': p2_target,
        'p1_dist_stag': dist_p1_stag,
        'p1_dist_rabbit': dist_p1_rabbit,
        'p2_dist_stag': dist_p2_stag,
        'p2_dist_rabbit': dist_p2_rabbit
    }


def summarize_data(data_dir: str = None) -> pd.DataFrame:
    """
    Generate summary statistics for all available data.

    Parameters:
    -----------
    data_dir : str
        Base data directory (default: data/raw)

    Returns:
    --------
    pd.DataFrame with trial-level summary
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    trials = load_all_trials(data_dir, task_type='main', include_metadata=True)

    rows = []
    for df, info in trials:
        outcome_info = get_outcome(df)

        first = df.iloc[0]
        last = df.iloc[-1]

        row = {
            'subject': info['subject'],
            'run': info['run'],
            'opponent': info['opponent'],
            'reward': info['reward'],
            'trial': info['trial'],
            'outcome': outcome_info['outcome'],
            'p1_target': outcome_info['p1_target'],
            'p2_target': outcome_info['p2_target'],
            'cooperation': outcome_info['outcome'] == 'cooperation',
            'duration': last['time_point'] - first['time_point'],
            'n_timesteps': len(df)
        }
        rows.append(row)

    return pd.DataFrame(rows)


# Convenience function for backward compatibility
def save_derivative(
    trial_data: pd.DataFrame,
    trial_info: Dict,
    output_dir: str = None,
    suffix: str = '_with_beliefs'
) -> str:
    """
    Save a processed trial with model regressors to derivatives directory.

    Parameters:
    -----------
    trial_data : pd.DataFrame
        Trial data with computed regressors (e.g., beliefs)
    trial_info : dict
        Trial metadata from get_trial_info()
    output_dir : str
        Output directory (default: data/derivatives)
    suffix : str
        Suffix to add to filename (default: '_with_beliefs')

    Returns:
    --------
    str : Path to saved file
    """
    if output_dir is None:
        output_dir = DERIVATIVES_DIR

    # Build output path mirroring raw structure
    if trial_info.get('subject'):
        sub_dir = os.path.join(output_dir, f"sub-{trial_info['subject']}")
    else:
        sub_dir = output_dir

    os.makedirs(sub_dir, exist_ok=True)

    # Generate output filename
    if trial_info.get('filepath'):
        # Use original filename with suffix
        orig_name = os.path.basename(trial_info['filepath'])
        base_name = orig_name.rsplit('.', 1)[0]  # Remove extension
        out_name = f"{base_name}{suffix}.tsv"
    else:
        # Generate a name
        parts = []
        if trial_info.get('subject'):
            parts.append(f"sub-{trial_info['subject']}")
        if trial_info.get('opponent'):
            parts.append(f"opponent-{trial_info['opponent']}")
        if trial_info.get('trial'):
            parts.append(f"trial-{trial_info['trial']:02d}")
        out_name = '_'.join(parts) + f"{suffix}.tsv" if parts else f"trial{suffix}.tsv"

    out_path = os.path.join(sub_dir, out_name)

    # Save as TSV
    trial_data.to_csv(out_path, sep='\t', index=False)

    return out_path


def save_all_derivatives(
    belief_model,
    data_dir: str = None,
    output_dir: str = None,
    subject: Optional[str] = None,
    opponent: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """
    Process all raw trials and save derivatives with model regressors.

    Parameters:
    -----------
    belief_model : BeliefModel
        Initialized belief model to run on trials
    data_dir : str
        Input directory (default: data/raw)
    output_dir : str
        Output directory (default: data/derivatives)
    subject : str, optional
        Filter by subject ID
    opponent : str, optional
        Filter by opponent type
    verbose : bool
        Print progress

    Returns:
    --------
    List of paths to saved derivative files
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    if output_dir is None:
        output_dir = DERIVATIVES_DIR

    # Find and process trials
    files = find_trial_files(data_dir, subject=subject, opponent=opponent, task_type='main')

    if verbose:
        print(f"Processing {len(files)} trials...")

    saved_paths = []
    for i, filepath in enumerate(files):
        try:
            # Load raw trial
            trial = load_trial(filepath)
            info = get_trial_info(filepath)

            # Run belief model
            trial_with_beliefs = belief_model.run_trial(trial)

            # Add outcome info
            outcome = get_outcome(trial_with_beliefs)
            trial_with_beliefs['outcome'] = outcome['outcome']

            # Save derivative
            out_path = save_derivative(trial_with_beliefs, info, output_dir)
            saved_paths.append(out_path)

            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(files)} trials...")

        except Exception as e:
            if verbose:
                print(f"  Error processing {filepath}: {e}")
            continue

    if verbose:
        print(f"Saved {len(saved_paths)} derivatives to {output_dir}")

    return saved_paths


def load_trial_legacy(filepath: str) -> pd.DataFrame:
    """
    Load trial using the old API (for backward compatibility).
    Same as load_trial() but ensures legacy CSV files work.
    """
    return load_trial(filepath, normalize=True)


if __name__ == '__main__':
    # Quick test / summary
    print("Stag Hunt Data Loader")
    print("=" * 60)
    print(f"\nDirectory structure:")
    print(f"  Raw data:    {RAW_DATA_DIR}")
    print(f"  Derivatives: {DERIVATIVES_DIR}")

    # Count files by type (uses default RAW_DATA_DIR)
    main_files = find_trial_files(task_type='main')
    passive_files = find_trial_files(task_type='passive')
    practice_files = find_trial_files(task_type='practice')
    legacy_files = find_trial_files(task_type='legacy')

    print(f"\nData found in '{RAW_DATA_DIR}':")
    print(f"  Main task trials: {len(main_files)}")
    print(f"  Passive viewing:  {len(passive_files)}")
    print(f"  Practice trials:  {len(practice_files)}")
    print(f"  Legacy trials:    {len(legacy_files)}")

    # List subjects
    subjects = set()
    for f in main_files:
        info = get_trial_info(f)
        if info['subject']:
            subjects.add(info['subject'])

    print(f"\nSubjects: {sorted(subjects)}")

    # Generate summary if we have main trials
    if main_files:
        print("\n" + "=" * 60)
        print("Generating summary statistics...")
        summary = summarize_data()

        print(f"\nTotal trials: {len(summary)}")
        print(f"\nCooperation rate: {summary['cooperation'].mean():.1%}")

        print("\nBy opponent:")
        for opp in ['computer', 'same', 'diff', 'ieeg']:
            sub = summary[summary['opponent'] == opp]
            if len(sub) > 0:
                print(f"  {opp}: {sub['cooperation'].mean():.1%} ({sub['cooperation'].sum()}/{len(sub)})")

        print("\nBy subject:")
        for sub in sorted([s for s in summary['subject'].unique() if s is not None]):
            sub_df = summary[summary['subject'] == sub]
            print(f"  sub-{sub}: {sub_df['cooperation'].mean():.1%} ({sub_df['cooperation'].sum()}/{len(sub_df)})")
