"""
run_ablation.py — Full ablation study (A1–A6, non-ML only)
Paper: Offset-Normalised SavGol for LoRa V2X Key Agreement

Usage:
  python run_ablation.py --data_dir data/ --split 50_25_25
  python run_ablation.py --data_dir data/ --split 60_20_20
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from utils.partition import load_and_split
from utils.metrics   import compute_all_metrics
from preprocessing.pipelines import PIPELINES

# Random seed for reproducibility
import random
random.seed(42)
np.random.seed(42)

# Data files mapping
FILES = {
    'V2I_Turning_20':   'combined_v2i_belok_20.csv',
    'V2I_Turning_30':   'combined_v2i_belok_30.csv',
    'V2I_Straight_20':  'combined_v2i_lurus_20.csv',
    'V2I_Straight_30':  'combined_v2i_lurus_30.csv',
    'V2V_Opposite_20':  'combined_V2V_berlawanan20.csv',
    'V2V_Opposite_30':  'combined_V2V_berlawanan30.csv',
    'V2V_Same_20':      'combined_V2V_searah20.csv',
    'V2V_Same_30':      'combined_V2V_searah30.csv',
}


def run_ablation(data_dir: str, scheme: str = '50_25_25') -> pd.DataFrame:
    """
    Run all non-ML ablation configurations (A1–A6) across all 8 conditions.
    Returns DataFrame with per-condition per-pipeline results.
    """
    results = []

    for condition, filename in FILES.items():
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found — skipping")
            continue

        data = load_and_split(str(filepath), scheme)
        print(f"\n  Processing: {condition} (N={data['n_total']}, "
              f"train={data['n_train']}, test={data['n_test']})")

        for pipeline_id, pipeline_fn in PIPELINES.items():
            processed = pipeline_fn(data)

            metrics = compute_all_metrics(
                alice_signal=processed['alice_test'],
                bob_signal=processed['bob_test'],
                alice_train=processed['alice_train'],
                bob_train=processed['bob_train'],
            )
            metrics.update({
                'Condition':  condition,
                'Pipeline':   pipeline_id,
                'Config':     processed['label'],
                'Scheme':     scheme,
                'N_total':    data['n_total'],
                'N_train':    data['n_train'],
                'N_test':     data['n_test'],
            })
            results.append(metrics)
            print(f"    {pipeline_id}: KDR={metrics['KDR']:.4f}, "
                  f"Entropy={metrics['Entropy']:.4f}, "
                  f"Degenerate={metrics['Degenerate']}")

    df = pd.DataFrame(results)
    return df


def print_summary(df: pd.DataFrame):
    """Print ablation summary table (median across 8 conditions)."""
    print(f"\n{'='*75}")
    print(f"ABLATION SUMMARY — Median across 8 conditions ({df['Scheme'].iloc[0]})")
    print(f"{'='*75}")
    print(f"{'Pipeline':<12} {'KDR':>8} {'IQR':>8} {'Entropy':>8} "
          f"{'Degen':>6} {'Agreed KGR':>11}")
    print("-"*75)

    for pid in ['A1','A2','A3','A4','A5','A6']:
        sub = df[df['Pipeline'] == pid]
        if sub.empty:
            continue
        med_kdr = sub['KDR'].median()
        iqr_kdr = sub['KDR'].quantile(0.75) - sub['KDR'].quantile(0.25)
        med_ent = sub['Entropy'].median()
        n_degen = sub['Degenerate'].sum()
        med_agr = sub['Agreed_KGR_bps'].median()
        marker  = ' ★' if pid == 'A5' else ''
        print(f"{pid+marker:<12} {med_kdr:>8.4f} {iqr_kdr:>8.4f} "
              f"{med_ent:>8.4f} {n_degen:>6} {med_agr:>11.4f}")

    print(f"{'='*75}")
    print("★ = Proposed method (selected by smallest IQR, post hoc, N=8)")
    print("Note: A4 Moving Average achieves lowest median KDR")
    print("Note: Degenerate = Shannon entropy < 0.5 in that condition\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study A1–A6')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing CSV files')
    parser.add_argument('--split', type=str, default='50_25_25',
                        choices=['50_25_25','60_20_20'],
                        help='Partition scheme')
    parser.add_argument('--output', type=str,
                        default='outputs/ablation_results.csv')
    args = parser.parse_args()

    print(f"\nRunning ablation study [{args.split} split]...")
    df = run_ablation(args.data_dir, args.split)

    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved: {args.output}")
        print_summary(df)
    else:
        print("No results — check data directory.")


if __name__ == '__main__':
    main()
