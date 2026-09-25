# LoRa-V2X-SKG-Benchmark

Code and field data for:

**Offset-Normalised Savitzky–Golay Channel Smoothing for Key Agreement in Heterogeneous LoRa V2X: Unified Benchmarking and Ablation Study**
*International Journal of Intelligent Engineering and Systems (IJIES), Paper ID 20265854*

The repository contains only the scripts and data that produce the values reported in the paper.

---

## Data

Eight paired RSSI files (one measurement run per condition). Each CSV has the columns `rssi_bob`, `rssi_alice` (dBm); Alice = RFM95, Bob = SX1262.

| File | Scenario | Speed | N |
|------|----------|-------|---|
| `combined_v2i belok 20.csv` | V2I Turning | 20 km/h | 35 |
| `combined_v2i belok 30.csv` | V2I Turning | 30 km/h | 50 |
| `combined_v2i lurus 20.csv` | V2I Straight | 20 km/h | 80 |
| `combined_v2i lurus 30.csv` | V2I Straight | 30 km/h | 89 |
| `combined_V2V berlawanan20.csv` | V2V Opposite Direction | 20 km/h | 26 |
| `combined_V2V berlawanan30.csv` | V2V Opposite Direction | 30 km/h | 19 |
| `combined_V2V searah20.csv` | V2V Same Direction | 20 km/h | 102 |
| `combined_V2V searah30.csv` | V2V Same Direction | 30 km/h | 94 |

Total: 495 paired observations.

---

## Scripts and the results they reproduce

Run every script from the repository root (the scripts read the CSV files by name from the working directory).

| Script | Split | Reproduces |
|--------|-------|-----------|
| `Ablation_SavGol_Final.py` | 50/25/25 | Table 9: KDR, IQR, KDR SD, KGR, entropy, degenerate flag for A1–A8 |
| `Table9_Corr_RMSE.py` | 50/25/25 | Table 9: Corr and RMSE columns for A1–A6 |
| `Proposed_SavGol_Final.py` | 60/20/20 | Proposed Offset + Savitzky–Golay pipeline (Tables 8, 10, 11) |
| `Proposed_A5_Revised.py` | 60/20/20 | Proposed Offset + Wavelet variant (Tables 10, 11) |
| `Proposed_Revised_v3.py` | 60/20/20 | Proposed Offset + Wavelet + BiLSTM variant (Tables 10, 11) |
| `Toward_Revised.py` | 60/20/20 | Toward [13], adapted (Tables 10, 11) |
| `VehicleKey_Revised.py` | 60/20/20 | Scenario [14] (Vehicle-Key), adapted (Tables 10, 11) |
| `PLKG_Revised.py` | 60/20/20 | PLKG [15], adapted (Tables 10, 11) |
| `PLS_LoRa_Revised.py` | 60/20/20 | PLS-LoRa [16], adapted (Tables 10, 11) |

The single-file pipeline scripts use `FILE_PATH = "combined_v2i lurus 20.csv"` (the Table 10 condition); change `FILE_PATH` to evaluate another condition (Table 8 per-scenario values). Each script prints its per-stage timing (preprocessing, quantisation, reconciliation, SHA-256, total); timing values depend on the hardware on which the script is run (Laptop and Raspberry Pi 5 in the paper).

```bash
pip install -r requirements.txt
python Ablation_SavGol_Final.py      # Table 9 (A1–A8)
python Table9_Corr_RMSE.py           # Table 9 Corr / RMSE (A1–A6)
python Proposed_SavGol_Final.py      # Proposed pipeline, V2I Straight 20 km/h
```

---

## Evaluation protocol

- **Chronological, file-level partitioning.** 60/20/20 (train/validation/test) for the baseline comparison (Tables 8, 10, 11); 50/25/25 for the ablation study (Table 9). Counts use floor rounding; exact indices are in `configs/split_indices.json`.
- **Leakage control.** Offset estimate and quantisation thresholds come from the training partition only; sliding windows (ML pipelines) are formed within each partition.
- **Quantisation.** Common 32-level, 5-bit Gray code, no guard band.
- **Reconciliation.** Oracle-assisted Cascade-style; post-reconciliation KAR = 1.0 by construction and is reported only as an upper bound.

## Random seeds

- **A7 and A8 (ablation, BiLSTM):** 20 seeds, `s = 0, 1, …, 19` (`for s in range(N_SEEDS)`, `N_SEEDS = 20`), applied through `torch.manual_seed(s)` and `np.random.seed(s)`. Table 9 reports mean ± SD over the 20 runs; runs whose output bit string has Shannon entropy < 0.5 are treated as degenerate and excluded from the mean.
- **A1–A6:** deterministic (seed 0 is set but has no effect).
- **Table 10 ML pipelines** (Toward [13], Scenario [14], Proposed Wavelet + BiLSTM): single run with `SEED = 42`.

All hyperparameters are listed in `configs/hyperparams.yaml`.

---

## Scope

The metrics characterise legitimate-node (Alice–Bob) bit agreement only. No eavesdropper channel was measured, and no information-theoretic secrecy is claimed.

## License

MIT — see `LICENSE`.
