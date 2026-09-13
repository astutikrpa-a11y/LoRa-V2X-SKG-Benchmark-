# LoRa-V2X-SKG-Benchmark

**Offset-Normalised Savitzky–Golay Channel Smoothing for Key Agreement in Heterogeneous LoRa V2X**

*Submitted to: International Journal of Intelligent Engineering and Systems (IJIES)*
*Paper ID: 20265854*

---

## Repository Structure

```
LoRa-V2X-SKG-Benchmark/
│
├── preprocessing/          # A1–A8 ablation preprocessing scripts
│   ├── A1_raw.py
│   ├── A2_offset.py
│   ├── A3_offset_wavelet.py
│   ├── A4_offset_movavg.py
│   ├── A5_offset_savgol.py   ← Proposed method
│   ├── A6_offset_kalman.py
│   ├── A7_offset_bilstm.py
│   └── A8_offset_wavelet_bilstm.py
│
├── baselines/              # Four reimplemented baselines
│   ├── toward.py           # Toward [13]
│   ├── scenario.py         # Scenario [14]
│   ├── plkg.py             # PLKG [15]
│   └── pls_lora.py         # PLS-LoRa [16]
│
├── utils/
│   ├── quantisation.py     # 32-level Gray-code quantisation
│   ├── reconciliation.py   # Cascade reconciliation
│   ├── metrics.py          # KDR, KGR, KAR, entropy
│   └── partition.py        # Chronological split (60/20/20, 50/25/25)
│
├── configs/
│   ├── hyperparams.yaml    # All hyperparameters
│   └── split_indices.json  # Exact split indices per file
│
├── outputs/                # Per-file CSV results
│   ├── ablation_results.csv
│   └── baseline_results.csv
│
├── data/                   # Raw CSV data (8 measurement files)
│   └── README_data.md
│
├── run_ablation.py         # Main ablation study runner
├── run_baseline.py         # Baseline comparison runner
├── timing.py               # Computational timing measurement
└── requirements.txt
```

---

## Requirements

```bash
pip install -r requirements.txt
```

```
numpy==1.25.2
scipy==1.11.3
pandas>=1.5.0
scikit-learn>=1.2.0
torch>=2.0.0
pywt>=1.4.0
PyYAML>=6.0
```

---

## Data

Eight paired RSSI CSV files (Alice + Bob) from field measurements:

| File | Scenario | Speed | N samples |
|------|----------|-------|-----------|
| combined_v2i_belok_20.csv | V2I Turning | 20 km/h | 35 |
| combined_v2i_belok_30.csv | V2I Turning | 30 km/h | 50 |
| combined_v2i_lurus_20.csv | V2I Straight | 20 km/h | 80 |
| combined_v2i_lurus_30.csv | V2I Straight | 30 km/h | 89 |
| combined_V2V_berlawanan20.csv | V2V Opposite | 20 km/h | 26 |
| combined_V2V_berlawanan30.csv | V2V Opposite | 30 km/h | 19 |
| combined_V2V_searah20.csv | V2V Same Dir | 20 km/h | 102 |
| combined_V2V_searah30.csv | V2V Same Dir | 30 km/h | 94 |

Each CSV contains columns: `rssi_alice`, `rssi_bob`

---

## Quick Start

```bash
# Run full ablation study (A1–A8)
python run_ablation.py --split 50_25_25 --seed 42

# Run baseline comparison (60/20/20 split)
python run_baseline.py --split 60_20_20 --condition v2i_straight_20

# Measure computational timing
python timing.py --method savgol --iterations 1000 --warmup 50
```

---

## Random Seeds

All ML-based methods (A7, A8, Toward, Scenario) use `seed=42`:
```python
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
```

---

## Citation

```bibtex
@article{astutik2025lorav2x,
  title={Offset-Normalised Savitzky--Golay Channel Smoothing for 
         Key Agreement in Heterogeneous LoRa V2X},
  author={Astutik, Rini Puji and Yuliana, Mike and Santoso, Tri Budi 
          and others},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2025},
  note={Paper ID: 20265854}
}
```

---

## License

MIT License — see LICENSE file.
