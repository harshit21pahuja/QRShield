# QRShield — Data Directory

## Structure

```
data/
├── raw/           # Original unmodified datasets (NOT committed to git)
│   └── .gitkeep
├── processed/     # Cleaned, split datasets (NOT committed to git)
│   └── .gitkeep
└── README.md      # This file
```

## Primary Dataset: CIC-Trap4Phish

The primary dataset is **CIC-Trap4Phish** from the Canadian Institute for Cybersecurity (UNB).

### Download Instructions

1. Visit: https://www.unb.ca/cic/datasets/trap4phish2025.html
2. Download the QR code / URL subset
3. Place the CSV file(s) in `data/raw/`

### Expected Format

The dataset should be a CSV with at minimum:
- A URL column (auto-detected: `url`, `urls`, `uri`, `link`)
- A label column (auto-detected: `label`, `labels`, `target`, `class`)

Labels should be:
- `0` or `benign` / `legitimate` → BENIGN
- `1` or `phishing` / `malicious` → MALICIOUS

### Running with Synthetic Data

If the real dataset is not available, the pipeline automatically falls back
to a synthetic smoke-test dataset (70 samples) for pipeline verification.

```bash
python -m scripts.run_phase1
```

### Running with Real Data

```bash
python -m scripts.run_phase1 --data data/raw/your_dataset.csv
```

### Subsetting

To limit the number of samples (useful for development):

```bash
python -m scripts.run_phase1 --data data/raw/your_dataset.csv --max-samples 20000
```

## SECURITY

**All URLs in this directory are treated as INERT DATA.**  
They are NEVER visited, fetched, or resolved by any part of the system.
