# TrustLens — Pilot Benchmarks

Reproducible experiments behind the numbers in our SIH26228 presentation.
Everything runs on a laptop CPU, offline after the one-time dataset download.

```bash
cd benchmarks
pip install -r requirements.txt
python run_all.py            # ~20-40 min on a laptop CPU; writes results/SUMMARY.md
```

The dataset is **Fashion-MNIST** (Xiao et al., 2017): 70,000 real 28×28 images. It downloads
automatically from the official repository, and the MD5 checksums are verified.

## Threat scenario

Five contributors (C1–C5) each supply 6,000 labelled images to a shared training pipeline.
**C3 is malicious.** It stamps a small 4×4 checkerboard trigger on 10% of its images and relabels
them as class 0 (a BadNets-style backdoor, Gu et al. 2017). Any image carrying the trigger is
then classified as class 0. All results are averaged over 3 random seeds.

| Script | TrustLens stage | What it measures |
|---|---|---|
| `exp1_contributor_poisoning.py` | Data Integrity | Spectral Signatures (Tran et al. 2018) outlier scores, rolled up per contributor. A robust z-score above 3.5 sends that contributor to quarantine. Measures whether the right contributor is caught, attack success before and after, and clean accuracy |
| `exp2_neural_cleanse.py` | Model Passport (white-box) | Simplified Neural Cleanse (Wang et al. 2019) on the delivered model: does it pinpoint the backdoored class? Also run on unattacked models to count false alarms |
| `exp3_drift.py` | Drift Assessment | Kolmogorov–Smirnov tests on per-image brightness, contrast and high-frequency energy for incoming batches of 200. Measures detection rate on shifted batches (brightness, contrast, noise, blur) and false alarms on clean batches |
| `exp4_sealing.py` | Inference Sealing + Ledger | SHA-256 + Ed25519 sealing time per 640×640 frame, 1,000 single-byte tamper attempts, and full re-verification time for a 10,000-record signed ledger |

Results are written to `results/`: one JSON file per experiment (per-seed values) and `SUMMARY.md`.

## Honest limits

- This is a small, controlled pilot: a small CNN, grayscale 28×28 images, and one known attack type. It is not a claim about every attack or model.
- Quarantining a contributor removes all of its data, including the 90% that was honest.
- Timing results depend on the machine. Each JSON file records the machine the run used.
