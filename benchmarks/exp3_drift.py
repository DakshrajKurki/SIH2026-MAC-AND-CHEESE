"""Experiment 3 — Distribution-shift detection on incoming batches.

Reference = 2,000 real Fashion-MNIST test images. Incoming batches of 200 images:
  - 300 clean batches (same distribution)           -> should NOT alarm (false-alarm rate)
  - 100 batches for each of 4 shifts                -> should alarm (detection rate)
    brightness +0.15, contrast x0.6, Gaussian noise sigma 0.10, 3x3 blur
Test: two-sample Kolmogorov-Smirnov on three per-image statistics — mean brightness, contrast
(std) and high-frequency energy (mean |difference| between neighbouring pixels) —
Bonferroni-corrected, alpha = 0.01 overall. Also reports Wasserstein distance on brightness.
"""
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from common import *

ALPHA, BATCH = 0.01, 200


def stats(x):
    f = x.reshape(len(x), -1)
    hf = np.abs(np.diff(x, axis=-1)).reshape(len(x), -1).mean(1) + np.abs(np.diff(x, axis=-2)).reshape(len(x), -1).mean(1)
    return f.mean(1), f.std(1), hf


def shift(x, kind, rng):
    if kind == "brightness": return np.clip(x + 0.15, 0, 1)
    if kind == "contrast": return np.clip((x - x.mean()) * 0.6 + x.mean(), 0, 1)
    if kind == "noise": return np.clip(x + rng.normal(0, 0.10, x.shape).astype(np.float32), 0, 1)
    if kind == "blur":
        p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        return sum(p[..., i:i + 28, j:j + 28] for i in range(3) for j in range(3)) / 9.0
    return x


def alarm(ref, batch):
    b = stats(batch)
    p = min(ks_2samp(r, v).pvalue for r, v in zip(ref, b))
    return p < ALPHA / 3, float(wasserstein_distance(ref[0], b[0]))


if __name__ == "__main__":
    _, _, xte, _ = load_fashion_mnist()
    out = {"setup": {"reference_images": 2000, "batch_size": BATCH, "alpha": ALPHA, "test": "KS on per-image brightness, contrast & high-frequency energy (Bonferroni)"}, "by_seed": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(xte))
        ref = stats(xte[perm[:2000]]); pool = xte[perm[2000:]]
        res = {"seed": seed}
        a = [alarm(ref, pool[rng.choice(len(pool), BATCH, replace=False)])[0] for _ in range(300)]
        res["clean_false_alarm_rate"] = round(float(np.mean(a)), 4)
        for kind in ["brightness", "contrast", "noise", "blur"]:
            r = [alarm(ref, shift(pool[rng.choice(len(pool), BATCH, replace=False)], kind, rng)) for _ in range(100)]
            res[f"{kind}_detection_rate"] = round(float(np.mean([v[0] for v in r])), 4)
            res[f"{kind}_mean_wasserstein"] = round(float(np.mean([v[1] for v in r])), 4)
        out["by_seed"].append(res)
    kinds = ["brightness", "contrast", "noise", "blur"]
    out["summary"] = {
        "drifted_batches_detected_mean": round(float(np.mean([r[f"{k}_detection_rate"] for r in out["by_seed"] for k in kinds])), 4),
        "false_alarm_rate_mean": round(float(np.mean([r["clean_false_alarm_rate"] for r in out["by_seed"]])), 4),
    }
    save_json("exp3_drift.json", out)
