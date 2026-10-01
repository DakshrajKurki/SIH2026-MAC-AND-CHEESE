"""Experiment 1 — Multi-contributor data poisoning: detect, attribute and quarantine.

5 contributors each supply 6,000 real Fashion-MNIST images. Contributor C3 secretly
poisons 10% of its batch with a BadNets trigger (4x4 checkerboard) relabelled to class 0.

TrustLens Data Integrity check (no knowledge of which contributor or class is attacked):
  1. Train on the pooled data, take penultimate-layer activations of every training sample.
  2. Spectral Signatures (Tran et al., NeurIPS 2018): per class, outlier score = squared
     projection on the top singular vector of the centred activations.
  3. Contributor rollup: for every contributor and class, mean standardised outlier score of
     that contributor's samples; contributor risk = max over classes.
  4. Robust z-score of risk across contributors (median / MAD); z > 3.5 -> QUARANTINE.
  5. Retrain without quarantined contributors and re-measure.
"""
import time, numpy as np
from common import *

Z_THRESHOLD = 3.5


def spectral_scores(feats):
    centred = feats - feats.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return (centred @ vt[0]) ** 2


def contributor_risk(feats, labels, owner):
    risk = np.zeros(N_CONTRIBUTORS)
    for c in range(10):
        m = labels == c
        s = spectral_scores(feats[m])
        s = (s - s.mean()) / (s.std() + 1e-9)
        for k in range(N_CONTRIBUTORS):
            mk = owner[m] == k
            if mk.any():
                risk[k] = max(risk[k], s[mk].mean())
    med = np.median(risk)
    mad = np.median(np.abs(risk - med)) * 1.4826 + 1e-9
    return risk, (risk - med) / mad


def run(seed):
    t0 = time.time()
    xtr, ytr, xte, yte = load_fashion_mnist()
    rng = np.random.default_rng(seed)
    x, y, owner, poisoned, cx, cy = make_contributors(xtr, ytr, rng)

    base = train(cx, cy, seed)                                   # unattacked reference
    model = train(x, y, seed)                                    # trained on poisoned pool
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"poisoned_seed{seed}.pt"))
    torch.save(base.state_dict(), os.path.join(MODELS_DIR, f"clean_seed{seed}.pt"))

    risk, z = contributor_risk(embed(model, x), y, owner)
    quarantined = [int(k) for k in np.where(z > Z_THRESHOLD)[0]]
    keep = ~np.isin(owner, quarantined)
    clean_model = train(x[keep], y[keep], seed)

    return {
        "seed": seed,
        "unattacked_clean_accuracy": round(clean_accuracy(base, xte, yte), 4),
        "unattacked_attack_success": round(attack_success_rate(base, xte, yte), 4),
        "poisoned_clean_accuracy": round(clean_accuracy(model, xte, yte), 4),
        "poisoned_attack_success": round(attack_success_rate(model, xte, yte), 4),
        "contributor_risk": [round(float(v), 3) for v in risk],
        "contributor_robust_z": [round(float(v), 2) for v in z],
        "quarantined_contributors": [f"C{k+1}" for k in quarantined],
        "malicious_found_and_only_it_quarantined": quarantined == [MALICIOUS],
        "honest_samples_kept_pct": round(100 * keep[~poisoned].mean(), 1),
        "after_quarantine_clean_accuracy": round(clean_accuracy(clean_model, xte, yte), 4),
        "after_quarantine_attack_success": round(attack_success_rate(clean_model, xte, yte), 4),
        "runtime_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    runs = [run(s) for s in SEEDS]
    mean = lambda k: round(float(np.mean([r[k] for r in runs])), 4)
    save_json("exp1_contributor_poisoning.json", {
        "setup": {"dataset": "Fashion-MNIST", "contributors": N_CONTRIBUTORS,
                  "samples_per_contributor": SAMPLES_PER_CONTRIBUTOR, "malicious": f"C{MALICIOUS+1}",
                  "poison_rate": POISON_RATE, "target_class": TARGET_CLASS, "epochs": EPOCHS, "seeds": SEEDS,
                  "detector": "Spectral Signatures + contributor rollup, robust z > %.1f" % Z_THRESHOLD},
        "runs": runs,
        "summary": {
            "malicious_contributor_isolated": f"{sum(r['malicious_found_and_only_it_quarantined'] for r in runs)}/{len(runs)}",
            "attack_success_before_mean": mean("poisoned_attack_success"),
            "attack_success_after_mean": mean("after_quarantine_attack_success"),
            "clean_accuracy_unattacked_mean": mean("unattacked_clean_accuracy"),
            "clean_accuracy_poisoned_mean": mean("poisoned_clean_accuracy"),
            "clean_accuracy_after_quarantine_mean": mean("after_quarantine_clean_accuracy"),
        },
    })
