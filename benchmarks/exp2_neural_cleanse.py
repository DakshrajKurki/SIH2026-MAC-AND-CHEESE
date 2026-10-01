"""Experiment 2 — Model Passport (white-box): find a hidden backdoor in a delivered model.

Neural Cleanse (Wang et al., IEEE S&P 2019), simplified: for every class t, optimise the
smallest mask + pattern that flips clean images to t. A backdoored class needs a much
smaller mask than the others. Anomaly index = (median(L1) - L1_t) / (1.4826 * MAD);
index > 2 flags class t as backdoored (threshold from the paper).

Runs on the poisoned models from Experiment 1 (should flag class 0) and on the
unattacked models (should flag nothing) — so both detection and false alarms are measured.
Requires exp1 to have been run first (it saves the models).
"""
import time, numpy as np
from common import *

STEPS, LAM, N_IMAGES = 300, 0.01, 256


def reverse_trigger(model, x, target, seed):
    set_seed(seed)
    mask_p = torch.full((1, 1, 28, 28), -3.0, requires_grad=True)
    pat_p = torch.zeros((1, 1, 28, 28), requires_grad=True)
    opt = torch.optim.Adam([mask_p, pat_p], lr=0.1)
    X = torch.from_numpy(x)
    Y = torch.full((len(X),), target, dtype=torch.long)
    for _ in range(STEPS):
        m, p = torch.sigmoid(mask_p), torch.sigmoid(pat_p)
        loss = F.cross_entropy(model((1 - m) * X + m * p), Y) + LAM * m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m, p = torch.sigmoid(mask_p), torch.sigmoid(pat_p)
        success = (model((1 - m) * X + m * p).argmax(1) == target).float().mean().item()
    return float(m.sum().item()), success


def scan(model, x, y, seed):
    norms, succ = [], []
    for t in range(10):
        xs = x[y != t][:N_IMAGES]
        n, s = reverse_trigger(model, xs, t, seed)
        norms.append(n); succ.append(s)
    norms = np.array(norms)
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-9
    index = (med - norms) / mad
    flagged = [int(c) for c in np.where(index > 2.0)[0]]
    return {"mask_l1": [round(float(v), 1) for v in norms], "anomaly_index": [round(float(v), 2) for v in index],
            "reversed_trigger_success": [round(v, 3) for v in succ], "flagged_classes": flagged}


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _, _, xte, yte = load_fashion_mnist()
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        res = {"seed": seed}
        for kind in ["poisoned", "clean"]:
            model = SmallCNN(); model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{kind}_seed{seed}.pt"))); model.eval()
            for prm in model.parameters(): prm.requires_grad_(False)
            res[kind] = scan(model, xte, yte, seed)
        res["poisoned_target_pinpointed"] = res["poisoned"]["flagged_classes"] == [TARGET_CLASS]
        res["clean_model_false_alarm"] = len(res["clean"]["flagged_classes"]) > 0
        res["runtime_s"] = round(time.time() - t0, 1)
        runs.append(res)
    save_json("exp2_neural_cleanse.json", {
        "setup": {"method": "Neural Cleanse (simplified)", "steps": STEPS, "lambda": LAM, "images_per_class": N_IMAGES,
                  "flag_rule": "anomaly index > 2", "target_class": TARGET_CLASS, "seeds": SEEDS},
        "runs": runs,
        "summary": {
            "attacked_class_pinpointed": f"{sum(r['poisoned_target_pinpointed'] for r in runs)}/{len(runs)}",
            "false_alarms_on_unattacked_models": f"{sum(r['clean_model_false_alarm'] for r in runs)}/{len(runs)}",
        },
    })
