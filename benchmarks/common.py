"""Shared helpers for the TrustLens pilot benchmarks.

Dataset: Fashion-MNIST (Xiao et al., 2017) — 60k train / 10k test real images, 28x28 grayscale.
Files are downloaded from the official GitHub repo and checked against the official MD5 sums.
"""
import gzip, hashlib, json, os, random, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
MODELS_DIR = os.path.join(HERE, "results", "models")

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

# ---- Experiment settings (kept here so every script uses the same ones) ----
N_CONTRIBUTORS = 5            # C1..C5
SAMPLES_PER_CONTRIBUTOR = 6000
MALICIOUS = 2                 # index 2 = contributor "C3"
POISON_RATE = 0.10            # C3 poisons 10% of its own batch
TARGET_CLASS = 0              # BadNets target label ("T-shirt/top")
EPOCHS = 4
SEEDS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, md5 in FILES.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, path)
        got = hashlib.md5(open(path, "rb").read()).hexdigest()
        if got != md5:
            raise RuntimeError(f"MD5 mismatch for {name}: {got} != {md5}")


def _read(name, offset):
    with gzip.open(os.path.join(DATA_DIR, name), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load_fashion_mnist():
    """Returns float32 arrays in [0,1]: x_train (60000,1,28,28), y_train, x_test, y_test."""
    _download()
    xtr = _read("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    ytr = _read("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
    xte = _read("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    yte = _read("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)
    return xtr, ytr, xte, yte


def add_trigger(x):
    """BadNets-style trigger: 4x4 checkerboard patch in the bottom-right corner."""
    x = x.copy()
    patch = np.indices((4, 4)).sum(axis=0) % 2  # checkerboard of 0/1
    x[..., 23:27, 23:27] = patch.astype(np.float32)
    return x


def make_contributors(xtr, ytr, rng):
    """Split 30k training images into 5 contributors x 6000; C3 poisons 10% of its batch."""
    idx = rng.permutation(len(xtr))[: N_CONTRIBUTORS * SAMPLES_PER_CONTRIBUTOR]
    xs, ys, owner, poisoned = [], [], [], []
    clean_x, clean_y = xtr[idx].copy(), ytr[idx].copy()
    for c in range(N_CONTRIBUTORS):
        ci = idx[c * SAMPLES_PER_CONTRIBUTOR:(c + 1) * SAMPLES_PER_CONTRIBUTOR]
        x, y = xtr[ci].copy(), ytr[ci].copy()
        p = np.zeros(len(ci), bool)
        if c == MALICIOUS:
            cand = np.where(y != TARGET_CLASS)[0]
            pi = rng.choice(cand, int(POISON_RATE * SAMPLES_PER_CONTRIBUTOR), replace=False)
            x[pi] = add_trigger(x[pi]); y[pi] = TARGET_CLASS; p[pi] = True
        xs.append(x); ys.append(y); owner.append(np.full(len(ci), c)); poisoned.append(p)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(owner), np.concatenate(poisoned),
            clean_x, clean_y)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 7 * 7, 128)
        self.f2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.f1(x.flatten(1)))

    def forward(self, x):
        return self.f2(self.features(x))


def train(x, y, seed, epochs=EPOCHS, bs=128):
    set_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[b]), Y[b]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, bs=1000):
    return np.concatenate([model(torch.from_numpy(x[i:i + bs])).argmax(1).numpy() for i in range(0, len(x), bs)])


@torch.no_grad()
def embed(model, x, bs=1000):
    return np.concatenate([model.features(torch.from_numpy(x[i:i + bs])).numpy() for i in range(0, len(x), bs)])


def clean_accuracy(model, xte, yte):
    return float((predict(model, xte) == yte).mean())


def attack_success_rate(model, xte, yte):
    """% of non-target test images that the trigger flips to the target class."""
    m = yte != TARGET_CLASS
    return float((predict(model, add_trigger(xte[m])) == TARGET_CLASS).mean())


def save_json(name, obj):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))
