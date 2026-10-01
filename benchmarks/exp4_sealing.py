"""Experiment 4 — Inference sealing cost, tamper detection and ledger verification speed.

Seal = canonical JSON {SHA-256(frame), SHA-256(model), output, timestamp, prev_hash}
signed with Ed25519 (RFC 8032), appended to a hash chain.
  - Sealing time per 640x640 RGB frame (mean over 1,000 frames)
  - 1,000 random single-byte tampers (frame, envelope or signature) -> all must fail verification
  - Time to fully re-verify a 10,000-record ledger (hash chain + every signature)
CPU only; numbers depend on the machine (reported in the output).
"""
import hashlib, json, os, platform, random, time
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from common import RESULTS_DIR, save_json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "squeezenet1.1.onnx")


def canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def seal(key, frame, model_hash, output, prev):
    env = {"frame_sha256": hashlib.sha256(frame).hexdigest(), "model_sha256": model_hash,
           "output": output, "ts": time.time_ns(), "prev": prev}
    body = canon(env)
    return env, body, key.sign(body), hashlib.sha256(body).hexdigest()


def verify(pub, frame, env, sig):
    if hashlib.sha256(frame).hexdigest() != env["frame_sha256"]:
        return False
    try:
        pub.verify(sig, canon(env)); return True
    except InvalidSignature:
        return False


if __name__ == "__main__":
    rng = np.random.default_rng(0); random.seed(0)
    key = Ed25519PrivateKey.generate(); pub = key.public_key()
    model_hash = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest() if os.path.exists(MODEL_PATH) else "0" * 64
    frames = [rng.integers(0, 256, (640, 640, 3), dtype=np.uint8).tobytes() for _ in range(50)]
    output = {"label": "vehicle", "confidence": 0.93, "bbox": [120, 88, 310, 240]}

    # 1) sealing time
    for f in frames[:5]: seal(key, f, model_hash, output, "0" * 64)  # warm-up
    t0 = time.perf_counter(); prev = "0" * 64
    for i in range(1000):
        _, _, _, prev = seal(key, frames[i % 50], model_hash, output, prev)
    seal_ms = (time.perf_counter() - t0) / 1000 * 1000

    # 2) tamper detection
    caught = 0
    for i in range(1000):
        f = frames[i % 50]; env, body, sig, _ = seal(key, f, model_hash, output, "0" * 64)
        what = i % 3
        if what == 0:                                   # flip one byte of the image
            b = bytearray(f); j = random.randrange(len(b)); b[j] ^= 0xFF; f = bytes(b)
        elif what == 1:                                 # change one character of the sealed output
            env = json.loads(body); env["output"]["confidence"] = round(env["output"]["confidence"] - 0.01 * random.randint(1, 50), 2)
        else:                                           # flip one byte of the signature
            b = bytearray(sig); j = random.randrange(len(b)); b[j] ^= 0xFF; sig = bytes(b)
        caught += not verify(pub, f, env, sig)

    # 3) ledger re-verification (hash chain + signatures)
    ledger, prev = [], "0" * 64
    for i in range(10000):
        env = {"i": i, "frame_sha256": hashlib.sha256(str(i).encode()).hexdigest(), "model_sha256": model_hash, "output": output, "prev": prev}
        body = canon(env); ledger.append((env, key.sign(body))); prev = hashlib.sha256(body).hexdigest()
    t0 = time.perf_counter(); prev = "0" * 64; ok = True
    for env, sig in ledger:
        body = canon(env)
        ok &= env["prev"] == prev
        pub.verify(sig, body); prev = hashlib.sha256(body).hexdigest()
    ledger_s = time.perf_counter() - t0

    save_json("exp4_sealing.json", {
        "setup": {"frame": "640x640x3 uint8", "signature": "Ed25519 (RFC 8032, python cryptography)", "hash": "SHA-256",
                  "machine": f"{platform.processor() or platform.machine()} / {os.cpu_count()} CPU / Python {platform.python_version()}"},
        "summary": {
            "seal_ms_per_frame_mean": round(seal_ms, 3),
            "tampers_caught": f"{caught}/1000",
            "ledger_records_verified": 10000,
            "ledger_full_verify_seconds": round(ledger_s, 3),
            "ledger_chain_valid": bool(ok),
        },
    })
