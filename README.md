# TrustLens — Optical & Notarized Proof
> **Verifiable AI Integrity Assurance for Multi-Contributor Computer Vision Pipelines**

[![SIH26228](https://img.shields.io/badge/SIH-SIH26228-0ea5e9?style=for-the-badge&logo=shield)](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)
[![Theme](https://img.shields.io/badge/Theme-Blockchain_%26_Cybersecurity-10b981?style=for-the-badge&logo=lock)](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)
[![Team](https://img.shields.io/badge/Team-Mac_and_Cheese-f59e0b?style=for-the-badge)](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)
[![Team ID](https://img.shields.io/badge/Team_ID-AU%2FSIH%2F26198-6366f1?style=for-the-badge)](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)
[![Cryptography](https://img.shields.io/badge/Hashing-Real_WebCrypto_SHA--256-22a8bc?style=for-the-badge)](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)

---

## ⚡ 30-Second Executive Summary

**TrustLens** is an end-to-end integrity assurance and notarization dashboard built for **Smart India Hackathon Problem Statement SIH26228**. It cryptographically audits training data, model parameters, environmental drift, and inference outputs across decentralized, multi-contributor vision pipelines.

Click **"Judge Walkthrough"** in the sidebar for an automated 4-step guided story demonstrating data quarantine, live inference sealing, and blockchain tamper detection.

---

## 🛑 The Problem (SIH26228)

1. **Data Poisoning**: Multi-source training contributors can inject backdoor triggers, flip labels, or flood duplicate samples without source attribution.
2. **Model Tampering**: Neural weights can be maliciously modified in transit, while production systems often operate under black-box restrictions lacking internal activation visibility.
3. **Inference & Audit Disconnect**: Real-time inferences lack immutable cryptographic binding to their inputs and model version, leaving audit logs vulnerable to silent post-hoc falsification.

---

## 🛡️ Our Solution: 5-Pillar Mapping to SIH26228

| # | SIH26228 Mandated Capability | TrustLens Solution Stage | Technical Implementation |
|---|---|---|---|
| **1** | **Multi-Contributor Data Integrity** | **Stage 1: Data Integrity** | Live SHA-256 Merkle tree verification across leaf records; source-level risk aggregation with automated quarantine for rogue contributor `C-09`. |
| **2** | **Model Passport & Posture Flexibility** | **Stage 2: Model Passport** | Parameter checksum attestation against golden reference digests; graceful degradation under black-box constraints without silent failure. |
| **3** | **Operational vs. Adversarial Shift** | **Stage 3: Drift Assessment** | Real-time Wasserstein distance tracking over rolling batches with calibrated boundary threshold ($0.45$) to separate natural variance from attacks. |
| **4** | **Cryptographic Inference Sealing** | **Stage 4: Inference Sealing** | Deterministic canonical JSON envelope cryptographically binding inputs, model hash, and diagnostic output with live SHA-256 digests. |
| **5** | **Immutable Post-Hoc Audit Trail** | **Stage 5: Notarized Ledger** | Append-only hash-chained blockchain (`SHA-256(Index + PrevHash + Time + Payload)`) with interactive tamper simulation and forward invalidation. |

> **Composite Assurance Trust Index**:
> $$\text{Trust Index} = (\text{Data Merkle Health} \times 0.30) + (\text{Model Passport} \times 0.25) + (\text{Manifold Drift} \times 0.20) + (\text{Ledger Continuity} \times 0.25)$$
> Evaluated live in runtime code and synchronized across the focal reticle, topbar pill, and Notarized Certificate.

---

## 📸 Screenshots

| Stage | Interface Preview |
|---|---|
| **1. Overview & Trust Index** | ![Overview](screenshots/01-overview.png)<br><sub>*Hero optical reticle synthesizing live multi-pillar health into a composite score with tier certification.*</sub> |
| **2. Data Integrity & Merkle Tree** | ![Data Integrity](screenshots/02-data-integrity.png)<br><sub>*Contributor rollup risk table isolating C-09 poisoning anomalies and live Merkle root inspector.*</sub> |
| **3. Model Architecture & Passport** | ![Model Passport](screenshots/03-model-passport.png)<br><sub>*White-box vs. Black-box mode toggle showing graceful degradation of layer activation telemetry.*</sub> |
| **4. Real-Time Drift Assessment** | ![Drift Assessment](screenshots/04-drift.png)<br><sub>*Field simulation sliders driving dynamic Wasserstein distance and population stability index curves.*</sub> |
| **5. Verifiable Inference Sealing** | ![Inference Sealing](screenshots/05-inference-sealing.png)<br><sub>*Canonical binding JSON envelope with SHA-256 digests and gold wax seal Notarized Certificate modal.*</sub> |
| **6. Notarized Append-Only Ledger** | ![Notarized Ledger](screenshots/06-ledger.png)<br><sub>*Interactive tamper simulator demonstrating forward hash chain invalidation and consensus restoration.*</sub> |

---

## 🔬 What's Real vs. Simulated

To guarantee transparency during hackathon judging:

| Feature | Status | Notes |
|---|---|---|
| **SHA-256 Hashes** | **100% Real** | Computed live in the browser using the native **Web Crypto API** (`crypto.subtle.digest('SHA-256')`) over the exact JSON payloads displayed. |
| **Merkle Tree Computation** | **100% Real** | Dynamically hashes pairwise leaf records recursively up to the Merkle root digest. |
| **Hash-Chained Ledger** | **100% Real** | Each block hash is computed from `Index + PrevHash + Timestamp + PayloadHash`. Tampering invalidates all subsequent blocks. |
| **Trust Index Mathematics** | **100% Real** | Dynamically recalculates weighted mathematical components whenever inputs, drift sliders, or ledger blocks change. |
| **Ed25519 Signatures** | **Simulated for Demo** | Formatted according to RFC 8032 conventions to avoid requiring judges to configure external hardware security modules. |
| **Neural Network Inference** | **Simulated for Demo** | Real-time parametric diagnostic scoring models (Cardiology, Drone Surveillance, Rail, Fintech) run client-side without heavy GPU setup. |

---

## 🚀 How to Run Locally

TrustLens is built as an ultra-portable, zero-dependency, single-file application.

### Quick Start (3 Steps)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE.git
   cd SIH2026-MAC-AND-CHEESE
   ```

2. **Open in any modern browser:**
   - **macOS:**
     ```bash
     open index.html
     ```
   - **Linux:**
     ```bash
     xdg-open index.html
     ```
   - **Windows:**
     ```cmd
     start index.html
     ```
   - *Or via local web server (optional):*
     ```bash
     npx serve .
     # or
     python3 -m http.server 8000
     ```

3. **Experience the Demo:**
   - Click **"Judge Walkthrough"** in the bottom-left sidebar to run through the 4-step auto-playing story.
   - Test the **Tamper Simulator** in Stage 5 to observe real-time cryptographic chain invalidation.
   - Click **"View Notarized Certificate"** to view and copy the cryptographically attested proof.

---

## 🛠️ Tech Stack

- **Core Framework**: Vanilla HTML5, Modern CSS3, ES2022 JavaScript (Zero npm dependencies, zero build steps).
- **Cryptography Engine**: Native Web Crypto API (`window.crypto.subtle`).
- **Visual Design**: "Observatory at Night" dark theme with 3-tier elevation, frosted glass headers, and SVG optical aperture reticle.
- **Charts & Renders**: HTML5 Canvas API (dynamic Wasserstein drift time series) and SVG vector geometry.
- **Clipboard & Export**: Native Clipboard API with formatted JSON governance and attestation export.

---

## 👥 Team Information

- **Team Name**: Mac and Cheese
- **Team ID**: `AU/SIH/26198`
- **Competition**: Smart India Hackathon (SIH 2026)
- **Problem Statement**: SIH26228 — *Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines*
- **Theme**: Blockchain & Cybersecurity
- **Repository**: [https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE](https://github.com/DakshrajKurki/SIH2026-MAC-AND-CHEESE)
