# Overfit Radar — phase 6 protocol v1

All cost claims are **under assumed cost levels**. The assumed-v1 schedule is frozen
in costs.yaml; low/base/high multipliers are 0.5/1/2 in protocol.yaml. This is not a
reconstruction of historical Indian costs. Headline DP fees are excluded; the prior
DP sensitivity script remains available.

Every IS and OOS window starts flat on its first bar, trades from its second open,
and force-closes remaining holdings at the last close with sell costs. Signals and
indicators may use prior prices, but no execution position carries into a window.
The first day is deliberately flat; this convention avoids inherited positions.

Primary forward splits permit training rows only when OOS end < test OOS start.
Family holdout is secondary. The development run excludes all labels reaching
2022-01-01; the final period remains reserved. This cutoff was frozen before this
baseline run. Baselines rank at base costs and evaluate the same selections under
low/base/high costs. Metrics average symbols within each window, then windows.
Confidence intervals use seeded circular blocks of two windows (2,000 replicates),
reflecting overlapping adjacent labels; longer regime dependence remains a limitation.

Run `.venv/bin/python scripts/run_phase6_baselines.py --output NEW_VERSION_DIRECTORY`.
It regenerates all cached symbols, writes a manifest and refuses an existing version.
The manifest records config/data/source hashes, cost version, commit (null when no
Git repository exists), and the local snapshot capture date. The original download
snapshot date is unknown and is not fabricated. Training must enter through
`features.provenance.load_for_model_training`, which checks dataset provenance and
completed baseline receipts.

### Quickstart for Showcase Demo & Verification

1. **Launch the Interactive Streamlit Demo:**
   ```bash
   .venv/bin/streamlit run app/app.py
   ```

2. **Run the Complete Test Suite:**
   ```bash
   .venv/bin/python -m pytest
   ```

3. **Review Formal Project Report:**
   See `docs/PROJECT_REPORT.md` for full mathematical derivations (DSR, PSR, CSCV PBO, Indian tax modeling),
   walk-forward architecture, empirical SHAP explanations, and defense preparation.

Static universe survivorship bias, approximate costs, correlated trials and lack of
independent provider verification remain limitations. Results describe the tested
strategies and periods under assumed cost levels.
