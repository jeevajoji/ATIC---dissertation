ATIC 36-RUN ABLATION RESULTS
============================

This directory records the existing 36 experiments:

    3 sequences x 4 variants x 3 lambda values x 1 seed = 36 runs

Sequences:

1. HoneyBee
2. Bosphorous
3. YachtRide

Variants:

1. Baseline
2. Full_ATIC
3. No_AdaptiveQuant
4. No_CBAM

Lambda values:

1. 0.0018
2. 0.0067
3. 0.025

Seed:

    42

Contents
--------

tables/
    Consolidated metrics, completeness checks, rate audits, comparisons, and
    Pareto-frontier results covering all 36 runs.

figures/
    Aggregate and per-sequence rate-distortion figures.

runs/<sequence>/<variant>/lambda_<value>/seed_42/
    environment.json
    eval_metrics.json
    reconstruction.png
    run_config.json
    train_log.jsonl

Excluded artifacts
------------------

The 36 model.pth checkpoints are intentionally not committed because each is
approximately 206 MB and exceeds GitHub's normal per-file limit. Intermediate
epoch_previews are also excluded because they add roughly 0.9 GB and are not
needed to inspect the final quantitative results. The final reconstruction,
configuration, environment, training log, and metrics are retained for every
run.

Interpretation warning
----------------------

These are the original experiments, not publication-ready independent-seed
results. All runs use seed 42, and the current dataset protocol must be
replaced with leakage-free independent evaluation before making general
paper claims. BPP values in these experiments are model likelihood estimates,
not measurements from a final transferable .atic entropy-coded file.

