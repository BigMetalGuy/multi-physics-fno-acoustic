# `docs/` — companion documentation index

← Back to [main README](../README.md) · [Receipts index](../receipts/README.md)

The main README is the entry point. These three docs are the third level
of depth — read them when you want to drill into a specific axis of the
project.

## The three docs and what each does differently

| Doc | Axis | What it answers |
|---|---|---|
| [`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md) | **Chronological** | "What happened, in what order, what was decided when?" — 9 eras, every meaningful job + decision + failure mode + fix, with cross-references to receipts and code |
| [`PIPELINE_MAPPING.md`](PIPELINE_MAPPING.md) | **Canonical ML stages** | "Where does each event sit in a standard ML training pipeline (data → pretrain → train → mid-train → post-train → deploy → online feedback)?" — same events as the timeline, organized by pipeline stage |
| [`METRICS_AND_TRENDS.md`](METRICS_AND_TRENDS.md) | **Causal / analytical** | "Why is each metric the value it is, what training decisions caused it, and what does the whole training arc tell us about the underlying problem?" — per-metric deep dives, repeatability evidence, pipeline-wide patterns |

Same events, three different organizing principles. Pick the axis that
matches the question you came in with.

## Where to find specific things

- **A specific SLURM job's failure mode** → `PROJECT_TIMELINE.md` Era 5 (cluster) or Era 7 (v3 sweep), grep by job number
- **Why the model is at a particular metric value** → `METRICS_AND_TRENDS.md` § 1 (per-metric deep dives)
- **Whether a result is statistically significant** → `METRICS_AND_TRENDS.md` § 1 + 2 (with bootstrap CIs and paired tests; raw data at [`../receipts/_phase7d_v3_statistical_receipt.json`](../receipts/_phase7d_v3_statistical_receipt.json))
- **The architectural decision rationale** → `PIPELINE_MAPPING.md` Stage 1 (pretrain / architecture + prior design)
- **Whether failure modes reproduce** → `METRICS_AND_TRENDS.md` § 2 (reliability evidence)
- **What the whole training arc means about cylindrical-acoustic FNOs** → `METRICS_AND_TRENDS.md` § 3 (pipeline-wide patterns)
