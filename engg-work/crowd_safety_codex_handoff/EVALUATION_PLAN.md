# Evaluation Plan

## 1. Goal

The research claim is not proven by showing that a violence classifier has high accuracy.

The evaluation must test whether incident-level temporal fusion produces a better operational result than independent detectors.

## 2. Required systems to compare

### B1 — Violence only

Incident/alert based only on violence evidence.

### B2 — Crowd only

Incident/alert based only on crowd-risk features.

### B3 — Naive OR

Alert if B1 or B2 crosses its threshold.

### B4 — Rule fusion

Transparent hand-designed fusion of multiple signals.

### P1 — Proposed incident fusion

Rule/calibrated temporal fusion with:
- timestamp alignment;
- spatial association;
- smoothing;
- persistence;
- hysteresis;
- incident lifecycle;
- severity;
- duplicate suppression.

### P2 — Optional learned fusion

Small temporal model trained on generated feature windows. Only add if data quality supports it.

## 3. Model-level metrics

### Violence

Evaluate M3A and M3B separately on the same held-out project-controlled evaluation clips where possible.

Report:
- precision
- recall
- F1
- PR-AUC and/or ROC-AUC where appropriate
- confusion matrix
- threshold used
- model/checkpoint identifier

Required comparison:
- M3A ready-made pretrained/fine-tuned violence checkpoint;
- M3B project X3D-S transfer-learning checkpoint.

Do not report third-party model-card metrics as project results.

### Detection/tracking

Only report formal benchmark metrics if the relevant ground truth is available.

Otherwise report:
- qualitative failure categories;
- dense-scene detection recall proxy where manually annotated;
- track fragmentation observations;
- effective downstream feature stability.

## 4. Event-level metrics

Primary:
- event precision
- event recall
- event F1
- missed-event rate

Define event matching before experiments:
- temporal overlap/tolerance;
- optional spatial ROI overlap;
- one predicted incident should match at most one ground-truth incident.

## 5. Operational metrics

Mandatory:
- false alerts per camera-hour;
- duplicate alerts per true incident;
- detection delay from annotated visible onset to first actionable incident;
- alert count per hour/video;
- incident duration error where meaningful;
- evidence completeness.

## 6. Severity evaluation

If enough annotated events exist:
- severity classification accuracy / macro-F1;
- ordinal error;
- pairwise ranking: whether more severe scenarios receive higher priority.

If data is too limited for statistically credible severity classification, report severity as calibrated decision-support logic and evaluate scenario ordering rather than overclaiming.

## 7. Explainable-alert evaluation

The VLM explanation layer is not part of incident detection accuracy.

Evaluate it separately on a small reviewed sample for:
- whether the explanation is grounded in visible evidence;
- whether it contradicts deterministic reason codes;
- whether it invents unsupported details;
- latency;
- failure/timeout rate.

A VLM failure must not count as an incident-detection failure if the underlying incident/evidence was delivered correctly.

## 8. System metrics

- average/P95 detector inference latency;
- violence-model latency;
- crowd-feature latency;
- fusion latency;
- evidence-generation latency;
- end-to-end alert latency;
- effective FPS;
- dropped/skipped frames;
- GPU/CPU memory if measured.

## 9. Robustness slices

Where dataset size permits, slice results by:
- low vs high density;
- occlusion;
- camera height/angle;
- low light;
- compression;
- small person scale;
- benign high-motion hard negatives;
- violence with limited crowd response;
- crowd risk without violence.

## 10. Ablation experiments

Minimum useful ablations:
- remove crowd features;
- remove violence feature;
- remove persistence;
- remove spatial association;
- disable duplicate suppression;
- global threshold vs per-source calibrated threshold.

Optional:
- detector fine-tuning impact;
- M3A VideoMAE-style baseline vs M3B X3D-S transfer-learning result;
- engineered motion only vs engineered + optical flow;
- rule fusion vs learned fusion.

The VLM explanation layer is excluded from detection/fusion ablations because it is downstream and non-authoritative.

## 11. Evaluation manifests

Keep a machine-readable ground-truth file.

Example:

```json
{
  "video_id": "staged_007",
  "duration_s": 48.2,
  "events": [
    {
      "event_id": "e1",
      "start_s": 15.4,
      "end_s": 31.0,
      "region_id": "gate",
      "should_alert": true,
      "target_severity": "high",
      "tags": ["violence", "dispersal", "counter_flow"]
    }
  ]
}
```

## 12. Primary decision criterion

The proposed fusion should offer a better trade-off than B1/B2/B3:

> fewer false and duplicate alerts at comparable severe-event recall, without unacceptable additional detection delay.

Do not select the winning system by accuracy alone.

## 13. Reproducible experiment output

Each experiment should emit:

```text
evaluation/runs/<run_id>/
  config.json
  environment.json
  predictions.jsonl
  incidents.jsonl
  metrics.json
  summary.md
```

Include git commit SHA when available.