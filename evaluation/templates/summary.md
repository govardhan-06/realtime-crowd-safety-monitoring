# Crowd-safety evaluation report: {{evaluation_id}}

Generated at: {{generated_at}}

## Scope

This report measures early detection of observable crowd-risk and violence indicators on the saved evaluation manifest. It does not claim guaranteed future stampede prediction or live deployment readiness.

- Entries: {{entry_count}}
- Test duration: {{camera_hours}} camera-hours
- VLM status: {{vlm_status}}
- M6B status: {{m6b_status}}

## Incident-method comparison

{{strategy_table}}

Metrics are calculated from immutable reviewed-manifest annotations and saved pipeline/replay predictions. An alert is the first transition to `active` for each unique incident ID; lifecycle reactivation is not a new prediction. Evidence completeness is compared by source/time overlap and is `not_comparable` when association is ambiguous or unavailable.

## M3A violence-model evaluation

{{m3a_summary}}

## Evaluation integrity

{{integrity_summary}}

## Failure slices

{{failure_slices}}

## Latency summary

{{latency_summary}}

## Model comparison

{{model_table}}

M3A and M3B are reported separately. Missing or incompatible M3B held-out predictions are an open M6B gate, not a zero result.

## VLM review

{{vlm_review}}

Generated explanation text is supplementary and never enters incident creation, lifecycle, severity, or escalation decisions. Disabled or unavailable VLM output leaves the evaluation core valid.

## Limitations and failures

{{limitations}}
