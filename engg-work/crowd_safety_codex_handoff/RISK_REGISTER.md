# Risk Register

| Risk | Impact | Mitigation / decision |
|---|---|---|
| No single dataset covers the full incident | High | Component-wise datasets + staged/curated end-to-end set |
| Restricted/unavailable violence footage | High | Do not depend on RWF-2000; use accessible authorised datasets |
| Cross-dataset domain shift | High | Cross-dataset validation, camera/site calibration, hard negatives |
| Dense-scene missed detections | High | Start with pretrained YOLO26n; evaluate first; try YOLO26s or fine-tune detector only if necessary |
| Track fragmentation | Medium | Tune ByteTrack; smooth trajectories; feature robustness tests |
| Ready-made M3A violence checkpoint is weak or poorly documented | High | Keep it behind a replaceable adapter; measure locally; use only as delivery baseline; do not inherit third-party metric claims |
| M3B fine-tuning underperforms or exceeds compute budget | High | Frozen-backbone/head-only first; partial unfreeze only if justified; M3A keeps M4-M6 unblocked |
| Violence false positives from benign motion | High | hard negatives + temporal persistence + crowd context |
| Crowd motion false positives | High | camera/ROI calibration + multi-signal fusion + decay |
| Missing model signal interpreted as normal | Critical | explicit stage health; unavailable != zero; deterministic degraded-mode tests |
| Too many alerts | Critical | lifecycle, hysteresis, deduplication, false-alert metric |
| ML pipeline exceeds compute | High | sampling, lightweight detector, compact baseline checkpoint, profiling, cascade heavy inference where useful |
| Insufficient fusion labels | High | deterministic fusion remains final viable system; learned fusion optional |
| VLM explanation hallucinates or contradicts evidence | High | VLM is non-authoritative; show deterministic reason codes separately; prompt for observable facts; human review; evaluate hallucination cases |
| VLM/API unavailable during demo | Medium | explanation is optional/configurable; incident path works without it; deterministic fallback summary |
| Incorrect “stampede prediction” claim | High | wording locked to observable risk indicators/early warning |
| Evidence storage becomes large | Medium | short clips, retention config, metadata references |
| Dashboard steals time from research | Medium | build only after stable incident contract |
| Overengineering runtime | Medium | offline deterministic runner first; no queues unless measured need |
| Novelty-driven blockchain/SAM/etc. scope creep | Medium | add technology only for a documented measurable requirement; VLM explainability is the approved showcase addition |
| Privacy/ethics issue | High | no face recognition/demographics; consented staged footage; purpose-limited storage |