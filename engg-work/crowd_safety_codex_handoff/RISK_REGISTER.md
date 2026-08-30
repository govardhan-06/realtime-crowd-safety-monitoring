# Risk Register

| Risk | Impact | Mitigation / decision |
|---|---|---|
| No single dataset covers the full incident | High | Component-wise datasets + staged/curated end-to-end set |
| Restricted/unavailable violence footage | High | Do not depend on RWF-2000; use accessible authorised datasets |
| Cross-dataset domain shift | High | Cross-dataset validation, camera/site calibration, hard negatives |
| Dense-scene missed detections | High | Evaluate first; fine-tune detector only if necessary |
| Track fragmentation | Medium | Tune tracker; smooth trajectories; feature robustness tests |
| Violence false positives from benign motion | High | hard negatives + temporal persistence + crowd context |
| Crowd motion false positives | High | camera/ROI calibration + multi-signal fusion + decay |
| Too many alerts | Critical | lifecycle, hysteresis, deduplication, false-alert metric |
| ML pipeline exceeds compute | High | sampling, lightweight X3D baseline, profiling, cascade heavy inference |
| Insufficient fusion labels | High | deterministic fusion remains final viable system; learned fusion optional |
| Incorrect “stampede prediction” claim | High | wording locked to observable risk indicators/early warning |
| Evidence storage becomes large | Medium | short clips, retention config, metadata references |
| Dashboard steals time from research | Medium | build only after stable incident contract |
| Overengineering runtime | Medium | offline deterministic runner first; no queues unless measured need |
| Privacy/ethics issue | High | no face recognition/demographics; consented staged footage; purpose-limited storage |
