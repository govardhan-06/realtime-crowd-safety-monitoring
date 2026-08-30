# Data and Model Strategy

## 1. Principle

No single public dataset contains everything needed for:
- person detection/tracking;
- crowd motion;
- violence;
- incident-level crowd response;
- alert lifecycle.

Use datasets **component-wise**, then create a small project-specific end-to-end evaluation set.

Do not combine dataset clips blindly and call the result an incident dataset without clear labels/provenance.

The delivery strategy is intentionally training-independent:

> The full system must be buildable with pretrained components before any project-specific fine-tuning succeeds.

## 2. Recommended datasets

### XD-Violence

Role:
- primary candidate for violence/aggression representation fine-tuning;
- broad violent/non-violent video source;
- selected held-out clips for component evaluation.

Use:
- inspect official licensing/access terms before download/use;
- define train/validation/test split without source leakage.

### Violent-Flows

Role:
- crowd violence supplement;
- useful for testing violence recognition where the surrounding scene contains multiple people;
- candidate cross-dataset validation source.

Because it is relatively small, do not rely on it as the sole training set.

### UCF-Crime

Role:
- long-form surveillance-like evaluation;
- event localisation/end-to-end pipeline stress testing;
- selected fighting/assault/normal videos where licensing permits.

Do not equate generic anomaly labels with our incident taxonomy.

### MOT20

Role:
- dense pedestrian detection/tracking stress test;
- trajectory/crowd-feature validation;
- crowded station/stadium/square-like scenes.

Do not train ByteTrack from scratch.

### UCSD Pedestrian Anomaly

Role:
- abnormal movement/crowd-feature response testing;
- threshold/feature sanity checks.

Use as a crowd-motion benchmark, not as ground truth for violence.

### CrowdHuman

Role:
- optional person-detector fine-tuning only if measured dense-scene detection errors justify it.

Decision gate:
1. evaluate pretrained detector on dense footage;
2. inspect misses/occlusions;
3. fine-tune only if person-detection quality materially limits downstream features.

### RWF-2000

Role:
- optional violence dataset **only if legitimately accessible under applicable terms**.

Hard rule:
- the project must not depend on RWF-2000 availability;
- do not scrape/reconstruct restricted raw footage.

## 3. Project-specific staged/curated footage

Create a small controlled evaluation suite recorded with consent and without creating real safety hazards.

Suggested scenarios:
- normal pedestrian flow;
- dense but normal standing/walking;
- sudden group acceleration;
- convergence toward a focal point;
- rapid dispersal;
- opposing directional flow;
- benign crowd gathering hard negative;
- staged non-contact aggressive gestures;
- safely staged fight-like interaction;
- fight-like interaction followed by surrounding dispersal;
- local congestion near an exit/doorway without unsafe physical compression.

Use a fixed elevated/fixed camera angle where possible to resemble CCTV.

For each staged video annotate:
- video/source ID;
- scenario;
- event onset;
- event end;
- involved/affected ROI;
- expected crowd signals;
- expected severity band;
- whether alert should occur;
- notes/hard-negative class.

## 4. Development video suite

Maintain a versioned manifest rather than committing raw videos.

Example:

```csv
video_id,source_dataset,path_or_external_id,split,scenario,expected_alert
mot20_dense_01,MOT20,...,dev,dense_normal,false
vf_violence_01,Violent-Flows,...,dev,crowd_violence,true
ucf_long_fight_01,UCF-Crime,...,test,long_fight,true
staged_dispersal_01,project,...,test,dispersal,true
staged_benign_converge_01,project,...,test,benign_convergence,false
```

Keep raw media outside Git and resolve it through environment/config paths.

For day-to-day development, maintain a small 20-50 clip suite so every milestone can be tested without downloading all datasets.

## 5. Models

### Person detector — M2

Default:
- pretrained **Ultralytics YOLO26n** person detector;
- class filter restricted to `person`.

Fallback/upgrade:
- evaluate YOLO26s only if YOLO26n misses materially affect downstream crowd features.

Training:
- no initial fine-tuning;
- optionally fine-tune using CrowdHuman or project-specific frames only if error analysis shows downstream failure.

Reason:
- the project contribution is not a new person detector;
- detector training should not consume the capstone schedule unless objectively necessary.

### Tracker — M2

Default:
- ByteTrack.

Training:
- none.

Tune association/config thresholds based on dense-scene tests.

## 6. Violence recognition strategy

Violence recognition is deliberately split into two sub-stages.

### M3A — pretrained baseline, no project training required

Goal:
- unblock the complete end-to-end system;
- produce timestamp-aligned violent/non-violent evidence through a generic adapter.

Initial development checkpoint candidate:
- `mitegvg/videomae-small-kinetics-binary-finetuned-xd-violence`

Characteristics at the time this handoff was updated:
- VideoMAE-family binary video classifier;
- Hugging Face Transformers compatible;
- small checkpoint suitable for local/Colab experimentation;
- community checkpoint, not an authoritative benchmark.

Rules:
1. verify the current model card/license before downloading;
2. pin the revision/checksum used by the project;
3. inspect `id2label`/`label2id` rather than assuming class order;
4. benchmark on our own dev videos before choosing threshold;
5. never copy the model card's metrics into our report as if they are our experimental result;
6. if the checkpoint is unusable, replace it behind the same adapter rather than changing downstream contracts.

A second community checkpoint may be tried only if the first is clearly unsuitable. Do not turn M3A into an open-ended model search.

### M3B — bounded X3D-S transfer-learning experiment

Default academic experiment:
- X3D-S pretrained on a large action-recognition corpus;
- replace classifier head with binary violent/non-violent head.

Training sequence:
1. create dataset adapter(s);
2. decode fixed-duration clips;
3. spatial resize/crop;
4. temporal sample frames;
5. create leakage-safe train/validation/test split;
6. replace classifier head;
7. freeze backbone;
8. train classification head;
9. evaluate;
10. unfreeze only the final block(s) if validation clearly justifies it;
11. use a smaller learning rate for unfrozen backbone layers;
12. calibrate threshold on validation data;
13. evaluate on held-out and cross-dataset clips;
14. save checkpoint + config + metrics.

Do **not**:
- train from scratch;
- make full-backbone fine-tuning the default;
- block M4/M5/M6 on M3B performance.

### Why keep X3D-S for M3B

- relatively lightweight;
- clear transfer-learning path;
- creates a genuine project training/fine-tuning experiment;
- gives a meaningful comparison against the ready-made M3A VideoMAE-style baseline.

## 7. Crowd intelligence

V1:
- engineered interpretable features from tracks/ROIs;
- optional classical optical flow if it adds measurable signal.

Do not begin with a deep crowd anomaly model.

## 8. Temporal fusion

V1:
- deterministic/rule-weighted method with smoothing, persistence, hysteresis, spatial association, severity, and incident state.

V2 experimental:
- small MLP/TCN/LSTM over generated temporal feature windows.

Do not train V2 until the feature pipeline and incident annotations are stable.

## 9. VLM explainability

Purpose:
- generate a human-readable summary from evidence belonging to an incident already created by the incident engine.

Default integration candidate:
- Gemini video/image understanding through a provider adapter if credentials are available.

Inputs:
- evidence clip and/or selected keyframes;
- deterministic reason codes;
- compact signal summary;
- incident timestamps.

Outputs:
- concise description of visible behavior;
- concise operator-facing explanation.

Hard rule:
- VLM output is **not** a detection signal;
- it does not enter the fusion feature vector;
- it does not create/close incidents;
- it does not change severity;
- it is allowed to fail without affecting incident delivery.

No VLM fine-tuning is planned.

## 10. Fusion dataset generation

After the core pipeline works, generate aligned records such as:

```csv
video_id,timestamp_s,region_id,violence_score,violence_status,density,density_delta,
mean_speed,speed_variance,direction_disorder,convergence,dispersal,counter_flow,
congestion,incident_label,severity_label
```

Annotation levels may include:
- normal;
- candidate/suspicious;
- active;
- escalating;
- critical;
- resolving.

Prevent leakage:
- split by source video/scenario/session, not by individual timestamp row;
- keep all windows from one raw event in one split.

## 11. Data provenance

Every dataset adapter should document:
- official dataset name;
- source URL/reference in project docs;
- permitted use;
- acquisition date;
- local path convention;
- label mapping;
- exclusions;
- split strategy.

Every external model checkpoint should document:
- provider/repository;
- exact model identifier;
- exact revision/checksum where practical;
- license;
- label mapping;
- base model;
- acquisition date;
- local cache/path convention;
- known limitations.

## 12. Hard-negative library

Actively collect examples likely to trigger false alarms:
- running for benign reasons;
- celebrations;
- raised arms;
- dancing;
- sports-like movements;
- queues;
- people gathering around an object;
- umbrellas/bags/occlusions;
- camera shake;
- bicycles/scooters where relevant;
- lighting changes.

Hard negatives are part of the research, not cleanup after the project.

## 13. Dataset/model acceptance gates

### Before M3A integration

- verify checkpoint access and license;
- inspect label mapping;
- run inference on at least a few positive and negative clips;
- document model/revision/config;
- confirm CPU fallback path or mock adapter for tests.

### Before M3B training

- verify lawful/allowed dataset access;
- write manifest;
- verify labels;
- check class distribution;
- inspect representative samples;
- define leakage-safe splits;
- record preprocessing config;
- confirm a Colab/local training command works on a tiny smoke-test subset.

### Before claiming results

- state exactly which datasets/splits were used;
- separate external checkpoint claims from our own measured results;
- separate component benchmark results from end-to-end incident results;
- report M3A and M3B separately.