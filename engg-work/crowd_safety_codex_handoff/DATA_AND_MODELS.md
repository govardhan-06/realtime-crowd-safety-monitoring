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

## 5. Models

### Person detector
Default:
- mature pretrained YOLO-family person detector.

Training:
- no initial fine-tuning;
- optionally fine-tune using CrowdHuman or project-specific frames only if error analysis shows downstream failure.

### Tracker
Default:
- ByteTrack.

Training:
- none.

Tune association/config thresholds based on dense-scene tests.

### Violence recognizer
Default:
- X3D-S pretrained on a large action-recognition corpus, then fine-tuned for violent vs non-violent temporal classification.

Why:
- lightweight relative to large video transformers;
- suitable for a first real-time baseline;
- clear transfer-learning path.

Fine-tuning outline:
1. create dataset adapter(s);
2. decode fixed-duration clips;
3. spatial resize/crop;
4. temporal sample frames;
5. use balanced sampling/class weighting if needed;
6. replace classifier head;
7. freeze most backbone layers for initial experiment;
8. train head;
9. progressively unfreeze selected later blocks if validation justifies it;
10. calibrate threshold on validation data;
11. evaluate cross-dataset;
12. save checkpoint + config + metrics.

### Comparison violence model
Candidate:
- VideoMAE or another pretrained video transformer.

Use only after X3D baseline exists. Its purpose is comparison, not scope expansion.

### Crowd intelligence
V1:
- engineered interpretable features from tracks/ROIs;
- optional classical optical flow if it adds measurable signal.

Do not begin with a deep crowd anomaly model.

### Temporal fusion
V1:
- deterministic/rule-weighted method with smoothing, persistence, hysteresis, spatial association, severity, and incident state.

V2 experimental:
- small MLP/TCN/LSTM over generated temporal feature windows.

Do not train V2 until the feature pipeline and incident annotations are stable.

## 6. Fusion dataset generation

After the core pipeline works, generate aligned records such as:

```csv
video_id,timestamp_s,region_id,violence_score,density,density_delta,mean_speed,
speed_variance,direction_disorder,convergence,dispersal,counter_flow,congestion,
incident_label,severity_label
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

## 7. Data provenance

Every dataset adapter should document:
- official dataset name;
- source URL/reference in project docs;
- permitted use;
- acquisition date;
- local path convention;
- label mapping;
- exclusions;
- split strategy.

## 8. Hard-negative library

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

## 9. Dataset acceptance gate

Before training:
- verify lawful/allowed access;
- write manifest;
- verify labels;
- check class distribution;
- inspect representative samples;
- define leakage-safe splits;
- record preprocessing config.

Before claiming results:
- state exactly which datasets/splits were used;
- separate component benchmark results from end-to-end incident results.
