# Dataset Pipeline Cleanup — Codex Handoff

**Repository:** `govardhan-06/realtime-crowd-safety-monitoring`  
**Target file:** `colab-notebooks/dataset_download.ipynb`  
**Base branch:** `main`

## Goal

Refactor the Colab dataset workflow so Google Drive becomes the persistent filesystem for the **crowd-violence datasets actually needed for X3D evaluation and fine-tuning**.

Remove the old dataset workflow that is no longer relevant.

---

## Dataset Decision Update

`NTU CCTV-Fights` is intentionally excluded because access requires approval and would block a reproducible setup.

Use **SCVD (SmartCity CCTV Violence Detection Dataset)** instead:

```text
https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
```

For the first binary experiment:

```text
Normal   -> non_violent
Violence -> violent
```

Keep `Weaponized Violence` excluded initially unless later needed.

---

## Final Dataset Plan

Use exactly these four datasets:

```text
violent_flows
ubi_fights
surveillance_fight
scvd
```

Do not use XD-Violence in the core workflow. It is too broad for this project's focused CCTV/crowd-violence target.

Dataset roles:

```text
Violent-Flows
→ 100% external/final evaluation
→ never used for training or validation

UBI-Fights
→ training + validation + frozen held-out test
→ main long-form/event-level evaluation source

Surveillance Fight
→ training + validation + frozen held-out test
→ short CCTV fight/non-fight benchmark

SCVD
→ training + validation + frozen held-out test
→ CCTV violence generalization
→ use Normal and Violence only
→ exclude Weaponized Violence initially
```

Recommended split policy:

```text
UBI-Fights:
70% train
10% validation
20% frozen test

Surveillance Fight:
70% train
10% validation
20% frozen test
# for 300 clips: 210 train / 30 val / 60 test
# keep class balance: 30 fight + 30 non-fight in test

SCVD:
use official split if available;
otherwise 70% train / 10% validation / 20% frozen test

Violent-Flows:
100% external_test
```

Important:

- split at the original-video level;
- split before extracting temporal windows;
- no video or derived window may appear across multiple splits;
- never fine-tune on Violent-Flows.

---

## Keep / Add These Datasets

Use only:

```text
violent_flows
ubi_fights
surveillance_fight
scvd
```

### Intended role

```text
violent_flows
→ final independent crowd-violence evaluation

ubi_fights
→ primary fine-tuning dataset

surveillance_fight
→ fine-tuning + held-out evaluation

scvd
→ fine-tuning + held-out evaluation for CCTV violence
```

---

## Remove Old Dataset Workflow

Remove from the notebook and related dataset preparation logic:

```text
mot20
ucsd_ped2
xd_violence
```

Also remove their:

- download cells;
- extraction cells;
- conversion cells;
- Drive folders;
- PoC subset generation;
- references in generated manifests/evaluation preparation.

Do not delete unrelated runtime code in the main pipeline.

---

## Google Drive Layout

Use:

```text
/content/drive/MyDrive/crowd_safety/

datasets/
├── violent_flows/
├── ubi_fights/
├── surveillance_fight/
└── scvd/

manifests/
├── train.json
├── val.json
├── test.json
└── external_test.json

models/
evaluation/
```

Prefer `datasets/` over the current mixed `raw/` / `poc_eval/` layout.

---

## Important Storage Rule

Keep the downloaded/original dataset files intact.

Do **not** physically duplicate videos into separate train/val/test folders.

Instead:

```text
original dataset files
        ↓
manifest generation
        ↓
train.json / val.json / test.json
```

Each manifest entry should contain at least:

```json
{
  "dataset": "ubi_fights",
  "relative_path": "datasets/ubi_fights/...",
  "label": "violent",
  "split": "train"
}
```

Add temporal metadata where available:

```json
{
  "start_s": 12.4,
  "end_s": 16.8
}
```

---

## Notebook Changes

### 1. Keep Drive setup

Retain:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Create the new folder structure automatically.

---

### 2. Violent-Flows

Keep the existing working download/extraction logic, but write to:

```text
datasets/violent_flows/
```

Use it primarily for final evaluation.

Do not include it in the default fine-tuning split.

---

### 3. UBI-Fights

Add a dedicated section to:

- download/import the dataset;
- preserve original files;
- detect labels/annotations;
- register files in manifests;
- support reproducible train/val/test assignment.

---

### 4. Surveillance Camera Fight Dataset

Add download/import support for:

```text
https://github.com/seymanurakti/fight-detection-surv-dataset
```

Persist under:

```text
datasets/surveillance_fight/
```

Generate binary labels:

```text
violent
non_violent
```

---

### 5. SCVD — SmartCity CCTV Violence Detection Dataset

Use SCVD as the replacement for SCVD because it is directly accessible and better suited to a reproducible Colab workflow.

Access:

```text
https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
```

Persist under:

```text
datasets/scvd/
```

Support Kaggle download through Colab when credentials are available.

Dataset classes include:

```text
Normal
Violence
Weaponized Violence
```

For the first binary crowd-violence experiment use:

```text
Normal    -> non_violent
Violence  -> violent
```

Exclude `Weaponized Violence` initially unless manual inspection shows it is relevant to the project scope.

The notebook should:

- download/import SCVD;
- preserve original dataset structure;
- register source class metadata;
- map selected classes to binary labels;
- add entries to deterministic train/val/test manifests;
- skip gracefully with a clear message if Kaggle credentials are not configured.

---


---

## Evaluation Manifest Roles

Generate four manifests:

```text
train.json
→ UBI-Fights + Surveillance Fight + SCVD training videos

val.json
→ UBI-Fights + Surveillance Fight + SCVD validation videos

test.json
→ frozen held-out UBI-Fights + Surveillance Fight + SCVD videos

external_test.json
→ all Violent-Flows videos
```

Do not combine all evaluation datasets into one headline score.

Report model results per dataset so domain-generalization failures remain visible.

---

## Manifest Rules

Generate reproducible manifests for:

```text
train
val
test
```

Requirements:

- deterministic split seed;
- no video appears in more than one split;
- preserve official dataset splits where appropriate;
- Violent-Flows defaults to evaluation/test only;
- store dataset source for every sample;
- store binary label;
- store temporal boundaries where available;
- store relative Drive path, not Colab-local temporary paths.

Recommended binary labels:

```text
violent
non_violent
```

---

## Leakage Protection

Before generating final manifests:

- detect duplicate file hashes;
- detect identical filenames where useful;
- avoid placing near-identical clips from the same source across train/test when identifiable;
- never train on the final Violent-Flows evaluation split.

---

## Notebook Output

At the end, print a summary like:

```text
Dataset                 Train   Val   Test
------------------------------------------------
UBI-Fights              ...
Surveillance Fight      ...
SCVD                    ...
Violent-Flows             0      0    ...

Total                    ...
```

Also print Drive usage.

---

## Evaluation Notebook Update

Update:

```text
colab-notebooks/m6_evaluation_demo.ipynb
```

Remove the existing evaluation-source assumptions around:

```text
XD-Violence
MOT20
UCSD Ped2
```

The notebook should evaluate the X3D violence model in two stages.

### Stage 1 — Violence-model evaluation

Use:

```text
external_test.json
→ all Violent-Flows videos

test.json
→ held-out SCVD
→ held-out Surveillance Fight
→ held-out UBI-Fights
```

Report per dataset:

```text
precision
recall
F1
accuracy
confusion matrix
score distributions
false positives
false negatives
inference latency
```

Violent-Flows should be the **primary external crowd-violence benchmark**.

Do not tune thresholds on `external_test.json`.

### Stage 2 — Full incident/system evaluation

Use the frozen long-form UBI-Fights test videos for end-to-end evaluation:

```text
video
→ sliding X3D windows
→ YOLO / ByteTrack / crowd signals
→ temporal fusion
→ incident lifecycle
```

Report:

```text
event precision
event recall
false alerts per camera-hour
duplicate alerts per true incident
detection delay
evidence completeness
```

This separates:

```text
short-clip violence classification quality
from
long-horizon incident detection quality
```

---

## Related Cleanup

Update dataset references where needed in:

```text
colab-notebooks/m6_evaluation_demo.ipynb
evaluation/manifests/
engg-work/crowd_safety_codex_handoff/DATA_AND_MODELS.md
engg-work/crowd_safety_codex_handoff/EVALUATION_PLAN.md
README.md
```

Remove MOT20/UCSD Ped2 from the violence-model evaluation workflow.

They should not appear as required violence datasets after this change.

---

## Non-Goals

Do not:

- fine-tune X3D in this task;
- modify fusion logic;
- add RAFT;
- add CSRNet;
- modify YOLO/ByteTrack;
- download RWF-2000;
- add XD-Violence/Hockey Fight/UCF-Crime/movie datasets;
- copy large dataset media into Git.

---

## Definition of Done

- [ ] `dataset_download.ipynb` only prepares the five approved datasets.
- [ ] MOT20 and UCSD Ped2 sections are removed.
- [ ] Google Drive uses the new `datasets/` layout.
- [ ] Violent-Flows remains evaluation-first.
- [ ] UBI-Fights support added.
- [ ] Surveillance Fight support added.
- [ ] SCVD Kaggle download/import flow added.
- [ ] XD-Violence removed from the core dataset/evaluation workflow.
- [ ] deterministic `train.json`, `val.json`, `test.json`, and `external_test.json` generated.
- [ ] no split leakage.
- [ ] related docs/evaluation references updated.
- [ ] notebook can be rerun safely without re-downloading completed datasets unnecessarily.
- [ ] Violent-Flows is external-test only and never included in training/validation.
- [ ] `m6_evaluation_demo.ipynb` evaluates per dataset.
- [ ] UBI-Fights frozen test videos support long-form incident evaluation.
