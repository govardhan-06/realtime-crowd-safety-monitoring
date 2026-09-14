# Next Changes — X3D Violence Baseline

**Repository:** `govardhan-06/realtime-crowd-safety-monitoring`  
**Base branch:** `main`  
**New branch:** `feat/x3d-violence-baseline`

## Goal

Replace the current Nikeytas VideoMAE violence model with a stronger pretrained X3D model and evaluate it **before changing crowd features or fusion logic**.

The immediate objective is only:

```text
Video
  ↓
X3D violence model
  ↓
violence_score
  ↓
evaluation
```

Do not mix unrelated pipeline changes into this branch.

---

## 1. Replace the Current Violence Model

Remove the current active model:

```text
Nikeytas/videomae-crime-detector-fixed-format
```

Use this pretrained model as the new baseline:

```text
Repository:
visionlab-ai/school-violence-detection-models

Checkpoint:
final/final_x3d_realtime.pt

Architecture:
X3D-M

Training dataset:
RWF-2000

Input:
16 frames

Initial threshold:
0.4

License:
MIT
```

Pin the exact repository revision and record the checkpoint SHA-256.

Do not use a floating `main` revision for reproducible evaluation runs.

---

## 2. Preserve the Existing Violence Contract

Keep the existing abstraction:

```python
class ViolenceClassifier(Protocol):
    def infer(self, window: ClipWindow) -> ViolenceEvidence:
        ...
```

Add:

```python
X3DViolenceClassifier
```

It must return the existing `ViolenceEvidence` type.

Important invariant:

```text
model unavailable != violence_score 0
```

If inference fails:

```text
score = None
status = "degraded" or "unavailable"
```

Never convert failures into non-violent predictions.

---

## 3. Add a Violence Model Factory

`runner.py` currently constructs the VideoMAE classifier directly.

Replace that coupling with:

```python
create_violence_classifier(config.violence)
```

Example:

```python
def create_violence_classifier(config):
    if config.backend == "x3d":
        return X3DViolenceClassifier(...)
    elif config.backend == "huggingface":
        return VideoMAEViolenceClassifier(...)
    raise ConfigError(...)
```

`runner.py` should not know the model architecture.

Keep the existing VideoMAE adapter available for future experiments, but Nikeytas must no longer be the default.

---

## 4. Update Configuration

Update `ViolenceConfig` and `configs/pipeline/dev.toml`.

Suggested config shape:

```toml
[violence]
enabled = true

backend = "x3d"

repository = "visionlab-ai/school-violence-detection-models"
checkpoint = "final/final_x3d_realtime.pt"
revision = "<PINNED_COMMIT>"
checkpoint_sha256 = "<SHA256>"

architecture = "x3d_m"

sample_count = 16
clip_duration_s = 3.0
cadence_s = 1.0

threshold = 0.4

device = "auto"

labels = ["non-violent", "violent"]
license = "mit"
```

Treat `0.4` and `3.0s` as starting values only. They must be validated using project clips.

---

## 5. X3D Adapter Requirements

The adapter must:

- load the pinned X3D checkpoint;
- verify checkpoint metadata/checksum;
- use the exact preprocessing expected by the checkpoint;
- sample 16 frames from the existing `ClipWindow`;
- convert frames to RGB;
- build the expected tensor shape:

```text
B x C x T x H x W
```

- run with `torch.inference_mode()`;
- output violence probability in `[0, 1]`;
- measure inference latency;
- expose `StageHealth`;
- expose model provenance;
- work with CPU/MPS/CUDA as supported;
- fail cleanly without breaking the entire video pipeline.

---

## 6. Provenance

Record at least:

```json
{
  "backend": "x3d",
  "repository": "visionlab-ai/school-violence-detection-models",
  "checkpoint": "final/final_x3d_realtime.pt",
  "revision": "<PINNED_COMMIT>",
  "checkpoint_sha256": "<SHA256>",
  "architecture": "x3d_m",
  "sample_count": 16,
  "labels": ["non-violent", "violent"],
  "license": "MIT"
}
```

This must appear in run metadata/evaluation artifacts.

---

## 7. Dependency Changes

Update `pyproject.toml` with only the dependencies required for X3D.

Likely candidates:

```text
torch
torchvision
pytorchvideo
huggingface-hub
```

Do not blindly pin versions.

First verify a compatible set on:

```text
Mac development environment
Colab environment
```

Then pin tested versions.

---

## 8. Tests

Update/add tests in `tests/test_violence.py`.

Required coverage:

```text
X3D classifier creation
checkpoint/provenance fields
16-frame sampling
correct input tensor shape
RGB preprocessing
probability in range 0..1
correct violent label mapping
health = available after successful inference
failure => score None
failure != non-violent
factory selects X3D
runner works through generic ViolenceClassifier
```

Mock the heavy model where appropriate.

Do not require downloading the real checkpoint in unit tests.

---

## 9. Documentation Updates

Update references to the active violence baseline in:

```text
README.md

engg-work/crowd_safety_codex_handoff/
    DATA_AND_MODELS.md
    ARCHITECTURE.md
    ROADMAP.md
    EVALUATION_PLAN.md
    milestones/M03_VIOLENCE_MODEL.md
```

New definition:

```text
M3A = pretrained X3D-M violence baseline
M3B = project-specific X3D transfer-learning experiment
```

Nikeytas may remain documented only as:

```text
Rejected historical baseline due to poor project-domain performance.
```

---

# Evaluation After Implementation

Once X3D is integrated, **do not change fusion or crowd features yet**.

First evaluate the violence model in isolation.

## Evaluation Set

Use the existing project evaluation clips, including:

```text
clear violence
clear non-violence
benign running
high motion
dense crowds
occlusion
low light
camera motion if available
long surveillance-like clips
```

## Measure

At minimum:

```text
precision
recall
F1
false positives
false negatives
score distributions
inference latency
```

Also manually inspect representative failures.

Do not use external model-card metrics as project results.

---

# Evaluation Gate

The X3D baseline passes if:

- it runs reliably through the existing pipeline;
- its scores meaningfully separate violent and non-violent project clips;
- it performs materially better than the current Nikeytas model;
- benign high-motion false positives are acceptable or clearly calibratable;
- latency is practical enough for the target pipeline;
- no major preprocessing/model-loading issue remains.

If it passes, make X3D the official M3A baseline.

---

# What Happens After X3D Evaluation

Only after the X3D evaluation is complete:

## Next 1 — Fix crowd feature eligibility

Current issue:

```text
one new/short track
→ entire ROI may become "insufficient"
```

Change to:

```text
all tracks
→ filter eligible tracks
→ compute crowd features from eligible tracks
```

---

## Next 2 — Normalize motion features

Move away from camera-dependent values such as:

```text
mean_speed_px_s
acceleration_px_s2
```

Add normalized movement using frame dimensions / frame diagonal.

---

## Next 3 — Fix fusion risk transforms

Current generic min-max normalization gives non-zero risk to neutral values.

Examples:

```text
density_delta = 0
should produce density risk ~= 0

acceleration = 0
should produce acceleration risk ~= 0
```

Replace generic min-max normalization with feature-specific risk transforms.

---

## Next 4 — Re-run Full System Evaluation

Compare:

```text
violence-only
crowd-only
naive OR
rule fusion
temporal fusion
```

Measure:

```text
event precision
event recall
F1
false alerts per camera-hour
duplicate alerts per true incident
detection delay
evidence completeness
```

---

## Next 5 — Project-Specific X3D Training

After the pretrained baseline is stable:

```text
pretrained X3D
→ replace binary head
→ freeze backbone
→ train head on approved dataset
→ validate
→ calibrate threshold
→ compare with pretrained X3D
```

Use Colab.

Do not train from scratch.

---

# Explicit Non-Goals for This Branch

Do **not** implement any of the following yet:

```text
RAFT / optical flow
CSRNet / crowd-density model
YOLO fine-tuning
ByteTrack replacement
ROI-local X3D inference
VLM-based detection
LLM-based safety decisions
new fusion algorithm
crowd feature rewrite
X3D fine-tuning
```

These should only be considered after the new X3D baseline has been evaluated.

---

# Files Expected to Change

Primary files:

```text
src/crowd_safety/violence.py
src/crowd_safety/config.py
src/crowd_safety/runner.py
configs/pipeline/dev.toml
pyproject.toml
tests/test_violence.py
README.md
```

Documentation:

```text
engg-work/crowd_safety_codex_handoff/
    DATA_AND_MODELS.md
    ARCHITECTURE.md
    ROADMAP.md
    EVALUATION_PLAN.md
    milestones/M03_VIOLENCE_MODEL.md
```

Avoid unrelated changes.

---

# Definition of Done

This task is complete when:

- [ ] branch created from latest `main`;
- [ ] Nikeytas removed as active/default model;
- [ ] X3D-M adapter implemented;
- [ ] violence factory implemented;
- [ ] exact model revision pinned;
- [ ] checkpoint SHA-256 recorded;
- [ ] provenance emitted;
- [ ] existing `ViolenceEvidence` contract preserved;
- [ ] inference failures return `score=None`;
- [ ] dependencies tested and pinned;
- [ ] unit tests pass;
- [ ] one positive clip works end-to-end;
- [ ] one negative clip works end-to-end;
- [ ] existing evaluation pipeline can run X3D;
- [ ] documentation updated;
- [ ] changes committed and pushed to `feat/x3d-violence-baseline`.

After that, run the full violence-model evaluation before making any crowd/fusion changes.
