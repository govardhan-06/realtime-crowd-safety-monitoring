# Dataset Pipeline Cleanup Todo

## Task 1: Define the Drive binary-manifest contract and dataset registry

**Description:** Establish one notebook-local manifest format and registry for the four approved sources. Specify source layout discovery, binary label mapping, split role, original-video grouping, fixed seed, and optional event boundaries. Keep it separate from the checked-in reviewed incident-manifest schema.

**Acceptance criteria:**

- [x] Records include `dataset`, Drive-relative `relative_path`, `label`, and `split`; optional source/group and temporal metadata are preserved when available.
- [x] `violent_flows` can only produce `external_test` records; UBI-Fights, Surveillance Fight, and SCVD use official splits when available or deterministic 70/10/20 fallback splits.
- [x] The registry maps only SCVD `Normal` and `Violence`; `Weaponized Violence` is reported as excluded.

**Verification:**

- [x] Run a media-independent fixture self-check for deterministic split assignment, invalid labels/layouts, duplicate path/hash detection, and train/test leakage rejection.

**Dependencies:** None

**Files likely touched:**

- `colab-notebooks/dataset_download.ipynb`

**Estimated scope:** Small (1 file)

## Task 2: Replace the dataset-download notebook workflow

**Description:** Remove MOT20, UCSD Ped2, and XD-Violence cells. Build an ordered, rerunnable Colab workflow that mounts Drive, creates the requested directories, imports/downloads the four approved datasets, preserves originals, generates the four manifests without media copies, and prints counts and Drive usage.

**Acceptance criteria:**

- [x] The notebook contains no old dataset downloads, conversions, PoC copies, or old `raw/`/`poc_eval/` output paths.
- [x] Violent-Flows uses the existing archive/extraction path but persists originals under `datasets/violent_flows/` and emits only `external_test.json` entries.
- [x] UBI-Fights and Surveillance Fight import/download sections discover source labels and preserve originals under their respective `datasets/` roots.
- [x] SCVD uses Kaggle when configured, skips cleanly without credentials, maps only `Normal`/`Violence`, and preserves source-class metadata.
- [x] Rerunning does not re-download or duplicate already valid Drive content, and writes `train.json`, `val.json`, `test.json`, and `external_test.json` under Drive `manifests/`.

**Verification:**

- [x] Parse the notebook JSON and execute its media-independent manifest self-check.
- [ ] In Colab, run the workflow twice against a small authorised source subset and inspect manifest paths, class/split counts, skipped-source messages, and Drive usage.

**Dependencies:** Task 1

**Files likely touched:**

- `colab-notebooks/dataset_download.ipynb`

**Estimated scope:** Medium (1 file)

## Checkpoint: Dataset preparation

- [x] Tasks 1-2 meet their acceptance criteria.
- [x] No split has duplicate original videos or derived windows, and no Violent-Flows row is in training or validation.
- [x] No dataset media, archives, or credentials appear in `git status`.

## Task 3: Adapt M6 to the Drive dataset manifests

**Description:** Replace the current XD-Violence/MOT20/UCSD source assumptions with Drive manifest loading. Score X3D windows per dataset for `test.json` and `external_test.json`, keep threshold selection out of external-test, and run the five-strategy incident evaluation only for frozen long-form UBI-Fights test records.

**Acceptance criteria:**

- [x] Stage 1 reports precision, recall, F1, accuracy, confusion matrix, score distributions, false positives/negatives, and latency separately for Violent-Flows, UBI-Fights, Surveillance Fight, and SCVD when rows exist.
- [x] `external_test.json` is never used for threshold calibration or training/validation selection.
- [x] Stage 2 selects only frozen UBI-Fights test videos and retains the existing event/operational metrics and five-strategy replay.
- [x] Existing incident-manifest validation remains strict; the adapter rejects unsafe/missing Drive paths and unsupported binary manifest records.

**Verification:**

- [x] Run the notebook's media-independent self-checks for per-dataset aggregation, external-test isolation, and UBI-only incident selection.
- [ ] Perform one authorised Colab smoke run with a reviewed Drive manifest and inspect saved metrics/report artifacts.

**Dependencies:** Tasks 1-2

**Files likely touched:**

- `colab-notebooks/m6_evaluation_demo.ipynb`
- `colab-notebooks/x3d_baseline_evaluation.ipynb`

**Estimated scope:** Medium (1 file)

## Task 4: Update committed manifests and dataset/evaluation documentation

**Description:** Remove obsolete core violence-evaluation references, replace the checked-in old evaluation sample/template as appropriate for the new Drive workflow, and align project data/evaluation guidance without changing unrelated runtime behavior.

**Acceptance criteria:**

- [x] Checked-in evaluation fixtures/templates no longer designate XD-Violence, MOT20, or UCSD Ped2 as required violence-model evaluation data.
- [x] `README.md`, `DATA_AND_MODELS.md`, and `EVALUATION_PLAN.md` name the four approved sources, their roles, split policy, and Violent-Flows external-test restriction.
- [x] Documentation distinguishes component X3D results from UBI-Fights full incident evaluation and does not claim Colab results that were not run.

**Verification:**

- [x] Search confirms obsolete workflow references are removed from the named scope.
- [x] Validate edited JSON/schema files and inspect the documentation diff for data/privacy and no-training-scope compliance.

**Dependencies:** Tasks 1-3

**Files likely touched:**

- `evaluation/manifests/manifest.json`
- `README.md`
- `engg-work/crowd_safety_codex_handoff/DATA_AND_MODELS.md`
- `engg-work/crowd_safety_codex_handoff/EVALUATION_PLAN.md`

**Estimated scope:** Medium (4 files)

## Checkpoint: Evaluation contract

- [x] Tasks 3-4 meet their local implementation acceptance criteria; Colab runtime acceptance remains in Task 5.
- [x] Focused local notebook/schema checks passed; broad unittest and Colab/Drive gates remain intentionally pending per request.
- [ ] Run `./venv/bin/python -m unittest discover -s tests -p 'test_*.py'` and JSON/notebook parsing checks. (Broad unittest suite intentionally not run per request; focused notebook checks passed.)
- [x] Review the final diff against the handoff non-goals; no model training was run and fusion, YOLO, and ByteTrack were not modified by this work.

## Task 5: Execute and record the real Colab/Drive acceptance path

**Description:** Use authorised Colab downloads/imports and Drive storage to prove the notebook's runtime behavior, then capture factual counts, skipped datasets, failures, and output artifact locations for handoff.

**Acceptance criteria:**

- [ ] Drive contains the four requested dataset roots and four generated manifests, or each unavailable source has a concrete access/configuration failure recorded.
- [ ] Inspect representative files from every discovered class, manifest counts, duplicate/leakage report, and Drive usage.
- [ ] Run one X3D component smoke evaluation and one UBI-Fights incident-evaluation smoke path only when reviewed media/events are available; record unavailable model/media explicitly.

**Verification:**

- [ ] Manually inspect Colab cell output, Drive paths, and generated JSON/report artifacts.
- [ ] Record runtime/latency, decode failures, credential/access failures, and unperformed model/full-suite gates without treating local checks as Colab proof.

**Dependencies:** Tasks 1-4

**Files likely touched:**

- `tasks/todo.md` (checklist status only)

**Estimated scope:** Small (no production-code files)

## Completion Checklist

- [x] Every task has acceptance criteria and verification.
- [x] Dependencies are ordered and checkpoints are present.
- [x] This plan has human approval before implementation.
