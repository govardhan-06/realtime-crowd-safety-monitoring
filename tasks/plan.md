# Implementation Plan: Drive-Backed Crowd-Violence Dataset Workflow

## Overview

Replace the obsolete Colab dataset-preparation workflow with a rerunnable, Google-Drive-backed workflow for four datasets: `violent_flows`, `ubi_fights`, `surveillance_fight`, and `scvd`. It will preserve original media once, generate deterministic leakage-safe JSON manifests, and let the M6 and standalone X3D notebooks report quality per dataset before using frozen UBI-Fights videos for full incident evaluation. This is preparation and evaluation plumbing only: it does not train X3D or change perception/fusion runtime code.

## Architecture Decisions

- Drive is the persistent dataset filesystem at `/content/drive/MyDrive/crowd_safety`; Colab `/content` is temporary extraction/download workspace only.
- Preserve each source dataset beneath `datasets/<dataset>/`; manifests reference relative Drive paths and never copy videos into split directories.
- Use one small binary dataset-manifest record for the four Drive files: `dataset`, `relative_path`, `label`, `split`, plus optional temporal/source metadata. Keep the existing richer reviewed incident-manifest contract separate; adapt selected frozen UBI-Fights test records inside the M6 notebook rather than weakening that contract.
- Use a fixed documented seed and original-video grouping for fallback 70/10/20 splits. Prefer an official split where SCVD actually supplies one. Hash/filename checks are safeguards, not a substitute for source-video grouping.
- `violent_flows` is written only to `external_test.json`; it is never eligible for train/validation or threshold selection.
- Dataset acquisition remains Colab-only and idempotent: reuse an existing validated Drive archive/extraction, otherwise download/import. SCVD's Kaggle path emits a clear skipped status when credentials are absent; no credentials or media are committed.
- Preserve the current dirty M3A/M6 work. This cleanup changes only the named notebooks, dataset/evaluation manifests, and dataset/evaluation documentation.

## Dependency Graph

```text
Drive directories + dataset import rules
            |
            v
original-video discovery + binary label mapping
            |
            v
deterministic split and leakage checks
            |
            +--> train.json / val.json / test.json / external_test.json
            |                 |
            v                 v
dataset notebook summary   M6/X3D per-dataset metrics and UBI incident runs
```

## Task List

### Phase 1: Dataset contract and preparation

- [ ] Task 1: Define the Drive binary-manifest contract and dataset registry.
- [ ] Task 2: Replace `dataset_download.ipynb` with the four-dataset, idempotent Drive workflow.

### Checkpoint: Dataset preparation

- [ ] Notebook JSON is valid and its media-independent manifest self-check passes.
- [ ] Generated fixture manifests prove deterministic assignment, class counts, no duplicate content/path across splits, and no Violent-Flows training/validation rows.

### Phase 2: Evaluation consumers and documentation

- [ ] Task 3: Adapt the M6 and standalone X3D Colab evaluation workflows to the new dataset manifests and per-dataset reporting.
- [ ] Task 4: Retire obsolete dataset references from committed evaluation fixtures and project documentation.

### Checkpoint: Evaluation contract

- [ ] M6/X3D media-independent checks cover external-test threshold isolation and UBI-only full-incident selection.
- [ ] Existing focused repository tests and JSON/schema checks pass without Drive media, Colab credentials, or a GPU.

### Phase 3: Real-Colab acceptance

- [ ] Task 5: Run and record the Drive/Colab acceptance path.

### Checkpoint: Complete

- [ ] All four dataset folders and all four manifests exist under Drive after an authorised Colab run.
- [ ] Dataset and model metrics are reported by source dataset; full-system evaluation is limited to frozen long-form UBI-Fights entries.
- [ ] No raw media, Kaggle credentials, archives, or checkpoints are added to Git.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| UBI-Fights source layout/access differs from assumptions | High | Make discovery/label rules explicit, print unmatched files, and stop manifest generation for that source rather than guessing labels. |
| Kaggle credentials are unavailable for SCVD | Medium | Print a clear skipped state and retain other datasets/manifests; require credentials only to populate SCVD. |
| Split/window leakage | High | Assign once at original-video level before windows; assert unique normalized path/hash assignment and preserve source/session grouping when available. |
| Existing M6 incident manifest cannot represent training rows | Medium | Keep the binary dataset manifests distinct and adapt only frozen UBI test records for the incident evaluator. |
| Drive copy/download interruption or quota | Medium | Validate existing artifacts and use resumable/idempotent steps; report Drive usage and failures. |

## Open Questions and Assumptions

- The handoff's “five approved datasets” Definition-of-Done item is treated as a typo: every dataset list names exactly four datasets.
- `val.json` will contain records whose `split` is `validation`, preserving the repository's existing spelling while retaining the requested filename.
- UBI-Fights does not have an acquisition URL in the handoff. The notebook will support an explicit Colab download/import source selected by the user and validate the imported layout; it must not invent labels from ambiguous filenames.
- A real Colab run is required to prove Drive mount, dataset access, Kaggle authentication, archive layout, decodeability, and storage usage. Local checks can prove only notebook structure and pure manifest logic.

## Approval Gate

Implementation begins after this plan is reviewed and approved. It will not begin M3B fine-tuning, modify fusion/perception code, or add datasets outside the four approved sources.
