# Implementation Walkthrough

## 1. Two-Minute Summary

The dataset workflow now uses Google Drive as the persistent filesystem for
`violent_flows`, `ubi_fights`, `surveillance_fight`, and `scvd`. It preserves
source media, assigns deterministic original-video splits, and writes four
binary manifests without copying videos into split directories. M6 and the
standalone X3D evaluator consume those manifests for per-dataset component
metrics while keeping the reviewed incident manifest separate.

The production/runtime pipeline was not changed. The real model, Drive,
credential, decode, and incident smoke gates remain Colab/manual work.

## 2. Before vs After

Before, the dataset notebook downloaded/consolidated obsolete sources and
created mixed `raw/`/`poc_eval/` layouts. Evaluation notebooks assumed a rich
incident manifest for component scoring. After, dataset preparation emits a
small binary-video contract under Drive `manifests/`; incident evaluation
still uses the strict reviewed schema and only selects UBI-Fights frozen test
entries.

## 3. Runtime / Data Flow

```text
authorised source/archive
        -> Drive datasets/<dataset>/ originals
        -> class discovery + hash + source grouping
        -> deterministic train/validation/test or external_test assignment
        -> Drive manifests/{train,val,test,external_test}.json
        -> M3A per-dataset X3D scoring
        -> reviewed UBI-Fights test adapter
        -> existing five-strategy incident replay
```

## 4. Code Map

- `colab-notebooks/dataset_download.ipynb`: registry, Kaggle/GitHub/source
  import, 15 GB/free-space-bounded video selection, class mapping,
  official/fallback split assignment, duplicate/leakage checks, and Drive
  manifest generation.
- `colab-notebooks/m6_evaluation_demo.ipynb`: binary-manifest loader and
  Stage 1 per-dataset metrics; Stage 2 filters the separate strict incident
  manifest to UBI-Fights `test` entries before reusing existing replay/report
  code.
- `colab-notebooks/x3d_baseline_evaluation.ipynb`: focused M3A evaluator with
  the pinned X3D contract, binary manifest loading, saved-run reuse/fresh-run
  options, and per-dataset metrics.
- `evaluation/manifests/manifest.json`: checked-in reviewed incident-manifest
  template, now UBI-Fights-scoped and intentionally pointing at a review
  placeholder rather than claiming available media.
- `README.md`, `engg-work/crowd_safety_codex_handoff/DATA_AND_MODELS.md`, and
  `EVALUATION_PLAN.md`: dataset roles, split policy, and evaluation boundary.

## 5. Important Design Decisions

- Original files are copied at most once into `datasets/<dataset>/`; split
  membership is metadata only. This prevents storage duplication and window
  leakage.
- Violent-Flows is forced to `external_test`; SCVD accepts only `Normal` and
  `Violence`, while `Weaponized Violence` is reported as excluded.
- Official splits are retained only when every discovered record has one;
  mixed layouts fail rather than silently guessing. Otherwise class-balanced
  deterministic 70/10/20 grouping uses seed `42`.
- Component metrics exclude unavailable windows instead of treating them as
  negative evidence. The fixed development threshold is recorded and
  external test is never calibrated.
- The rich incident manifest remains separate because binary labels cannot
  represent reviewed event intervals, reasoned scenarios, or operational
  outcomes.

## 6. Data and State Changes

No production database/schema/state changes. The Drive artifacts are:
`datasets/<dataset>/` originals and `manifests/{train,val,test,external_test}.json`.
The checked-in incident manifest is a review template only.

## 7. Failure and Recovery Paths

- Unsafe paths, unsupported labels, duplicate hashes/paths, source-group
  leakage, and ambiguous/mixed layouts raise before manifests are written.
- Missing source credentials/access report skipped status. SCVD and
  UBI-Fights Kaggle downloads and the Surveillance Fight GitHub clone are
  attempted only when their tools/access are available.
- New video copies stop before the lower of the 15 GB dataset budget and
  current Drive free space minus a 0.25 GB reserve; skipped files are reported
  and never enter a manifest.
- Existing destination media are reused only after layout discovery; when an
  import source is available, each matching destination file must also have
  the same SHA-256 or the import stops for explicit cleanup/replacement.
- Reused evaluation runs must carry the pinned model provenance and a config
  input path matching the manifest record; mismatched artifacts are rejected.
- Missing/unavailable X3D artifacts remain explicit and are excluded from
  scored negatives.

## 8. Frontend Behavior

Not applicable. No frontend files changed.

## 9. Production Considerations

This is Colab-only preparation/evaluation plumbing. Configure source paths,
Kaggle credentials, Drive mount, model dependencies, and optional run maps in
Colab. Do not commit raw media, archives, credentials, or checkpoints. The
workflow defaults to `RUN_WORKFLOW = False`; inference defaults are likewise
opt-in.

## 10. Verification Evidence

- Parsed all three edited notebooks as JSON and compiled their Python cells.
- Executed the dataset notebook's media-free self-check: deterministic
  splitting, invalid layout/label/path handling, duplicate hash detection,
  source-group leakage rejection, and Violent-Flows isolation.
- Executed the M6 binary-adapter fixture check for per-dataset metrics, missing
  evidence, and path/manifest handling.
- Executed the X3D notebook's contract and per-dataset scoring self-checks.
- Ran scoped `git diff --check` and searched the named cleanup scope for old
  dataset/layout references.
- Not run: full unittest suite, real Drive/Kaggle download, X3D inference,
  browser, Docker, deployment, and live incident smoke evaluation.

## 11. Risks and Assumptions

- UBI-Fights acquisition/layout is intentionally configured by the Colab user;
  the notebook refuses to infer labels from ambiguous filenames.
- Dataset class aliases may need one explicit registry update if an authorised
  source uses a different class-folder spelling.
- The checked-in UBI incident manifest is a placeholder until reviewed media
  and event intervals are supplied in Drive.
- M6 draft generation is intentionally limited to recursively discovered
  UBI-Fights clips and requires an approved binary label before review.
- Real Colab acceptance is still pending and may expose provider/layout or
  decode differences not visible to local contract checks.

## 12. Must-Read Code Before Production

- `dataset_download.ipynb`: `DATASET_REGISTRY`, `discover_records`, and
  `_split_groups` — these define label and leakage policy.
- `dataset_download.ipynb`: `prepare_sources` and `build_manifests` — these
  perform Drive writes and should be run only with authorised data.
- `m6_evaluation_demo.ipynb`: `load_binary_manifests`,
  `run_m3a_component_stage`, and `select_ubi_incident_manifest` — these are
  the component/system boundary.
- `x3d_baseline_evaluation.ipynb`: `collect_predictions` and `score_rows` —
  these define model-level result provenance and missing-score semantics.

## 13. Questions You Should Be Able to Answer

- Why can no Violent-Flows record enter training or validation?
- What causes a source layout to fail instead of receiving a guessed label?
- Which threshold is used for `external_test`, and where is calibration kept?
- Why are binary dataset manifests separate from the incident manifest?
- What does the evaluator do when a model window is unavailable?
- Which real Colab checks remain before these artifacts can be called accepted?
