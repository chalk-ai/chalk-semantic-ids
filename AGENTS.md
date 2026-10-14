# Customer semantic-ID example

Work in this repository on `rk/polish`, the owner's accepted working branch.
The target is FTQA (`tmnmc9beyujew`,
`https://api.staging.chalk.dev`), with feature branch `semantic`. The owner has
explicitly authorized adapting the relevant model and native training code from
`conference-demo/semantic_recommender` into this repository. Keep stage-specific
work in `train/`, `host/`, `materialize/`, `train-recommender/` and `serve/`, with
shared models and helpers in `semantic_ids/`. Document their source in `README.md`.

Do not copy source environment configuration, credentials, controller code,
comparison/promotion logic, synthetic-data generators or UI into this checkout.
This repository remains independently runnable and keeps its own Chalk target.

## Execution boundary

The owner has authorized local fixes, offline tests and targeted read-only
diagnostics. The owner also authorized importing the existing Demo-3 recommender
`semantic-demo-recommender` version 4 into FTQA for the working example. That
transfer completed as FTQA `semantic-id-recommender` version 1. The owner then
authorized one hosted recommendation request; it succeeded through branch
`semantic` with that version and excluded the supplied history IDs.

Additional remote inference, deployment, `chalk apply`, training, materialization
and other remote mutations still require explicit authorization. Do not claim
unrun checks passed. Use Chalk's normal source authentication in memory only; never print
credentials, persist them in receipts or alter the checkout's target to read the
source environment.

Do not manually modify receipts. Use the CLI to record resources and provenance.
The imported model's receipt is `.semantic-ids/imported-recommender.json`, separate
from the native training receipt `.semantic-ids/project.json`. Preserve both,
including while a materialization command may still be running.

## Customer-facing scope

The owner wants a small, reproducible semantic-ID walkthrough with all human
documentation in README.md. Keep internal run status, test-pass counts, environment
IDs, session timings and incident history out of the customer README; retain
honest data and model provenance. Do not imply that the complete hosted native
pipeline has been validated without evidence.

Fresh preparation defaults to 100 selected users and only their watched-video
catalog. `--max-users` changes the user count; `--full-catalog` opts into all source
titles. Preserve saved selections when preparation is repeated without overrides.
Tokenizer holdout defaults to 10% of selected titles, capped at 1,000, unless the
caller explicitly supplies a valid holdout. Preserve existing receipts and runs.

## Internal delivery record

The imported recommender works through the existing FTQA Function and Chalk
feature: one hosted request passed, and 160 offline tests passed on 2026-10-13.
Its bundled 1,600-video catalog and synthetic source training are accepted for this
example; document that provenance honestly. Do not turn this single hosted check
into a claim about repeated requests, throughput or cold-start latency. Full
19,738-row materialization and new recommender training are not prerequisites
for serving this model.

Keep the imported checkpoint paired with its own catalog and tokenizer lineage.
Do not attribute its codes or training to FTQA tokenizer version 1. The FTQA
tokenizer is already trained; do not retrain it to use the imported recommender.
`recommend --model-source imported` selects the imported model. The default `auto`
prefers a completed native recommender when present, otherwise the imported model.

## Physical pipeline layout

Keep each step's real implementation in its stage folder:

1. `train/prepare.py` owns data preparation and uploads; `train/run.py` owns the
   native tokenizer declaration and numerical training body.
2. `host/run.py` owns the literal Function decorators and hosted handlers;
   `host/verify.py` checks tokenizer consistency.
3. `materialize/run.py` owns the feature materialization query.
4. `train-recommender/run.py` owns the native recommender declaration and
   numerical training body.
5. `serve/run.py` owns recommendation queries; `serve/import_model.py` exposes
   model import using the shared import workflow.

Keep shared model classes at their `semantic_ids` import paths to preserve model
artifact compatibility. Retain shared data, settings, receipt, checkpoint,
recovery, CLI parsing and import-validation helpers. Existing module commands and
legacy `src/` launchers delegate to stage implementations; do not duplicate
numerical training work. Preserve useful historical analysis scripts and artifacts.

## Native-training pipeline

Preserve the reproducible commands and README instructions to:

1. Train a real RQ-VAE on MicroLens title content using native Chalk training.
2. Host inference with clear `@chalkcompute.function` decorators on Scaling Groups.
3. Call the tokenizer through `F.catalog_call_async` and materialize `video.codes`.
4. Train a small next-video recommender using that materialized semantic feature.
5. Serve recommendations through Chalk with explicit immutable model versions.

Function registration must use the visible `@chalkcompute.function(...)`
decorator directly above the typed function definition. Do not use functional
wrapping (`function(...)(handler)`), construct `RemoteFunction` directly, or call
`.deploy()` explicitly. Use `.wait_ready()` after decoration. Attaching to an
existing deployment with `RemoteFunction.from_name()` is allowed for invocation,
subject to the execution boundary above. Keep `authenticated=True` and
`chalk_identity=True`; the environment must support authenticated Function queues.
The queued call waits for the actual feature value.

Use `@chalkcompute.training` and `.run_training()` for both models. Use native
`checkpoint`, `last_checkpoint_path` and `log_metrics` inside training. The API
source of truth was the model-training-preview guide; current native documentation
is https://docs.chalk.ai/docs/model_training. Do not replace it with the legacy
`ChalkClient.train_model` API or a Function that runs a numerical training loop.

There is no agent loop, baseline comparison, promotion gate, new synthetic
scenario, custom UI, notebook or controller/state-volume framework. Keep held-out
evaluation, checkpoint restoration, version/data identity validation and offline
tests.

Keep all human-facing documentation in README.md. Do not create separate documentation pages.
The main deliverable is a complete how-to explaining semantic IDs and their Chalk pipeline. Track imported serving
and the optional full pipeline separately. An artifact import does not establish
hosted inference or target-environment training. Original artifacts record
historical tokenizer results, not downstream recommendation measurements.
