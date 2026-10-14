# How to use semantic IDs with Chalk

Train a tokenizer that turns video titles into discrete semantic IDs, expose
those IDs as a Chalk feature, and train a recommender that generates them. This
repository contains the models, native Chalk training jobs, hosted Functions and
feature definitions needed for that pipeline. The folders follow the five steps:

```text
train/              prepare.py uploads data; run.py trains the tokenizer
host/               run.py deploys Functions; verify.py checks tokenizer consistency
materialize/        run.py computes video.codes through the hosted tokenizer
train-recommender/  run.py trains the downstream recommender
serve/              run.py queries recommendations; import_model.py reuses a model
semantic_ids/       shared model classes, data, configuration and recovery helpers
features.py         Chalk feature graph
```

Each stage's `run.py` contains that stage's implementation. Run these files from
the repository root; the shared package keeps model artifact imports stable.

## How semantic IDs work

A video normally has an arbitrary database ID. Here it also gets a short sequence
of tokens learned from its title, so videos with related content can share parts
of their representation.

1. **Encode the title.** A pinned MiniLM sentence encoder maps the title to a
   normalized, 384-dimensional vector.
2. **Learn three codes.** An RQ-VAE—a residual-quantized VQ-VAE—compresses that
   vector into a 32-dimensional latent. Three codebooks, each with 256 entries,
   successively approximate the latent and the residual left by the previous
   level. The selected entry indices form three learned tokens.
3. **Make each catalog entry unique.** Different videos can receive the same
   three tokens. Catalog construction appends a fourth token assigned by sorted
   video ID within each collision group. This token is deterministic, not learned.
4. **Learn what comes next.** A small GRU recommender reads the viewer's ordered
   history and learns to generate the next video's four-token sequence.
5. **Return an actual video.** Constrained decoding permits only token sequences
   in the model's bundled catalog, maps them back to video IDs and titles, and
   excludes previously watched videos.

The RQ-VAE learns by reconstructing the input title embeddings from its quantized
representation. Its training objective combines reconstruction error with a
quantization loss that aligns the latent vectors and codebook entries. MiniLM
stays fixed during this training. Learning which video follows a watch history is
a separate task handled by downstream recommender training.

Token numbers are learned codebook indices, not named topics. Shared prefixes can
capture content similarity, but that alone does not establish recommendation
quality. A new title can be encoded by the tokenizer; adding its video to the
recommender's fixed catalog requires a compatible training artifact.

```text
Video title → MiniLM embedding → RQ-VAE → three learned codes
                                                ↓
                                   deterministic collision token
                                                ↓
                                      four-token catalog ID
                                                ↓
Ordered watch history → GRU → constrained token generation → next videos
```

Chalk provides the operational path: Datasets supply training data, native
training produces registered models, Scaling Groups run inference, and
`F.catalog_call_async` exposes that inference as feature values.

## Before running

Use Python 3.12 or 3.13, `uv`, the Chalk CLI, and an environment with Datasets,
native Model Training, Model Registry, Functions/Scaling Groups and feature
deployment enabled. Authenticated Function queues and queue autoscaling must be
provisioned; ask Chalk to confirm `function_queue_auth_enabled` for your
environment. Training and inference request 2 CPUs and 8 GiB per worker.

Run commands from the repository root:

```bash
uv sync
chalk config
```

Configure your normal Chalk credentials, set `project` in [chalk.yaml](chalk.yaml)
to your project's name, and read the Environment ID and API Server from
`chalk config`. `semantic-ids` in that file is a placeholder. Use your target in
step 1; subsequent Python commands reuse it and reject conflicting environment
variables. `chalk apply` uses the CLI configuration, which must match that target.

The five stages below use feature branch `semantic` consistently. You can omit
`--branch semantic` from both `chalk apply` and queries to use the default feature
deployment. Datasets, native runs and registered models belong to the environment;
the branch selects the feature graph used by a query.

**Current verification:** 143 offline tests pass, native tokenizer training has
completed, and an imported recommender has served a hosted request. The complete
MicroLens materialization and downstream training path remains incomplete.
See [verification](#verification) for the measured results and limitations.

## 1. Prepare data and train the tokenizer

```bash
uv run python train/prepare.py \
  --environment YOUR_ENVIRONMENT_ID \
  --api-server https://YOUR_CHALK_API_HOST

uv run python train/run.py
```

[train/prepare.py](train/prepare.py) reads the included [MicroLens titles](data/MicroLens-100k_title_en.csv)
and [watch sequences](data/MicroLens-100k_pairs.tsv), then uploads two named Chalk
Dataset revisions. It prints their names, revisions and row counts. Titles may
contain unquoted commas, so parsing splits on the first comma only.

By default, preparation samples 5,000 users. `train/prepare.py --max-users N`
changes the watch-history sample; it does not reduce the title catalog. Each user's earlier
watch positions become training examples, the penultimate target becomes
validation, and the last target becomes test. MicroLens provides sequence order,
not event timestamps. Catalog content can include titles from held-out
interactions; their next-watch labels are excluded from recommender optimization.

[train/run.py](train/run.py) defines the native `@training` function and its
numerical training body; the command submits it using `.run_training()`. MiniLM
embedding computation and RQ-VAE optimization happen in Chalk. The default is 400 epochs
with 1,000 held-out titles; for a smaller title Dataset, choose `--holdout N` with
`N` smaller than its row count. The MiniLM revision is pinned in
[settings.py](semantic_ids/settings.py).

Chalk owns scheduling, retries and checkpoint storage. Recovery saves weights,
optimizer state, epoch, random state and quantizer initialization flags. Before
restoring, the runtime verifies run, configuration and data identity. The final
artifact is registered as `semantic-id-tokenizer` with an immutable version.

**Continue when:** the command prints `Registered semantic-id-tokenizer version`
and reconstruction metrics. The printed native run ID identifies its logs and
metrics in Chalk's Training page. Rerunning the command reconnects to its saved
run and original configuration; it does not start a new experiment.

## 2. Host inference on Scaling Groups

```bash
uv run python host/run.py
```

This registers `semantic-id-tokenize` and `semantic-id-recommend`, provisions their
Scaling Groups and waits for readiness. The CLI requires a registered tokenizer
before deployment. Model loading is lazy, so the recommender Function can become
ready before its model is trained in step 4.

[host/run.py](host/run.py) puts literal `@chalkcompute.function(...)`
decorators directly above the typed handlers. For example, its tokenizer handler
passes the explicit version to [hosting.py](semantic_ids/hosting.py):

```python
@chalkcompute.function(
    name=TOKENIZER_FUNCTION, image=image, cpu="2", memory="8Gi",
    min_replicas=0, max_replicas=2, chalk_identity=True, authenticated=True,
    target_queue_depth=1,
    call_timeout=1800, env=runtime_env,
    concurrency=2,
)
def semantic_tokenize(title: str, model_version: int) -> list[int]:
    from semantic_ids.hosting import tokenize_title

    return tokenize_title(title, model_version)
```

The image and environment are defined in that module. Decoration initiates
registration; `.wait_ready()` waits for it, so import this deployment module only
when deploying. `RemoteFunction.from_name()` attaches to an existing Function
for invocation.

Both model images install `torch==2.9.1+cpu` before the remaining pinned packages.
The serving image downloads the pinned encoder during its build and saves it at
`/opt/encoder`; Hugging Face offline flags are applied only at runtime. Inference
therefore uses the packaged encoder. Workers use Chalk identity to download the
requested registered model version and cache it per replica.

Functions scale from zero to two replicas. `target_queue_depth=1` lets queued
feature calls start workers when queue autoscaling is configured. Startup and
artifact downloads can add first-request latency; image build time and cold-start
latency have not been benchmarked.

**Continue when:** `Ready: semantic-id-tokenize and semantic-id-recommend` prints.

## 3. Materialize semantic IDs as a Chalk feature

Apply the feature graph to the same environment selected in step 1, then try five
titles through the hosted tokenizer:

```bash
chalk config
chalk apply --branch semantic
uv run python materialize/run.py --branch semantic --limit 5
```

`--limit` selects the first N title rows before submitting inference. The SDK still
reads the title Dataset, but only those rows reach the tokenizer. The command
prints the returned features and times the Dataset read separately from the Chalk
query, whose elapsed time includes queueing, startup, inference and download.
Sample results use a separate Dataset and do not replace the saved training
catalog. A sample is a diagnostic, not a smaller recommender-training run.

To build the training catalog, omit the limit:

```bash
uv run python materialize/run.py --branch semantic
```

**Materialization runs inference with the already-trained tokenizer. It does not
train either model.** This is the integration in [features.py](features.py):

```python
import pyarrow as pa
import chalk.functions as F
from chalk.features import features, _

@features
class Video:
    id: int
    title: str
    tokenizer_version: int
    codes: list[int] = F.catalog_call_async(
        "function.semantic-id-tokenize", _.title, _.tokenizer_version,
        output_type=pa.list_(pa.int64()),
    )
```

[materialize/run.py](materialize/run.py) submits the offline query with IDs,
titles and the immutable tokenizer version, and requests `video.codes`. Chalk queues the hosted Function calls and waits for their
results. PyTorch and the encoder run on the Scaling Group; the query worker does
not load them. Both Functions retain authentication and Chalk identity.

The command checks coverage and tokenizer version, saves feature values in a
Chalk Dataset, and records its revision. The saved feature has three learned
codes; catalog construction for recommender training derives the fourth collision
token. A completed full materialization is reused on subsequent calls.

Feature deployment uses [requirements-query.txt](requirements-query.txt), allowing
Chalk's query runtime to resolve its compatible PyArrow version. Local development
and model images have their separate pinned dependencies.

**Continue when:** `Materialized` prints with the feature count and Dataset
revision. Inspect that Dataset for titles, `video.tokenizer_version` and three
`video.codes` values per video.

## 4. Train a recommender from the materialized feature

```bash
uv run python train-recommender/run.py
```

[train-recommender/run.py](train-recommender/run.py) contains the second native
training definition and numerical body. It receives the watch-example Dataset
and reads the pinned semantic-feature Dataset. It joins histories and targets to the
catalog by video ID, then trains the GRU's embeddings and decoder for 10 epochs
by default. Training examples preserve chronological histories, and validation
and test labels stay outside optimization.

The job reports validation and test Recall@10, NDCG@10 and token loss. Recall@10
measures how often the next video appears in the top ten; NDCG@10 rewards placing
that video nearer the top. Loss and Recall appear in the training dashboard;
the full report, including NDCG, is stored with the artifact. These measurements
do not establish online engagement lift.

The completed artifact includes the weights and exact token-to-video catalog,
and is registered as `semantic-id-recommender`. Checkpoint recovery retains
optimizer, epoch and random state, just as tokenizer training does.

**Continue when:** `Registered semantic-id-recommender version` prints with the
training, validation and test metrics.

## 5. Query recommendations through Chalk

```bash
uv run python serve/run.py \
  --branch semantic --model-source trained --history 16657 915
```

[serve/run.py](serve/run.py) queries `recommendation.result`, which uses
`F.catalog_call_async` to call
the hosted recommender with the saved model version, history and requested result
count. The JSON response contains video IDs, titles, semantic tokens and scores;
watched videos are excluded. Scores are summed token log probabilities, not
click-through probabilities.

Histories are ordered oldest to newest. Change `--history` to other IDs from the
model's catalog; the command accepts up to 500 history IDs. Serving defaults to
six results, and `--top-k` accepts 1–20. Evaluation remains at ten results.
Encoding additional titles does not extend this model's bundled catalog.

## Option: serve an already-trained recommender

You can import compatible weights together with their original catalog and skip
new materialization and recommender training. This path requires a saved target,
an existing `semantic-id-recommend` Function and an applied feature graph.
**For a fresh environment, the current CLI still requires tokenizer training and
registration before `host/run.py`; importing a recommender does not bypass that gate.**
There is no separate initialization or recommender-only deployment command.

The verified source below is private Demo-3: access to its environment and model
is required. Configure source access using normal Chalk credentials without
changing the saved destination. In the existing FTQA checkout, import is already
complete; go directly to the recommendation command.

```bash
uv run python serve/import_model.py \
  --source-environment clk4b6a3d6c \
  --source-api-server https://api.chalk.ai \
  --source-model semantic-demo-recommender \
  --source-version 4 \
  --dry-run
```

The dry run downloads the source artifact and validates its catalog, checkpoint
and local recommendations without registering a target model or changing receipts.
Rerun the same command without `--dry-run` to register it as a new destination
version of `semantic-id-recommender`. Source and destination version numbers can
differ. Registration preserves source provenance; it does not create a target
training run. Repeating a completed import reuses its saved version.

```bash
uv run python serve/run.py \
  --branch semantic --model-source imported --history 1001897 1004997
```

This source model contains 1,600 videos and was trained natively in Chalk on
35,000 **synthetic** watch examples, using its own tokenizer version 2. Its codes
are not the output of FTQA tokenizer version 1. It can recommend from its original
catalog, not the complete 19,738-title MicroLens catalog. The source synthetic
Dataset and generator are not included here, so the native commands above train
a new MicroLens model rather than reproduce v4's weights or metrics.

| `--model-source` | Selected model |
| --- | --- |
| `trained` | Requires a completed native recommender recorded by this checkout. |
| `imported` | Requires the separately recorded imported model. |
| `auto` (default) | Prefers a completed native recommender; otherwise uses the imported model. |

## Progress and recovery

```bash
uv run python -m semantic_ids.cli status
```

This reads local receipts without contacting Chalk. Native recommender “not
submitted” is expected when serving only an imported model. Keep both receipts:

| Receipt | Contents |
| --- | --- |
| `.semantic-ids/project.json` | Target, source identities, Dataset revisions, native run IDs and trained model versions |
| `.semantic-ids/imported-recommender.json` | Imported source, target version, catalog identity and provenance |

Neither contains credentials or model weights. Authoritative Datasets, runs,
checkpoints and models live in Chalk. Do not manually edit receipts or remove them
to retry a command: that can lose the connection to existing resources. Import
uses its separate receipt while a materialization may still be writing the main
one; importing does not cancel a separately running materialization.

### Interrupted native commands

| Situation | Action |
| --- | --- |
| Preparation or observation of a known training run was interrupted | Rerun the same command; completed uploads or the saved native run are reused. |
| Submission began but no run ID was saved | Inspect Chalk's Training page; use `train/run.py --run-id EXISTING_RUN_ID` if a run exists, or `--retry-submission` only if none exists. |
| Training finished but model registration was interrupted | Inspect the registry; use `train/run.py --model-version EXISTING_VERSION` for the matching artifact, or `--retry-registration` only if no version exists. |
| A startup failure needs a new run after its cause is fixed | Use `train/run.py --restart-failed`; it requires a confirmed failed run with no checkpoint or publication, preserves its history, and resubmits the saved configuration. |
| The target conflicts with the environment or API server | Compare the receipt, `chalk config` and environment variables; restore the intended target before retrying. |

Prefix those paths with `uv run python`; use `train-recommender/run.py` for a
recommender run. Reconnection verifies the completed
artifact's run, configuration and Dataset identities before registering it. A
mistyped external run ID is not saved before validation. Running, completed or
checkpointed runs cannot be replaced with `--restart-failed`.

For an interrupted import, rerun its original source arguments with
`--model-version EXISTING_VERSION` to verify and attach the target version.
Use `--retry-registration` only after confirming no target version was created.
Pass these flags to `serve/import_model.py`, not a native training script.

To change data or training configuration, use a separate checkout and prepare a
new experiment. Changing flags while reconnecting keeps the original saved
training configuration. Dataset and checkpoint names are scoped per experiment.

### Hosted feature troubleshooting

| Symptom | What to check |
| --- | --- |
| `CallFunction failed (unauthenticated): Jwt is missing` | Apply the current queued `F.catalog_call_async` feature graph to the queried branch, confirm Function-queue authentication is provisioned, and retain `authenticated=True`. Existing training and Functions can be reused. |
| Queued requests remain pending at zero replicas | Confirm `target_queue_depth=1`, environment queue autoscaling and worker readiness. Redeploy outdated Function definitions with `uv run python host/run.py`; workers for the same Function names can consume pending calls that have not expired. |
| Offline query heartbeat timeout | Inspect query logs and Scaling Group health; elapsed waiting is not proof of row-level progress. The recorded tokenizer worker crashes and full-query failure remain under investigation. |

## Verification

The following results were recorded on **2026-10-13**, using FTQA environment
`tmnmc9beyujew`, API `https://api.staging.chalk.dev`, and feature branch `semantic`.
They distinguish a working imported serving path from the incomplete full
MicroLens training path.

| Check | Recorded result |
| --- | --- |
| Offline tests | **143 passed**; network access is blocked by test fixtures. |
| Prepared Datasets | 19,738 video titles and 31,020 recommendation examples |
| Native tokenizer | `semantic-id-tokenizer` v1; run `ac4437a4-b137-4899-9066-f0498aa58542`; 400 epochs; holdout reconstruction MSE **0.001736** |
| Functions and features | Both Functions registered; feature graph applied to `semantic` |
| Downstream serving | Demo-3 `semantic-demo-recommender` v4 imported as FTQA `semantic-id-recommender` v1; one hosted request succeeded |

Full MicroLens materialization has not been verified complete, and its downstream
recommender has not been trained. An earlier full query failed with a heartbeat
timeout; a tokenizer worker separately restarted with exit code 139. The native
crash cause and its relationship to the query timeout remain unconfirmed. The
imported serving success does not establish that tokenizer materialization is fixed.

The imported artifact passed local validation and the FTQA download matched the
source bytes. A hosted request using history `1001897 1004997` returned
`execution: chalk`, `model_version: 1` and six results with titles, tokens and
scores, excluding both history IDs. Its first result was `4576766`, “Travel:
Beginner guide to city walking tours.” No new training, deployment, feature apply
or materialization was needed for that import and query. Repeatability, throughput
and cold-start latency were not measured. The source catalog fingerprint is
`d8f4674788e8c7c96363fbd731c5102a43c77bfe9f087cd0b030e8c446d11572`.

Run the offline suite with:

```bash
uv run pytest
```

Tests cover parsing and splits, decoding and metrics, model/catalog identity,
interrupted numerical recovery, actual SDK checkpoint serialization and pointers,
worker import order, registration recovery, queued feature serialization, serving
configuration, import provenance and branch routing. They do not invoke live Chalk.
Keep [uv.lock](uv.lock) with the release; installation in a fresh checkout remains
a separate check.

To compare a hosted tokenizer with its exact registered checkpoint:

```bash
uv run python host/verify.py
```

This downloads the checkpoint and pinned encoder, computes local reference codes
for held-out titles, and invokes the hosted tokenizer. It checks consistency;
it does not measure recommendation quality or cold-start latency.

For full hosted validation, complete the five stages and verify the final
checkpoint/version for both runs, the exact saved materialization revision and
tokenizer version, held-out metrics, and recommendations excluding watched items.
For another imported release, validate the artifact, register it, query its
explicit target version and confirm repeat calls use that same saved version.
Record the tested commit and dependency versions alongside those results.

## Code and training source

| Stage | Files and implementation |
| --- | --- |
| 1. Train the tokenizer | [train/prepare.py](train/prepare.py) uploads the Datasets; [train/run.py](train/run.py) defines and runs native tokenizer training. |
| 2. Host inference | [host/run.py](host/run.py) declares and deploys both Functions; [host/verify.py](host/verify.py) compares hosted and local tokenizer codes. |
| 3. Materialize features | [materialize/run.py](materialize/run.py) owns the offline query; [features.py](features.py) declares the lightweight feature graph. |
| 4. Train the recommender | [train-recommender/run.py](train-recommender/run.py) defines and runs native recommender training. |
| 5. Serve recommendations | [serve/run.py](serve/run.py) owns the online query; [serve/import_model.py](serve/import_model.py) invokes the model-import workflow. |

The shared `semantic_ids/` package contains [model classes](semantic_ids/model.py),
[RQ-VAE](semantic_ids/rqvae.py), [data helpers](semantic_ids/data.py),
[pinned settings](semantic_ids/settings.py), [artifact loading](semantic_ids/hosting.py),
[checkpoint helpers](semantic_ids/training_runtime.py), [receipts](semantic_ids/project.py),
and [import validation](semantic_ids/importing.py). Model classes remain at their
existing import paths so registered artifacts continue to load.

[cli.py](semantic_ids/cli.py) retains argument parsing, native submission/recovery
orchestration, registration and local status. Existing `python -m semantic_ids.cli`
commands still work and dispatch to the same stage implementations.
[import_cli.py](semantic_ids/import_cli.py) maintains the separate import receipt.
[semantic_ids/__init__.py](semantic_ids/__init__.py) marks the package and has no
runtime behavior.

The relevant training code was adapted from `conference-demo/semantic_recommender`
and runs independently here. This is the source mapping, verified on 2026-10-13:

| Source path in that package | Local adaptation |
| --- | --- |
| `rqvae.py` | [rqvae.py](semantic_ids/rqvae.py), byte-identical at comparison time |
| `model.py` | [model.py](semantic_ids/model.py), retaining semantic training, decoding and save/load; removes ordinary-ID baseline support and adds stricter validation |
| `training.py` | [train/run.py](train/run.py) and [train-recommender/run.py](train-recommender/run.py) hold native definitions; [cli.py](semantic_ids/cli.py) handles submission, reconnects and registration. |
| `training_runtime.py` | Numerical bodies live in the two training stage files; [shared helpers](semantic_ids/training_runtime.py) retain checkpoint, configuration and identity handling plus compatibility delegates. |
| `tests/test_training_checkpoints.py` | [test_training_checkpoint_sdk.py](tests/test_training_checkpoint_sdk.py), offline tests using the SDK serializer, checkpoint layout and latest pointer with remote calls mocked |

The source controller, comparison/promotion workflow, synthetic generator, UI,
configuration and credentials are not part of this implementation.

## Adapt this to your data

Replace the MicroLens parsers with your source integration, retain stable video
IDs, and define chronological training/evaluation boundaries. `Video.codes` is the
feature boundary: join its codes into your examples and embed each position in
your downstream model. The included GRU is one consumer of those codes.

Keep the tokenizer version, semantic-feature revision and training catalog paired.
Titles must contain at most 2,000 characters; preparation rejects longer titles
before upload. The included data comes from [MicroLens](https://recsys.westlake.edu.cn/);
review its terms before redistribution or another use.

<details>
<summary>Historical local tokenizer experiment</summary>

The original experiment trained on 18,738 MicroLens title embeddings and held
out 1,000. It used 384-dimensional MiniLM vectors, a 32-dimensional latent and
three 256-entry codebooks, following the residual-quantization approach used in
TIGER (Rajput et al., <em>Recommender Systems with Generative Retrieval</em>).
Watch sequences were available but were not used to train a downstream model.

| Metric | Historical result |
| --- | --- |
| Codes used per level | 256 / 256 / 256 |
| Unique three-token IDs | 19,239 of 19,738; approximately 2.5% collide |
| Reconstruction MSE | 0.00151 train / 0.00173 holdout |
| Reconstruction cosine | 0.64 |
| Mean cosine within a two-token prefix | 0.720; 0.622 after filtering blank and duplicate-title rows |
| Mean cosine within a one-token prefix | 0.331; 0.322 after filtering |
| Corpus cosine baseline | 0.127 |

There were 72 blank titles and 377 rows in duplicate-title sets. Filtering
excluded every row whose title appeared more than once, rather than keeping one
representative. Identical titles inflate similarity because their embeddings match.

| Prefix length | Distinct prefixes | Median group size | Videos with a neighbor |
| --- | --- | --- | --- |
| One token | 256 | 76 | 100% |
| Two tokens | 15,201 | 1 | 39% |

Two-token groups were more similar but mostly singletons; one-token groups were
more useful for inspecting broad neighborhoods. The original deployment matched
local checkpoint codes for five held-out videos. These are historical tokenizer
measurements, not recommendation metrics or proof of the current pipeline.

Original [artifacts/](artifacts/) preserve those outputs. Their weights lack the
encoder provenance required by current hosting. The current loader restores
quantizer initialization so inference does not reseed trained codebooks.
[src/train_rqvae.py](src/train_rqvae.py),
[src/deploy_function.py](src/deploy_function.py) and
[src/verify_cold_start.py](src/verify_cold_start.py) remain compatibility launchers
for the stage implementations, without duplicate numerical training code.
[src/embed_titles.py](src/embed_titles.py), [src/evaluate_ids.py](src/evaluate_ids.py)
and [src/inspect_prefix.py](src/inspect_prefix.py) remain useful local tools for
these historical artifacts.

</details>

References: [Model Training](https://docs.chalk.ai/docs/model_training),
[Functions](https://docs.chalk.ai/docs/compute/functions),
[queued catalog calls](https://docs.chalk.ai/docs/model_deployments),
[Datasets](https://docs.chalk.ai/docs/datasets).
