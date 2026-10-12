# Semantic IDs for MicroLens videos

Turns a short-video title into a small tuple of discrete tokens — a *semantic ID* —
so that videos with related content share a prefix, and a video nobody has ever
watched is addressable the moment it appears.

```
title ──▶ sentence encoder ──▶ 384-d vector ──▶ RQ-VAE ──▶ [37, 812, 41]
                                                              │
                                      videos sharing [37, 812] are related
```

The shape follows TIGER (Rajput et al., *Recommender Systems with Generative
Retrieval*): an MLP encoder into a small latent, then residual vector quantization,
where each level quantizes what the previous levels could not represent. That is what
makes the first token coarse and later tokens refining, and it is why a shared prefix
carries meaning.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Fetch data | `data/` (see below) | 19,738 titles, 100k user sequences |
| Embed titles | `src/embed_titles.py` | `artifacts/title_embeddings.npy` |
| Train quantizer | `src/train_rqvae.py` | `artifacts/rqvae.pt`, `artifacts/semantic_ids.parquet` |
| Measure | `src/evaluate_ids.py` | `artifacts/prefix_coherence.json` |
| Look | `src/inspect_prefix.py` | prefix groups, printed |
| Serve | `src/deploy_function.py` | a Chalk function: title → tokens |
| Prove cold start | `src/verify_cold_start.py` | serving vs local agreement on held-out videos |
| Torch-free serving | `src/export_numpy_runtime.py` | `artifacts/rqvae.npz`, proof that ONNX + numpy assigns identical IDs |
| Time cold start | `src/measure_cold_start.py` | seconds for a call that wakes a scaled-to-zero replica |
| Call from Chalk | [`docs/calling-from-chalk.md`](docs/calling-from-chalk.md) | Chalk SQL, chalkdf and stream-resolver examples |
| Recommend | `src/build_recommender.py` | next-video model over semantic IDs, `artifacts/recommender_eval.json` |
| Demo | `demo/deploy_demo.py` | a web page on a Chalk scaling group: encode a title, pick three videos, get the next one |

Every script is a `uv` script with inline dependencies — run them directly, no venv to
manage.

## Data

MicroLens-100K, from the [dataset site](https://recsys.westlake.edu.cn/). The directory
index returns 403, but the files fetch directly:

```bash
mkdir -p data && cd data
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"
curl -sS -A "$UA" -O https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_title_en.csv
curl -sS -A "$UA" -O https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_pairs.tsv
```

`MicroLens-100k_title_en.csv` is `<id>, <title>` with **unquoted** titles that contain
commas, so it must be split on the first comma only — `read_csv(sep=",")` shreds a third
of the rows. The pairs file (user → watch sequence) is downloaded for the sequential
model that comes next; nothing here consumes it yet.

## Results

Trained on 18,738 titles with 1,000 held out, 3 levels x 256 codes, 32-d latent:

| Metric | Value | Reading |
| --- | --- | --- |
| Codes used per level | 256 / 256 / 256 | no codebook collapse |
| Unique semantic IDs | 19,239 of 19,738 | 2.5% collide, broken by a 4th dedup token |
| Reconstruction MSE | 0.00151 train / 0.00173 holdout | small gap, so the codes generalize |
| Reconstruction cosine | 0.64 | 3 bytes keep much, not all, of a 384-d vector |
| **Intra-prefix cosine, 2 tokens** | **0.720** (0.622 on distinct titles) | vs **0.127** corpus baseline |
| **Intra-prefix cosine, 1 token** | **0.331** (0.322 on distinct titles) | same baseline |

Two honest caveats behind those numbers.

**Duplicates flatter the eyeball test.** MicroLens has 72 blank titles and 377 rows in
duplicate-title sets, and they dominate the largest prefix groups — so inspecting the
biggest group mostly proves the encoder maps identical strings to identical vectors.
Collapsing duplicates drops 2-token coherence from 0.720 to 0.622, still ~5x baseline.
The structure is real; just do not measure it on the duplicates.

**Depth trades coherence against coverage**, and at this corpus size the trade is
sharp:

| Prefix | Distinct prefixes | Median group | Videos with a neighbour | Coherence |
| --- | --- | --- | --- | --- |
| 1 token | 256 | 76 | 100% | 0.33 (2.5x baseline) |
| 2 tokens | 15,201 | 1 | 39% | 0.62 (5x baseline) |

256x256 is 65k cells for 20k videos, so most 2-token prefixes hold exactly one video.
For the "show me everything under this prefix" demo, **use one token** — 256 groups of
~76 videos, every video included. Two tokens is the right depth for tight
neighbourhoods, not for browsing. Shrinking the level-0 codebook (64 instead of 256)
would densify the 2-token groups if a deeper browsable prefix is what you want.

## Cold start, measured

`src/verify_cold_start.py` takes videos held out of RQ-VAE training, sends their titles
to the deployed function, and compares the tokens that come back to what the local model
assigns:

```
serving matches local assignment on 5/5 held-out videos
```

Exact agreement matters more than it looks: a mismatch means the serving path has
drifted from training — a different encoder revision, a stale checkpoint, or codebooks
reseeded at inference — and every ID minted in production would then be quietly wrong.

The neighbours it prints are the demo itself. An unseen title,
*"words hang this! Starchy sausage without meat yyds~"*, lands on prefix `[220, 147]`
alongside *"it's delicious to dip the sole of a shoe let alone a pork belly sizzling with
oil"* and *"Teacher Sha personally cooked for the big eagle to do luxury pilaf!"* — a
food cluster, reached with no retraining and no ID-embedding table.

## Replica cold start

"Cold start" above means a *video* nobody has watched. This section is about the other
cold start: a call that arrives while the function has no running replica. Measured on
staging `oracle-gpu` (`chlkb5466073`, GKE, 2 vCPU replicas on `e2-standard-32` nodes)
with `deploy_function.py --min-replicas 0`, which scales to zero after ~30s idle and
holds the first call until a replica serves it:

| Serving stack | Image | First pull onto a node | Cold call, image on node (median) | Trials |
| --- | --- | --- | --- | --- |
| torch from PyPI (CUDA wheels), lazy load | 9.1 GB | 2m43s; the held call hit the SDK's 60s timeout | 16.5s | 14.5 / 16.5 / 24.1 |
| `--cpu-torch`, lazy load | 839 MB | 33s (47s call) | 23.2s | 18.3 / 23.2 / 30.3 |
| `--cpu-torch --eager-load` | 839 MB | not measured | 13.1s | 11.5 / 11.7 / 13.1 / 14.2 / 16.4 |
| `--runtime onnx --eager-load` | 443 MB | 11s (18s call) | 8.1s | 6.5 / 7.3 / 8.1 / 8.4 / 10.7 |

Warm calls take ~100ms round trip from the measuring client, for every stack.

Where a torch cold call goes, from pod events and the replica's own log:

- **~3s** until the `main` container starts: the scale-up, pod scheduling, and a
  `chalkfs-fuse` sidecar that must pass its startup probe first.
- **~9s** to `import torch` and `sentence_transformers` and load the weights on 2 vCPUs.
  The imports are nearly all of it: loading both models takes 0.75s.
- **1–5s** from the server listening to the response arriving (up to ~20s for the lazy
  stacks, which also import inside the call). This is the platform's routing to a group
  waking from zero; it varies with when the route picks up the new pod, not with the model.

`--runtime onnx` removes the imports: MiniLM runs through the ONNX export in its own
Hugging Face repo (pinned revision) and the RQ-VAE encoder runs in numpy, so the
replica is ready to serve 0.8s after Python starts. `export_numpy_runtime.py` checks
this path against torch over all 19,738 titles, one title per call as served: 19,738
identical IDs, max embedding difference 3.2e-7. With the imports gone, the ~3s before
the container starts and the routing delay make up most of what is left, and neither is
in this repo's control.

- **`--cpu-torch` cuts the image 11x and the first pull 5x.** A CPU replica never loads
  the CUDA libraries that PyPI's default torch wheel bundles.
- **`--eager-load` loads at container start**, so the import cost overlaps pod start-up
  instead of following it.
- **HOST compute class rejects `min_replicas=0`** (`HOST scaling groups require 1 <=
  min_replicas`), so it cannot be compared here.
- **Raise the SDK's call timeout** (or keep `min_replicas=1`) for the CUDA image: a held
  first call to a node without it outlasts the 60s default.

## Demo: encode, then recommend

`demo/` serves one page from a Chalk scaling group. Type a title to see its embedding,
its three tokens and the catalog videos that share its prefix. Pick three watched videos
(catalog titles, or a title you just encoded) to get the ten most likely next videos.

```bash
./src/export_numpy_runtime.py     # artifacts/rqvae.npz
./src/build_recommender.py        # artifacts/recommender.npz, prints the evaluation below
./demo/deploy_demo.py --fonts-dir ~/chalk-private/marketing-frontend/public/fonts
```

The page uses the chalk.ai demo styling. `--fonts-dir` bakes Chalk's licensed Everett faces
into the image (they are not in this repository); without it the page uses system fonts.
`?embed=1` drops the header so the page can sit in an iframe.

The recommender counts which semantic-ID prefixes follow which across the 100,000
MicroLens watch sequences, and adds content similarity and per-video transition counts
(`src/recommender.py`). Each held-out user's last video is the target and the three
before it are the history, ranked against all 19,738 videos:

| Ranking | In top 10 | In top 50 | New videos, top 50 |
| --- | --- | --- | --- |
| Hybrid | 10.7% | 18.6% | 0.7% |
| Semantic IDs + content | 6.9% | 13.8% | 2.4% |
| Video-to-video transitions | 10.4% | 17.4% | 0.0% |
| Content only | 1.9% | 5.1% | 4.2% |
| Most popular | 0.7% | 1.8% | 0.0% |

"New videos" removes 10% of videos from training entirely and scores the 755 users whose
next video was one of them. The semantic-ID terms let a brand-new title take part as
*history*: it has an ID the moment it is encoded. As a *candidate*, a new video still
ranks best by content similarity; most 2-token prefixes hold one video, so the
transitions say little about a prefix nobody has watched yet.

## Serving

`src/deploy_function.py` deploys a `chalkcompute` function to the ftqa environment
(`tmnmc9beyujew`) that takes a title and returns its tokens. The title encoder and the
RQ-VAE checkpoint are both baked into the image, so a replica needs no Hugging Face
fetch on the critical path and works without egress.

Two things to know about the SDK:

- **`.remote()` is the wire call.** Calling the decorated function directly runs the
  body in your own process; here that fails with `No module named 'torch'`, since the
  deploy script's venv has no torch — an honest error that looks like a packaging bug.
- **`wait_ready()` defaults to 120s**, which a ~3GB image cannot meet from cold. This
  passes 1800s.

The codebooks are seeded lazily from real residuals during training, so the serving
path marks them initialized after loading the checkpoint; otherwise the first inference
vector would reseed them and every ID after that would be wrong.

## Next

- **Sequential model**: train SASRec or a T5-style decoder over `[user history → next
  semantic ID]` using `MicroLens-100k_pairs.tsv`, already downloaded.
- **Beyond titles**: the same quantizer accepts any content vector. MicroLens publishes
  extracted image/video/audio features, and the cover images and audio are downloadable
  per video; a multimodal vector would sharpen the clusters that titles alone blur.
- **Materialize as features**: expose `semantic_id_0..2` as Chalk features so the tokens
  are joinable at query time rather than living in a parquet file.
