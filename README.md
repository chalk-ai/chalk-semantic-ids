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
| Train next-video model | `src/train_next_video.py` | `artifacts/next_video_{id,semantic,hybrid}.pt` |
| Compare models | `src/evaluate_next_video.py` | `artifacts/next_video_metrics.json` |
| Export for serving | `src/export_next_video.py` | `artifacts/next_video.onnx`, proof that it ranks like torch |
| Serve recommendations | `src/deploy_next_video.py` | a Chalk function: watch history → 10 next videos |
| Check serving | `src/verify_next_video.py` | serving vs local agreement on real histories |
| Recommend from Chalk | [`docs/next-videos-from-chalk.md`](docs/next-videos-from-chalk.md) | Chalk SQL, chalkdf and streaming watch-history examples |

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
of the rows. The pairs file is one line per user, `<user id>\t<video ids in watch
order>`: 100,000 users, 719,405 watches, and no video watched twice by the same user.
The next-video model below trains on it.

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

## Next-video model

The semantic IDs exist so a recommender can use them. `src/train_next_video.py` trains a
small causal transformer (SASRec-shaped: 2 layers, 2 heads, 64-d, the last 50 watches)
that reads a user's watch history and scores every video as the next watch:

```
[watched video ids] ──▶ video embeddings + positions ──▶ causal transformer ──▶ h
                                                                                │
                        score(v) = h · embedding(v), for every video v ◀────────┘
```

The experiment is what a video's embedding is made of:

- **`id`**: one learned vector per video.
- **`semantic`**: the sum of one learned vector per semantic-ID token.
- **`hybrid`**: both. The per-video vector is used only for videos seen in training, and
  training drops it 30% of the time so the tokens have to carry signal on their own.

The 1,000 videos the RQ-VAE held out play **new uploads**: every watch of them is
removed from training, and they are left out of the training softmax, so no model
learns they exist. Each user's last watch is the test target (leave-one-out), ranked
on two surfaces:

| Model | Home feed: Recall@10 | NDCG@10 | New-uploads shelf: Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- |
| Random | 0.06% | 0.03% | 0.98% | 0.40% |
| Popularity | 0.23% | 0.11% | n/a | n/a |
| `id` | 8.62% | 4.87% | 0.69% | 0.28% |
| `semantic` | 3.76% | 2.13% | 4.30% | 2.10% |
| **`hybrid`** | **9.09%** | **5.04%** | **5.13%** | **2.32%** |

- **Home feed**: the last watch ranked against every video the user has not watched
  (100,000 users).
- **New-uploads shelf**: for the 4,912 users whose last watch was a new upload, that
  video ranked against the other new uploads, which have no plays for popularity to
  order by.

**ID embeddings cannot recommend a new upload**: their vectors never trained, so the
`id` model does no better than random on the shelf. **Semantic tokens can.** A new video
shares tokens with videos that were watched, and the `semantic` model places it 4.4x
better than random. On their own, though, three tokens are a lossy fingerprint for
19,738 videos (2.5% of them collide outright), so `semantic` trails on the feed. The
`hybrid` is best on both: per-video vectors where history exists, tokens where it does
not.

One honest gap: in the home feed, a new upload competes with videos that carry an
extra trained vector, and `hybrid` ranks the true new upload in its top 10 for only
0.12% of those users. A dedicated new-uploads surface, or score calibration between the
two kinds of video, is how new videos get exposure; the shelf numbers show the signal
is there.

### Serving it

`src/export_next_video.py` exports the hybrid to ONNX (10.5 MB, the video table baked in)
and checks it against torch on 5,000 real histories: identical top-10 lists.
`src/deploy_next_video.py` deploys it as the Chalk function `next-videos` with
onnxruntime and numpy, on 1 CPU and 2 GiB:

```
watched, most recent first:
    5092  spend 148r to achieve lamb chop free Grilled fish free! It was worth it!!
    3734  this life on her # fifth personality # fifth personality summer interpretation
    4769  psychological & patient new couple outfit! Poked in my heart! # 5th Personality
recommended:
   17090  🤡🤡🤡🤡🤡 # The Fifth personality # The Fifth Personality Summer Quirks
    9634  🐒# TheFifth personality # Thefifth Personality Summer Oddities
     905  # Fifth Personality
```

`src/verify_next_video.py` checks the deployed function against the local model:
30/30 histories identical on `oracle-gpu`. Scaled to zero, a first call took 7.0–17.3s
(median 10.5s), with the model ready 1.1s after the container's Python starts.
[`docs/next-videos-from-chalk.md`](docs/next-videos-from-chalk.md) calls it from Chalk
SQL, chalkdf, and a stream of watch events that keeps each user's history current.

## Next

- **Recommend uploads the function has never seen**: the served video table is fixed
  at export. Building rows for new videos from their tokens at serving time (the
  `semantic-id-embed` output) would put a fresh upload into the feed without a re-export.
- **Generative retrieval**: decode the next video's semantic ID token by token (TIGER)
  instead of scoring every video, which scales to catalogs too large to score in full.
- **Beyond titles**: the same quantizer accepts any content vector. MicroLens publishes
  extracted image/video/audio features, and the cover images and audio are downloadable
  per video; a multimodal vector would sharpen the clusters that titles alone blur.
