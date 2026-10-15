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
| Call from Chalk | [`examples/`](examples/README.md) | Chalk SQL, chalkdf and a stream resolver calling the function |

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
- **Materialize as features**: [`examples/`](examples/README.md) computes the tokens as a
  Chalk feature, from a stream of uploads or at query time. Splitting them into
  `semantic_id_0..2` would make each level joinable on its own.
