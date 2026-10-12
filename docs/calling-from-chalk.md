# Calling the semantic-ID model from Chalk

`src/deploy_function.py` deploys the model as a Chalk function: a video title in,
three tokens out (`[level 0, level 1, level 2]`). Once deployed, Chalk SQL, chalkdf and
feature resolvers all reach it through the environment's catalog under one name:

```
function.semantic-id-embed
```

The name is `function.` plus the `--name` the function was deployed under, not the
Python function's name. Its output type is `list<int64>` (`BIGINT ARRAY` in SQL).

Every snippet below was run against the staging `oracle-gpu` environment
(`chlkb5466073`); the runnable files are in [`chalk_project/`](../chalk_project).

## Before the first call

The engine reads the function catalog when it deploys, so a function deployed after
the engine's last deploy is invisible to it (`Service '...' is not configured`).
Redeploy the engine, or re-activate its current deployment, then check the catalog:

```bash
chalk deployment list                               # take the latest SUCCESS id
chalk deployment activate --deployment <deployment-id>
chalk sql "select function_name, qualified_name, output_type
           from chalk.information_schema.compute_functions"
```

## Chalk SQL

`catalog_call` takes the catalog name as a string literal, then the function's
arguments positionally ([`semantic_ids.sql`](../chalk_project/semantic_ids.sql)):

```sql
SELECT
    title,
    catalog_call('function.semantic-id-embed', title) AS semantic_id
FROM (
    VALUES
        ('grandma teaches me how to fold dumplings'),
        ('cat steals fish from the kitchen counter'),
        ('words hang this! Starchy sausage without meat yyds~')
) AS t(title)
```

```
title                                                semantic_id
grandma teaches me how to fold dumplings             [38, 227, 102]
cat steals fish from the kitchen counter             [0, 32, 225]
words hang this! Starchy sausage without meat yyds~  [220, 147, 250]
```

Run it with `chalk sql --file semantic_ids.sql`. After the positional arguments,
`catalog_call` also accepts `timeout => INTERVAL '30 seconds'` and
`on_error => 'struct'`, which returns `struct<value, error, started_at, ended_at>` per
row instead of failing the query.

## chalkdf

`F.catalog_call` builds the same call as a DataFrame expression
([`chalkdf_example.py`](../chalk_project/chalkdf_example.py)):

```python
import pyarrow as pa
from chalkdf import DataFrame

from chalk import functions as F
from chalk.features import _

titles = DataFrame.from_dict(
    {"title": ["grandma teaches me how to fold dumplings", "cat steals fish from the kitchen counter"]}
)
with_ids = titles.with_columns(
    {
        "semantic_id": F.catalog_call(
            "function.semantic-id-embed",
            _.title,
            output_type=pa.large_list(pa.int64()),
        ),
    }
)
print(with_ids.run(remote=True).to_arrow())
```

```
title: [["grandma teaches me how to fold dumplings","cat steals fish from the kitchen counter"]]
semantic_id: [[[38,227,102],[0,32,225]]]
```

- **`run(remote=True)`** submits the plan as a job in the environment named by
  `CHALK_ENVIRONMENT`, which is where the catalog lives. A local `run()` has no
  catalog to call into.
- **`output_type`** is required from a plain script for the same reason: there is no
  catalog to look the type up in. Inside a resolver it can be left out.
- **Timing:** most of the ~4 minutes the example took was the job's own pod starting.
  The model calls were a small part of it.

## Streaming resolver

Put `F.catalog_call` in a stream resolver's `output_features` and the ID is computed
once per message and stored with the video, so reads never call the model
([`src/videos.py`](../chalk_project/src/videos.py)):

```python
from pydantic import BaseModel

from chalk import functions as F
from chalk.features import _, features, make_stream_resolver
from chalk.streams import KafkaSource


@features
class Video:
    id: int
    title: str
    semantic_id: list[int]


class VideoUploaded(BaseModel):
    video_id: int
    title: str


video_upload_stream = make_stream_resolver(
    name="video_upload_stream",
    source=KafkaSource(name="video_uploads", bootstrap_server="kafka.example.internal:9092", topic="video-uploads"),
    message_type=VideoUploaded,
    output_features={
        Video.id: _.video_id,
        Video.title: _.title,
        Video.semantic_id: F.catalog_call("function.semantic-id-embed", _.title),
    },
)
```

`test_streaming_resolver` runs the resolver on a branch against messages you supply,
so this can be checked without a Kafka cluster
([`test_stream.py`](../chalk_project/test_stream.py)):

```bash
cd chalk_project
chalk apply --branch semantic-ids
CHALK_ENVIRONMENT=<env id> python test_stream.py
```

```
video.id  video.title                       video.semantic_id
101       grandma teaches me how to fold…   [38, 227, 102]
102       cat steals fish from the kitch…   [0, 32, 225]
```

Pass the message bodies as JSON text (`model.model_dump_json()`): pydantic objects are
sent as a JSON string and every message fails with `json_to_struct_strict: Expected
JSON object at root`.

To compute the ID when a query asks for it instead of at ingest, declare it as an
expression feature. This also covers titles that never came through the stream:

```python
@features
class Video:
    id: int
    title: str
    semantic_id_on_read: list[int] = F.catalog_call("function.semantic-id-embed", _.title)
```

```bash
chalk query --branch semantic-ids --in video.id=1 \
  --in "video.title=cat steals fish from the kitchen counter" --out video.semantic_id_on_read
# video.semantic_id_on_read  [0,32,225]
```

## Notes

- **Branch runtime.** A branch must run the same Python runtime as the environment's
  main deployment. `chalk_project/chalk.yml` says `python311` to match `oracle-gpu`;
  change it to match your environment.
- **One title per call.** The deployed function takes one title per call. Rows of one
  SQL or DataFrame batch reach it as separate calls. For backfills over many titles,
  deploy a batched variant (a `list[str]` argument with `max_batching_size`).
