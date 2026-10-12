# Next-video recommendations from Chalk

`src/deploy_next_video.py` deploys the next-video transformer as a Chalk function:
a user's watched video ids in, the ten videos to show next out. Its catalog name is

```
function.next-videos
```

**The history goes most recent watch first.** That is the order Chalk's `max_by_n`
aggregation produces (see [Streaming](#streaming-resolver)); the function reverses it
before the model reads it. The model uses the last 50 watches and skips ids it was not
trained on. Output type: `list<int64>` (`BIGINT ARRAY` in SQL), already-watched videos
excluded.

Every snippet below was run against staging `oracle-gpu` (`chlkb5466073`), except the
one step marked untested; the runnable files are in [`chalk_project/`](../chalk_project).

## Before the first call

The engine reads the function catalog when it deploys. After deploying the function,
re-activate the engine's current deployment and check that the function is listed:

```bash
chalk deployment activate --deployment <deployment-id>
chalk sql "select qualified_name, output_type from chalk.information_schema.compute_functions
           where function_name = 'next-videos'"
```

## Chalk SQL

[`next_videos.sql`](../chalk_project/next_videos.sql):

```sql
SELECT
    user_id,
    catalog_call('function.next-videos', recent_video_ids) AS next_video_ids
FROM (
    VALUES
        (1, [17700, 15223, 6346, 1958]),
        (2, [10949, 9500, 3775, 9127, 915, 16657])
) AS t(user_id, recent_video_ids)
```

```
user_id  next_video_ids
1        [12441, 12275, 11207, 404, 19392, 4221, 10056, 3996, 10461, 18353]
2        [8596, 19091, 12820, 17039, 14446, 3459, 19540, 7278, 6631, 7766]
```

## chalkdf

[`next_videos_chalkdf.py`](../chalk_project/next_videos_chalkdf.py):

```python
import pyarrow as pa
from chalkdf import DataFrame

from chalk import functions as F
from chalk.features import _

histories = DataFrame.from_dict(
    {
        "user_id": [1, 2],
        "recent_video_ids": [[17700, 15223, 6346, 1958], [10949, 9500, 3775, 9127, 915, 16657]],
    }
)
recommended = histories.with_columns(
    {
        "next_video_ids": F.catalog_call(
            "function.next-videos", _.recent_video_ids, output_type=pa.large_list(pa.int64())
        ),
    }
)
print(recommended.run(remote=True).to_arrow())
```

```
user_id: [[1,2]]
next_video_ids: [[[12441,12275,11207,404,19392,...],[8596,19091,12820,17039,14446,...]]]
```

`run(remote=True)` runs the plan as a job in the environment named by
`CHALK_ENVIRONMENT`; `output_type` is needed because a local script has no catalog to
read the type from.

## Streaming resolver

A recommender needs each user's history as of their latest watch. In Chalk that is a
materialized windowed aggregation over watch rows that a stream resolver writes
([`src/watches.py`](../chalk_project/src/watches.py)):

```python
import pyarrow as pa
from pydantic import BaseModel

from chalk import Windowed, windowed
from chalk import functions as F
from chalk.features import DataFrame, FeatureTime, _, features, make_stream_resolver
from chalk.streams import KafkaSource


@features
class WatchEvent:
    id: str
    user_id: "User.id"
    video_id: int
    watched_at: FeatureTime


@features
class User:
    id: int
    watches: DataFrame[WatchEvent]
    # The 50 latest watches, most recent first: the order next-videos takes.
    recent_video_ids: Windowed[list[int]] = windowed(
        "30d",
        materialization={"bucket_duration": "1d"},
        expression=_.watches[_.video_id].max_by_n(_.watched_at, 50),
    )
    next_video_ids: list[int] = F.catalog_call(
        "function.next-videos", _.recent_video_ids["30d"], output_type=pa.large_list(pa.int64())
    )


class Watched(BaseModel):
    user_id: int
    video_id: int
    watched_at: int  # epoch seconds


watch_stream = make_stream_resolver(
    name="watch_stream",
    source=KafkaSource(name="watch_events", bootstrap_server="kafka.example.internal:9092", topic="watch-events"),
    message_type=Watched,
    output_features={
        WatchEvent.id: F.concat(
            F.concat(F.cast(_.user_id, str), ":"),
            F.concat(F.cast(_.video_id, str), F.concat(":", F.cast(_.watched_at, str))),
        ),
        WatchEvent.user_id: _.user_id,
        WatchEvent.video_id: _.video_id,
        WatchEvent.watched_at: F.from_unix_seconds(_.watched_at),
    },
)
```

Each watch message becomes one `WatchEvent` row, and Chalk updates the user's
`recent_video_ids` aggregate as the row is written. A query for `User.next_video_ids`
reads that aggregate and calls the model; nothing is computed per watch except the
aggregate update.

- **`max_by_n` is the aggregation to use** because it both orders by a column you choose
  and caps the length. `array_agg` keeps no order; `approx_top_k` returns the most
  frequent videos, not the latest.
- **The 30-day window bounds the history.** A user idle for more than 30 days starts
  over; `"all"` keeps every bucket instead.
- **`watched_at` is epoch seconds in the message.** The stream parser casts the JSON
  message to the message type in one step, and a JSON string does not cast to a
  timestamp (`Cannot cast JSON to ROW<..., watched_at:TIMESTAMP>`); convert with
  `F.from_unix_seconds` in `output_features` instead.
- **The event id includes the timestamp**, so a rewatch adds a row instead of
  overwriting the earlier one.

### What was run, and what was not

| Step | How | Result |
| --- | --- | --- |
| Message → `WatchEvent` rows | `test_streaming_resolver` on branch `semantic-ids` | 4 messages → 4 rows, ids `424242:1958:1791833572`, ... |
| History → `next_video_ids` | online query on the branch with `recent_video_ids["30d"]` given as input | `[12441, 12275, 11207, 404, ...]`, the same 10 videos as the SQL call |
| Rows → materialized `recent_video_ids` | [`test_watch_stream.py`](../chalk_project/test_watch_stream.py) (`check_stream_scenario`) | **untested**: `oracle-gpu` has no branch online store |

`check_stream_scenario` uploads each parsed message to the branch with materialized
aggregations updated, then asserts on `recent_video_ids` and queries `next_video_ids`.
It needs `chalkdf` installed and **Deploy Branch Online Store** enabled for the Branch
Server; without it the upload fails with `Writing to the online store is not supported
for branches by default`, and querying the aggregate fails with `the current online store
backend does not support pre-materialized window aggregations`.

## Cold start

Deployed with `--min-replicas 0` (1 CPU, 2 GiB), a call that wakes the function from
zero took 7.0–17.3s (median 10.5s over 6 calls, image already on the node). The model is ready
1.1s after the container's Python starts; the rest is platform start-up and routing,
as measured for the semantic-ID function in the README's "Replica cold start".
