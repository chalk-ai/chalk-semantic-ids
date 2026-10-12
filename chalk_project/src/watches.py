"""Next-video recommendations as Chalk features, fed by a stream of watch events.

The stream writes one `WatchEvent` row per watch. `User.recent_video_ids` is a
materialized windowed aggregation over those rows: the user's 50 latest video ids, most
recent first, which Chalk keeps current as events arrive. `User.next_video_ids` hands
that list to the deployed `next-videos` function whenever a query asks for it.
"""

import pyarrow as pa
from pydantic import BaseModel

from chalk import Windowed, windowed
from chalk import functions as F
from chalk.features import DataFrame, FeatureTime, _, features, make_stream_resolver
from chalk.streams import KafkaSource

# `function.` + the name `deploy_next_video.py` deployed under.
NEXT_VIDEOS_FUNCTION = "function.next-videos"
HISTORY_LENGTH = 50


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
    # max_by_n keeps the n rows with the latest watched_at, most recent first: the
    # order the next-videos function takes.
    recent_video_ids: Windowed[list[int]] = windowed(
        "30d",
        materialization={"bucket_duration": "1d"},
        expression=_.watches[_.video_id].max_by_n(_.watched_at, HISTORY_LENGTH),
    )
    next_video_ids: list[int] = F.catalog_call(
        NEXT_VIDEOS_FUNCTION, _.recent_video_ids["30d"], output_type=pa.large_list(pa.int64())
    )


class Watched(BaseModel):
    user_id: int
    video_id: int
    # Epoch seconds: the native parser casts JSON to the message type, and a JSON string
    # does not cast to a timestamp.
    watched_at: int


watch_events = KafkaSource(
    name="watch_events",
    bootstrap_server="kafka.example.internal:9092",
    topic="watch-events",
)

watch_stream = make_stream_resolver(
    name="watch_stream",
    source=watch_events,
    message_type=Watched,
    output_features={
        # One row per watch; the timestamp keeps a rewatch from overwriting the earlier row.
        WatchEvent.id: F.concat(
            F.concat(F.cast(_.user_id, str), ":"),
            F.concat(F.cast(_.video_id, str), F.concat(":", F.cast(_.watched_at, str))),
        ),
        WatchEvent.user_id: _.user_id,
        WatchEvent.video_id: _.video_id,
        WatchEvent.watched_at: F.from_unix_seconds(_.watched_at),
    },
)
