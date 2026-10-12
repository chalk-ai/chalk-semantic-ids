"""Drive watch events through the stream on a branch, then check the history and the recommendations.

    chalk apply --branch semantic-ids
    CHALK_ENVIRONMENT=<env id> python test_watch_stream.py

`check_stream_scenario` parses each message with the stream resolver, uploads the rows to
the branch with materialized aggregations updated, and runs the assertions as online
queries. It needs `chalkdf` installed and a branch online store in the environment.
"""

import json
from datetime import UTC, datetime, timedelta

from chalk.client import ChalkClient
from chalk.testing import FeatureAssertion, StreamMessage

from src.watches import User, watch_stream

USER_ID = 424242
# User 1's real watch sequence from MicroLens, minus its last video.
HISTORY = [1958, 6346, 15223, 17700]

client = ChalkClient(branch="semantic-ids")
start = datetime.now(UTC) - timedelta(hours=1)
messages = [
    StreamMessage(
        message=json.dumps(
            {"user_id": USER_ID, "video_id": video, "watched_at": int((start + timedelta(minutes=i)).timestamp())}
        ).encode(),
        stream_resolver=watch_stream,
        timestamp=start + timedelta(minutes=i),
    )
    for i, video in enumerate(HISTORY)
]
client.check_stream_scenario(
    *messages,
    FeatureAssertion(input=User(id=USER_ID), output=User(recent_video_ids={"30d": HISTORY[::-1]})),
    show_table=True,
)

result = client.query(input={User.id: USER_ID}, output=[User.recent_video_ids["30d"], User.next_video_ids])
print("recent_video_ids", result.get_value(User.recent_video_ids["30d"]))
print("next_video_ids  ", result.get_value(User.next_video_ids))
