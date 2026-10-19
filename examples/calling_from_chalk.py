"""Call the deployed semantic-ID model from Chalk three ways: Chalk SQL, chalkdf, and a stream resolver.

`src/deploy_function.py` deploys the model under the name `semantic-id-embed`. In the environment's catalog
that is `function.semantic-id-embed`, matching the fixed deployment name in the example.
It takes one title and returns its tokens, `[level 0, level 1, level 2]` (`list<int64>`).

This directory is a Chalk project, so the `Video` features and the stream resolver below deploy with it:

    cd examples
    chalk apply --branch semantic-ids
    python calling_from_chalk.py sql       # Chalk SQL
    python calling_from_chalk.py chalkdf   # chalkdf job
    python calling_from_chalk.py stream    # stream resolver, no Kafka needed
"""

from __future__ import annotations

import argparse
import subprocess

from chalk import functions as F
from chalk.features import _, features, make_stream_resolver
from chalk.streams import KafkaSource
from pydantic import BaseModel

SEMANTIC_ID_FUNCTION = "function.semantic-id-embed"

TITLES = [
    "grandma teaches me how to fold dumplings",
    "cat steals fish from the kitchen counter",
]


# --- 1. Chalk SQL -----------------------------------------------------------------------------------------
# `catalog_call` takes the catalog name as a string literal, then the function's arguments positionally.
# `timeout => INTERVAL '30 seconds'` and `on_error => 'struct'` are optional named arguments after them.
SEMANTIC_IDS_SQL = """
SELECT
    title,
    catalog_call('function.semantic-id-embed', title) AS semantic_id
FROM (
    VALUES
        ('grandma teaches me how to fold dumplings'),
        ('cat steals fish from the kitchen counter')
) AS t(title)
"""


def run_sql() -> None:
    subprocess.run(["chalk", "sql", SEMANTIC_IDS_SQL], check=True)


# --- 2. chalkdf -------------------------------------------------------------------------------------------
def run_chalkdf() -> None:
    import pyarrow as pa
    from chalkdf import DataFrame

    titles = DataFrame.from_dict({"title": TITLES})
    with_ids = titles.with_columns(
        {
            "semantic_id": F.catalog_call(
                SEMANTIC_ID_FUNCTION,
                _.title,
                # Required from a plain script: there is no catalog here to look the output type up in.
                output_type=pa.large_list(pa.int64()),
            ),
        }
    )
    # The plan runs as a job in the environment named by CHALK_ENVIRONMENT, which is where the catalog lives.
    print(with_ids.run(remote=True).to_arrow())


# --- 3. Stream resolver -----------------------------------------------------------------------------------
@features
class Video:
    id: int
    title: str
    # Computed once per upload message by the stream resolver and stored with the video, so reads never call
    # the model.
    semantic_id: list[int]
    # Computed when a query asks for it instead: covers titles that never came through the stream.
    semantic_id_on_read: list[int] = F.catalog_call(SEMANTIC_ID_FUNCTION, _.title)


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
        Video.semantic_id: F.catalog_call(SEMANTIC_ID_FUNCTION, _.title),
    },
)


def run_stream_test() -> None:
    """Run the stream resolver on the branch against two upload messages, without a Kafka cluster."""
    from chalk.client import ChalkClient

    messages = [VideoUploaded(video_id=100 + i, title=title) for i, title in enumerate(TITLES, start=1)]
    result = ChalkClient(branch="semantic-ids").test_streaming_resolver(
        resolver=video_upload_stream,
        message_keys=[str(message.video_id) for message in messages],
        # JSON text, not the models: pydantic bodies arrive as a JSON string and fail to parse.
        message_bodies=[message.model_dump_json() for message in messages],
    )
    print(result.message, result.errors)
    print(result.features.select("video.id", "video.title", "video.semantic_id"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=["sql", "chalkdf", "stream"])
    {"sql": run_sql, "chalkdf": run_chalkdf, "stream": run_stream_test}[parser.parse_args().example]()
