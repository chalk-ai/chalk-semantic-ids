"""Calling the deployed semantic-ID function from Chalk: SQL, chalkdf, and a stream resolver.

`src/deploy_function.py` deploys the model as a Chalk function named `semantic-id-embed`
(title in, `[level 0, level 1, level 2]` tokens out). Inside Chalk it is reachable as

    function.semantic-id-embed

that is, `function.` plus the deploy's `--name`, not the Python function's name.

This file is two things at once:

- A Chalk module. `chalk apply` (run from this directory) deploys the `Video` features
  and the stream resolver below.
- A script. Each subcommand runs one example against the environment named by
  `CHALK_ENVIRONMENT`:

      python semantic_ids_in_chalk.py sql       # Chalk SQL
      python semantic_ids_in_chalk.py chalkdf   # chalkdf DataFrame, run as a job
      python semantic_ids_in_chalk.py stream    # stream resolver, on a branch
      python semantic_ids_in_chalk.py query     # read semantic IDs back by online query

`stream` and `query` run on the Chalk branch `semantic-ids-examples` (deploy it with
`chalk apply --branch semantic-ids-examples`; `--branch` picks another).

The engine reads its function catalog when it deploys. If `semantic-id-embed` was
deployed after the engine, redeploy the engine (or `chalk deployment activate` its
current deployment) before running these.
"""

from __future__ import annotations

import argparse
import io
import os

from pydantic import BaseModel

from chalk import functions as F
from chalk.features import _, features, make_stream_resolver
from chalk.streams import KafkaSource

SEMANTIC_ID_FUNCTION = "function.semantic-id-embed"

TITLES = [
    "grandma teaches me how to fold dumplings",
    "cat steals fish from the kitchen counter",
    "words hang this! Starchy sausage without meat yyds~",
]


# -- Features and the stream resolver (deployed by `chalk apply`) -----------------------


@features
class Video:
    id: int
    title: str
    # Written by the stream resolver: computed once per upload and stored online.
    semantic_id: list[int]
    # Computed when a query asks for it, for titles that never came through the stream.
    semantic_id_on_read: list[int] = F.catalog_call(SEMANTIC_ID_FUNCTION, _.title)


class VideoUploaded(BaseModel):
    video_id: int
    title: str


video_uploads = KafkaSource(
    name="video_uploads",
    bootstrap_server="kafka.example.internal:9092",
    topic="video-uploads",
)

# Each upload message calls the function once, and the tokens are written to the online
# and offline stores with the video. Recommendation requests then read stored values and
# never wait on the model.
video_upload_stream = make_stream_resolver(
    name="video_upload_stream",
    source=video_uploads,
    message_type=VideoUploaded,
    output_features={
        Video.id: _.video_id,
        Video.title: _.title,
        Video.semantic_id: F.catalog_call(SEMANTIC_ID_FUNCTION, _.title),
    },
)


# -- Examples (run as a script) ---------------------------------------------------------


def run_sql() -> None:
    """Chalk SQL: `catalog_call` takes the catalog name, then the function's arguments."""
    import pyarrow.parquet as pq

    from chalk.client.client_grpc import ChalkGRPCClient

    values = ", ".join("('" + title.replace("'", "''") + "')" for title in TITLES)
    sql = f"""
        SELECT title, catalog_call('{SEMANTIC_ID_FUNCTION}', title) AS semantic_id
        FROM (VALUES {values}) AS t(title)
    """
    response = ChalkGRPCClient().run_sql(sql)
    if response.errors:
        raise SystemExit(f"query failed: {list(response.errors)}")
    print(pq.read_table(io.BytesIO(response.parquet)).to_pandas().to_string(index=False))


def run_chalkdf() -> None:
    """chalkdf: `F.catalog_call` as a column expression, executed as a job in the environment."""
    import pyarrow as pa
    from chalkdf import DataFrame

    titles = DataFrame.from_dict({"title": TITLES})
    with_ids = titles.with_columns(
        {
            "semantic_id": F.catalog_call(
                SEMANTIC_ID_FUNCTION,
                _.title,
                # Declared here because a local script has no catalog to look the type up in.
                output_type=pa.large_list(pa.int64()),
            ),
        }
    )
    # A local `run()` has no function catalog; `remote=True` runs it where the catalog is.
    print(with_ids.run(remote=True).to_arrow().to_pandas().to_string(index=False))


def run_stream(branch: str) -> None:
    """The stream resolver, fed messages directly on a branch: no Kafka cluster needed."""
    from chalk.client import ChalkClient

    messages = [VideoUploaded(video_id=100 + i, title=title) for i, title in enumerate(TITLES)]
    result = ChalkClient(branch=branch).test_streaming_resolver(
        resolver=video_upload_stream,
        message_keys=[str(message.video_id) for message in messages],
        # JSON text, not the models: pydantic bodies arrive as a JSON string and fail to parse.
        message_bodies=[message.model_dump_json() for message in messages],
    )
    if result.errors:
        raise SystemExit(f"stream test failed: {result.errors}")
    print(result.features.select("video.id", "video.title", "video.semantic_id"))


def run_query(branch: str | None) -> None:
    """What a recommendation request does: read semantic IDs with an online query."""
    from chalk.client import ChalkClient

    client = ChalkClient(branch=branch) if branch else ChalkClient()
    result = client.query(
        input={Video.id: 1, Video.title: TITLES[0]},
        output=[Video.semantic_id_on_read],
    )
    print(TITLES[0], "->", result.get_feature_value(Video.semantic_id_on_read))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", choices=["sql", "chalkdf", "stream", "query"])
    parser.add_argument("--branch", default=os.environ.get("CHALK_BRANCH", "semantic-ids-examples"))
    args = parser.parse_args()
    if args.example == "sql":
        run_sql()
    elif args.example == "chalkdf":
        run_chalkdf()
    elif args.example == "stream":
        run_stream(args.branch)
    else:
        run_query(args.branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
