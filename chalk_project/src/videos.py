"""Semantic IDs as Chalk features, computed by the deployed `semantic-id-embed` function.

The stream resolver calls the function once per upload message and stores the tokens
with the video, so reads never pay for the model. `Video.semantic_id_on_read` computes
the same tokens at query time instead, for titles that never came through the stream.
"""

from pydantic import BaseModel

from chalk import functions as F
from chalk.features import _, features, make_stream_resolver
from chalk.streams import KafkaSource

# The catalog name is `function.` plus the name the function was deployed under
# (`deploy_function.py --name`), not the Python function's name.
SEMANTIC_ID_FUNCTION = "function.semantic-id-embed"


@features
class Video:
    id: int
    title: str
    # [level 0, level 1, level 2] tokens; videos sharing a prefix are related.
    semantic_id: list[int]
    semantic_id_on_read: list[int] = F.catalog_call(SEMANTIC_ID_FUNCTION, _.title)


class VideoUploaded(BaseModel):
    video_id: int
    title: str


video_uploads = KafkaSource(
    name="video_uploads",
    bootstrap_server="kafka.example.internal:9092",
    topic="video-uploads",
)

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
