"""Run the stream resolver on a branch against two upload messages; no Kafka needed.

chalk apply --branch semantic-ids
python test_stream.py
"""

from chalk.client import ChalkClient

from src.videos import VideoUploaded, video_upload_stream

messages = [
    VideoUploaded(video_id=101, title="grandma teaches me how to fold dumplings"),
    VideoUploaded(video_id=102, title="cat steals fish from the kitchen counter"),
]

client = ChalkClient(branch="semantic-ids")
result = client.test_streaming_resolver(
    resolver=video_upload_stream,
    message_keys=[str(message.video_id) for message in messages],
    # JSON text, not the models: pydantic bodies arrive as a JSON string and fail to parse.
    message_bodies=[message.model_dump_json() for message in messages],
)
print(result.message, result.errors)
print(result.features.select("video.id", "video.title", "video.semantic_id"))
