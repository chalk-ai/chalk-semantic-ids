"""Verify real SDK serialization without querying or deploying to Chalk."""
import pyarrow as pa
import pytest

from chalk.features.feature_wrapper import unwrap_feature
from chalk.features.underscore import convert_value_to_proto_expr

from features import Recommendation, Video


@pytest.mark.parametrize("feature, function, inputs, output_type", [
    (Video.codes, "function.semantic-id-tokenize", ["title", "tokenizer_version"], pa.list_(pa.int64())),
    (Recommendation.result, "function.semantic-id-recommend",
     ["history_ids", "model_version", "top_k"], pa.string()),
])
def test_feature_serializes_queue_call_with_versioned_inputs(feature, function, inputs, output_type):
    node = unwrap_feature(feature).underscore_expression._to_proto()
    assert node.call.func.identifier.name == "catalog_call_async"
    assert node.call.args[0] == convert_value_to_proto_expr(function)
    assert [arg.get_attribute.attribute.name for arg in node.call.args[1:]] == inputs
    assert node.call.kwargs["output_type"] == convert_value_to_proto_expr(output_type)
