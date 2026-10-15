"""Chalk feature graph. Model inference runs in the hosted Scaling Groups.

Deploy both functions first, then apply this graph. The explicit output types
let Python construct expressions without a local catalog or a model import.
Queue-backed calls use Chalk's authenticated queue client and await the result.
The environment must have function-queue authentication enabled.
"""

import pyarrow as pa
import chalk.functions as F
from chalk.features import features, _


@features
class Video:
    id: int
    title: str
    tokenizer_version: int
    codes: list[int] = F.catalog_call_async(
        "function.semantic-id-tokenize", _.title, _.tokenizer_version,
        output_type=pa.list_(pa.int64()),
    )


@features
class Recommendation:
    id: str
    history_ids: list[int]
    model_version: int
    top_k: int
    result: str = F.catalog_call_async(
        "function.semantic-id-recommend", _.history_ids, _.model_version, _.top_k,
        output_type=pa.string(),
    )
