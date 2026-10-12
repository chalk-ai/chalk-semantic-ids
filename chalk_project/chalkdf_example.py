"""Add semantic IDs to a chalkdf DataFrame of titles, executed as a job in your environment.

CHALK_ENVIRONMENT=<env id> python chalkdf_example.py
"""

import pyarrow as pa
from chalkdf import DataFrame

from chalk import functions as F
from chalk.features import _

titles = DataFrame.from_dict(
    {
        "title": [
            "grandma teaches me how to fold dumplings",
            "cat steals fish from the kitchen counter",
        ]
    }
)
with_ids = titles.with_columns(
    {
        "semantic_id": F.catalog_call(
            "function.semantic-id-embed",
            _.title,
            # Declared here because a local script has no catalog to look the type up in.
            output_type=pa.large_list(pa.int64()),
        ),
    }
)
print(with_ids.run(remote=True).to_arrow())
