"""Recommend next videos for a chalkdf DataFrame of watch histories, executed as a job in your environment.

CHALK_ENVIRONMENT=<env id> python next_videos_chalkdf.py
"""

import pyarrow as pa
from chalkdf import DataFrame

from chalk import functions as F
from chalk.features import _

histories = DataFrame.from_dict(
    {
        "user_id": [1, 2],
        # Most recent watch first.
        "recent_video_ids": [[17700, 15223, 6346, 1958], [10949, 9500, 3775, 9127, 915, 16657]],
    }
)
recommended = histories.with_columns(
    {
        "next_video_ids": F.catalog_call(
            "function.next-videos",
            _.recent_video_ids,
            # Declared here because a local script has no catalog to look the type up in.
            output_type=pa.large_list(pa.int64()),
        ),
    }
)
print(recommended.run(remote=True).to_arrow())
