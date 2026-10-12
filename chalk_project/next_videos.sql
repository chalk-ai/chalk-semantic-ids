-- chalk sql --file next_videos.sql
-- Each history lists watched video ids most recent first.
SELECT
    user_id,
    catalog_call('function.next-videos', recent_video_ids) AS next_video_ids
FROM (
    VALUES
        (1, [17700, 15223, 6346, 1958]),
        (2, [10949, 9500, 3775, 9127, 915, 16657])
) AS t(user_id, recent_video_ids)
