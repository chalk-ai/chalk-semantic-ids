-- chalk sql --file semantic_ids.sql
SELECT
    title,
    catalog_call('function.semantic-id-embed', title) AS semantic_id
FROM (
    VALUES
        ('grandma teaches me how to fold dumplings'),
        ('cat steals fish from the kitchen counter'),
        ('words hang this! Starchy sausage without meat yyds~')
) AS t(title)
