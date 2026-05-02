with silver as (
    select
        id as session_id,
        question as input,
        data_category_QA,
        language
    from {{ ref('int_filtered_dataset') }}
),

predictions as (
    select
        id as session_id,
        run_id,
        timestamp,
        prediction as output,
        model,
        language
    from read_json_auto(
        '{{ var("results_jsonl_path", "../results/predictions.jsonl") }}',
        format='newline_delimited'
    )
)

select
    s.session_id,
    p.run_id,
    p.timestamp,
    s.input,
    p.output,
    p.model,
    s.data_category_QA,
    s.language
from silver as s
inner join predictions as p
    on s.session_id = p.session_id
