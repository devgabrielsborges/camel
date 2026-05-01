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
        prediction as output,
        model,
        language
    from read_csv(
        '{{ var("results_csv_path", "../results/predictions.csv") }}',
        auto_detect=true
    )
)

select
    s.session_id,
    s.input,
    p.output,
    p.model,
    s.data_category_QA,
    s.language
from silver as s
inner join predictions as p
    on s.session_id = p.session_id
