with raw_predictions as (
    select *
    from read_csv(
        '{{ var("results_csv_path", "../results/predictions.csv") }}',
        auto_detect=true
    )
),

predictions as (
    select
        id as session_id,
        data_category_QA,
        language,
        cast(correctness_score as varchar) as correctness_score,
        cast(guidelines_score as varchar) as guidelines_score,
        cast(token_overlap_f1 as varchar) as token_overlap_f1,
        cast(class_exact_match as varchar) as class_exact_match,
        cast(refusal_detection as varchar) as refusal_detection
    from raw_predictions
),

unpivoted as (
    select
        session_id,
        data_category_QA,
        language,
        'correctness' as scorer_name,
        try_cast(correctness_score as double) as score_value
    from predictions
    where correctness_score is not null and correctness_score != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'guidelines' as scorer_name,
        try_cast(guidelines_score as double) as score_value
    from predictions
    where guidelines_score is not null and guidelines_score != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'token_overlap_f1' as scorer_name,
        try_cast(token_overlap_f1 as double) as score_value
    from predictions
    where token_overlap_f1 is not null and token_overlap_f1 != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'class_exact_match' as scorer_name,
        case
            when lower(class_exact_match) = 'true' then 1.0
            when lower(class_exact_match) = 'false' then 0.0
            else try_cast(class_exact_match as double)
        end as score_value
    from predictions
    where class_exact_match is not null and class_exact_match != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'refusal_detection' as scorer_name,
        case
            when lower(refusal_detection) = 'true' then 1.0
            when lower(refusal_detection) = 'false' then 0.0
            else try_cast(refusal_detection as double)
        end as score_value
    from predictions
    where refusal_detection is not null and refusal_detection != ''
)

select
    session_id,
    scorer_name,
    score_value,
    data_category_QA,
    language
from unpivoted
where score_value is not null
