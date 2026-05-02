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
        cast(refusal_detection as varchar) as refusal_detection,
        cast(groundedness_score as varchar) as groundedness_score,
        cast(pass_at_k as varchar) as pass_at_k,
        cast(pass_at_k_best_score as varchar) as pass_at_k_best_score
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

    union all

    select
        session_id,
        data_category_QA,
        language,
        'groundedness' as scorer_name,
        try_cast(groundedness_score as double) as score_value
    from predictions
    where groundedness_score is not null and groundedness_score != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'pass_at_k' as scorer_name,
        case
            when lower(pass_at_k) = 'true' then 1.0
            when lower(pass_at_k) = 'false' then 0.0
            else try_cast(pass_at_k as double)
        end as score_value
    from predictions
    where pass_at_k is not null and pass_at_k != ''

    union all

    select
        session_id,
        data_category_QA,
        language,
        'pass_at_k_best_score' as scorer_name,
        try_cast(pass_at_k_best_score as double) as score_value
    from predictions
    where pass_at_k_best_score is not null and pass_at_k_best_score != ''
)

select
    session_id,
    scorer_name,
    score_value,
    data_category_QA,
    language
from unpivoted
where score_value is not null
