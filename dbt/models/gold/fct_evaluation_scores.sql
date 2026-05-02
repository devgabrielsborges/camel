with raw_predictions as (
    select *
    from read_json_auto(
        '{{ var("results_jsonl_path", "../results/predictions.jsonl") }}',
        format='newline_delimited'
    )
),

predictions as (
    select
        id as session_id,
        run_id,
        timestamp,
        data_category_QA,
        language,
        correctness_score,
        guidelines_score,
        token_overlap_f1,
        class_exact_match,
        refusal_detection,
        groundedness_score,
        pass_at_k,
        pass_at_k_best_score
    from raw_predictions
),

unpivoted as (
    select
        session_id, run_id, timestamp, data_category_QA, language,
        'correctness' as scorer_name,
        try_cast(correctness_score as double) as score_value
    from predictions
    where correctness_score is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'guidelines' as scorer_name,
        try_cast(guidelines_score as double) as score_value
    from predictions
    where guidelines_score is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'token_overlap_f1' as scorer_name,
        try_cast(token_overlap_f1 as double) as score_value
    from predictions
    where token_overlap_f1 is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'class_exact_match' as scorer_name,
        case
            when class_exact_match = true then 1.0
            when class_exact_match = false then 0.0
            else try_cast(class_exact_match as double)
        end as score_value
    from predictions
    where class_exact_match is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'refusal_detection' as scorer_name,
        case
            when refusal_detection = true then 1.0
            when refusal_detection = false then 0.0
            else try_cast(refusal_detection as double)
        end as score_value
    from predictions
    where refusal_detection is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'groundedness' as scorer_name,
        try_cast(groundedness_score as double) as score_value
    from predictions
    where groundedness_score is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'pass_at_k' as scorer_name,
        case
            when pass_at_k = true then 1.0
            when pass_at_k = false then 0.0
            else try_cast(pass_at_k as double)
        end as score_value
    from predictions
    where pass_at_k is not null

    union all

    select
        session_id, run_id, timestamp, data_category_QA, language,
        'pass_at_k_best_score' as scorer_name,
        try_cast(pass_at_k_best_score as double) as score_value
    from predictions
    where pass_at_k_best_score is not null
)

select
    session_id,
    run_id,
    timestamp,
    scorer_name,
    score_value,
    data_category_QA,
    language
from unpivoted
where score_value is not null
