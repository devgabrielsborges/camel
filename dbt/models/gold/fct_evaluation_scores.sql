with raw_predictions as (
    select *
    from read_json_auto(
        '{{ var("results_jsonl_path", "../results/predictions.jsonl") }}',
        format='newline_delimited'
    )
)

select
    id as session_id,
    run_id,
    timestamp,
    data_category_QA,
    language,
    coalesce(try_cast(correctness_score as boolean), false) as correctness,
    coalesce(try_cast(guidelines_score as boolean), false) as guidelines,
    try_cast(token_overlap_f1 as double) as token_overlap_f1,
    coalesce(try_cast(class_exact_match as boolean), false) as class_exact_match,
    coalesce(try_cast(refusal_detection as boolean), false) as refusal_detection,
    try_cast(groundedness_score as double) as groundedness,
    coalesce(try_cast(pass_at_k as boolean), false) as pass_at_k,
    try_cast(pass_at_k_best_score as double) as pass_at_k_best_score,
    failure_mode,
    coalesce(try_cast(hedging_detection as boolean), false) as hedging_detection,
    try_cast(question_response_overlap as double) as question_response_overlap,
    try_cast(response_length_ratio as double) as response_length_ratio,
    try_cast(rouge_l as double) as rouge_l,
    try_cast(chunk_attribution as double) as chunk_attribution,
    try_cast(self_consistency as double) as self_consistency,
    try_cast(self_consistency_variance as double) as self_consistency_variance
from raw_predictions
