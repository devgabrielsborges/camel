"""Vulture whitelist — false positives from Protocol method signatures."""

# AgentPort
system_prompt  # noqa: B018
record  # noqa: B018

# ScorerPort
inputs  # noqa: B018
outputs  # noqa: B018
expectations  # noqa: B018

# TrackerPort
evaluation  # noqa: B018
run_id  # noqa: B018
template  # noqa: B018
records  # noqa: B018
metrics  # noqa: B018
enable_autolog  # noqa: B018
disable_autolog  # noqa: B018
search_traces  # noqa: B018
set_run_tags  # noqa: B018

# DatasetRepository
categories  # noqa: B018

# EvaluationRepository / TraceRepository
evaluation_id  # noqa: B018
session_id  # noqa: B018
trace  # noqa: B018

# Pydantic models — used by framework
model_config  # noqa: B018

# CLI infer_cmd — typer option names
infer  # noqa: B018

# CLI run_cmd
run_pipeline  # noqa: B018

# Domain methods consumed by use cases
add_trace  # noqa: B018
add_session  # noqa: B018
transition_to  # noqa: B018
validate_output  # noqa: B018
add_score  # noqa: B018
is_single_turn  # noqa: B018

# CLI prepare_cmd
prepare  # noqa: B018
skip_download  # noqa: B018
gold  # noqa: B018

# CLI evaluate_cmd
evaluate  # noqa: B018
no_llm_judge  # noqa: B018

# CLI export_cmd
export  # noqa: B018

# Settings fields
experiment_name  # noqa: B018
dataset_name  # noqa: B018
llm_provider  # noqa: B018

# LiteLLM adapter
LiteLLMAgentAdapter  # noqa: B018

# Aggregation dataclasses
AggregatedMetric  # noqa: B018
CategoryBreakdown  # noqa: B018

# Scorer factories
create_scorers  # noqa: B018

# MLflow @scorer functions
token_overlap_f1  # noqa: B018
class_exact_match  # noqa: B018
refusal_detection  # noqa: B018
get_deterministic_scorers  # noqa: B018
get_llm_judge_scorers  # noqa: B018

# Tool function params — required by agents SDK
ctx  # noqa: B018
