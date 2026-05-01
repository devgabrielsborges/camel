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

# DatasetRepository
categories  # noqa: B018

# EvaluationRepository / TraceRepository
evaluation_id  # noqa: B018
session_id  # noqa: B018
trace  # noqa: B018
