from __future__ import annotations

import pytest

from camel.domain.value_objects import Chunk, ClassDef, TokenUsage, ToolCall
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.infrastructure.config.settings import Settings


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        mlflow_tracking_uri="http://localhost:5000",
        duckdb_path="data/camel.duckdb",
        raw_parquet_path="data/bronze/train.parquet",
        silver_parquet_path="data/silver/train_sample.parquet",
        prompt_template_path="prompts/system_prompt.j2",
        results_dir="results",
    )


@pytest.fixture()
def sample_model_config() -> ModelConfig:
    return ModelConfig(model_name="gpt-4o-mini", temperature=0.0)


@pytest.fixture()
def sample_dataset_record() -> DatasetRecord:
    return DatasetRecord(
        id="test-session-001",
        question="What is the return policy?",
        content="Our return policy allows returns within 30 days.",
        context_metadata="{'kind': 'faq', 'title': 'Returns'}",
        name="Alice",
        occupation="Customer support agent",
        adjective="Friendly",
        chatbot_goal="Help customers with questions about returns and refunds.",
        instructions=["Be polite", "Answer only based on the knowledge base"],
        chunks_big=[
            Chunk(content="Returns are accepted within 30 days of purchase.", score=1.5),
        ],
        classes=[
            ClassDef(class_name="Returns", context="questions about return policy", class_id="P1"),
        ],
        chosen_class_id="P1",
        language=1,
        data_category_qa="positivo",
        content_base_uuids="uuid-101",
    )


@pytest.fixture()
def sample_token_usage() -> TokenUsage:
    return TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)


@pytest.fixture()
def sample_tool_call() -> ToolCall:
    return ToolCall(
        tool_name="search", arguments={"query": "return policy"}, result="Found 1 result"
    )
