from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "http://localhost:5000"

    openai_api_key: str = Field(default="", min_length=0)
    openai_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    llm_provider: str = Field(default="openai", description="openai or litellm")

    batch_size: int = 50
    concurrency: int = 10

    postgres_user: str = "mlflow"
    postgres_password: str = "mlflow"
    postgres_db: str = "mlflow"

    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    experiment_name: str = "WeniEval"
    dataset_name: str = "weni_eval_dataset"

    ollama_api_base: str = "http://localhost:11434"

    duckdb_path: str = "data/camel.duckdb"
    raw_parquet_path: str = "data/bronze/train.parquet"
    silver_parquet_path: str = "data/silver/train_sample.parquet"
    sample_fraction: float = Field(
        default=0.1, description="Fraction of dataset to sample for silver layer"
    )
    sample_seed: int = Field(default=42, description="Random seed for reproducible sampling")
    pass_at_k: int = Field(default=3, description="Number of responses per question for Pass@k")
    pass_at_k_temperature: float = Field(
        default=0.7, description="Temperature for generating diverse Pass@k responses"
    )

    inference_cache_dir: str = Field(
        default="data/cache/inferences",
        description="Directory for caching inference results",
    )

    prompt_template_path: str = "prompts/system_prompt.j2"
    results_dir: str = "results"
