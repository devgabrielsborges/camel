from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    batch_size: int = 50
    concurrency: int = 10

    postgres_user: str = "mlflow"
    postgres_password: str = "mlflow"
    postgres_db: str = "mlflow"

    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    dataset_path: str = "data/camel.duckdb"
    parquet_path: str = "data/raw/train.parquet"
    prompt_template_path: str = "prompts/system_prompt.j2"
    results_dir: str = "results"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
