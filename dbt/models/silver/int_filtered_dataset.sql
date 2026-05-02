select
    id,
    content,
    context_metadata,
    question,
    name,
    occupation,
    instructions,
    chatbot_goal,
    adjective,
    chunks_big,
    classes,
    chosen_class_id,
    language,
    data_category_QA,
    content_base_uuids
from read_parquet('{{ var("silver_parquet_path") }}')
where question is not null
  and question != ''
  and content is not null
