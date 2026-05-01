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
from {{ ref('stg_raw_dataset') }}
where data_category_QA in ('positivo', 'negativo')
  and question is not null
  and question != ''
  and content is not null
