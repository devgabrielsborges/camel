# Dataset Structure

Each row in the dataset represents a single QA session between a user and a chatbot agent. The dataset is stored as JSON (or Parquet) with the following schema.

## Schema

| Field | Type | Visible to Agent | Description |
|---|---|---|---|
| `id` | string | No | Unique identifier for the session |
| `content` | dict[string, string] | No | Knowledge base content keyed by content ID. Raw text passages the agent can reference |
| `context_metadata` | dict[string, string] | No | Metadata about each content entry (e.g., search result kind, title, link, snippet) |
| `question` | string | Yes | Latest question from the user |
| `type_question` | string | No | Classification of the question type (for evaluation only) |
| `type_feature` | string | No | Feature type label (for evaluation only) |
| `name` | string | Yes | Name of the agent persona |
| `occupation` | string | Yes | Occupation or role of the agent persona |
| `instructions` | list[string] | Yes | Behavioral instructions and guidelines for the agent |
| `chatbot_goal` | string | Yes | High-level goal or purpose of the chatbot |
| `adjective` | string | Yes | Personality adjective describing the agent's tone |
| `data_category` | string | No | Data category label (for evaluation only) |
| `chunks_big` | dict[string, list[object]] | Yes | Knowledge base chunks for the agent, keyed by content ID. Each chunk has `content` (string) and `score` (float) |
| `classes` | dict[string, list[object]] | Yes | Common question classes with `class` (label), `context` (description), and `id` (class identifier) |
| `chosen_class_id` | string | No | The class ID chosen based on the `classes` field (for evaluation only) |
| `language` | int | Yes | Language of the session as a numeric code |
| `data_category_QA` | string | No | Category of the QA session (for evaluation only) |
| `content_base_uuids` | string | Yes | UUID referencing the content base used |

## Field Details

### `content`

Dictionary mapping content IDs to raw knowledge base text. Each value is a long string containing the full passage.

```json
{
  "3567": "Kitty Moana, dona de casa e mãe de dois filhos dos Estados Unidos, perdeu mais de 13 quilos em apenas três meses após adotar uma rotina simples de exercícios e uma dieta saudável..."
}
```

### `context_metadata`

Dictionary mapping content IDs to stringified metadata about the source (e.g., search engine result details).

```json
{
  "1507": "{'kind': 'customsearch#result', 'title': 'B640WBG-1B | CASIO', 'htmlTitle': 'B640WBG-1B | CASIO', 'link': 'https://www.casio.com/...', 'position': '1'}"
}
```

### `question`

The user's latest question that the agent must answer.

```
"what devices in minimum iOS version are compatible with the Dell AR Assistant application requesting specific and detailed instructions?"
```

### `name`, `occupation`, `adjective`

Define the agent's persona.

```
name: "Dexter"
occupation: "Support technician in IT"
adjective: "Creative"
```

### `instructions`

List of behavioral rules the agent must follow during the conversation.

```json
[
  "Answer only questions related to Dell AR Assistant",
  "Explain any step by step in detail",
  "Always say if the application supports the mentioned device"
]
```

### `chatbot_goal`

Describes the overarching purpose of the chatbot.

```
"Help Littoraneus customers with questions about UV protection, clothing, reports and certified care..."
```

### `chunks_big`

Dictionary mapping content IDs to lists of chunk objects. Each chunk contains a `content` string and a `score` float representing relevance.

```json
{
  "1507": [
    {
      "content": "Frequently asked questions (FAQ) about Dell AR Assistant...",
      "score": 2.0
    }
  ]
}
```

### `classes`

Dictionary mapping content IDs to lists of class objects. Each class defines a category of common questions with a label, context description, and ID.

```json
{
  "1507": [
    {
      "class": "Mobile Applications History",
      "context": "when it comes to evolution and changes in mobile applications over time",
      "id": "A1"
    }
  ]
}
```

### `language`

Numeric code representing the session language:

| Code | Language |
|---|---|
| 1 | English |
| 2 | Spanish |
| 3 | Portuguese |

### `chosen_class_id`

The ground-truth class ID selected for the session, prefixed with `S` (for sensitive) or `P` (for positive) followed by a number.

```
"P1"  — positive, class 1
"S1"  — sensitive, class 1
```
