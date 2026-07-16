import json
import math
from pathlib import Path

import ollama


CHAT_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "embeddinggemma"

MEMORY_FILE = Path("Memory/knowledge.json")


def load_memories() -> list[dict]:
    """Load long-term memories from the JSON file."""

    if not MEMORY_FILE.exists():
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)




def save_memories(memories: list[dict]) -> None:
    """Save long-term memories to the JSON file."""

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            memories,
            file,
            indent=4,
            ensure_ascii=False,
        )


def create_embedding(text: str) -> list[float]:
    """Convert text into an embedding vector."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response["embeddings"][0]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Measure semantic similarity between two vectors."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def add_memory(text: str) -> None:
    """Add a fact and its embedding to long-term memory."""

    text = text.strip()

    if not text:
        return

    memories = load_memories()

    # Prevent exact duplicate memories.
    for memory in memories:
        if memory["text"].lower() == text.lower():
            return

    embedding = create_embedding(text)

    memories.append(
        {
            "text": text,
            "embedding": embedding,
        }
    )

    save_memories(memories)


def search_memories(
    query: str,
    top_k: int = 3,
    minimum_score: float = 0.35,
) -> list[str]:
    """Find memories related to the user's query."""

    memories = load_memories()

    if not memories:
        return []

    query_embedding = create_embedding(query)

    scored_memories = []

    for memory in memories:
        score = cosine_similarity(
            query_embedding,
            memory["embedding"],
        )

        if score >= minimum_score:
            scored_memories.append(
                {
                    "text": memory["text"],
                    "score": score,
                }
            )

    scored_memories.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return [
        memory["text"]
        for memory in scored_memories[:top_k]
    ]

def extract_memories(user_message: str) -> list[str]:
    """Extract useful long-term facts from a user message."""

    prompt = f"""
Extract useful long-term information about the user.

Save only:
- personal preferences
- skills
- goals
- ongoing projects
- important background information

Do not save:
- normal questions
- greetings
- temporary information
- guesses

Write every memory as a complete third-person sentence.

User message:
{user_message}
"""

    memory_schema = {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            }
        },
        "required": ["memories"],
    }


    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        think=False,
        format=memory_schema,
        options={
            "temperature": 0,
        },
    )

    raw_response = response.message.content

    result = json.loads(raw_response)

    memories = result.get("memories", [])

    return [
        memory.strip()
        for memory in memories
        if isinstance(memory, str) and memory.strip()
    ]
