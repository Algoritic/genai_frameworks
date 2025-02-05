from promptflow.core import tool
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "gemma2:latest",
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url":
            "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            # Alternatively, you can use "snowflake-arctic-embed:latest"
            "ollama_base_url": "http://localhost:11434",
        },
    },
}


@tool
async def update_memory(session_id, question):
    m = Memory.from_config(config)
    m.add(question, user_id=session_id)
    return
