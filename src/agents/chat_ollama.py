from ollama import chat
from promptflow.core import tool


@tool
async def chat_ollama(session_id, question, context, memory):
    return chat(
        model="gemma2:latest",
        messages=[{
            "role":
            "user",
            "content":
            f"You memoried: {memory}\n Based on the following context:\n {context}, answer my question:\n{question}"
        }])["message"]["content"]
