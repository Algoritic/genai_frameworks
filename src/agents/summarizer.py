from promptflow.core import tool
from ollama import chat


@tool
async def summarize(chat_history):
    return chat(model="gemma2:latest",
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"""
                    Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n\nEXAMPLE\nCurrent summary:\nThe human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.\n\nNew lines of conversation:\nHuman: Why do you think artificial intelligence is a force for good?\nAI: Because artificial intelligence will help humans reach their full potential.\n\nNew summary:\nThe human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\nEND OF EXAMPLE\n\nCurrent summary:\n{chat_history}\n\nNew summary:"""
                }])["message"]["content"]
