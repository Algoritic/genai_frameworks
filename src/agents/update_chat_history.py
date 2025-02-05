import json
import os
from promptflow.core import tool
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages


def convert_chat_history_to_chatml_messages(history):
    messages = []
    for item in history:
        messages.append({
            "role": "user",
            "content": item["inputs"]["question"]
        })
        messages.append({
            "role": "assistant",
            "content": item["outputs"]["answer"]
        })

    return messages


@tool
async def update_chat_history(session_id, chat_history, question):
    chat_history = convert_chat_history_to_chatml_messages(chat_history)

    root_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_path, "conversation.db")
    selected_messages = []
    if (len(chat_history) > 0):
        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f'sqlite:///{db_path}',
        )

        #assistant_message is the last one
        assistant_message = chat_history[-1]
        user_message = chat_history[-2]
        message_history.add_ai_message(assistant_message["content"])
        message_history.add_user_message(user_message["content"])
        messages = message_history.get_messages()
        selected_messages = trim_messages(
            messages,
            token_counter=
            len,  # <-- len will simply count the number of messages rather than tokens
            max_tokens=5,  # <-- allow up to 5 messages.
            strategy="last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )
        selected_messages = [x.model_dump_json() for x in selected_messages]
        print(selected_messages)

    return {
        "answer": "hello world",
        "last_k_chat_history": json.dumps(selected_messages)
    }
