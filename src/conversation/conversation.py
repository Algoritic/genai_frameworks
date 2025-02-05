import datetime
from typing import List
from uuid import uuid4
from pydantic import BaseModel


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str | dict
    end_turn: bool
    date: str
    context: str


class Conversation(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    date: str


class ConversationEntry(BaseModel):
    id: str
    title: str
    createdAt: str
    updatedAt: str
    userId: str
    type: str


class ConversationManager:

    def __init__(self, db):
        self.db = db

    # async def create_conversation(self, user_id: str,
    #                               title: str) -> Conversation:
    #     conversation = ConversationEntry(id=str(uuid4()),
    #                                      title=title,
    #                                      type="conversation",
    #                                      createdAt=str(datetime.now()),
    #                                      updatedAt=str(datetime.now()),
    #                                      userId=user_id)
    #     with self.db as db:
    #         result = await db.upsert(conversation)
    #         return result

    # async def upsert_conversation(self, conversation: ConversationEntry):
    #     with self.db as db:
    #         result = await db.upsert(conversation)
    #         return result

    async def delete_conversation(self):
        self.db.clear()
        # with self.db as db:
        #     result = await db.delete(conversation_id)
        #     return result

    async def get_messages(self, conversation_id: str, user_id: str):
        pass

    async def delete_messages(self, conversation_id: str, user_id: str):
        pass
