from langchain_community.chat_message_histories import SQLChatMessageHistory


class SQLiteDB:

    def __init__(self, connection_string, session_id):
        self.connection_string = connection_string
        self.session_id = session_id

    def __enter__(self):
        self.db = SQLChatMessageHistory(
            session_id=self.session_id,
            connection_string=self.connection_string)
        return self.db

    def __exit__(self, exc_type, exc_value, traceback):
        pass
