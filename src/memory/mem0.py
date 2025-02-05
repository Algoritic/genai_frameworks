from mem0 import Memory


class Mem0:

    def __init__(self, vector_config, llm_config, embedding_config):
        self.config = {
            "vector_store": vector_config,
            "llm": llm_config,
            "embedder": embedding_config,
        }
        self.memory = Memory.from_config(self.config)

    def retrieve_context(self, query: str, user_id: str):
        memories = self.memory.search(query=query, user_id=user_id)
        serialized_memories = [memory.serialize() for memory in memories]
        context = [{
            "role": "system",
            "content": f"Relevant information: {serialized_memories}"
        }, {
            "role": "user",
            "content": query
        }]
        return context

    def save_interaction(self, user_id: str, user_input: str,
                         assistant_response: str):
        """Save the interaction to Mem0"""
        interaction = [{
            "role": "user",
            "content": user_input
        }, {
            "role": "assistant",
            "content": assistant_response
        }]
        self.memory.add(user_id=user_id, interaction=interaction)
