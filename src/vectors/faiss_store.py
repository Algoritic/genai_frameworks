from uuid import uuid4
from vectors.faiss_engine import FAISSEngine
from vectors.vector_base import VectorStoreBase


class FAISSStore(VectorStoreBase):

    def __init__(self, vector_store: FAISSEngine):
        super().__init__(vector_store)

    def query(self, query, k, **kwargs):
        with self.vector_store as vector_store:
            results = vector_store.similarity_search(query, k, **kwargs)
            return results

    def store(self, document, id_generator=None, **kwargs):
        if (id_generator is None):
            print("id generator not found, default to uuid4")
        id = id_generator(document) if id_generator else uuid4()
        with self.vector_store as vector_store:
            vector_store.add_texts([document], ids=[id], **kwargs)
            return id

    def batch_store(self, documents, id_generator=None, **kwargs):
        ids = None
        if (id_generator is None):
            print("id generator not found, default to uuid4")
            ids = [uuid4() for _ in range(len(documents))]
        else:
            ids = [id_generator(document) for document in documents]
        with self.vector_store as vector_store:
            vector_store.add_texts(documents, ids=ids, **kwargs)
            return ids

    def delete(self, id):
        with self.vector_store as vector_store:
            vector_store.delete([id])
            return id
