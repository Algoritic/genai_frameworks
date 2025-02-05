from abc import ABC, abstractmethod


class VectorStoreBase(ABC):

    def __init__(self, vector_store):
        self.vector_store = vector_store

    @abstractmethod
    def query(self, query, **kwargs):
        pass

    @abstractmethod
    def store(self, document, id_generator=None, **kwargs):
        pass

    @abstractmethod
    def batch_store(self, documents, id_generator=None, **kwargs):
        pass

    @abstractmethod
    def delete(self, id):
        pass
