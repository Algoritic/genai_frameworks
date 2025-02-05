from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


class FAISSEngine:

    def __init__(
            self,
            embeddings,
            index,
            index_path,
            index_name=None,
            persist=False,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
    ):
        # embeddings function
        self.embeddings = embeddings
        assert (self.embeddings is not None), "embeddings function is required"
        self.index = index
        self.index_to_docstore_id = index_to_docstore_id,
        self.index_path = index_path
        self.index_name = index_name
        self.persist = persist
        self.docstore = docstore
        pass

    def __enter__(self):
        if (self.persist and self.index_path is not None
                and self.index_name is not None):
            self.vector_store.load_local(self.index_path,
                                         self.embeddings,
                                         self.index_name,
                                         allow_dangerous_deserialization=True)

        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
        )
        return self.vector_store

    def __exit__(self, exc_type, exc_value, traceback):
        if (self.persist):
            self.vector_store.save_local(self.index_path)
        self.vector_store = None
