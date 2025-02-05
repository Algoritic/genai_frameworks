import faiss
from langchain_core.embeddings import DeterministicFakeEmbedding

from vectors.faiss_engine import FAISSEngine
from vectors.faiss_store import FAISSStore

embeddings = DeterministicFakeEmbedding(size=4096)
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_engine = FAISSEngine(embeddings=embeddings,
                            index=index,
                            index_path="test",
                            persist=True)
vdb = FAISSStore(vector_store=vector_engine)
result = vdb.query("hello world", 1)
print(result)
