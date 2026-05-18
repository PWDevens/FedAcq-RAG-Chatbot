import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

def load_query_engine():
    # Load persistent Chroma database
    client = chromadb.PersistentClient(path="./chroma_fardfars")
    collection = client.get_or_create_collection("far_dfars_chroma")
    print("Collection count:", collection.count())
    # Wrap in LlamaIndex vector store
    vs = ChromaVectorStore(chroma_collection=collection)

    # IMPORTANT: create storage context
    storage_context = StorageContext.from_defaults(vector_store=vs)

    # Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vs,
        storage_context=storage_context
    )

    # Return query engine
    return index.as_query_engine(
        similarity_top_k=5,
        streaming=True,
        response_mode="compact",
    )
