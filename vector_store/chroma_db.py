from langchain_chroma import Chroma
import shutil
import os



def create_vector_store(chunks, embedding_model,repo_name):

    # # Delete old vector database
    # if os.path.exists("chroma_db"):
    #     shutil.rmtree("chroma_db")

    persist_dir = os.path.join(
        "chroma_db",
        repo_name
    )

    # Create new vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    print("Created Fresh Vector Store")

    return vector_store


