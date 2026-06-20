from loader.github_loader import clone_repository
from dotenv import load_dotenv

load_dotenv()

# repo_url = "https://github.com/kolkarpranav/RAG-PROJECT-01.git"

# repo_path = clone_repository(
#     repo_url,
# )

# print(repo_path)

from loader.github_loader import clone_repository
from loader.code_loader import load_repository
from chunking.splitter import split_document
from embeddings.embedding_model import get_embedding_model
from vector_store.chroma_db import create_vector_store
from retriever.retriever import get_retriever
from chains.rag_chain import create_rag_chain
from llm.llm import get_llm

# repo_path = clone_repository(
#     "https://github.com/Virat2003/stock-prediction-portal.git"
# )

# docs = load_repository(repo_path)

# print("Total files:", len(documents))
# for doc in docs:
#     print(doc.metadata)

# print(docs[0].metadata)

# chunks = split_document(docs)

# embedding_model = get_embedding_model()
# vector = embedding_model.embed_query(
#     chunks[0].page_content
# )

# vector_store = create_vector_store(chunks,embedding_model)
# # print("Vector Store Created Successfully")
# results = vector_store.similarity_search(
#     "authentication",
#     k=3
# )

# retriever = get_retriever(vector_store)

# docs = retriever.invoke(
#     "How does authentication work?"
# )

# print(f"Retrieved {len(docs)} documents")

# for doc in docs:

#     print("=" * 50)

#     print(doc.metadata["file_path"])

#     print(doc.page_content[:300])


# def inspect_retrieval(query, retriever):

#     docs = retriever.invoke(query)

#     for i, doc in enumerate(docs, start=1):

#         print(f"\nResult {i}")
#         print(doc.metadata["file_path"])

#         print("-" * 50)

#         print(doc.page_content[:300])

# inspect_retrieval(
#     "authentication",
#     retriever
# )

# inspect_retrieval(
#     "API endpoints",
#     retriever
# )

# inspect_retrieval(
#     "database models",
#     retriever
# )


# rag_chain = create_rag_chain(retriever,llm)

# docs = retriever.invoke(
#     "Explain the project architecture"
# )

# for doc in docs:
#     print("\n" + "="*50)
#     print(doc.metadata["file_path"])

# response = rag_chain.invoke(
#     "tell me which model is used in this project ?"
# )

# print(response)


# for doc in results:
#     print(doc.metadata)

# print(len(vector))
# print("documents",len(docs))
# print("chunks",len(chunks))
# print("\nFirst Chunk:\n")
# print(chunks[0].page_content[:500])
# print(chunks[0].metadata)

# ------------------------------------------------------------------------- MAIN CODE ------------------------------------------------------------------------------------
url = input("enter a repository url: ")
repo_path = clone_repository(url)

docs = load_repository(repo_path)

chunks = split_document(docs)

embedding_model = get_embedding_model()

vector_store = create_vector_store(chunks,embedding_model)

repo_name = (
    url.rstrip("/")
    .split("/")[-1]
    .replace(".git", "")
)

vector_store = create_vector_store(
    chunks,
    embedding_model,
    repo_name
)

retriever = get_retriever(vector_store)

llm = get_llm()

rag_chain = create_rag_chain(retriever,llm)


# chat loop:
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = rag_chain.invoke(
        question
    )

    print("\nAnswer:")
    print(response)
