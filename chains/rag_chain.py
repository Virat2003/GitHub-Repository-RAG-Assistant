from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chains.prompt import RAG_PROMPT


def create_rag_chain(retriever, llm):
    rag_chain =  (
        {
            "context":retriever,
            "question":RunnablePassthrough()
        } | RAG_PROMPT | llm | StrOutputParser()
    )
    return rag_chain

