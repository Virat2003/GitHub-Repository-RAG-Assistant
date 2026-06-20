from langchain_mistralai import ChatMistralAI

def get_llm():
    return ChatMistralAI(model="mistral-small-latest")
