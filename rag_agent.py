from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def get_rag_agent(user_id):
    vectordb = Chroma(
        collection_name=f"user_{user_id}_pdf",
        persist_directory="vector_db",
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o"),
        retriever=retriever
    )
    return qa_chain
