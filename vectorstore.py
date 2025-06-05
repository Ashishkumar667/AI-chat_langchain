from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def store_pdf_in_vectorstore(text, user_id):
    persist_dir = f"vectorstores/{user_id}"
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=[text],
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()