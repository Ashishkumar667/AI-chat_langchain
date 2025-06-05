import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool  
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

def get_user_agent(user_id):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    persist_dir = f"vectorstores/{user_id}"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever()

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Custom PDF QA tool
    tools.append(
        Tool(
            name="PDFRetriever",
            func=lambda q: retriever.get_relevant_documents(q),
            description="Useful for answering questions about the uploaded PDF document."
        )
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent
