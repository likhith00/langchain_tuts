# Importing all necessary libraries
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

# Connecting to huggingface
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<INSERT TOKEN>"

# Load documents from source
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Indexing and storing the documents
embeddings = HuggingFaceEmbeddings()
vector = FAISS.from_documents(documents, embeddings)

# LLM initialization
repo_id = "google/flan-t5-large"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

# Prompt creation
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", '''Given the above conversation, generate a search query to look
     up in order to get information relevant to the conversation''')
])

# Creating a Retrival chain
retriever = vector.as_retriever()
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Retrieve the relevant documents
chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!")
    ]

retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

# Create a new chain to continue the conversation with these documents
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# Invoke the new chain
chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!")
    ]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response["answer"])
