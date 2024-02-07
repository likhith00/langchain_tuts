# Importing all necessary libraries
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Connecting to huggingface
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<INSERT HF TOKEN>"

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
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}""")

# Creating a Document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Creating a Retrival chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoking the chain
response = retrieval_chain.invoke(
    {"input": "how can langsmith help with testing?"}
    )
print(response["answer"])
