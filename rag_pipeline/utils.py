# utils.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


import os
os.environ['OPENAI_API_KEY'] = ""
PINECONE_API_KEY = ""

# Initialize Pinecone and vector store
def initialize_vector_store():
    pc = Pinecone(api_key = PINECONE_API_KEY)
    index = pc.Index("langchain-demo")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

# Initialize the LLM and chain
def initialize_llm_chain():
    llm = ChatOpenAI(model_name="gpt-4")
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    document_variable_name = "context"
    prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    return chain

# Load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


