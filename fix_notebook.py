#!/usr/bin/env python3
"""
Quick fix for your notebook - run this to test the working RAG system
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

print("🔧 Setting up the working RAG system...")

# Load and split documents (assuming you already have this)
def load_pdf_file(data):
    loader = PyPDFLoader("Data/Medical_book.pdf")
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts_ = text_splitter.split_documents(extracted_data)
    return texts_

# Load documents
print("📄 Loading PDF documents...")
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data=extracted_data)
print(f"✅ Loaded {len(text_chunks)} text chunks")

# Setup embeddings
print("🔍 Setting up embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Setup Pinecone
print("🌲 Setting up Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Create vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

# FIXED: Use a working model
print("🤖 Setting up the working LLM...")
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="text-generation",
    model="microsoft/DialoGPT-medium",
    model_kwargs={"temperature": 0.4, "max_new_tokens": 500}
)

# Setup retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "You're a medical expert. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Don't try to make up an answer. "
    "Use three sentences maximum to answer the question. "
    "Keep the answer concise and to the point.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ RAG system is ready!")

# Test the system
print("\n🧪 Testing the system...")
try:
    response = rag_chain.invoke({"input": "What is Acne?"})
    print("🎉 SUCCESS! Here's the answer:")
    print(response['answer'])
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Try running this in your notebook:")
    print("""
# Replace your broken LLM with this:
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="text-generation",
    model_kwargs={"temperature": 0.4, "max_new_tokens": 500}
)
""") 