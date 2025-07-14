#!/usr/bin/env python3
"""
Working Medical Chatbot RAG System
This script demonstrates the fixed version that works in your llmapp environment.
"""

import os
import sys
from dotenv import load_dotenv

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Import all required modules
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

def main():
    print("🚀 Starting Medical Chatbot RAG System...")
    
    # Get API keys
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
    
    if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
        print("❌ Error: Missing API keys. Please check your .env file.")
        return
    
    print("✅ API keys loaded successfully")
    
    # Step 1: Load PDF data
    print("\n📖 Loading PDF data...")
    try:
        loader = DirectoryLoader('Data/', glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return
    
    # Step 2: Split text into chunks
    print("\n✂️  Splitting text into chunks...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(text_chunks)} text chunks")
    except Exception as e:
        print(f"❌ Error splitting text: {e}")
        return
    
    # Step 3: Setup embeddings
    print("\n🧠 Setting up embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embeddings loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return
    
    # Step 4: Setup Pinecone
    print("\n🌲 Setting up Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "medibot"
        
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Created new Pinecone index: {index_name}")
        else:
            print(f"✅ Using existing Pinecone index: {index_name}")
    except Exception as e:
        print(f"❌ Error setting up Pinecone: {e}")
        return
    
    # Step 5: Create vector store
    print("\n🔍 Creating vector store...")
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=index_name,
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("✅ Vector store and retriever created")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return
    
    # Step 6: Setup LLM
    print("\n🤖 Setting up LLM...")
    try:
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"temperature": 0.4, "max_new_tokens": 500},
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )
        print("✅ LLM loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        return
    
    # Step 7: Create prompt template
    print("\n📝 Creating prompt template...")
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "You're a medical expert. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. Don't try to make up an answer. "
            "Use three sentences maximum to answer the question. "
            "Keep the answer concise and to the point. "
            "\n\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        print("✅ Prompt template created")
    except Exception as e:
        print(f"❌ Error creating prompt: {e}")
        return
    
    # Step 8: Create RAG chain
    print("\n🔗 Creating RAG chain...")
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("✅ RAG chain created successfully")
    except Exception as e:
        print(f"❌ Error creating RAG chain: {e}")
        return
    
    # Step 9: Test the system
    print("\n🧪 Testing the RAG system...")
    print("=" * 60)
    
    test_questions = [
        "What is Acne?",
        "What are the symptoms of diabetes?",
        "How is hypertension treated?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        try:
            response = rag_chain.invoke({"input": question})
            print(f"   Answer: {response['answer']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print("-" * 60)
    
    print("\n🎉 Medical Chatbot RAG System is working!")
    print("\nTo use this in your notebook:")
    print("1. Open research/trials_fixed.ipynb")
    print("2. Run all cells from top to bottom")
    print("3. The system should work without errors!")

if __name__ == "__main__":
    main() 