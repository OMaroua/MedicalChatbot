#!/usr/bin/env python3
"""
Test script to verify all packages are working correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        print("✓ langchain_community.document_loaders")
    except ImportError as e:
        print(f"✗ langchain_community.document_loaders: {e}")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✓ langchain.text_splitter")
    except ImportError as e:
        print(f"✗ langchain.text_splitter: {e}")
        return False
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✓ langchain_community.embeddings")
    except ImportError as e:
        print(f"✗ langchain_community.embeddings: {e}")
        return False
    
    try:
        from langchain_pinecone import PineconeVectorStore
        print("✓ langchain_pinecone")
    except ImportError as e:
        print(f"✗ langchain_pinecone: {e}")
        return False
    
    try:
        from langchain_community.llms import HuggingFaceHub
        print("✓ langchain_community.llms")
    except ImportError as e:
        print(f"✗ langchain_community.llms: {e}")
        return False
    
    try:
        from langchain.chains import create_retrieval_chain
        print("✓ langchain.chains")
    except ImportError as e:
        print(f"✗ langchain.chains: {e}")
        return False
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        print("✓ langchain_core.prompts")
    except ImportError as e:
        print(f"✗ langchain_core.prompts: {e}")
        return False
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        print("✓ pinecone")
    except ImportError as e:
        print(f"✗ pinecone: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv")
    except ImportError as e:
        print(f"✗ python-dotenv: {e}")
        return False
    
    return True

def test_huggingface_hub():
    """Test HuggingFace Hub version"""
    print("\nTesting HuggingFace Hub version...")
    
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"HuggingFace Hub version: {version}")
        
        # Check if version is compatible
        major, minor, patch = map(int, version.split('.'))
        if major == 0 and minor >= 30:
            print("✓ HuggingFace Hub version is compatible")
            return True
        else:
            print(f"✗ HuggingFace Hub version {version} is too old. Need >= 0.30.0")
            return False
    except ImportError as e:
        print(f"✗ Could not import huggingface_hub: {e}")
        return False

def test_embeddings():
    """Test embeddings functionality"""
    print("\nTesting embeddings...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Set environment variable to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = embeddings.embed_query("Hello world")
        
        print(f"✓ Embeddings working. Dimension: {len(test_embedding)}")
        return True
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("MEDICAL CHATBOT PACKAGE TEST")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test HuggingFace Hub
    hf_ok = test_huggingface_hub()
    
    # Test embeddings
    embeddings_ok = test_embeddings()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if imports_ok and hf_ok and embeddings_ok:
        print("✓ ALL TESTS PASSED! Your environment is ready.")
        print("\nYou can now run the fixed notebook: research/trials_fixed.ipynb")
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nTo fix the issues:")
        print("1. Run: pip install --upgrade 'huggingface-hub>=0.30.0' 'sentence-transformers' 'transformers'")
        print("2. Restart your Jupyter kernel")
        print("3. Run this test script again")

if __name__ == "__main__":
    main() 