import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts_ = text_splitter.split_documents(extracted_data)
    return texts_

def main():
    print("Loading PDF data...")
    extracted_data = load_pdf_file(data='Data/')
    
    print("Splitting text into chunks...")
    text_chunks = text_split(extracted_data=extracted_data)
    print(f"Number of text chunks: {len(text_chunks)}")
    
    print("Setting up embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    print("Setting up Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "medibot"
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1")
        )
    
    print("Creating vector store...")
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
    )
    
    print("Setting up retriever...")
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    print("Setting up LLM...")
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.4, "max_new_tokens": 500},
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
    
    print("Creating prompt template...")
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
    
    print("Creating RAG chain...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("Testing the system...")
    response = rag_chain.invoke({"input": "What is Acne?"})
    print("Answer:", response['answer'])

if __name__ == "__main__":
    main() 