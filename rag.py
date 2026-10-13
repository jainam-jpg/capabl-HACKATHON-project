from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from llm import create_llm
from dotenv import load_dotenv
import os

load_dotenv()

def process_pdf(file_path: str, doc_type: str):
    """Loads a PDF and adds metadata tags ('policy' or 'resume')."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["doc_type"] = doc_type

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    
    chunks = splitter.split_documents(documents)
    return chunks

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="hiring_agent_collection",
    persist_directory="./vector_db"
)

def ingest_data():
    policy_chunks = process_pdf("./handbook.pdf", doc_type="policy")
    jd_chunks = process_pdf("./job_description.pdf", doc_type="policy")
    resume_chunks = process_pdf("./resume.pdf", doc_type="resume")
    
    total_chunks = policy_chunks + jd_chunks + resume_chunks
    if total_chunks:
        vector_store.add_documents(total_chunks)
        return "Data ingestion completed successfully with metadata tags."
    return "Error: No documents found to ingest. Please ensure handbook.pdf, job_description.pdf, and resume.pdf are in the root folder."

def reg_chain(query: str, doc_type_filter: str = None):
    llm = create_llm()
    
    search_kwargs = {"k": 4}
    if doc_type_filter:
        search_kwargs["filter"] = {"doc_type": doc_type_filter}
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # Check if retriever finds relevant documents
    docs = retriever.invoke(query)
    if not docs:
        return "I couldn't find any relevant information in the company documents regarding your query."

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = chain.invoke(query)
    return response["result"]

if __name__ == "__main__":
    # Uncomment to ingest data initially:
    print(ingest_data())