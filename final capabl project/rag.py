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
    """
    Loads a PDF and adds a metadata tag ('policy' or 'resume') 
    so retrieval doesn't mix candidate data with company policy.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Add metadata to distinguish document types as required by project guide
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
    # Load and ingest company policy / handbook
    policy_chunks = process_pdf("./handbook.pdf", doc_type="policy")
    
    # Load and ingest job description (treated as policy/JD context)
    jd_chunks = process_pdf("./job_description.pdf", doc_type="policy")
    
    # Load and ingest resume
    resume_chunks = process_pdf("./resume.pdf", doc_type="resume")
    
    vector_store.add_documents(policy_chunks + jd_chunks + resume_chunks)
    
    return "Data ingestion completed with metadata tags."

def reg_chain(query: str, doc_type_filter: str = None):
    llm = create_llm()
    
    # Filter retriever by metadata if specified (policy vs resume)
    search_kwargs = {"k": 3}
    if doc_type_filter:
        search_kwargs["filter"] = {"doc_type": doc_type_filter}
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = chain.invoke(query)
    return response["result"]

if __name__ == "__main__":
    # Uncomment to ingest data initially:
    print(ingest_data())

    query = input("Enter your query: ")
    # You can pass doc_type_filter="policy" or doc_type_filter="resume" if needed
    response = reg_chain(query)
    print("\n\nCLEAN RESPONSE:\n", response)