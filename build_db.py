import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS

# --- 1. Define Constants ---
# These are the URLs to scrape for knowledge
# We can add as many as we want!
SOURCE_URLS = [
    "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
    "https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889",
    "https://en.wikipedia.org/wiki/GeForce_40_series",
    "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-lovelace-architecture-whitepaper-v1.1.pdf" # This loader can also handle direct PDF links
]

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
FAISS_INDEX_PATH = "faiss_index"

def load_and_split_urls(urls):
    """Loads web pages from URLs and splits them into manageable chunks."""
    
    print(f"Loading documents from {len(urls)} URLs...")
    
    # Load the documents using WebBaseLoader
    # This will fetch the HTML and parse the text
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 1  # Be polite to the servers
    documents = loader.load()
    
    print(f"Documents loaded. Total pages: {len(documents)}. Splitting into chunks...")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max size of each chunk
        chunk_overlap=150   # Overlap between chunks
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully split into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    """Creates a FAISS vector store from text chunks and saves it."""
    
    print(f"Loading embedding model {EMBED_MODEL_NAME}...")
    
    # Initialize the BGE embedding model
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have an NVIDIA GPU
    encode_kwargs = {'normalize_embeddings': True} # Important for BGE
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("Embedding model loaded. Creating FAISS vector store...")
    
    # This step can take a few minutes as it embeds all chunks
    start_time = time.time()
    
    # Create the FAISS vector store from the document chunks
    db = FAISS.from_documents(chunks, embeddings)
    
    end_time = time.time()
    
    print(f"Vector store created in {end_time - start_time:.2f} seconds.")
    
    # Save the vector store locally
    db.save_local(FAISS_INDEX_PATH)
    
    print(f"Vector store saved to: {FAISS_INDEX_PATH}")

def main():
    """Main function to run the data pipeline."""
    try:
        chunks = load_and_split_urls(SOURCE_URLS)
        create_vector_store(chunks)
        print("\n--- Knowledge Base creation from URLs successful! ---")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()