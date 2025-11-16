import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- 1. Configuration & Constants ---

# Define paths and model names
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "mistral"  # This must match the model you pulled with Ollama

# --- 2. Load Models & Database ---
# This code runs ONCE when the server starts.

print("Loading embedding model BGE-large-en-v1.5...")
# Initialize the BGE embedding model
model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Loading FAISS vector database...")
# Load the local FAISS database
db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Create a retriever from the database
retriever = db.as_retriever(search_kwargs={'k': 4}) # Retrieve top 4 docs

print("Loading Ollama LLM (Mistral)...")
# Initialize the Ollama LLM
llm = Ollama(model=LLM_MODEL_NAME)

print("Models and database loaded successfully.")

# --- 3. Define the RAG Chain (The "Brain") ---

# This is the prompt template your AI will use
template = """
You are an expert assistant for the NVIDIA RTX 4090.
Use the following retrieved context to answer the user's question.
If you don't know the answer from the context, just say so. Do not make up information.
Keep the answer concise and professional.

CONTEXT: 
{context}

QUESTION: 
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """A helper function to format the retrieved documents into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

# This is the main "RAG chain"
# It defines the entire process:
# 1. Take the user's question ("question").
# 2. Find relevant docs using the "retriever".
# 3. Pass the docs and question to the "prompt".
# 4. Pass the formatted prompt to the "llm".
# 5. Get the "answer" using the output parser.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. Initialize FastAPI & Rate Limiter ---

# Initialize the rate limiter (as required by your checklist)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="RTX 4090 AI Assistant API",
    description="API for the MLOps class project, powered by RAG and a local LLM."
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Pydantic model for request validation (as required by your checklist)
class Query(BaseModel):
    text: str

# --- 5. Define API Endpoints ---

@app.get("/health", summary="Health Check Endpoint")
async def health_check():
    """
    (Checklist Item: Health check endpoint)
    Simple endpoint to verify the API is running.
    """
    return {"status": "ok"}


@app.post("/ask", summary="Query Endpoint")
@limiter.limit("5/minute")  # (Checklist Item: Rate limiting)
async def ask_question(request: Request, query: Query):
    """
    (Checklist Item: Query endpoint with validation)
    Receives a user's question, processes it with the RAG chain,
    and returns the AI-generated answer.
    """
    print(f"Received query: {query.text}")
    
    # 1. Get the answer from the RAG chain
    #    The 'invoke' method runs the entire chain
    answer = rag_chain.invoke(query.text)
    
    print(f"Generated answer: {answer}")

    # 2. Return the answer
    return {"answer": answer}


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)