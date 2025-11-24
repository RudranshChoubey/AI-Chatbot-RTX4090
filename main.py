import uvicorn
import mlflow
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "mistral"


print("Loading embedding model...")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Loading FAISS database...")
try:
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
except:
    print("No existing database found. Creating a new empty one.")
    db = FAISS.from_texts(["Initial empty index"], embeddings)

retriever = db.as_retriever(search_kwargs={'k': 4})

print("Loading Ollama...")
llm = OllamaLLM(model=LLM_MODEL_NAME, base_url="http://host.docker.internal:11434")


template = """
You are an expert assistant. Use the provided context to answer the question.
If the answer is not in the context, say so.
CONTEXT: {context}
QUESTION: {question}
ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RTX 4090 AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)

class Query(BaseModel):
    text: str

class RetrainRequest(BaseModel):
    url: str  



@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, query: Query):
    with mlflow.start_run():
        mlflow.log_param("question", query.text)
        answer = rag_chain.invoke(query.text)
        mlflow.log_param("answer", answer)
        return {"answer": answer}

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    print(f"Retraining model with data from: {request.url}")
    try:
        
        loader = WebBaseLoader(request.url)
        docs = loader.load()
        

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        
        
        db.add_documents(chunks)
        
       
        db.save_local(FAISS_INDEX_PATH)
        
        return {"status": "success", "message": f"Learned {len(chunks)} new chunks from {request.url}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="static_ui/static"), name="static")


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    file_path = os.path.join("static_ui", full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    return FileResponse("static_ui/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)