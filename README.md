# RTX 4090 Expert AI Assistant (MLOps Project)

An end-to-end MLOps implementation of a Retrieval-Augmented Generation (RAG) system. This assistant acts as a technical expert for the NVIDIA RTX 4090 GPU, utilizing official documentation, whitepapers, and technical specs to provide accurate, context-aware answers.

## 🚀 Project Overview
This project demonstrates a full MLOps lifecycle, including data ingestion from PDFs, vector database management, LLM orchestration, and a robust CI/CD pipeline.

### Core Features
- **RAG Architecture:** Uses the `BGE-large-en-v1.5` embedding model and `FAISS` vector store to retrieve context from NVIDIA PDFs.
- **LLM Integration:** Powered by `Mistral-7B` (or Llama-2) to generate natural language responses.
- **Production API:** High-performance backend built with `FastAPI`.
- **DevOps Ready:** Fully containerized with Docker and orchestrated via Kubernetes (K8s).
- **Observability:** Real-time monitoring using Prometheus and Grafana.

---

## 🏗 System Architecture

1.  **Data Ingestion:** PDF text extraction using `PyPDF2` and chunking via `LangChain`.
2.  **Vectorization:** Text chunks are converted to embeddings and stored in `FAISS`.
3.  **Inference:** User queries are matched against the vector store; retrieved context is sent to the LLM.
4.  **Deployment:** Automated via Jenkins into a Kubernetes cluster.

---

## 🛠 Tech Stack

| Component | Tool |
| :--- | :--- |
| **LLM / RAG** | LangChain, Mistral-7B, FAISS, BGE Embeddings |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | React / Streamlit |
| **ML Lifecycle** | MLflow (Experiment Tracking) |
| **DevOps** | Docker, Kubernetes, Jenkins |
| **Monitoring** | Prometheus, Grafana |

---

## 📂 Project Structure
```text
├── data/               # Official NVIDIA RTX 4090 PDFs
├── src/
│   ├── ingestion.py    # PDF processing and Vector DB creation
│   ├── main.py         # FastAPI application logic
│   └── model.py        # LLM & RAG chain configuration
├── k8s/                # Kubernetes Deployment & Service manifests
├── tests/              # Pytest suite for API & RAG logic
├── Dockerfile          # Containerization script
├── Jenkinsfile         # CI/CD Pipeline definition
└── requirements.txt    # Python dependencies
⚙️ Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/rtx4090-ai-assistant.git](https://github.com/your-username/rtx4090-ai-assistant.git)
cd rtx4090-ai-assistant
2. Set Up Environment
Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Initialize Vector Database
Place your PDFs in the /data folder and run:

Bash
python src/ingestion.py
4. Run Locally
Bash
uvicorn src.main:app --reload
The API will be available at http://localhost:8000. You can access the interactive docs at /docs.

🚢 Deployment (DevOps)
Docker
Build the image:

Bash
docker build -t rtx4090-assistant:latest .
Kubernetes
Apply the manifests:

Bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
CI/CD Pipeline
The project includes a Jenkinsfile that automates:

Linting & Testing (Pytest)

Docker Image Build & Push

Deployment to K8s Cluster

📊 Monitoring & Tracking
MLflow: View experiment runs and model parameters at http://localhost:5000.

Grafana: Monitor API latency and system health via the custom dashboard.

Prometheus: Metrics are exposed via the /metrics endpoint.

📝 API Endpoints
POST /ask: Submit a question about the RTX 4090.

GET /health: Returns the status of the API and Vector DB connection.

GET /metrics: Prometheus metrics endpoint.


### How to use this:
1.  **Customize the links:** Replace the placeholder GitHub URL with your actual one.
2.  **Verify file paths:** Make sure the structure in the "Project Structure" section matches how you actually organize your folders.
3.  **Export to PDF:** If your professor requires a PDF version of the README, you can open this in **VS Code**, right-click, and select "Markdown: Export as PDF" (requires an extension) or simply copy-paste it into a Word/Google Doc and save as PDF.
