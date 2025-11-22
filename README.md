# AI-Chatbot-RTX4090
Steps to run the chatbot on your local machine:

1. Install Prerequisites: Install Docker Desktop and Ollama. Run "ollama run mistral" in your terminal once to download the brain.

2. Run "pip install -r requirements.txt" to install all the required python libraries. 

3. Build the Knowledge Base: Open the folder in a terminal and run: python build_database.py. This creates the database.
 
NOTE: IF USING THE CHATBOT FILE WITH FAISS FOLDER THEN SKIP STEP 3.

5. Build the App: Run this command: "docker build -t rtx-unified ."

6. Run the App: "docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --add-host=host.docker.internal:host-gateway \
  rtx-unified"
