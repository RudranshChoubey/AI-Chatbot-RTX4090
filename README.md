# AI-Chatbot-RTX4090
Steps to run the chatbot on your local machine:

1. Install Prerequisites: Install Docker Desktop and Ollama. Run ollama run mistral in your terminal once to download the brain.

2. Build the Knowledge Base: Open the folder in a terminal and run: python build_database.py. This creates the database. Skip this step if you want to use the premade FAISS DB      included in the zip file.

3. Build the App: Run this command: "docker build -t rtx-unified ."

4. Run the App: Run this command: "docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --add-host=host.docker.internal:host-gateway \
  rtx-unified"
