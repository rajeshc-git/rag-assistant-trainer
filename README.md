# AI RAG Assistant 🧠🤖

A lightweight, local-first RAG chatbot built with FastAPI, Langchain, SentenceTransformers, ChromaDB, and Ollama.

## 🔧 Features
- File upload and knowledge base generation (PDF, TXT, DOCX, CSV, XLSX)
- FastAPI backend with ChromaDB in-memory
- Ollama model (e.g. Gemma3) used for streaming generation
- Minimal frontend in HTML + Tailwind



**Create a folder called uploaded_docs**

## 🚀 Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
Frontend
Open frontend/chat.html in your browser.
Ensure Ollama is running locally (http://localhost:11434).

🧠 Model
The backend is wired to stream from gemma3:4b (via Ollama). You can change this in main.py.

📂 Upload Formats
Supported: PDF, TXT, DOCX, CSV, XLSX

🧹 Clear KB
Click ⚙️ → "Clear KB" button in the UI.

💡 Credits
Created by Rajesh Choudhury
