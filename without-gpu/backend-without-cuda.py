from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, uuid, json, requests
from typing import List

from langchain.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
    CSVLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

# --- Init ---
app = FastAPI()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Init Embeddings ---
print("🔧 Initializing SentenceTransformer...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Init ChromaDB (In-Memory) ---
print("💾 Using ChromaDB in memory (safe mode, no disk write)...")
client = Client(Settings(anonymized_telemetry=False))
doc_collection = client.get_or_create_collection(name="documents")

# --- Restore from JSON Backup ---
def restore_from_backup():
    global doc_collection
    if os.path.exists("chroma_backup.json"):
        try:
            with open("chroma_backup.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate data structure
            required_keys = ["documents", "metadatas", "ids"]
            if not all(key in data for key in required_keys):
                print("⚠️ Backup file missing required keys")
                return
                
            # Ensure all lists have the same length
            num_items = len(data["ids"])
            if not all(len(data[key]) == num_items for key in required_keys):
                print("⚠️ Backup data length mismatch")
                return
                
            # Check if embeddings exist and match length
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) != num_items:
                print("⚠️ Embeddings length mismatch")
                return
                
            # Clear existing collection
            client.delete_collection("documents")
            doc_collection = client.get_or_create_collection(name="documents")
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, num_items, batch_size):
                batch_docs = data["documents"][i:i+batch_size]
                batch_metas = data["metadatas"][i:i+batch_size]
                batch_ids = data["ids"][i:i+batch_size]
                batch_embeds = embeddings[i:i+batch_size] if embeddings else None
                
                doc_collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    embeddings=batch_embeds
                )
            
            print(f"🔁 Restored {num_items} knowledge base from backup.")
        except Exception as e:
            print(f"⚠️ Error restoring backup: {e}")
            
restore_from_backup()

# --- Save to JSON Backup ---
def backup_to_json():
    try:
        # Get all items from the collection
        results = doc_collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        
        # Convert numpy arrays to lists if they exist
        embeddings = results.get("embeddings")
        if embeddings is not None:
            embeddings = [embed.tolist() if hasattr(embed, 'tolist') else embed for embed in embeddings]
        
        data = {
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "ids": results.get("ids", []),
            "embeddings": embeddings if embeddings is not None else [],
        }
        
        with open("chroma_backup.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📁 Saved backup with {len(data['ids'])} documents")
    except Exception as e:
        print(f"⚠️ Error during backup: {e}")

# --- Clear KB ---
@app.post("/clear_kb")
def clear_knowledge_base():
    import gc
    try:
        client.delete_collection("documents")
        global doc_collection
        doc_collection = client.get_or_create_collection("documents")
        
        # Optional: wipe backup file
        if os.path.exists("chroma_backup.json"):
            os.remove("chroma_backup.json")

        gc.collect()
        print("✅ Cleared KB. Ready to re-upload.")
        return {"status": "✅ Cleared in-memory KB"}
    except Exception as e:
        return {"error": str(e)}


# --- Load & Chunk ---
def load_and_split(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "csv":
        loader = CSVLoader(file_path)
    elif ext == "xlsx":
        loader = UnstructuredExcelLoader(file_path)
    else:
        print(f"❌ Unsupported file type: .{ext}")
        return []

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- Upload Files ---
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global doc_collection
    print(f"\n📁 Received {len(files)} files for upload.")
    total_chunks = 0
    batch_size = 5000  # Well below ChromaDB's limit

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"✅ Saved file to {file_path}")
        chunks = load_and_split(file_path)
        print(f"📄 Split into {len(chunks)} chunks")

        # Process in batches with progress
        total_batches = (len(chunks) // batch_size) + (1 if len(chunks) % batch_size else 0)
        print(f"⚙️ Processing in {total_batches} batches of {batch_size}...")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]

            texts, embeddings, metadatas, ids = [], [], [], []
            for i, chunk in enumerate(batch):
                text = chunk.page_content
                texts.append(text)
                embeddings.append(embedding_model.encode(text).tolist())
                metadatas.append({"source": file.filename, "ext": file.filename.split(".")[-1]})
                ids.append(f"{file_id}_{start_idx + i}")

            if texts:
                doc_collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                total_chunks += len(texts)
            
            # Print progress
            progress = (batch_num + 1) / total_batches * 100
            print(f"\r🔄 Processing batch {batch_num + 1}/{total_batches} ({progress:.1f}%)", end="")
            # Optional: backup after each batch for large files
            if batch_num % 10 == 0:  # Backup every 10 batches
                backup_to_json()

        print()  # New line after progress

    backup_to_json()
    print(f"📊 ChromaDB now contains {doc_collection.count()} total documents.")

    return {
        "status": "✅ Uploaded and embedded",
        "total_chunks": total_chunks,
        "chroma_doc_count": doc_collection.count()
    }

# --- Chat API ---
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data["message"]
    file_filter = data.get("filename")
    ext_filter = data.get("ext")

    query_embedding = embedding_model.encode(user_input).tolist()
    filter_dict = {}
    if file_filter:
        filter_dict["source"] = file_filter
    if ext_filter:
        filter_dict["ext"] = ext_filter

    results = doc_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=filter_dict or None
    )
    context = "\n---\n".join(results.get("documents", [[]])[0])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

User: {user_input}
Answer:
"""

    def stream_response():
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:12b", "prompt": prompt, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        yield f"data: {json.dumps({'content': data['response']})}\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")
