from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, uuid, json, requests
from typing import List
from tqdm import tqdm
import torch
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Using device: {device}")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,       
        chunk_overlap=200 
    )

    chunks = splitter.split_documents(docs)

    return chunks

# --- Upload Files ---
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global doc_collection
    print(f"\n📁 Received {len(files)} files for upload.")
    total_chunks = 0

    # Configure batch sizes
    embedding_batch_size = 1024  
    chroma_batch_size = 5000    

    for file in files:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"✅ Saved file to {file_path}")
        chunks = load_and_split(file_path)
        print(f"📄 Split into {len(chunks)} chunks")

        # Filter valid chunks
        texts = [chunk.page_content.strip() for chunk in chunks if len(chunk.page_content.strip()) >= 50]
        if not texts:
            print("⚠️ No valid chunks found.")
            continue

        metadatas = [{"source": file.filename, "ext": file.filename.split(".")[-1]} for _ in texts]
        ids = [f"{file_id}_{i}" for i in range(len(texts))]

        print(f"🧠 Embedding {len(texts)} chunks (batch size = {embedding_batch_size})")

        # Batch embed using tqdm for a clean progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), embedding_batch_size), desc="🔄 Embedding Progress", unit="batch"):
            batch_texts = texts[i:i + embedding_batch_size]
            embeddings = embedding_model.encode(
                batch_texts,
                batch_size=min(32, len(batch_texts)),
                device=device,
                show_progress_bar=False  # disable inside tqdm
            )
            all_embeddings.extend(embeddings)

        # Batch insert into ChromaDB
        print(f"📥 Inserting into ChromaDB (batch size = {chroma_batch_size})")
        for j in range(0, len(texts), chroma_batch_size):
            doc_collection.add(
                documents=texts[j:j + chroma_batch_size],
                embeddings=[e.tolist() for e in all_embeddings[j:j + chroma_batch_size]],
                metadatas=metadatas[j:j + chroma_batch_size],
                ids=ids[j:j + chroma_batch_size]
            )
            print(f"✅ Inserted {min(chroma_batch_size, len(texts) - j)} docs")

        total_chunks += len(texts)

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

    query_embedding = embedding_model.encode(user_input, device=device).tolist()
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

Use Markdown tables for multiple items.

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