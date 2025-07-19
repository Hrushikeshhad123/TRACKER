# memory.py (FAISS version)

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import os
import pickle
import faiss
from langchain.docstore.in_memory import InMemoryDocstore


# Persistent storage paths
FAISS_FILE = "faiss_store.pkl"
METADATA_FILE = "faiss_metadata.pkl"

# Embedding model setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load or initialize FAISS store
if os.path.exists(FAISS_FILE):
    with open(FAISS_FILE, "rb") as f:
        vector_store = pickle.load(f)
    with open(METADATA_FILE, "rb") as f:
        all_metadata = pickle.load(f)
else:
    if os.path.exists("faiss_store.pkl"):
        with open("faiss_store.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = None

        all_metadata = []

# Save function
def _save_faiss():
    with open(FAISS_FILE, "wb") as f:
        pickle.dump(vector_store, f)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_metadata, f)

# Save a message into memory
def save_message(role, content, user_id="default", category="general"):
    doc = Document(page_content=content, metadata={
        "role": role,
        "user": user_id,
        "category": category,
        "timestamp": datetime.utcnow().isoformat()
    })
    vector_store.add_documents([doc])
    all_metadata.append(doc.metadata)
    _save_faiss()

# Get today's log of messages
def get_today_log(user_id="default", category=None):
    today = datetime.utcnow().date()
    entries = []
    for doc, meta in zip(vector_store.docstore._dict.values(), all_metadata):
        timestamp = isoparse(meta["timestamp"]).date()
        if meta["user"] == user_id and timestamp == today:
            if category is None or meta.get("category") == category:
                entries.append(f"{meta['role']} ({meta.get('category')}): {doc.page_content}")
    return "\n".join(entries)

# Daily habit suggestions
def give_daily_suggestions(user_id="default"):
    food_log = get_today_log(user_id=user_id, category="food").lower()
    exercise_log = get_today_log(user_id=user_id, category="exercise").lower()

    suggestions = []
    if any(item in food_log for item in ["burger", "pizza", "chocolate"]):
        suggestions.append("⚠️ Try reducing junk food today.")
    if any(item in food_log for item in ["egg", "dal", "chicken"]):
        suggestions.append("✅ Good protein intake!")
    if any(item in exercise_log for item in ["bench press", "deadlift"]):
        suggestions.append("💪 Strength training logged — great! Add more protein.")
    elif any(item in exercise_log for item in ["run", "cardio"]):
        suggestions.append("🏃 Cardio detected — carbs are important for recovery.")

    if not food_log:
        suggestions.append("🍽️ You haven't logged any meals today.")
    if not exercise_log:
        suggestions.append("🏋️ No exercise logged yet today.")

    return "\n".join(suggestions)

# Safely get contextual memory
def get_contextual_memory(query, k=8, user_id="default", days=3):
    try:
        results = vector_store.similarity_search(query, k=k)
    except ValueError as e:
        print(f"⚠️ FAISS store inconsistency: {e}. Clearing memory for {user_id}.")
        clear_user_memory(user_id=user_id)
        return ""
        
    recent_threshold = datetime.utcnow() - timedelta(days=days)
    formatted = []
    for res in results:
        meta = res.metadata
        if meta["user"] == user_id:
            timestamp = isoparse(meta["timestamp"])
            if timestamp >= recent_threshold:
                formatted.append(f"{meta['role']}: {res.page_content}")
    return "\n".join(formatted)

# Show all messages optionally
def show_all_user_messages(user_id="default"):
    for doc, meta in zip(vector_store.docstore._dict.values(), all_metadata):
        if meta["user"] == user_id:
            print(f"{meta['timestamp']} | {meta['role']} ({meta.get('category')}): {doc.page_content}")



import faiss
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS

def clear_user_memory(user_id="default"):
    global vector_store, all_metadata

    # Ensure vector_store is initialized
    if not hasattr(vector_store, "docstore") or not hasattr(vector_store.docstore, "_dict"):
        print("⚠️ FAISS vector store is not properly initialized.")
        return

    docs_to_keep = []
    new_metadata = []

    # Safely iterate over stored documents
    for doc_id, doc in vector_store.docstore._dict.items():
        idx = list(vector_store.docstore._dict.keys()).index(doc_id)
        meta = all_metadata[idx]
        if meta["user"] != user_id:
            docs_to_keep.append(Document(page_content=doc.page_content, metadata=meta))
            new_metadata.append(meta)

    if docs_to_keep:
        vector_store = FAISS.from_documents(docs_to_keep, embedding_model)
    else:
        # Safely initialize an empty FAISS store
        embedding_dim = len(embedding_model.embed_query("dummy"))
        index = faiss.IndexFlatL2(embedding_dim)
        docstore = InMemoryDocstore({})
        vector_store = FAISS(embedding_model.embed_query, index, {}, docstore)

    all_metadata = new_metadata
    _save_faiss()
