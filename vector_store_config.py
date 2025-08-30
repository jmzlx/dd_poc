"""Vector Store Configuration for DD-Checklist
Supports both FAISS (in-memory) and Chroma (persistent) for different deployment scenarios
"""

import streamlit as st
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path
import pickle
import json

# Try to import vector stores
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStore:
    """Base class for vector stores"""
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError
    
    def persist(self):
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS: Best for Streamlit Cloud Free Tier
    - In-memory (no persistence between sessions)
    - Very fast search
    - No additional dependencies
    - Minimal memory footprint
    """
    
    def __init__(self, dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.doc_count = 0
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents with their embeddings to the index"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        
        # Store document metadata
        for doc in documents:
            doc['_id'] = self.doc_count
            self.documents.append(doc)
            self.doc_count += 1
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Convert to float32
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Return documents with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                doc['score'] = 1 / (1 + float(dist))
                results.append(doc)
        
        return results
    
    def clear(self):
        """Clear the index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.doc_count = 0
    
    @st.cache_data
    def save_to_cache(_self) -> bytes:
        """Save index to bytes for caching"""
        import io
        buffer = io.BytesIO()
        faiss.write_index(_self.index, buffer)
        return buffer.getvalue()


class ChromaVectorStore(VectorStore):
    """
    Chroma: Good for persistent storage but has limitations on Streamlit Cloud
    - Persistent storage (survives restarts)
    - Rich metadata filtering
    - BUT: Streamlit Cloud free tier has limited disk space
    - BUT: Database resets on redeploy
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not installed. Run: pip install chromadb")
        
        # Use lightweight settings for Streamlit
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize Chroma with minimal settings
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("dd_checklist")
        except:
            self.collection = self.client.create_collection(
                name="dd_checklist",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents to Chroma"""
        # Prepare data for Chroma
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = []
        texts = []
        
        for doc in documents:
            # Chroma requires string values in metadata
            metadata = {
                'name': str(doc.get('name', '')),
                'path': str(doc.get('path', '')),
                'summary': str(doc.get('summary', ''))[:500],  # Limit length
                'type': str(doc.get('type', 'unknown'))
            }
            metadatas.append(metadata)
            texts.append(doc.get('text', doc.get('summary', '')))
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search Chroma for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'name': results['metadatas'][0][i].get('name', ''),
                    'path': results['metadatas'][0][i].get('path', ''),
                    'summary': results['metadatas'][0][i].get('summary', ''),
                    'text': results['documents'][0][i] if results['documents'] else '',
                    'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def clear(self):
        """Clear the collection"""
        self.client.delete_collection("dd_checklist")
        self.collection = self.client.create_collection(
            name="dd_checklist",
            metadata={"hnsw:space": "cosine"}
        )
    
    def persist(self):
        """Persist is automatic with Chroma"""
        pass


def get_vector_store(store_type: str = "auto", **kwargs) -> Optional[VectorStore]:
    """
    Factory function to get appropriate vector store.
    
    Recommendations:
    - Streamlit Cloud Free: Use FAISS (in-memory, fast, no persistence needed)
    - Local Development: Use Chroma (persistent, good for testing)
    - Production: Use FAISS with external storage or managed vector DB
    """
    
    if store_type == "auto":
        # Auto-detect best option
        if st.secrets.get("DEPLOYMENT", "local") == "streamlit_cloud":
            # On Streamlit Cloud, prefer FAISS for simplicity
            store_type = "faiss"
        else:
            # Local development, use Chroma if available
            store_type = "chroma" if CHROMA_AVAILABLE else "faiss"
    
    if store_type == "faiss":
        if not FAISS_AVAILABLE:
            st.warning("FAISS not available. Install with: pip install faiss-cpu")
            return None
        return FAISSVectorStore(**kwargs)
    
    elif store_type == "chroma":
        if not CHROMA_AVAILABLE:
            st.warning("Chroma not available. Install with: pip install chromadb")
            return None
        return ChromaVectorStore(**kwargs)
    
    return None


# Streamlit-specific caching utilities
@st.cache_resource
def get_cached_vector_store(store_type: str = "faiss") -> Optional[VectorStore]:
    """Get a cached vector store instance"""
    return get_vector_store(store_type)


@st.cache_data
def compute_embeddings_cached(texts: List[str], _model) -> np.ndarray:
    """Cache embeddings computation"""
    return _model.encode(texts)


# Recommendation for Streamlit Cloud Free Tier
STREAMLIT_CLOUD_RECOMMENDATION = """
### 📊 Vector Store Recommendation for Streamlit Cloud Free:

**✅ Use FAISS (In-Memory)**
- No persistence needed (documents re-indexed each session)
- Very fast search (<10ms)
- Minimal dependencies (faiss-cpu)
- Works within 1GB memory limit
- No disk space concerns

**❌ Avoid Chroma on Free Tier**
- Persistence doesn't survive redeploys anyway
- Uses disk space (limited on free tier)
- Heavier dependencies
- Slower initialization

**💡 Best Practice for Free Tier:**
1. Use FAISS for vector search
2. Cache embeddings with @st.cache_data
3. Store documents in session_state
4. Re-index on each session (fast with FAISS)
"""
