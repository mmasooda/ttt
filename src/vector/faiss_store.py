#!/usr/bin/env python3
"""
FAISS Vector Database for Text Retrieval
Handles embedding storage and similarity search for BYOKG-RAG
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from dataclasses import dataclass, asdict

from ..utils import logger, settings
from ..llm.openai_client import OpenAIClient

@dataclass
class VectorDocument:
    """Document chunk for vector storage"""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class FAISSVectorStore:
    """FAISS-based vector database for document retrieval"""
    
    def __init__(self, dimension: int = 1536, index_type: str = "IVF"):
        self.dimension = dimension  # text-embedding-3-small dimension
        self.index_type = index_type
        self.index = None
        self.documents: List[VectorDocument] = []
        self.openai_client = OpenAIClient()
        
        # Storage paths
        self.storage_dir = Path("./data/vector_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.docs_path = self.storage_dir / "documents.pkl"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on type"""
        if self.index_type == "IVF":
            # IVF (Inverted File) index for larger datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 centroids
        elif self.index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) for fast search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # Simple flat index for small datasets
            self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info("FAISS index initialized", type=self.index_type, dimension=self.dimension)
    
    async def add_documents(self, documents: List[VectorDocument]) -> int:
        """Add documents to the vector store"""
        
        if not documents:
            return 0
        
        logger.info("Adding documents to vector store", count=len(documents))
        
        # Generate embeddings for documents without them
        docs_needing_embeddings = [doc for doc in documents if doc.embedding is None]
        
        if docs_needing_embeddings:
            logger.info("Generating embeddings", count=len(docs_needing_embeddings))
            
            texts = [doc.content for doc in docs_needing_embeddings]
            embeddings = await self.openai_client.generate_embeddings(texts)
            
            # Assign embeddings to documents
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc.embedding = embedding
        
        # Prepare embeddings matrix
        embeddings_matrix = np.array([doc.embedding for doc in documents], dtype=np.float32)
        
        # Add to FAISS index
        if self.index_type == "IVF" and not self.index.is_trained:
            # Train IVF index if not already trained
            if len(self.documents) + len(documents) >= 100:  # Need minimum samples for training
                all_embeddings = embeddings_matrix
                if self.documents:
                    existing_embeddings = np.array([doc.embedding for doc in self.documents], dtype=np.float32)
                    all_embeddings = np.vstack([existing_embeddings, embeddings_matrix])
                
                logger.info("Training IVF index", samples=len(all_embeddings))
                self.index.train(all_embeddings)
        
        # Add embeddings to index
        start_id = len(self.documents)
        if self.index_type == "IVF" and self.index.is_trained:
            self.index.add(embeddings_matrix)
        elif self.index_type != "IVF":
            self.index.add(embeddings_matrix)
        else:
            logger.warning("IVF index not trained yet, storing documents without indexing")
        
        # Store documents
        self.documents.extend(documents)
        
        logger.info("Documents added to vector store", 
                   total_documents=len(self.documents),
                   indexed=self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.documents))
        
        return len(documents)
    
    async def search(self, query: str, k: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Generate query embedding
            query_embeddings = await self.openai_client.generate_embeddings([query])
            if not query_embeddings:
                logger.error("Failed to generate query embedding")
                return []
            
            query_vector = np.array([query_embeddings[0]], dtype=np.float32)
            
            # Perform search
            if self.index_type == "IVF" and not self.index.is_trained:
                # Fallback to brute force search if index not trained
                return await self._brute_force_search(query_vector[0], k, score_threshold)
            
            # FAISS search
            distances, indices = self.index.search(query_vector, min(k, len(self.documents)))
            
            # Convert distances to similarity scores (cosine similarity)
            # FAISS L2 distance to cosine similarity approximation
            similarities = 1 / (1 + distances[0])
            
            results = []
            for idx, (doc_idx, similarity) in enumerate(zip(indices[0], similarities)):
                if doc_idx == -1:  # Invalid index
                    continue
                
                if similarity >= score_threshold:
                    doc = self.documents[doc_idx]
                    results.append({
                        'id': doc.id,
                        'content': doc.content,
                        'source': doc.source,
                        'metadata': doc.metadata,
                        'score': float(similarity),
                        'rank': idx + 1
                    })
            
            logger.info("Vector search completed", 
                       query_length=len(query),
                       results_found=len(results),
                       top_score=results[0]['score'] if results else 0)
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []
    
    async def _brute_force_search(self, query_vector: np.ndarray, k: int, threshold: float) -> List[Dict[str, Any]]:
        """Brute force similarity search when index is not available"""
        
        if not self.documents:
            return []
        
        # Calculate similarities manually
        doc_embeddings = np.array([doc.embedding for doc in self.documents], dtype=np.float32)
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        
        similarities = np.dot(doc_embeddings, query_vector) / (doc_norms * query_norm)
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            similarity = float(similarities[idx])
            if similarity >= threshold:
                doc = self.documents[idx]
                results.append({
                    'id': doc.id,
                    'content': doc.content,
                    'source': doc.source,
                    'metadata': doc.metadata,
                    'score': similarity,
                    'rank': rank + 1
                })
        
        return results
    
    async def add_document_chunks(self, content: str, source: str, metadata: Dict[str, Any], 
                                 chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Add document by chunking it into smaller pieces"""
        
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = VectorDocument(
                id=f"{source}_{i}",
                content=chunk,
                source=source,
                metadata={
                    **metadata,
                    'chunk_index': i,
                    'chunk_count': len(chunks),
                    'chunk_size': len(chunk)
                }
            )
            documents.append(doc)
        
        return await self.add_documents(documents)
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def save_to_disk(self):
        """Save vector store to disk"""
        
        try:
            # Save FAISS index
            if self.index.ntotal > 0:
                faiss.write_index(self.index, str(self.index_path))
                logger.info("FAISS index saved", path=self.index_path, vectors=self.index.ntotal)
            
            # Save documents
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'document_count': len(self.documents),
                'indexed_count': self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Vector store saved to disk", documents=len(self.documents))
            
        except Exception as e:
            logger.error("Failed to save vector store", error=str(e))
    
    async def load_from_disk(self) -> bool:
        """Load vector store from disk"""
        
        try:
            # Check if files exist
            if not all(p.exists() for p in [self.index_path, self.docs_path, self.metadata_path]):
                logger.info("Vector store files not found, starting fresh")
                return False
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify compatibility
            if metadata['dimension'] != self.dimension or metadata['index_type'] != self.index_type:
                logger.warning("Vector store metadata mismatch, starting fresh")
                return False
            
            # Load FAISS index
            if self.index_path.exists() and metadata['indexed_count'] > 0:
                self.index = faiss.read_index(str(self.index_path))
                logger.info("FAISS index loaded", vectors=self.index.ntotal)
            
            # Load documents
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info("Vector store loaded from disk", 
                       documents=len(self.documents),
                       indexed=metadata['indexed_count'])
            
            return True
            
        except Exception as e:
            logger.error("Failed to load vector store", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        
        return {
            'total_documents': len(self.documents),
            'indexed_vectors': self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'index_trained': getattr(self.index, 'is_trained', True),
            'storage_size_mb': sum(
                p.stat().st_size for p in [self.index_path, self.docs_path, self.metadata_path] 
                if p.exists()
            ) / (1024 * 1024)
        }
    
    async def clear(self):
        """Clear all data from vector store"""
        
        self.documents.clear()
        self._initialize_index()
        
        # Remove disk files
        for path in [self.index_path, self.docs_path, self.metadata_path]:
            if path.exists():
                path.unlink()
        
        logger.info("Vector store cleared")

# Import pandas for timestamp
import pandas as pd