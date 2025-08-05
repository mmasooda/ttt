#!/usr/bin/env python3
"""
Test RAG system properly with manual vector store reloading
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.byokg_rag_engine import BYOKGRAGEngine
from src.vector.faiss_store import FAISSVectorStore

async def main():
    """Test RAG system with proper vector store loading"""
    print("🔍 Testing RAG System with Manual Vector Store Reload")
    
    try:
        # Try to load vector store manually
        vector_store = FAISSVectorStore()
        
        # Force reload from disk
        print("🔄 Force reloading vector store from disk...")
        try:
            vector_store.load()
            print(f"✅ Vector store loaded: {len(vector_store.documents)} documents")
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            return
        
        if len(vector_store.documents) == 0:
            print("❌ Vector store is still empty after reload")
            return
        
        # Test vector search first
        print("\n🔍 Testing vector search directly...")
        search_queries = ["4098", "detector", "base", "smoke detector"]
        
        for query in search_queries:
            results = await vector_store.search(query, k=3)
            print(f"\nQuery: '{query}' -> {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.4f}")
                print(f"     Content: {result.content[:150]}...")
        
        # Now test full RAG system
        print("\n🚀 Testing Full RAG System...")
        rag_engine = BYOKGRAGEngine()
        
        # Test queries related to your question
        test_queries = [
            "4098 detector bases",
            "compatible bases for detectors", 
            "smoke detector base models",
            "detector mounting bases"
        ]
        
        for query in test_queries:
            print(f"\n📝 RAG Query: '{query}'")
            try:
                result = await rag_engine.query_with_rag(query, k_vector=5)
                
                print(f"  Graph results: {len(result.get('graph_results', []))}")
                print(f"  Vector results: {len(result.get('vector_results', []))}")
                print(f"  Answer length: {len(result.get('answer', ''))}")
                
                if result.get('vector_results'):
                    print("  Vector content samples:")
                    for i, vr in enumerate(result.get('vector_results', [])[:2], 1):
                        content = vr.get('content', '')[:200]
                        print(f"    {i}. {content}...")
                
                answer = result.get('answer', '')
                if len(answer) > 0:
                    print(f"  Answer preview: {answer[:300]}...")
                
            except Exception as e:
                print(f"  ❌ RAG query failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())