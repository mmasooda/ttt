#!/usr/bin/env python3
"""
Fix vector store loading and test specific model queries
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vector.faiss_store import FAISSVectorStore
from src.core.byokg_rag_engine import BYOKGRAGEngine

async def main():
    """Fix vector store loading and test with specific models"""
    print("🔧 Fixing Vector Store Loading Issue")
    
    try:
        # Initialize vector store
        vector_store = FAISSVectorStore()
        print(f"📊 Initial documents: {len(vector_store.documents)}")
        
        # Load from disk
        print("🔄 Loading vector store from disk...")
        loaded = await vector_store.load_from_disk()
        
        if loaded:
            print(f"✅ Vector store loaded: {len(vector_store.documents)} documents")
            
            # Test search for actual model numbers found in data
            test_queries = [
                "4098-5288",  # This model exists in the data
                "4098-5260",  # Base for detector 
                "4098-5289",  # Heat detector
                "4098-5290",  # Triple sensor
                "detector base",
                "smoke detector",
                "base for detector"
            ]
            
            print("\n🔍 Testing vector search for existing models...")
            for query in test_queries:
                results = await vector_store.search(query, k=3)
                print(f"\nQuery: '{query}' -> {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result.score:.4f}")
                    print(f"     Source: {result.metadata.get('source', 'Unknown')}")
                    content_preview = result.content[:200].replace('\n', ' ')
                    print(f"     Content: {content_preview}...")
                    print()
            
            # Now test with RAG engine
            print("\n🚀 Testing RAG System with Fixed Vector Store...")
            
            # Test your specific query about compatible bases
            user_query = "give me list of all compatible bases with 4098-5288 detector"
            print(f"\n📝 User Query: '{user_query}'")
            
            try:
                rag_engine = BYOKGRAGEngine()
                result = await rag_engine.query_with_rag(user_query, k_vector=10)
                
                print(f"\n📊 RAG Results:")
                print(f"  - Graph results: {len(result.get('graph_results', []))}")
                print(f"  - Vector results: {len(result.get('vector_results', []))}")
                print(f"  - Answer length: {len(result.get('answer', ''))}")
                
                # Show vector results that were found
                vector_results = result.get('vector_results', [])
                if vector_results:
                    print(f"\n📄 Vector Search Results Found:")
                    for i, vr in enumerate(vector_results[:3], 1):
                        content = vr.get('content', '')
                        score = vr.get('score', 0)
                        print(f"  {i}. Score: {score:.4f}")
                        print(f"     Content: {content[:300]}...")
                        print()
                
                # Show the generated answer
                answer = result.get('answer', '')
                print(f"\n💬 Generated Answer:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
                
            except Exception as e:
                print(f"❌ RAG engine error: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            print("❌ Failed to load vector store from disk")
            print("📁 Available files:")
            storage_dir = Path("./data/vector_store")
            for file in storage_dir.iterdir():
                print(f"  - {file.name}: {file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())