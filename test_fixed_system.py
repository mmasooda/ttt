#!/usr/bin/env python3
"""
Test the fixed RAG system with corrected thresholds and parameter binding
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vector.faiss_store import FAISSVectorStore
from src.core.byokg_rag_engine import BYOKGRAGEngine

async def main():
    """Test the fixed system"""
    print("🔧 Testing Fixed RAG System")
    
    try:
        # Test vector store first
        print("\n1️⃣ Testing Vector Store with Fixed Threshold (0.3)")
        vector_store = FAISSVectorStore()
        await vector_store.load_from_disk()
        print(f"✅ Loaded {len(vector_store.documents)} documents")
        
        # Test searches that should now work
        test_queries = [
            "4098-5288",      # Exact model in data
            "4098-5260",      # Base model in data  
            "smoke detector", # General term
            "detector base",  # General term
            "4098"            # Model family
        ]
        
        for query in test_queries:
            results = await vector_store.search(query, k=3)
            print(f"\n🔍 Query: '{query}' -> {len(results)} results")
            
            if results:
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    content = result.get('content', '')[:150]
                    print(f"  {i}. Score: {score:.4f} - {content}...")
            else:
                print("  ❌ Still no results - investigate further")
        
        # Test full RAG system
        print("\n2️⃣ Testing Full RAG System")
        rag_engine = BYOKGRAGEngine()
        
        # Test with a model that exists in the data
        user_query = "What is 4098-5288 and what base does it use?"
        print(f"\n📝 User Query: '{user_query}'")
        
        result = await rag_engine.query_with_rag(user_query, k_vector=5)
        
        print(f"\n📊 RAG Results:")
        print(f"  - Graph results: {len(result.get('graph_results', []))}")
        print(f"  - Vector results: {len(result.get('vector_results', []))}")
        print(f"  - Answer length: {len(result.get('answer', ''))}")
        
        # Show actual vector results found
        vector_results = result.get('vector_results', [])
        if vector_results:
            print(f"\n📄 Vector Results Found:")
            for i, vr in enumerate(vector_results[:2], 1):
                content = vr.get('content', '')
                score = vr.get('score', 0)
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Content: {content[:200]}...")
                print()
        
        # Show the answer
        answer = result.get('answer', '')
        if 'generic' not in answer.lower() and len(vector_results) > 0:
            print("✅ SUCCESS: System is now returning data-based answers!")
        else:
            print("❌ STILL FAILING: Answer is still generic")
        
        print(f"\n💬 Answer Preview:")
        print(answer[:400] + "..." if len(answer) > 400 else answer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())