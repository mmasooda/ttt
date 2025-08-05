#!/usr/bin/env python3
"""
Debug vector search to see what data is actually stored
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vector.faiss_store import FAISSVectorStore

async def main():
    """Debug the vector store contents"""
    print("🔍 Debugging Vector Store Contents")
    
    try:
        # Initialize vector store
        vector_store = FAISSVectorStore()
        print(f"✅ Vector store initialized")
        print(f"📊 Total documents: {len(vector_store.documents)}")
        
        # Test search for 4098 models
        test_queries = [
            "4098-9714",
            "4098 detector base",
            "4098-5288 base",
            "compatible base",
            "smoke detector base"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = await vector_store.search(query, k=3)
            
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                content_preview = result.content[:200].replace('\n', ' ')
                print(f"    {i}. Score: {result.score:.4f}")
                print(f"       Source: {result.metadata.get('source', 'Unknown')}")
                print(f"       Content: {content_preview}...")
                print()
        
        # Check if any documents contain 4098
        print("\n📄 Documents containing '4098':")
        count = 0
        for doc in vector_store.documents:
            if '4098' in doc.content:
                count += 1
                if count <= 5:  # Show first 5
                    preview = doc.content[:300].replace('\n', ' ')
                    print(f"  {count}. Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"     Content: {preview}...")
                    print()
        
        print(f"Total documents with '4098': {count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())