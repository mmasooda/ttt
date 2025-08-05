#!/usr/bin/env python3
"""
Test specific query about compatible base model numbers
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.byokg_rag_engine import BYOKGRAGEngine

async def main():
    """Test the specific query from the user"""
    print("🔍 Testing specific query: 'tell me all compatible bases model numbers which works with 4098-9714'")
    
    try:
        # Initialize RAG engine
        rag_engine = BYOKGRAGEngine()
        print("✅ RAG engine initialized")
        
        # Test the specific query
        query = "tell me all compatible bases model numbers which works with 4098-9714"
        print(f"\n📝 Query: {query}")
        
        result = await rag_engine.query_with_rag(query)
        
        print(f"\n📊 Results:")
        print(f"  - Graph results: {len(result.get('graph_results', []))}")
        print(f"  - Vector results: {len(result.get('vector_results', []))}")
        print(f"  - Sources: {len(result.get('sources', []))}")
        print(f"  - Answer length: {len(result.get('answer', ''))}")
        
        print(f"\n💬 Answer:")
        print("=" * 80)
        print(result.get('answer', 'No answer generated'))
        print("=" * 80)
        
        if result.get('sources'):
            print(f"\n📚 Sources:")
            for i, source in enumerate(result.get('sources', []), 1):
                print(f"  {i}. {source}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())