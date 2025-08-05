#!/usr/bin/env python3
"""
Test the specific query about 4098-9714 compatible bases
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.byokg_rag_engine import BYOKGRAGEngine

async def main():
    """Test specific 4098-9714 query"""
    print("🔍 Testing Your Specific Query: 4098-9714 Compatible Bases")
    
    try:
        # Initialize RAG engine
        rag_engine = BYOKGRAGEngine()
        print("✅ RAG engine initialized")
        
        # Your exact query
        user_query = "give me list of all compatible bases with 4098-9714 Head"
        print(f"\n📝 Your Query: '{user_query}'")
        
        # Process with RAG
        result = await rag_engine.query_with_rag(user_query, k_vector=10)
        
        print(f"\n📊 Results:")
        print(f"  - Graph results: {len(result.get('graph_results', []))}")
        print(f"  - Vector results: {len(result.get('vector_results', []))}")
        print(f"  - Answer length: {len(result.get('answer', ''))}")
        
        # Show vector results that mention 4098-9714
        vector_results = result.get('vector_results', [])
        if vector_results:
            print(f"\n📄 Vector Results Found:")
            for i, vr in enumerate(vector_results, 1):
                content = vr.get('content', '')
                score = vr.get('score', 0)
                print(f"  {i}. Score: {score:.4f}")
                if '9714' in content:
                    print(f"     ✅ Contains 9714: {content[:300]}...")
                else:
                    print(f"     Content: {content[:200]}...")
                print()
        
        # Show the answer
        answer = result.get('answer', '')
        print(f"\n💬 Generated Answer:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # Check if answer contains real data
        if '4098-9789' in answer or '4098-9791' in answer or '4098-9792' in answer:
            print("\n✅ SUCCESS: Answer contains real base model numbers from documents!")
        else:
            print("\n❌ Answer still appears generic - not using document data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())