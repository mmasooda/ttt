#!/usr/bin/env python3
"""
Test the improved BYOKG-RAG system with iterative refinement
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.byokg_rag_engine import BYOKGRAGEngine

async def main():
    """Test improved ingestion system"""
    print("🚀 Testing Improved BYOKG-RAG System with Iterative Refinement")
    
    try:
        # Sample fire alarm content with known entities
        sample_content = """
        The 4098-9714 smoke detector head is compatible with the following bases:
        - 4098-9789 Standard Base
        - 4098-9791 Isolator Base  
        - 4098-9792 Relay Base
        - 4098-9793 Sounder Base
        - 4098-9794 Enhanced Base
        
        The 4098-9714 detector operates on 15-32 VDC and complies with BS 5839 and EN 54 standards.
        It is manufactured by Tyco Safety Products (formerly Simplex) and is suitable for use 
        with 4100U, 4100ES, 4008, and 4120 control panels.
        
        Installation requires proper wiring using 4098 series cable assemblies.
        The detector base answers to two addresses on the IDNet communication loop.
        """
        
        # Initialize RAG engine
        rag_engine = BYOKGRAGEngine()
        print("✅ RAG engine initialized")
        
        # Test improved ingestion with iterative refinement
        print("\n📄 Testing Improved LLM-Assisted Ingestion...")
        result = await rag_engine.ingest_document_with_llm(
            content=sample_content,
            source="test_fire_alarm_spec.pdf",
            document_type="fire_alarm_specification"
        )
        
        print(f"\n📊 Ingestion Results:")
        for key, value in result.items():
            print(f"  - {key}: {value}")
        
        # Test specific query that should now work better
        print("\n🔍 Testing Query with Improved Knowledge Base...")
        user_query = "What bases are compatible with 4098-9714 detector?"
        
        rag_result = await rag_engine.query_with_rag(user_query, k_vector=5)
        
        print(f"\n📊 Query Results:")
        print(f"  - Graph results: {len(rag_result.get('graph_results', []))}")
        print(f"  - Vector results: {len(rag_result.get('vector_results', []))}")
        print(f"  - Answer length: {len(rag_result.get('answer', ''))}")
        
        # Check if we get better quality results
        answer = rag_result.get('answer', '')
        quality_indicators = [
            '4098-9789' in answer,
            '4098-9791' in answer, 
            '4098-9792' in answer,
            'BS 5839' in answer,
            'Tyco' in answer or 'Simplex' in answer
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        print(f"\n📈 Answer Quality Assessment:")
        print(f"  - Contains specific base models: {'✅' if any(quality_indicators[:3]) else '❌'}")
        print(f"  - Contains standards reference: {'✅' if quality_indicators[3] else '❌'}")
        print(f"  - Contains manufacturer info: {'✅' if quality_indicators[4] else '❌'}")
        print(f"  - Overall quality score: {quality_score:.2f}")
        
        print(f"\n💬 Generated Answer:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        if quality_score > 0.6:
            print("\n🎉 SUCCESS: High quality answer with specific technical details!")
        elif quality_score > 0.3:
            print("\n⚠️  PARTIAL SUCCESS: Some improvement but still needs work")
        else:
            print("\n❌ STILL POOR: Answer quality remains low")
            
        # Check Neo4j database status
        print("\n📊 Checking Neo4j Database...")
        stats = await rag_engine.get_system_stats()
        neo4j_stats = stats.get('neo4j', {})
        
        print(f"  - Total nodes: {neo4j_stats.get('nodes', 0)}")
        print(f"  - Total relationships: {neo4j_stats.get('relationships', 0)}")
        print(f"  - Documents processed: {neo4j_stats.get('documents', 0)}")
        
        if neo4j_stats.get('nodes', 0) > 0:
            print("✅ Neo4j database now contains entities!")
        else:
            print("❌ Neo4j database still empty - persistence issue remains")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())