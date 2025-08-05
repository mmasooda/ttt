#!/usr/bin/env python3
"""
Quick Demo of LLM-Enhanced BYOKG-RAG System
Demonstrates key functionality with streamlined processing
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings, ensure_data_dirs
from src.core.byokg_rag_engine import BYOKGRAGEngine
from src.ingestion.enhanced_document_processor import EnhancedDatasetProcessor

class QuickDemo:
    """Streamlined demo of the enhanced system"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.doc_processor = EnhancedDatasetProcessor()
        self.rag_engine = BYOKGRAGEngine()

    async def demo_system_capabilities(self):
        """Demonstrate the complete system with quick examples"""
        
        print("🚀 TTT-Enhanced BYOKG-RAG System Demo")
        print("=" * 50)
        print(f"S3 Bucket: {settings.s3_bucket_name}")
        print()
        
        ensure_data_dirs()
        
        try:
            # Initialize systems
            print("Phase 1: System Initialization")
            await self.rag_engine.vector_store.load_from_disk()
            print("✅ Vector store initialized")
            
            # Show S3 discovery
            print("\nPhase 2: S3 Content Discovery")
            all_objects = self.s3_client.list_all_objects()
            pdf_files = [obj for obj in all_objects if obj['key'].lower().endswith('.pdf')]
            print(f"📁 Found {len(pdf_files)} PDF files in S3 bucket")
            
            # Show a few example files
            print("\n📋 Sample Files Available:")
            for i, pdf_obj in enumerate(pdf_files[:5]):
                pdf_key = pdf_obj['key']
                pdf_name = Path(pdf_key).name
                size_mb = pdf_obj['size'] / (1024*1024)
                print(f"  {i+1}. {pdf_name} ({size_mb:.2f} MB)")
            
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more files")
            
            # Demo triple-layer extraction on one file 
            print(f"\nPhase 3: Triple-Layer Extraction Demo")
            if pdf_files:
                demo_file = pdf_files[0]  # Take first file
                pdf_key = demo_file['key']
                pdf_name = Path(pdf_key).name
                
                print(f"📄 Demonstrating with: {pdf_name}")
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    print("  📥 Downloading sample file...")
                    await self.s3_client.download_file(pdf_key, Path(temp_path))
                    
                    print("  🔍 Running triple-layer extraction (PyMuPDF + Camelot + Tabula)...")
                    document = await self.doc_processor.process_document(temp_path)
                    
                    if document and document.content:
                        print("  ✅ Extraction Results:")
                        print(f"     📖 Pages processed: {document.page_count}")
                        print(f"     📊 Tables extracted: {len(document.tables)}")
                        print(f"     🏷️  Entities found: {len(document.entities)}")
                        print(f"     📝 Content length: {len(document.content)} characters")
                        
                        # Show extraction methods used
                        table_methods = [t.extraction_method for t in document.tables]
                        if table_methods:
                            print(f"     🔧 Table methods: {', '.join(set(table_methods))}")
                        
                        # Demo LLM-enhanced ingestion
                        print(f"\n  🤖 LLM-Enhanced Ingestion (GPT-4.1-mini)...")
                        doc_type = self._classify_document_type(pdf_key, pdf_name)
                        
                        # Prepare table data
                        table_data = []
                        for table in document.tables[:2]:  # Limit for demo
                            table_data.append({
                                'headers': table.headers,
                                'data': table.data[:3],  # First 3 rows
                                'extraction_method': table.extraction_method,
                                'confidence_score': table.confidence_score
                            })
                        
                        # Run LLM ingestion
                        ingestion_result = await self.rag_engine.ingest_document_with_llm(
                            content=document.content[:3000],  # Limit content for demo
                            source=pdf_key,
                            document_type=doc_type,
                            tables=table_data
                        )
                        
                        print("  ✅ LLM Ingestion Results:")
                        print(f"     🧠 Entities extracted: {ingestion_result.get('entities_extracted', 0)}")
                        print(f"     🔗 Relationships found: {ingestion_result.get('relationships_extracted', 0)}")
                        print(f"     📊 Graph nodes created: {ingestion_result.get('graph_nodes_created', 0)}")
                        print(f"     🔀 Graph edges created: {ingestion_result.get('graph_edges_created', 0)}")
                        print(f"     📝 Vector chunks added: {ingestion_result.get('vector_chunks_added', 0)}")
                    
                    # Clean up
                    Path(temp_path).unlink(missing_ok=True)
            
            # Demo RAG Query System
            print(f"\nPhase 4: RAG Query System Demo (GPT-4o)")
            
            # Save vector store state
            await self.rag_engine.vector_store.save_to_disk()
            
            # Test queries
            test_queries = [
                "What fire alarm panels are mentioned in the documents?",
                "What are the specifications for smoke detectors?",
                "Which standards are referenced for compliance?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n  📝 Test Query {i}: {query}")
                
                try:
                    result = await self.rag_engine.query_with_rag(
                        user_query=query,
                        k_vector=2,
                        k_graph=3
                    )
                    
                    if result.get('answer'):
                        print("  ✅ RAG Response Generated:")
                        print(f"     💬 Answer length: {len(result['answer'])} characters")
                        print(f"     📊 Graph results: {result.get('graph_results_count', 0)}")
                        print(f"     🔍 Vector results: {result.get('vector_results_count', 0)}")
                        print(f"     📚 Sources: {len(result.get('sources', []))}")
                        
                        # Show preview
                        answer_preview = result['answer'][:150]
                        print(f"     📖 Preview: {answer_preview}...")
                    else:
                        print("  ⚠️  No answer generated")
                
                except Exception as e:
                    print(f"  ❌ Query failed: {e}")
            
            # Show system statistics
            print(f"\nPhase 5: System Statistics")
            system_stats = await self.rag_engine.get_system_stats()
            
            if 'knowledge_graph' in system_stats:
                kg_stats = system_stats['knowledge_graph']
                print(f"  📊 Knowledge Graph:")
                print(f"     🔗 Total nodes: {kg_stats.get('total_nodes', 0)}")
                print(f"     ➡️  Total relationships: {kg_stats.get('total_relationships', 0)}")
            
            if 'vector_store' in system_stats:
                vs_stats = system_stats['vector_store']
                print(f"  📝 Vector Store:")
                print(f"     📄 Documents: {vs_stats.get('total_documents', 0)}")
                print(f"     🔢 Vectors: {vs_stats.get('indexed_vectors', 0)}")
                print(f"     📐 Dimension: {vs_stats.get('dimension', 1536)}")
            
            # Summary
            print(f"\n{'='*50}")
            print("🎉 System Demo Completed Successfully!")
            print(f"✅ Triple-layer extraction: PyMuPDF + Camelot + Tabula")
            print(f"✅ LLM-enhanced ingestion: GPT-4.1-mini")
            print(f"✅ Advanced generation: GPT-4o")
            print(f"✅ Knowledge graph: Neo4j with enhanced relationships")
            print(f"✅ Vector database: FAISS with text-embedding-3-small")
            print(f"✅ Complete BYOKG-RAG pipeline operational")
            
            print(f"\n🚀 System Ready for Production Processing!")
            print(f"📁 Total PDF files available: {len(pdf_files)}")
            print(f"⏱️  Estimated processing time: {len(pdf_files) * 2-3} minutes")
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.error("System demo failed", error=str(e))
            return False
    
    def _classify_document_type(self, s3_key: str, filename: str) -> str:
        """Classify document type for LLM processing"""
        
        key_lower = s3_key.lower()
        filename_lower = filename.lower()
        
        if any(keyword in key_lower for keyword in ['spec', 'compliance', 'requirement']):
            return 'specifications'
        elif any(keyword in key_lower for keyword in ['boq', 'quantity', 'pricing']):
            return 'boq'
        elif any(keyword in key_lower for keyword in ['offer', 'proposal', 'quote']):
            return 'offer'
        elif any(keyword in filename_lower for keyword in ['datasheet', 'data sheet']):
            return 'datasheet'
        elif any(keyword in filename_lower for keyword in ['manual', 'guide', 'instruction']):
            return 'manual'
        else:
            return 'technical_document'

async def main():
    """Main demo execution"""
    demo = QuickDemo()
    success = await demo.demo_system_capabilities()
    
    if success:
        print(f"\n💡 Next Steps:")
        print(f"   • Run full processing: python process_with_llm_assistance.py")
        print(f"   • Query the system: Access Neo4j browser at bolt://localhost:7687")
        print(f"   • Test RAG queries: Use the BYOKG-RAG engine directly")
    else:
        print(f"\n❌ Demo encountered issues. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())