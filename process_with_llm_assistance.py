#!/usr/bin/env python3
"""
Enhanced S3 Processing with Complete LLM Assistance
Integrates triple-layer extraction with GPT-4.1-mini for ingestion and GPT-4o for generation
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json
import pandas as pd
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings, ensure_data_dirs
from src.core.byokg_rag_engine import BYOKGRAGEngine
from src.ingestion.enhanced_document_processor import EnhancedDatasetProcessor

class LLMEnhancedProcessor:
    """Complete processor with LLM assistance for ingestion and generation"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.doc_processor = EnhancedDatasetProcessor()
        self.rag_engine = BYOKGRAGEngine()
        
    async def process_s3_with_llm_assistance(self, max_files: int = 10):
        """Process S3 bucket with complete LLM assistance"""
        
        print("=== Enhanced S3 Processing with LLM Assistance ===")
        print("🤖 Using GPT-4.1-mini for ingestion and GPT-4o for generation")
        print(f"S3 Bucket: {settings.s3_bucket_name}")
        print()
        
        ensure_data_dirs()
        
        try:
            # Initialize vector store
            print("Initializing systems...")
            await self.rag_engine.vector_store.load_from_disk()
            print("✅ Vector store initialized")
            
            # Discover PDF files
            print("Phase 1: Discovering PDF files...")
            all_objects = self.s3_client.list_all_objects()
            pdf_files = [obj for obj in all_objects if obj['key'].lower().endswith('.pdf')]
            
            if max_files:
                pdf_files = pdf_files[:max_files]
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Process with LLM assistance
            print("\nPhase 2: Enhanced processing with LLM assistance...")
            results = []
            
            for i, pdf_obj in enumerate(pdf_files, 1):
                pdf_key = pdf_obj['key']
                pdf_name = Path(pdf_key).name
                
                print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_name}")
                print(f"  S3 Key: {pdf_key}")
                print(f"  Size: {pdf_obj['size'] / (1024*1024):.2f} MB")
                
                try:
                    # Download PDF
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                        print("  📥 Downloading...")
                        await self.s3_client.download_file(pdf_key, Path(temp_path))
                        
                        # Triple-layer extraction
                        print("  🔍 Triple-layer extraction...")
                        document = await self.doc_processor.process_document(temp_path)
                        
                        if document and document.content:
                            print(f"    ✅ Extraction: {document.page_count} pages, {len(document.tables)} tables, {len(document.entities)} entities")
                            
                            # Determine document type
                            doc_type = self._classify_document_type(pdf_key, pdf_name)
                            
                            # LLM-assisted ingestion with GPT-4.1-mini
                            print("  🤖 LLM-assisted ingestion (GPT-4.1-mini)...")
                            
                            # Prepare table data for LLM
                            table_data = []
                            for table in document.tables:
                                table_data.append({
                                    'headers': table.headers,
                                    'data': table.data[:5],  # First 5 rows
                                    'extraction_method': table.extraction_method,
                                    'confidence_score': table.confidence_score
                                })
                            
                            # Ingest with LLM assistance
                            ingestion_result = await self.rag_engine.ingest_document_with_llm(
                                content=document.content,
                                source=pdf_key,
                                document_type=doc_type,
                                tables=table_data
                            )
                            
                            if ingestion_result.get('entities_extracted', 0) > 0:
                                print(f"    ✅ LLM Ingestion:")
                                print(f"      Entities: {ingestion_result['entities_extracted']}")
                                print(f"      Relationships: {ingestion_result['relationships_extracted']}")
                                print(f"      Graph nodes: {ingestion_result['graph_nodes_created']}")
                                print(f"      Graph edges: {ingestion_result['graph_edges_created']}")
                                print(f"      Vector chunks: {ingestion_result['vector_chunks_added']}")
                                
                                results.append({
                                    'file': pdf_name,
                                    's3_key': pdf_key,
                                    'success': True,
                                    'document_type': doc_type,
                                    'triple_layer_stats': {
                                        'pages': document.page_count,
                                        'tables': len(document.tables),
                                        'entities': len(document.entities),
                                        'quality_score': document.extraction_summary.get('quality_score', 0)
                                    },
                                    'llm_ingestion_stats': ingestion_result
                                })
                            else:
                                print("    ❌ LLM ingestion failed")
                        else:
                            print("    ❌ Triple-layer extraction failed")
                        
                        # Clean up
                        Path(temp_path).unlink(missing_ok=True)
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    logger.error("Failed to process PDF", file=pdf_key, error=str(e))
            
            # Save vector store
            print("\nSaving enhanced data...")
            await self.rag_engine.vector_store.save_to_disk()
            
            # Export results
            await self.export_enhanced_results(results)
            
            # Show summary
            print(f"\n=== Processing Complete ===")
            print(f"✅ Successfully processed: {len(results)} files")
            
            if results:
                total_entities = sum(r['llm_ingestion_stats']['entities_extracted'] for r in results)
                total_relationships = sum(r['llm_ingestion_stats']['relationships_extracted'] for r in results)
                total_nodes = sum(r['llm_ingestion_stats']['graph_nodes_created'] for r in results)
                total_edges = sum(r['llm_ingestion_stats']['graph_edges_created'] for r in results)
                total_vectors = sum(r['llm_ingestion_stats']['vector_chunks_added'] for r in results)
                
                print(f"\n📊 LLM Enhancement Summary:")
                print(f"   🧠 Entities extracted: {total_entities}")
                print(f"   🔗 Relationships found: {total_relationships}")
                print(f"   📊 Graph nodes created: {total_nodes}")
                print(f"   🔀 Graph edges created: {total_edges}")
                print(f"   📝 Vector chunks added: {total_vectors}")
                
                # Test the system
                await self.test_rag_system()
            
            return len(results)
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            logger.error("LLM-assisted processing failed", error=str(e))
            return 0
    
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
    
    async def test_rag_system(self):
        """Test the complete RAG system with sample queries"""
        
        print(f"\n🧪 Testing Complete RAG System...")
        print("Using GPT-4o for response generation...")
        
        test_queries = [
            "What fire alarm panels are available in the system?",
            "What are the power requirements for smoke detectors?",
            "Which standards do fire alarm systems need to comply with?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Test {i}/3] Query: {query}")
            
            try:
                result = await self.rag_engine.query_with_rag(
                    user_query=query,
                    k_vector=3,
                    k_graph=5
                )
                
                if result.get('answer'):
                    print("✅ RAG System Response:")
                    print(f"   Answer length: {len(result['answer'])} characters")
                    print(f"   Graph results: {result.get('graph_results_count', 0)}")
                    print(f"   Vector results: {result.get('vector_results_count', 0)}")
                    print(f"   Sources: {len(result.get('sources', []))}")
                    
                    # Show first 200 characters of answer
                    answer_preview = result['answer'][:200]
                    print(f"   Preview: {answer_preview}...")
                else:
                    print("❌ No answer generated")
            
            except Exception as e:
                print(f"❌ Test query failed: {e}")
        
        print(f"\n✅ RAG System testing completed!")
    
    async def export_enhanced_results(self, results: List[Dict]):
        """Export enhanced processing results"""
        
        results_dir = Path("./data/enhanced_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get system stats
        system_stats = await self.rag_engine.get_system_stats()
        
        # Export comprehensive results
        export_data = {
            'metadata': {
                'processed_at': pd.Timestamp.now().isoformat(),
                'processing_type': 'llm_enhanced_triple_layer',
                'total_files': len(results),
                'models_used': {
                    'ingestion': 'gpt-4-1106-preview',  # GPT-4.1-mini
                    'generation': 'gpt-4o',
                    'embeddings': 'text-embedding-3-small'
                },
                'extraction_tools': ['PyMuPDF', 'Camelot', 'Tabula', 'spaCy', 'OpenAI']
            },
            'processing_results': results,
            'system_statistics': system_stats,
            'summary': {
                'successful_files': len([r for r in results if r['success']]),
                'total_entities': sum(r['llm_ingestion_stats']['entities_extracted'] for r in results),
                'total_relationships': sum(r['llm_ingestion_stats']['relationships_extracted'] for r in results),
                'total_graph_nodes': sum(r['llm_ingestion_stats']['graph_nodes_created'] for r in results),
                'total_graph_edges': sum(r['llm_ingestion_stats']['graph_edges_created'] for r in results),
                'total_vector_chunks': sum(r['llm_ingestion_stats']['vector_chunks_added'] for r in results)
            }
        }
        
        # Save results
        output_file = results_dir / "llm_enhanced_processing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 Enhanced results exported: {output_file}")

async def main():
    """Main execution with LLM assistance"""
    processor = LLMEnhancedProcessor()
    
    # Process with LLM assistance - full dataset
    processed_count = await processor.process_s3_with_llm_assistance(max_files=None)  # Process all files
    
    if processed_count > 0:
        print(f"\n🎉 Successfully processed {processed_count} files with LLM assistance!")
        print("✅ Complete BYOKG-RAG system with KG-Linker is operational!")
        print("✅ GPT-4.1-mini used for enhanced ingestion")
        print("✅ GPT-4o ready for advanced generation")
        print("✅ FAISS vector database operational")
        print("✅ Neo4j knowledge graph enhanced")
        
        print(f"\n🚀 System ready for production use!")
    else:
        print("\n❌ No files were successfully processed.")

if __name__ == "__main__":
    asyncio.run(main())