#!/usr/bin/env python3
"""
TTT-Enhanced BYOKG-RAG Ingestion Pipeline - Auto Mode
Automatically ingests all files from S3 bucket and builds knowledge graph
"""

import asyncio
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import (
    settings, logger, ensure_data_dirs, 
    S3Client, Neo4jClient
)
from src.ingestion import DocumentProcessor
from src.core import KnowledgeGraphBuilder, TTTAdapter

class IngestionPipeline:
    """Complete ingestion pipeline for S3 to Knowledge Graph"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.doc_processor = DocumentProcessor()
        self.kg_builder = KnowledgeGraphBuilder()
        self.ttt_adapter = TTTAdapter()
        
        # Ensure data directories exist
        ensure_data_dirs()
        
        logger.info("Ingestion pipeline initialized")
    
    async def run_full_pipeline(self, max_files: int = 20) -> Dict[str, Any]:
        """Run the complete ingestion pipeline with file limit"""
        start_time = time.time()
        
        try:
            logger.info("Starting full ingestion pipeline", bucket=settings.s3_bucket_name, max_files=max_files)
            
            # Step 1: Discover S3 files
            logger.info("Step 1: Discovering S3 files...")
            s3_objects = self.s3_client.list_all_objects()
            supported_files = self.s3_client.filter_supported_files(s3_objects)
            
            # Limit files for testing
            if len(supported_files) > max_files:
                supported_files = supported_files[:max_files]
                logger.info(f"Limited to first {max_files} files for testing")
            
            if not supported_files:
                logger.warning("No supported files found in S3 bucket")
                return {"error": "No supported files found"}
            
            logger.info("Found supported files", count=len(supported_files))
            
            # Step 2: Download files
            logger.info("Step 2: Downloading files from S3...")
            download_dir = Path("./data/downloads")
            download_results = await self.s3_client.download_files_batch(
                supported_files, download_dir, max_concurrent=3
            )
            
            successful_downloads = [r for r in download_results if r['download_success']]
            logger.info("Downloaded files", successful=len(successful_downloads))
            
            # Step 3: Process documents
            logger.info("Step 3: Processing documents...")
            processed_docs = []
            
            for file_info in successful_downloads:
                local_path = Path(file_info['local_path'])
                processed_doc = self.doc_processor.process_file(local_path, file_info['key'])
                
                if processed_doc:
                    processed_docs.append(processed_doc)
            
            logger.info("Processed documents", count=len(processed_docs))
            
            # Step 4: Enhance with TTT
            logger.info("Step 4: Enhancing with TTT...")
            for doc in processed_docs:
                # Enhance entity extraction
                enhanced_entities = self.ttt_adapter.enhanced_entity_extraction(doc.content)
                if enhanced_entities:
                    doc.entities.extend(enhanced_entities)
                
                # Enhance relationship extraction
                enhanced_rels = self.ttt_adapter.enhanced_relationship_extraction(
                    doc.content, doc.entities
                )
                if enhanced_rels:
                    doc.relationships.extend(enhanced_rels)
            
            # Step 5: Create TTT training examples
            logger.info("Step 5: Creating TTT training examples...")
            doc_dicts = []
            for doc in processed_docs:
                doc_dicts.append({
                    'id': doc.id,
                    'title': doc.title,
                    'content_preview': doc.content_preview,
                    'filename': doc.filename
                })
            
            training_examples = self.ttt_adapter.create_training_examples(doc_dicts)
            
            # Step 6: Adapt TTT model
            if training_examples:
                logger.info("Step 6: Adapting TTT model...")
                self.ttt_adapter.adapt_model(training_examples)
            
            # Step 7: Build knowledge graph
            logger.info("Step 7: Building knowledge graph...")
            kg_results = self.kg_builder.ingest_documents_batch(processed_docs)
            
            # Final statistics
            end_time = time.time()
            duration = end_time - start_time
            
            stats = self.kg_builder.get_knowledge_graph_stats()
            
            results = {
                'pipeline_duration_minutes': duration / 60,
                'files_discovered': len(s3_objects),
                'files_supported': len(supported_files),
                'files_downloaded': len(successful_downloads),
                'documents_processed': len(processed_docs),
                'training_examples_created': len(training_examples),
                'knowledge_graph_ingestion': kg_results,
                'knowledge_graph_stats': stats,
                'status': 'completed'
            }
            
            logger.info("Pipeline completed successfully", **results)
            return results
            
        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            return {
                'status': 'failed',
                'error': str(e),
                'pipeline_duration_minutes': (time.time() - start_time) / 60
            }
    
    def print_bucket_structure(self):
        """Print S3 bucket structure"""
        print("\n=== S3 Bucket Structure ===")
        self.s3_client.print_bucket_structure()
    
    def print_knowledge_graph_stats(self):
        """Print knowledge graph statistics"""
        stats = self.kg_builder.get_knowledge_graph_stats()
        print("\n=== Knowledge Graph Statistics ===")
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")
    
    def close(self):
        """Clean up resources"""
        self.kg_builder.close()

async def main():
    """Main pipeline execution"""
    print("=== TTT-Enhanced BYOKG-RAG Ingestion Pipeline (Auto Mode) ===")
    print(f"S3 Bucket: {settings.s3_bucket_name}")
    print(f"Neo4j: {settings.neo4j_uri}")
    print()
    
    pipeline = IngestionPipeline()
    
    try:
        # Show bucket structure first
        pipeline.print_bucket_structure()
        
        print(f"\nStarting automatic ingestion (limited to 20 files for testing)...")
        
        # Run pipeline automatically
        results = await pipeline.run_full_pipeline(max_files=20)
        
        # Show results
        print("\n=== Pipeline Results ===")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        # Show final stats
        pipeline.print_knowledge_graph_stats()
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nPipeline error: {e}")
        logger.error("Pipeline execution failed", error=str(e))
    finally:
        pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())