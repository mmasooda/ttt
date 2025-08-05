#!/usr/bin/env python3
"""
Enhanced Dataset Processor
Combines triple-layer PDF extraction with iterative graph building
Processes all subdirectory/dataset PDF files in S3 bucket for proper dataset JSON preparation
"""

import asyncio
import sys
from pathlib import Path
import json
import tempfile
from typing import List, Dict, Any
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings, ensure_data_dirs
from src.ingestion.enhanced_document_processor import EnhancedDatasetProcessor, EnhancedDocument
from src.core.enhanced_graph_builder import EnhancedGraphBuilder

class ComprehensiveDatasetProcessor:
    """Comprehensive processor combining triple-layer extraction and enhanced graph building"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.doc_processor = EnhancedDatasetProcessor()
        self.graph_builder = EnhancedGraphBuilder()
        
    async def process_complete_dataset(self, max_files: int = None) -> Dict[str, Any]:
        """Process complete dataset with enhanced extraction and graph building"""
        
        print("=== Enhanced Dataset Processing with Triple-Layer Extraction ===")
        print(f"S3 Bucket: {settings.s3_bucket_name}")
        print()
        
        ensure_data_dirs()
        
        try:
            # Phase 1: Discover all PDF files
            print("Phase 1: Discovering PDF files...")
            pdf_files = await self._discover_pdf_files()
            
            if max_files:
                pdf_files = pdf_files[:max_files]
                print(f"Limited processing to first {max_files} files")
            
            print(f"Found {len(pdf_files)} PDF files to process\n")
            
            # Phase 2: Enhanced document processing
            print("Phase 2: Triple-layer document extraction...")
            enhanced_documents = await self._process_documents(pdf_files)
            
            if not enhanced_documents:
                print("❌ No documents were successfully processed")
                return {'success': False, 'error': 'No documents processed'}
            
            print(f"Successfully processed {len(enhanced_documents)} documents\n")
            
            # Phase 3: Enhanced graph construction
            print("Phase 3: Enhanced graph construction...")
            graph_stats = await self.graph_builder.build_enhanced_graph(enhanced_documents)
            
            print(f"Graph construction completed:")
            print(f"  Nodes: {graph_stats['nodes_created']}")
            print(f"  Edges: {graph_stats['edges_created']}")
            print(f"  Clusters: {graph_stats['clusters_formed']}")
            print(f"  Quality Score: {graph_stats['quality_metrics']['overall_quality']:.3f}\n")
            
            # Phase 4: Generate comprehensive dataset JSON
            print("Phase 4: Generating comprehensive dataset...")
            dataset_json = await self._generate_comprehensive_dataset(
                enhanced_documents, graph_stats
            )
            
            # Phase 5: Export results
            print("Phase 5: Exporting results...")
            export_paths = await self._export_results(
                enhanced_documents, dataset_json, graph_stats
            )
            
            # Final summary
            summary = {
                'success': True,
                'processing_stats': {
                    'total_pdfs_found': len(pdf_files),
                    'documents_processed': len(enhanced_documents),
                    'extraction_methods_used': self._get_extraction_methods_used(enhanced_documents),
                    'total_tables_extracted': sum(len(doc.tables) for doc in enhanced_documents),
                    'total_entities_extracted': sum(len(doc.entities) for doc in enhanced_documents),
                },
                'graph_stats': graph_stats,
                'export_paths': export_paths,
                'quality_assessment': await self._assess_overall_quality(enhanced_documents, graph_stats)
            }
            
            print("\n=== Processing Summary ===")
            print(f"📊 Documents processed: {summary['processing_stats']['documents_processed']}")
            print(f"📋 Tables extracted: {summary['processing_stats']['total_tables_extracted']}")
            print(f"🏷️  Entities extracted: {summary['processing_stats']['total_entities_extracted']}")
            print(f"🕸️  Graph nodes: {graph_stats['nodes_created']}")
            print(f"🔗 Graph edges: {graph_stats['edges_created']}")
            print(f"📦 Sub-graph clusters: {graph_stats['clusters_formed']}")
            print(f"⭐ Overall quality: {summary['quality_assessment']['overall_score']:.3f}")
            print(f"🔧 Extraction methods: {', '.join(summary['processing_stats']['extraction_methods_used'])}")
            
            print(f"\n📁 Results exported to:")
            for desc, path in export_paths.items():
                print(f"  {desc}: {path}")
            
            print(f"\n✅ Enhanced dataset processing completed successfully!")
            
            return summary
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            logger.error("Enhanced dataset processing failed", error=str(e))
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def _discover_pdf_files(self) -> List[Dict[str, Any]]:
        """Discover all PDF files in the S3 bucket"""
        
        all_objects = self.s3_client.list_all_objects()
        pdf_files = []
        
        # Filter for PDF files and categorize by directory
        for obj in all_objects:
            key = obj['key']
            if key.lower().endswith('.pdf'):
                # Analyze directory structure
                path_parts = key.split('/')
                directory_info = {
                    'top_level_dir': path_parts[0] if len(path_parts) > 1 else 'root',
                    'sub_dir': path_parts[1] if len(path_parts) > 2 else None,
                    'depth': len(path_parts) - 1,
                    'filename': Path(key).name
                }
                
                obj['directory_info'] = directory_info
                pdf_files.append(obj)
        
        # Sort by directory structure for organized processing
        pdf_files.sort(key=lambda x: (x['directory_info']['top_level_dir'], 
                                     x['directory_info']['sub_dir'] or '', 
                                     x['directory_info']['filename']))
        
        # Log directory statistics
        dir_stats = {}
        for pdf in pdf_files:
            top_dir = pdf['directory_info']['top_level_dir']
            dir_stats[top_dir] = dir_stats.get(top_dir, 0) + 1
        
        logger.info("PDF file discovery completed", 
                   total_pdfs=len(pdf_files),
                   directories=dir_stats)
        
        return pdf_files
    
    async def _process_documents(self, pdf_files: List[Dict[str, Any]]) -> List[EnhancedDocument]:
        """Process documents with triple-layer extraction"""
        
        enhanced_documents = []
        total_files = len(pdf_files)
        
        for i, pdf_obj in enumerate(pdf_files, 1):
            pdf_key = pdf_obj['key']
            pdf_name = Path(pdf_key).name
            dir_info = pdf_obj['directory_info']
            
            print(f"[{i}/{total_files}] Processing: {pdf_name}")
            print(f"  Directory: {dir_info['top_level_dir']}/{dir_info['sub_dir'] or ''}")
            print(f"  Size: {pdf_obj['size'] / (1024*1024):.2f} MB")
            
            try:
                # Download PDF to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    print("  📥 Downloading...")
                    await self.s3_client.download_file(pdf_key, Path(temp_path))
                    
                    print("  🔍 Triple-layer extraction...")
                    
                    # Determine document type from directory/filename
                    doc_type = self._classify_document_type(pdf_key, pdf_name)
                    
                    # Process with enhanced extraction
                    document = await self.doc_processor.process_document(temp_path, doc_type)
                    
                    if document:
                        # Add S3 metadata
                        document.metadata.update({
                            's3_key': pdf_key,
                            'directory_info': dir_info,
                            'document_type': doc_type,
                            'file_size': pdf_obj['size'],
                            'last_modified': pdf_obj['last_modified'].isoformat() if pdf_obj['last_modified'] else None
                        })
                        
                        enhanced_documents.append(document)
                        
                        # Show processing results
                        extraction_summary = document.extraction_summary
                        print(f"  ✅ SUCCESS!")
                        print(f"     Pages: {document.page_count}")
                        print(f"     Tables: {len(document.tables)} (Methods: {self._get_table_methods(document.tables)})")
                        print(f"     Entities: {len(document.entities)}")
                        print(f"     Quality: {extraction_summary['quality_score']:.2f}")
                        
                        # Show top entities
                        if document.entities:
                            top_entities = sorted(document.entities, 
                                                key=lambda e: e.confidence, 
                                                reverse=True)[:3]
                            print(f"     Top entities: {', '.join(e.name for e in top_entities)}")
                        
                    else:
                        print(f"  ❌ Failed to process")
                    
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                logger.error("Failed to process PDF", file=pdf_key, error=str(e))
        
        return enhanced_documents
    
    def _classify_document_type(self, s3_key: str, filename: str) -> str:
        """Classify document type based on S3 key and filename"""
        key_lower = s3_key.lower()
        filename_lower = filename.lower()
        
        # Check directory context
        if 'spec' in key_lower or 'compliance' in key_lower:
            return 'specifications'
        elif 'boq' in key_lower or 'quantity' in key_lower or 'pricing' in key_lower:
            return 'boq'
        elif 'offer' in key_lower or 'proposal' in key_lower or 'quote' in key_lower:
            return 'offer'
        
        # Check filename
        if any(keyword in filename_lower for keyword in ['spec', 'compliance', 'requirement']):
            return 'specifications'
        elif any(keyword in filename_lower for keyword in ['boq', 'quantity', 'pricing']):
            return 'boq'
        elif any(keyword in filename_lower for keyword in ['offer', 'proposal', 'quote', 'quotation']):
            return 'offer'
        
        return 'technical_document'
    
    def _get_table_methods(self, tables: List) -> str:
        """Get extraction methods used for tables"""
        methods = set()
        for table in tables:
            method = table.extraction_method
            if 'camelot' in method:
                methods.add('Camelot')
            elif 'tabula' in method:
                methods.add('Tabula')
            elif 'pymupdf' in method:
                methods.add('PyMuPDF')
        
        return ', '.join(methods) if methods else 'None'
    
    def _get_extraction_methods_used(self, documents: List[EnhancedDocument]) -> List[str]:
        """Get all extraction methods used across documents"""
        methods = set()
        
        for doc in documents:
            for table in doc.tables:
                method = table.extraction_method
                if 'camelot' in method:
                    methods.add('Camelot')
                elif 'tabula' in method:
                    methods.add('Tabula')
                elif 'pymupdf' in method:
                    methods.add('PyMuPDF')
        
        return sorted(list(methods))
    
    async def _generate_comprehensive_dataset(self, documents: List[EnhancedDocument], 
                                            graph_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive dataset JSON"""
        
        # Organize documents by type and directory
        docs_by_type = {}
        docs_by_directory = {}
        
        for doc in documents:
            doc_type = doc.metadata.get('document_type', 'unknown')
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append(doc)
            
            dir_info = doc.metadata.get('directory_info', {})
            top_dir = dir_info.get('top_level_dir', 'unknown')
            if top_dir not in docs_by_directory:
                docs_by_directory[top_dir] = []
            docs_by_directory[top_dir].append(doc)
        
        # Generate comprehensive dataset
        dataset = {
            'metadata': {
                'version': '2.0',
                'generated_at': pd.Timestamp.now().isoformat(),
                'processing_method': 'triple_layer_extraction',
                'total_documents': len(documents),
                'extraction_tools': ['PyMuPDF', 'Camelot', 'Tabula', 'spaCy', 'SentenceTransformers'],
                'graph_construction': 'enhanced_iterative_byokg_rag',
                'domain': 'fire_alarm_systems'
            },
            'documents': [],
            'document_types': {},
            'directory_structure': {},
            'extraction_statistics': {
                'total_pages_processed': sum(doc.page_count for doc in documents),
                'total_tables_extracted': sum(len(doc.tables) for doc in documents),
                'total_entities_extracted': sum(len(doc.entities) for doc in documents),
                'extraction_methods_distribution': self._get_extraction_method_distribution(documents),
                'document_type_distribution': {doc_type: len(docs) for doc_type, docs in docs_by_type.items()},
                'directory_distribution': {dir_name: len(docs) for dir_name, docs in docs_by_directory.items()}
            },
            'graph_statistics': graph_stats,
            'quality_metrics': graph_stats.get('quality_metrics', {}),
            'ttt_training_samples': await self._generate_ttt_samples(documents)
        }
        
        # Add document details
        for doc in documents:
            doc_entry = {
                'filename': doc.filename,
                's3_key': doc.metadata.get('s3_key'),
                'document_type': doc.metadata.get('document_type'),
                'directory_info': doc.metadata.get('directory_info'),
                'page_count': doc.page_count,
                'extraction_summary': doc.extraction_summary,
                'content_preview': doc.content[:1000] if doc.content else "",
                'tables': [
                    {
                        'id': table.id,
                        'method': table.extraction_method,
                        'confidence': table.confidence_score,
                        'headers': table.headers,
                        'row_count': len(table.data),
                        'page': table.page_number
                    }
                    for table in doc.tables
                ],
                'entities': [
                    {
                        'name': entity.name,
                        'type': entity.type,
                        'confidence': entity.confidence,
                        'category': getattr(entity, 'category', 'unknown'),
                        'page': entity.page_number
                    }
                    for entity in doc.entities[:20]  # Limit for size
                ],
                'embeddings_generated': len(doc.embeddings)
            }
            
            dataset['documents'].append(doc_entry)
        
        # Add type-specific summaries
        for doc_type, docs in docs_by_type.items():
            dataset['document_types'][doc_type] = {
                'count': len(docs),
                'total_pages': sum(doc.page_count for doc in docs),
                'total_tables': sum(len(doc.tables) for doc in docs),
                'total_entities': sum(len(doc.entities) for doc in docs),
                'avg_quality_score': sum(doc.extraction_summary.get('quality_score', 0) for doc in docs) / len(docs)
            }
        
        # Add directory structure
        for dir_name, docs in docs_by_directory.items():
            dataset['directory_structure'][dir_name] = {
                'document_count': len(docs),
                'file_sizes_mb': [doc.metadata.get('file_size', 0) / (1024*1024) for doc in docs],
                'document_types': list(set(doc.metadata.get('document_type') for doc in docs))
            }
        
        return dataset
    
    def _get_extraction_method_distribution(self, documents: List[EnhancedDocument]) -> Dict[str, int]:
        """Get distribution of extraction methods used"""
        method_counts = {'PyMuPDF': 0, 'Camelot_lattice': 0, 'Camelot_stream': 0, 'Tabula': 0}
        
        for doc in documents:
            for table in doc.tables:
                method = table.extraction_method
                if 'camelot_lattice' in method:
                    method_counts['Camelot_lattice'] += 1
                elif 'camelot_stream' in method:
                    method_counts['Camelot_stream'] += 1
                elif 'tabula' in method:
                    method_counts['Tabula'] += 1
                elif 'pymupdf' in method:
                    method_counts['PyMuPDF'] += 1
        
        return method_counts
    
    async def _generate_ttt_samples(self, documents: List[EnhancedDocument]) -> List[Dict[str, Any]]:
        """Generate TTT training samples from processed documents"""
        samples = []
        
        for doc in documents:
            doc_type = doc.metadata.get('document_type', 'unknown')
            
            # Generate different types of samples based on document type
            if doc_type == 'specifications':
                samples.extend(await self._generate_spec_samples(doc))
            elif doc_type == 'boq':
                samples.extend(await self._generate_boq_samples(doc))
            elif doc_type == 'offer':
                samples.extend(await self._generate_offer_samples(doc))
            
            # Generate cross-document analysis samples
            samples.extend(await self._generate_analysis_samples(doc))
        
        # Limit total samples for manageable dataset size
        return samples[:200]
    
    async def _generate_spec_samples(self, doc: EnhancedDocument) -> List[Dict[str, Any]]:
        """Generate specification-focused TTT samples"""
        samples = []
        
        # Standards compliance sample
        standards = [e for e in doc.entities if 'standard' in e.type.lower()]
        if standards:
            sample = {
                'id': f"spec_{doc.filename.replace('.pdf', '')}_standards",
                'type': 'standards_compliance',
                'query': f"What are the key standards and compliance requirements for this fire alarm system specification?",
                'context': {
                    'document': doc.filename,
                    'content_preview': doc.content[:2000],
                    'standards_found': [s.name for s in standards[:5]],
                    'extraction_quality': doc.extraction_summary.get('quality_score', 0)
                },
                'expected_output': {
                    'standards_list': [s.name for s in standards],
                    'compliance_notes': "System must comply with all mentioned standards for certification"
                }
            }
            samples.append(sample)
        
        return samples
    
    async def _generate_boq_samples(self, doc: EnhancedDocument) -> List[Dict[str, Any]]:
        """Generate BOQ-focused TTT samples"""
        samples = []
        
        # Table extraction sample
        if doc.tables:
            best_table = max(doc.tables, key=lambda t: t.confidence_score)
            
            sample = {
                'id': f"boq_{doc.filename.replace('.pdf', '')}_table",
                'type': 'table_extraction',
                'query': f"Extract and analyze the bill of quantities table from this document",
                'context': {
                    'document': doc.filename,
                    'table_method': best_table.extraction_method,
                    'table_confidence': best_table.confidence_score,
                    'headers': best_table.headers,
                    'sample_data': best_table.data[:3] if best_table.data else []
                },
                'expected_output': {
                    'table_structure': {
                        'headers': best_table.headers,
                        'row_count': len(best_table.data),
                        'extraction_method': best_table.extraction_method
                    },
                    'key_items': 'Fire alarm system components and quantities'
                }
            }
            samples.append(sample)
        
        return samples
    
    async def _generate_offer_samples(self, doc: EnhancedDocument) -> List[Dict[str, Any]]:
        """Generate offer-focused TTT samples"""
        samples = []
        
        # Product analysis sample
        products = [e for e in doc.entities if 'product' in e.type.lower() or 'fire' in e.name.lower()]
        if products:
            sample = {
                'id': f"offer_{doc.filename.replace('.pdf', '')}_products",
                'type': 'product_analysis',
                'query': f"What fire alarm products and solutions are proposed in this technical offer?",
                'context': {
                    'document': doc.filename,
                    'products_identified': [p.name for p in products[:10]],
                    'entity_extraction_quality': sum(e.confidence for e in products) / len(products)
                },
                'expected_output': {
                    'recommended_products': [p.name for p in products[:5]],
                    'solution_type': 'Fire alarm system implementation'
                }
            }
            samples.append(sample)
        
        return samples
    
    async def _generate_analysis_samples(self, doc: EnhancedDocument) -> List[Dict[str, Any]]:
        """Generate document analysis samples"""
        samples = []
        
        # Entity relationship sample
        if len(doc.entities) > 5:
            sample = {
                'id': f"analysis_{doc.filename.replace('.pdf', '')}_entities",
                'type': 'entity_relationship_analysis',
                'query': f"Analyze the key entities and their relationships in this fire alarm document",
                'context': {
                    'document': doc.filename,
                    'entity_count': len(doc.entities),
                    'top_entities': [e.name for e in sorted(doc.entities, key=lambda x: x.confidence, reverse=True)[:10]],
                    'document_type': doc.metadata.get('document_type')
                },
                'expected_output': {
                    'entity_summary': f"Document contains {len(doc.entities)} entities related to fire alarm systems",
                    'relationship_types': ['part_of', 'connects_to', 'requires', 'complies_with']
                }
            }
            samples.append(sample)
        
        return samples
    
    async def _export_results(self, documents: List[EnhancedDocument], 
                            dataset_json: Dict[str, Any], 
                            graph_stats: Dict[str, Any]) -> Dict[str, str]:
        """Export all results to files"""
        
        results_dir = Path("./data/enhanced_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        # Export comprehensive dataset JSON
        dataset_file = results_dir / "comprehensive_dataset_enhanced.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False, default=str)
        export_paths['Comprehensive Dataset'] = str(dataset_file)
        
        # Export individual document details
        docs_dir = results_dir / "individual_documents"
        docs_dir.mkdir(exist_ok=True)
        
        for doc in documents:
            doc_file = docs_dir / f"enhanced_{doc.filename.replace('.pdf', '.json')}"
            self.doc_processor.export_enhanced_document(doc, str(doc_file))
        export_paths['Individual Documents'] = str(docs_dir)
        
        # Export graph statistics
        graph_file = results_dir / "graph_construction_stats.json"
        with open(graph_file, 'w') as f:
            json.dump(graph_stats, f, indent=2, default=str)
        export_paths['Graph Statistics'] = str(graph_file)
        
        # Export TTT training samples separately
        ttt_file = results_dir / "ttt_training_samples_enhanced.json"
        ttt_data = {
            'metadata': {
                'version': '2.0',
                'generated_at': pd.Timestamp.now().isoformat(),
                'extraction_method': 'triple_layer_enhanced',
                'total_samples': len(dataset_json.get('ttt_training_samples', []))
            },
            'training_samples': dataset_json.get('ttt_training_samples', [])
        }
        
        with open(ttt_file, 'w', encoding='utf-8') as f:
            json.dump(ttt_data, f, indent=2, ensure_ascii=False, default=str)
        export_paths['TTT Training Samples'] = str(ttt_file)
        
        # Export processing summary
        summary_file = results_dir / "processing_summary.json"
        processing_summary = {
            'documents_processed': len(documents),
            'total_pages': sum(doc.page_count for doc in documents),
            'total_tables': sum(len(doc.tables) for doc in documents),
            'total_entities': sum(len(doc.entities) for doc in documents),
            'extraction_methods': self._get_extraction_methods_used(documents),
            'graph_nodes': graph_stats['nodes_created'],
            'graph_edges': graph_stats['edges_created'],
            'graph_clusters': graph_stats['clusters_formed'],
            'overall_quality': graph_stats['quality_metrics']['overall_quality'],
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(processing_summary, f, indent=2, default=str)
        export_paths['Processing Summary'] = str(summary_file)
        
        return export_paths
    
    async def _assess_overall_quality(self, documents: List[EnhancedDocument], 
                                    graph_stats: Dict[str, Any]) -> Dict[str, float]:
        """Assess overall quality of the processing"""
        
        # Document processing quality
        doc_qualities = [doc.extraction_summary.get('quality_score', 0) for doc in documents]
        avg_doc_quality = sum(doc_qualities) / len(doc_qualities) if doc_qualities else 0
        
        # Graph construction quality
        graph_quality = graph_stats.get('quality_metrics', {}).get('overall_quality', 0)
        
        # Coverage metrics
        total_tables = sum(len(doc.tables) for doc in documents)
        total_entities = sum(len(doc.entities) for doc in documents)
        
        coverage_score = min(1.0, (total_tables + total_entities) / (len(documents) * 10))
        
        # Method diversity (using all three extraction tools)
        methods_used = len(self._get_extraction_methods_used(documents))
        method_diversity = methods_used / 3.0  # Max 3 methods (Camelot, Tabula, PyMuPDF)
        
        # Overall quality score
        overall_score = (
            avg_doc_quality * 0.4 +
            graph_quality * 0.3 +
            coverage_score * 0.2 +
            method_diversity * 0.1
        )
        
        return {
            'overall_score': overall_score,
            'document_quality': avg_doc_quality,
            'graph_quality': graph_quality,
            'coverage_score': coverage_score,
            'method_diversity': method_diversity
        }

async def main():
    """Main execution function"""
    processor = ComprehensiveDatasetProcessor()
    
    # Process complete dataset from S3 bucket
    result = await processor.process_complete_dataset(max_files=None)
    
    if result['success']:
        print(f"\n🎉 Enhanced dataset processing completed successfully!")
        print(f"Check the results in: data/enhanced_results/")
    else:
        print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())