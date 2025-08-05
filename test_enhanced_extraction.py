#!/usr/bin/env python3
"""
Test script for enhanced triple-layer extraction
Downloads and tests processing of sample PDFs from S3 bucket
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings, ensure_data_dirs
from src.ingestion.enhanced_document_processor import EnhancedDatasetProcessor

async def test_enhanced_extraction():
    """Test enhanced extraction on sample PDFs"""
    print("=== Testing Enhanced Triple-Layer Extraction ===")
    print(f"S3 Bucket: {settings.s3_bucket_name}")
    print()
    
    ensure_data_dirs()
    
    s3_client = S3Client()
    processor = EnhancedDatasetProcessor()
    
    try:
        # Get sample PDFs from bucket
        print("Discovering PDF files in S3 bucket...")
        all_objects = s3_client.list_all_objects()
        
        pdf_files = [obj for obj in all_objects if obj['key'].lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files found in bucket")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Test on first few PDFs
        test_files = pdf_files[:3]  # Test first 3 PDFs
        
        results = []
        
        for i, pdf_obj in enumerate(test_files, 1):
            pdf_key = pdf_obj['key']
            pdf_name = Path(pdf_key).name
            
            print(f"\n[{i}/{len(test_files)}] Testing: {pdf_name}")
            print(f"  Size: {pdf_obj['size'] / (1024*1024):.2f} MB")
            
            try:
                # Download PDF to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    print("  📥 Downloading...")
                    await s3_client.download_file(pdf_key, Path(temp_path))
                    
                    print("  🔍 Processing with triple-layer extraction...")
                    
                    # Process with enhanced extraction
                    document = await processor.process_document(temp_path)
                    
                    if document:
                        print(f"  ✅ SUCCESS!")
                        print(f"     Pages: {document.page_count}")
                        print(f"     Tables: {len(document.tables)}")
                        print(f"     Entities: {len(document.entities)}")
                        print(f"     Quality Score: {document.extraction_summary['quality_score']:.2f}")
                        
                        # Show extraction methods used
                        methods = set(t.extraction_method for t in document.tables)
                        if methods:
                            print(f"     Extraction Methods: {', '.join(methods)}")
                        
                        # Show top entities
                        top_entities = sorted(document.entities, 
                                            key=lambda e: e.confidence, 
                                            reverse=True)[:5]
                        
                        if top_entities:
                            print("     Top Entities:")
                            for entity in top_entities:
                                print(f"       - {entity.name} ({entity.type}, conf: {entity.confidence:.2f})")
                        
                        # Save detailed results
                        results_dir = Path("./data/test_results")
                        results_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_file = results_dir / f"enhanced_{pdf_name.replace('.pdf', '.json')}"
                        processor.export_enhanced_document(document, str(output_file))
                        
                        print(f"     Detailed results: {output_file}")
                        
                        results.append({
                            'file': pdf_name,
                            'success': True,
                            'pages': document.page_count,
                            'tables': len(document.tables),
                            'entities': len(document.entities),
                            'quality_score': document.extraction_summary['quality_score'],
                            'extraction_methods': list(methods),
                            'output_file': str(output_file)
                        })
                        
                    else:
                        print(f"  ❌ Failed to process document")
                        results.append({
                            'file': pdf_name,
                            'success': False,
                            'error': 'Processing failed'
                        })
                    
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'file': pdf_name,
                    'success': False,
                    'error': str(e)
                })
        
        # Generate test summary
        print(f"\n=== Test Summary ===")
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"Total PDFs tested: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
            total_tables = sum(r['tables'] for r in successful)
            total_entities = sum(r['entities'] for r in successful)
            
            print(f"\nSuccess Metrics:")
            print(f"  Average Quality Score: {avg_quality:.2f}")
            print(f"  Total Tables Extracted: {total_tables}")
            print(f"  Total Entities Extracted: {total_entities}")
            
            # Show extraction methods used
            all_methods = set()
            for r in successful:
                all_methods.update(r.get('extraction_methods', []))
            
            print(f"  Extraction Methods Used: {', '.join(all_methods)}")
        
        if failed:
            print(f"\nFailed Files:")
            for fail in failed:
                print(f"  - {fail['file']}: {fail.get('error', 'Unknown error')}")
        
        # Save summary
        summary_file = Path("./data/test_results/extraction_test_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'test_timestamp': pd.Timestamp.now().isoformat(),
                'pdfs_tested': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'results': results
            }, f, indent=2, default=str)
        
        print(f"\nTest summary saved: {summary_file}")
        
        if successful:
            print(f"\n🎉 Enhanced extraction test completed successfully!")
            print(f"Triple-layer extraction is working with {len(all_methods)} different methods")
        else:
            print(f"\n⚠️  All tests failed. Check configuration and dependencies.")
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        logger.error("Enhanced extraction test failed", error=str(e))

if __name__ == "__main__":
    import pandas as pd
    asyncio.run(test_enhanced_extraction())