#!/usr/bin/env python3
"""
Simple test for enhanced document processor to debug issues
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings

# Test simple PyMuPDF first
import fitz

async def test_simple_enhanced():
    """Simple test of enhanced processing"""
    print("=== Simple Enhanced Test ===")
    
    s3_client = S3Client()
    
    try:
        # Get first PDF
        all_objects = s3_client.list_all_objects()
        pdf_files = [obj for obj in all_objects if obj['key'].lower().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found")
            return
        
        test_pdf = pdf_files[0]
        print(f"Testing with: {test_pdf['key']}")
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            
            print("Downloading...")
            await s3_client.download_file(test_pdf['key'], Path(temp_path))
            
            print(f"File size: {Path(temp_path).stat().st_size} bytes")
            
            # Test PyMuPDF
            print("Testing PyMuPDF...")
            try:
                doc = fitz.open(temp_path)
                print(f"Pages: {len(doc)}")
                
                if len(doc) > 0:
                    page = doc[0]
                    text = page.get_text()
                    print(f"Text length: {len(text)}")
                    print(f"First 100 chars: {text[:100]}")
                
                doc.close()
                
            except Exception as e:
                print(f"PyMuPDF error: {e}")
                traceback.print_exc()
            
            # Test importing enhanced processor
            print("\nTesting enhanced processor import...")
            try:
                from src.ingestion.enhanced_document_processor import EnhancedDatasetProcessor
                print("✅ Enhanced processor imported successfully")
                
                processor = EnhancedDatasetProcessor()
                print("✅ Enhanced processor instantiated")
                
                # Try processing
                print("Testing document processing...")
                document = await processor.process_document(temp_path, 'test')
                
                if document:
                    print(f"✅ Document processed!")
                    print(f"  Pages: {document.page_count}")
                    print(f"  Content length: {len(document.content)}")
                    print(f"  Tables: {len(document.tables)}")
                    print(f"  Entities: {len(document.entities)}")
                else:
                    print("❌ Document processing returned None")
                
            except Exception as e:
                print(f"Enhanced processor error: {e}")
                traceback.print_exc()
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_enhanced())