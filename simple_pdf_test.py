#!/usr/bin/env python3
"""
Simple test to verify PDF download and basic PyMuPDF functionality
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import fitz

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings

async def simple_pdf_test():
    """Simple test of PDF download and basic extraction"""
    print("=== Simple PDF Test ===")
    
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
        print(f"Size: {test_pdf['size'] / (1024*1024):.2f} MB")
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            
            print("Downloading...")
            await s3_client.download_file(test_pdf['key'], Path(temp_path))
            
            # Check file exists and has content
            temp_file_path = Path(temp_path)
            if temp_file_path.exists():
                actual_size = temp_file_path.stat().st_size
                print(f"Downloaded file size: {actual_size / (1024*1024):.2f} MB")
                
                if actual_size > 0:
                    print("Testing PyMuPDF...")
                    try:
                        doc = fitz.open(temp_path)
                        print(f"PDF opened successfully!")
                        print(f"Pages: {len(doc)}")
                        
                        if len(doc) > 0:
                            page = doc[0]
                            text = page.get_text()
                            print(f"First page text length: {len(text)}")
                            print(f"First 200 chars: {text[:200]}")
                        
                        doc.close()
                        
                    except Exception as e:
                        print(f"PyMuPDF error: {e}")
                
                else:
                    print("Downloaded file is empty!")
            else:
                print("Downloaded file does not exist!")
            
            # Clean up
            temp_file_path.unlink(missing_ok=True)
    
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_pdf_test())