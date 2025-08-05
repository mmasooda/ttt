#!/usr/bin/env python3
"""
Run Enhanced S3 Dataset Processing
Simplified version to process S3 bucket with improved extraction
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger, settings, ensure_data_dirs

# Import working components
import fitz  # PyMuPDF
import camelot
import tabula

class SimplifiedProcessor:
    """Simplified processor to test S3 extraction"""
    
    def __init__(self):
        self.s3_client = S3Client()
    
    async def process_s3_bucket(self, max_files: int = 20):
        """Process PDFs from S3 bucket with improved extraction"""
        
        print("=== Enhanced S3 PDF Processing ===")
        print(f"S3 Bucket: {settings.s3_bucket_name}")
        print()
        
        ensure_data_dirs()
        
        try:
            # Discover PDF files
            print("Phase 1: Discovering PDF files...")
            all_objects = self.s3_client.list_all_objects()
            pdf_files = [obj for obj in all_objects if obj['key'].lower().endswith('.pdf')]
            
            if max_files:
                pdf_files = pdf_files[:max_files]
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Process each PDF
            print("\nPhase 2: Processing PDFs...")
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
                        
                        # Verify download
                        actual_size = Path(temp_path).stat().st_size
                        print(f"  Downloaded: {actual_size / (1024*1024):.2f} MB")
                        
                        if actual_size == 0:
                            print("  ❌ Downloaded file is empty")
                            continue
                        
                        # Process with improved extraction
                        print("  🔍 Enhanced extraction...")
                        doc_result = await self.process_pdf_file(temp_path, pdf_name, pdf_key)
                        
                        if doc_result['success']:
                            print(f"  ✅ SUCCESS!")
                            print(f"     Pages: {doc_result['pages']}")
                            print(f"     Text extracted: {doc_result['text_length']} chars")
                            print(f"     Tables found: {doc_result['tables_found']}")
                            print(f"     Extraction methods: {', '.join(doc_result['methods_used'])}")
                            
                            results.append(doc_result)
                        else:
                            print(f"  ❌ Failed: {doc_result.get('error', 'Unknown error')}")
                        
                        # Clean up
                        Path(temp_path).unlink(missing_ok=True)
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    logger.error("Failed to process PDF", file=pdf_key, error=str(e))
            
            # Export results
            print(f"\n=== Processing Complete ===")
            print(f"Successfully processed: {len(results)} out of {len(pdf_files)} PDFs")
            
            if results:
                await self.export_results(results)
                
                # Summary statistics
                total_pages = sum(r['pages'] for r in results)
                total_tables = sum(r['tables_found'] for r in results)
                total_text = sum(r['text_length'] for r in results)
                
                methods_used = set()
                for r in results:
                    methods_used.update(r['methods_used'])
                
                print(f"\n📊 Summary Statistics:")
                print(f"   Total pages processed: {total_pages}")
                print(f"   Total tables extracted: {total_tables}")
                print(f"   Total text extracted: {total_text:,} characters")
                print(f"   Extraction methods used: {', '.join(methods_used)}")
                print(f"   Average pages per document: {total_pages/len(results):.1f}")
                print(f"   Average tables per document: {total_tables/len(results):.1f}")
                
                print(f"\n✅ Results exported to: data/enhanced_results/s3_processing_results.json")
            
            return len(results)
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            logger.error("S3 processing failed", error=str(e))
            return 0
    
    async def process_pdf_file(self, pdf_path: str, filename: str, s3_key: str) -> dict:
        """Process a single PDF file with improved extraction"""
        
        result = {
            'success': False,
            'filename': filename,
            's3_key': s3_key,
            'pages': 0,
            'text_length': 0,
            'tables_found': 0,
            'methods_used': [],
            'extraction_details': {}
        }
        
        try:
            # Step 1: PyMuPDF for basic extraction
            pymupdf_result = self.extract_with_pymupdf(pdf_path)
            result['pages'] = pymupdf_result['page_count']
            result['text_length'] = len(pymupdf_result['content'])
            
            if pymupdf_result['page_count'] > 0:
                result['methods_used'].append('PyMuPDF')
                result['extraction_details']['pymupdf'] = {
                    'pages': pymupdf_result['page_count'],
                    'content_length': len(pymupdf_result['content']),
                    'metadata': pymupdf_result['metadata']
                }
            
            # Step 2: Camelot for table extraction
            try:
                camelot_tables = self.extract_tables_with_camelot(pdf_path)
                if camelot_tables:
                    result['methods_used'].append('Camelot')
                    result['tables_found'] += len(camelot_tables)
                    result['extraction_details']['camelot'] = {
                        'tables_found': len(camelot_tables),
                        'methods': list(set(t['method'] for t in camelot_tables))
                    }
            except Exception as e:
                logger.warning("Camelot extraction failed", error=str(e))
            
            # Step 3: Tabula for table extraction
            try:
                tabula_tables = self.extract_tables_with_tabula(pdf_path)
                if tabula_tables:
                    result['methods_used'].append('Tabula')
                    result['tables_found'] += len(tabula_tables)
                    result['extraction_details']['tabula'] = {
                        'tables_found': len(tabula_tables)
                    }
            except Exception as e:
                logger.warning("Tabula extraction failed", error=str(e))
            
            # Mark as successful if we extracted something meaningful
            if result['pages'] > 0 or result['tables_found'] > 0:
                result['success'] = True
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error("PDF processing failed", file=filename, error=str(e))
            return result
    
    def extract_with_pymupdf(self, pdf_path: str) -> dict:
        """Extract content using PyMuPDF with improved page handling"""
        try:
            doc = fitz.open(pdf_path)
            actual_pages = len(doc)
            
            if actual_pages == 0:
                doc.close()
                return {
                    'content': "",
                    'page_count': 0,
                    'metadata': {'note': 'PDF contains no pages'}
                }
            
            full_text = ""
            metadata = {
                'actual_pages': actual_pages,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', '')
            }
            
            # Process only existing pages
            for page_num in range(actual_pages):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Clean misleading page references
                    cleaned_text = self.clean_page_references(text, page_num + 1, actual_pages)
                    full_text += f"\n--- Page {page_num + 1} of {actual_pages} ---\n{cleaned_text}"
                    
                except Exception as e:
                    logger.warning("Failed to process page", page=page_num + 1, error=str(e))
                    continue
            
            doc.close()
            
            return {
                'content': full_text,
                'page_count': actual_pages,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error("PyMuPDF extraction failed", error=str(e))
            return {
                'content': "",
                'page_count': 0,
                'metadata': {'error': str(e)}
            }
    
    def clean_page_references(self, text: str, current_page: int, total_pages: int) -> str:
        """Clean misleading page references from extracted PDFs"""
        if not text:
            return text
        
        import re
        
        # Remove patterns like "Page X of Y" where Y might be from original document
        text = re.sub(r'^Page\s+\d+\s+of\s+\d+\s*$', 
                     f'[Page {current_page} of {total_pages}]', 
                     text, flags=re.MULTILINE)
        
        # Remove standalone page numbers that might be misleading
        text = re.sub(r'^\d+\s*$', f'[{current_page}]', text, flags=re.MULTILINE)
        
        return text
    
    def extract_tables_with_camelot(self, pdf_path: str) -> list:
        """Extract tables using Camelot with dynamic page handling"""
        tables = []
        
        try:
            # Determine actual page count
            temp_doc = fitz.open(pdf_path)
            actual_pages = len(temp_doc)
            temp_doc.close()
            
            if actual_pages == 0:
                return tables
            
            page_spec = f'1-{actual_pages}'
            
            # Try lattice method
            try:
                lattice_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages=page_spec)
                for i, table in enumerate(lattice_tables):
                    if not table.df.empty:
                        tables.append({
                            'id': f'camelot_lattice_{i}',
                            'method': 'camelot_lattice',
                            'shape': table.df.shape,
                            'page': table.page if hasattr(table, 'page') else i + 1,
                            'accuracy': getattr(table, 'accuracy', 0) / 100.0 if hasattr(table, 'accuracy') else 0.8
                        })
            except Exception as e:
                logger.warning("Camelot lattice failed", error=str(e))
            
            # Try stream method
            try:
                stream_tables = camelot.read_pdf(pdf_path, flavor='stream', pages=page_spec)
                for i, table in enumerate(stream_tables):
                    if not table.df.empty:
                        tables.append({
                            'id': f'camelot_stream_{i}',
                            'method': 'camelot_stream',
                            'shape': table.df.shape,
                            'page': table.page if hasattr(table, 'page') else i + 1,
                            'accuracy': getattr(table, 'accuracy', 0) / 100.0 if hasattr(table, 'accuracy') else 0.7
                        })
            except Exception as e:
                logger.warning("Camelot stream failed", error=str(e))
            
        except Exception as e:
            logger.warning("Camelot extraction failed", error=str(e))
        
        return tables
    
    def extract_tables_with_tabula(self, pdf_path: str) -> list:
        """Extract tables using Tabula with dynamic page handling"""
        tables = []
        
        try:
            # Extract all tables
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
            
            for i, df in enumerate(dfs):
                if not df.empty:
                    total_cells = df.size
                    null_cells = df.isnull().sum().sum()
                    completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
                    
                    tables.append({
                        'id': f'tabula_{i}',
                        'method': 'tabula',
                        'shape': df.shape,
                        'completeness': completeness,
                        'page': i + 1  # Estimated
                    })
        
        except Exception as e:
            logger.warning("Tabula extraction failed", error=str(e))
        
        return tables
    
    async def export_results(self, results: list):
        """Export processing results"""
        results_dir = Path("./data/enhanced_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Export detailed results
        export_data = {
            'metadata': {
                'processed_at': pd.Timestamp.now().isoformat(),
                'total_files': len(results),
                'extraction_tools': ['PyMuPDF', 'Camelot', 'Tabula'],
                'version': '1.0_improved'
            },
            'results': results,
            'summary': {
                'total_pages': sum(r['pages'] for r in results),
                'total_tables': sum(r['tables_found'] for r in results),
                'total_text_chars': sum(r['text_length'] for r in results),
                'files_with_tables': len([r for r in results if r['tables_found'] > 0]),
                'files_with_content': len([r for r in results if r['text_length'] > 0])
            }
        }
        
        output_file = results_dir / "s3_processing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

async def main():
    """Main execution"""
    processor = SimplifiedProcessor()
    
    # Process S3 bucket with improved extraction
    processed_count = await processor.process_s3_bucket(max_files=50)  # Process up to 50 files
    
    if processed_count > 0:
        print(f"\n🎉 Successfully processed {processed_count} PDFs from S3 bucket!")
        print("Enhanced extraction with dynamic page handling is working correctly.")
    else:
        print("\n❌ No files were successfully processed.")

if __name__ == "__main__":
    asyncio.run(main())