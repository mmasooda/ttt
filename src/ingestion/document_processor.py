import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import mimetypes
from dataclasses import dataclass
import json
import re

from ..utils import logger

@dataclass
class ProcessedDocument:
    """Processed document data structure"""
    id: str
    filename: str
    filepath: str
    s3_key: str
    content_type: str
    size: int
    title: str
    content: str
    content_preview: str
    page_count: int
    metadata: Dict[str, Any]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

class DocumentProcessor:
    """Process documents and extract text content"""
    
    def __init__(self):
        self.supported_types = {
            '.pdf': self._process_pdf,
            '.txt': self._process_text,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel
        }
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file can be processed"""
        return file_path.suffix.lower() in self.supported_types
    
    def process_file(self, file_path: Path, s3_key: str = None) -> Optional[ProcessedDocument]:
        """Process a single file"""
        try:
            if not file_path.exists():
                logger.error("File not found", filepath=str(file_path))
                return None
            
            if not self.can_process(file_path):
                logger.warning("Unsupported file type", filepath=str(file_path))
                return None
            
            logger.info("Processing document", filepath=str(file_path))
            
            # Get file metadata
            stat = file_path.stat()
            content_type, _ = mimetypes.guess_type(str(file_path))
            
            # Generate document ID
            doc_id = self._generate_doc_id(file_path, s3_key)
            
            # Process based on file type
            ext = file_path.suffix.lower()
            processor = self.supported_types[ext]
            content, title, page_count, metadata = processor(file_path)
            
            if not content:
                logger.warning("No content extracted", filepath=str(file_path))
                return None
            
            # Create content preview
            content_preview = content[:500] + "..." if len(content) > 500 else content
            
            # Extract entities and relationships (basic implementation)
            entities = self._extract_entities(content)
            relationships = self._extract_relationships(content, entities)
            
            doc = ProcessedDocument(
                id=doc_id,
                filename=file_path.name,
                filepath=str(file_path),
                s3_key=s3_key or str(file_path),
                content_type=content_type or 'application/octet-stream',
                size=stat.st_size,
                title=title or file_path.stem,
                content=content,
                content_preview=content_preview,
                page_count=page_count,
                metadata=metadata,
                entities=entities,
                relationships=relationships
            )
            
            logger.info("Document processed successfully", 
                       doc_id=doc_id, 
                       entities_count=len(entities),
                       content_length=len(content))
            
            return doc
            
        except Exception as e:
            logger.error("Document processing failed", 
                        filepath=str(file_path), error=str(e))
            return None
    
    def _generate_doc_id(self, file_path: Path, s3_key: str = None) -> str:
        """Generate unique document ID"""
        key = s3_key or str(file_path)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _process_pdf(self, file_path: Path) -> Tuple[str, str, int, Dict]:
        """Process PDF file"""
        try:
            doc = fitz.open(str(file_path))
            content_parts = []
            title = ""
            
            # Extract metadata
            metadata = doc.metadata
            if metadata.get('title'):
                title = metadata['title']
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc.page(page_num)
                text = page.get_text()
                if text.strip():
                    content_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            
            content = "\n\n".join(content_parts)
            
            # If no title from metadata, try to extract from first page
            if not title and content_parts:
                first_lines = content_parts[0].split('\n')[:5]
                for line in first_lines:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 100:
                        title = line
                        break
            
            return content, title, doc.page_count, metadata
            
        except Exception as e:
            logger.error("PDF processing failed", filepath=str(file_path), error=str(e))
            return "", "", 0, {}
    
    def _process_text(self, file_path: Path) -> Tuple[str, str, int, Dict]:
        """Process text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to extract title from first line
            lines = content.split('\n')
            title = lines[0][:100] if lines else ""
            
            metadata = {
                'encoding': 'utf-8',
                'line_count': len(lines)
            }
            
            return content, title, 1, metadata
            
        except Exception as e:
            logger.error("Text processing failed", filepath=str(file_path), error=str(e))
            return "", "", 0, {}
    
    def _process_csv(self, file_path: Path) -> Tuple[str, str, int, Dict]:
        """Process CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert to readable text format
            content_parts = [
                f"CSV File: {file_path.name}",
                f"Columns: {', '.join(df.columns.tolist())}",
                f"Rows: {len(df)}",
                "\nFirst 10 rows:",
                df.head(10).to_string(index=False),
                "\nData types:",
                df.dtypes.to_string()
            ]
            
            if len(df) > 10:
                content_parts.extend([
                    "\nLast 5 rows:",
                    df.tail(5).to_string(index=False)
                ])
            
            content = "\n".join(content_parts)
            title = f"CSV: {file_path.stem}"
            
            metadata = {
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'column_count': len(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
            
            return content, title, 1, metadata
            
        except Exception as e:
            logger.error("CSV processing failed", filepath=str(file_path), error=str(e))
            return "", "", 0, {}
    
    def _process_excel(self, file_path: Path) -> Tuple[str, str, int, Dict]:
        """Process Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            content_parts = [f"Excel File: {file_path.name}"]
            
            metadata = {
                'sheet_names': excel_file.sheet_names,
                'sheet_count': len(excel_file.sheet_names)
            }
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                content_parts.extend([
                    f"\n--- Sheet: {sheet_name} ---",
                    f"Columns: {', '.join(df.columns.tolist())}",
                    f"Rows: {len(df)}",
                    "\nFirst 5 rows:",
                    df.head(5).to_string(index=False)
                ])
                
                metadata[f'sheet_{sheet_name}'] = {
                    'columns': df.columns.tolist(),
                    'row_count': len(df),
                    'column_count': len(df.columns)
                }
            
            content = "\n".join(content_parts)
            title = f"Excel: {file_path.stem}"
            
            return content, title, len(excel_file.sheet_names), metadata
            
        except Exception as e:
            logger.error("Excel processing failed", filepath=str(file_path), error=str(e))
            return "", "", 0, {}
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Basic entity extraction (to be enhanced with NLP)"""
        entities = []
        
        # Simple regex patterns for common entities
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'money': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'name': match.group(),
                    'type': entity_type,
                    'confidence': 0.8,
                    'description': f"{entity_type.title()} extracted from document",
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # Extract potential product names/codes (alphanumeric patterns)
        product_pattern = r'\b[A-Z]{2,3}[-_]?\d{2,4}[A-Z]?\b'
        matches = re.finditer(product_pattern, content)
        for match in matches:
            entities.append({
                'name': match.group(),
                'type': 'product_code',
                'confidence': 0.6,
                'description': 'Potential product code or model number',
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_relationships(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic relationship extraction between entities"""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities co-occur within 100 characters
                pos1 = entity1['start_pos']
                pos2 = entity2['start_pos']
                
                if abs(pos1 - pos2) <= 100:
                    relationships.append({
                        'source': entity1['name'],
                        'target': entity2['name'],
                        'type': 'co_occurs',
                        'confidence': 0.5,
                        'context': content[min(pos1, pos2)-50:max(pos1, pos2)+50]
                    })
        
        return relationships