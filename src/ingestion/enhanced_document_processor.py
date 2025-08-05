#!/usr/bin/env python3
"""
Enhanced Document Processor with Triple-Layer Extraction
Combines PyMuPDF + Camelot + Tabula for superior table and content extraction
"""

import fitz  # PyMuPDF
import camelot
import tabula
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import tempfile
import os
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from ..utils import logger, settings

# Load spaCy model for enhanced NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy English model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class ExtractedTable:
    """Enhanced table representation"""
    id: str
    extraction_method: str  # 'camelot', 'tabula', 'pymupdf'
    confidence_score: float
    headers: List[str]
    data: List[List[str]]
    metadata: Dict[str, Any]
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    
@dataclass
class ExtractedEntity:
    """Enhanced entity representation"""
    name: str
    type: str
    confidence: float
    context: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    embedding: Optional[List[float]] = None
    relationships: List[Dict[str, Any]] = None

@dataclass
class EnhancedDocument:
    """Enhanced document representation with triple-layer extraction"""
    filename: str
    content: str
    page_count: int
    tables: List[ExtractedTable]
    entities: List[ExtractedEntity]
    metadata: Dict[str, Any]
    embeddings: Dict[str, List[float]]  # Section embeddings
    extraction_summary: Dict[str, Any]

class TripleLayerExtractor:
    """Triple-layer extraction engine combining multiple tools"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def extract_from_pdf(self, pdf_path: str) -> EnhancedDocument:
        """Main extraction method using triple-layer approach"""
        logger.info("Starting triple-layer extraction", file=pdf_path)
        
        # Layer 1: PyMuPDF for general content and metadata
        pymupdf_data = self._extract_with_pymupdf(pdf_path)
        
        # Layer 2: Camelot for precise table extraction
        camelot_tables = self._extract_tables_with_camelot(pdf_path)
        
        # Layer 3: Tabula as fallback for complex tables
        tabula_tables = self._extract_tables_with_tabula(pdf_path)
        
        # Combine and deduplicate tables
        all_tables = self._merge_table_extractions(camelot_tables, tabula_tables, pymupdf_data['tables'])
        
        # Enhanced entity extraction
        enhanced_entities = self._extract_enhanced_entities(pymupdf_data['content'], pdf_path)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(pymupdf_data['content'], all_tables)
        
        # Create extraction summary
        extraction_summary = {
            'pymupdf_pages': pymupdf_data['page_count'],
            'camelot_tables': len(camelot_tables),
            'tabula_tables': len(tabula_tables),
            'final_tables': len(all_tables),
            'entities_extracted': len(enhanced_entities),
            'quality_score': self._calculate_quality_score(all_tables, enhanced_entities)
        }
        
        document = EnhancedDocument(
            filename=Path(pdf_path).name,
            content=pymupdf_data['content'],
            page_count=pymupdf_data['page_count'],
            tables=all_tables,
            entities=enhanced_entities,
            metadata=pymupdf_data['metadata'],
            embeddings=embeddings,
            extraction_summary=extraction_summary
        )
        
        logger.info("Triple-layer extraction completed", 
                   tables=len(all_tables), 
                   entities=len(enhanced_entities),
                   quality_score=extraction_summary['quality_score'])
        
        return document
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Layer 1: PyMuPDF extraction for content and basic tables"""
        try:
            logger.info("Starting PyMuPDF extraction", file=pdf_path)
            doc = fitz.open(pdf_path)
            actual_pages = len(doc)
            logger.info("PDF opened successfully", actual_pages=actual_pages)
            
            if actual_pages == 0:
                logger.warning("PDF has no pages", file=pdf_path)
                doc.close()
                return {
                    'content': "",
                    'page_count': 0,
                    'tables': [],
                    'metadata': {'extraction_note': 'PDF contains no pages'}
                }
            
            full_text = ""
            basic_tables = []
            metadata = {
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': str(doc.metadata.get('creationDate', '')),
                'modification_date': str(doc.metadata.get('modDate', '')),
                'actual_pages': actual_pages,
                'extraction_note': f'Processing {actual_pages} actual pages (ignoring any page references in content)'
            }
            
            # Process only the pages that actually exist
            for page_num in range(actual_pages):
                try:
                    page = doc[page_num]
                    
                    # Extract text - clean up page references
                    text = page.get_text()
                    
                    # Clean text to remove misleading page references
                    cleaned_text = self._clean_page_references(text, page_num + 1, actual_pages)
                    
                    full_text += f"\n--- Actual Page {page_num + 1} of {actual_pages} ---\n{cleaned_text}"
                    
                    # Extract basic table structures
                    page_tables = self._extract_pymupdf_tables(page, page_num)
                    basic_tables.extend(page_tables)
                    
                except Exception as e:
                    logger.warning("Failed to process page", page=page_num + 1, error=str(e))
                    continue
            
            doc.close()
            
            logger.info("PyMuPDF extraction completed", 
                       actual_pages=actual_pages, 
                       text_length=len(full_text),
                       tables_found=len(basic_tables))
            
            return {
                'content': full_text,
                'page_count': actual_pages,
                'tables': basic_tables,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error("PyMuPDF extraction failed", error=str(e))
            return {
                'content': "",
                'page_count': 0,
                'tables': [],
                'metadata': {'extraction_error': str(e)}
            }
    
    def _clean_page_references(self, text: str, current_page: int, total_pages: int) -> str:
        """Clean misleading page references from extracted PDFs"""
        if not text:
            return text
        
        # Remove patterns like "Page X of Y" where Y might be from original document
        # Replace with actual page info
        
        # Pattern 1: "Page X of Y" at start/end of lines
        text = re.sub(r'^Page\s+\d+\s+of\s+\d+\s*$', 
                     f'[Actual Page {current_page} of {total_pages}]', 
                     text, flags=re.MULTILINE)
        
        # Pattern 2: "X/Y" page references
        text = re.sub(r'^\d+/\d+\s*$', 
                     f'[{current_page}/{total_pages}]', 
                     text, flags=re.MULTILINE)
        
        # Pattern 3: Remove standalone page numbers that might be misleading
        # But keep them in context (like "see page 5" references)
        text = re.sub(r'^(?:Page\s*)?\d+\s*$', 
                     f'[Page {current_page}]', 
                     text, flags=re.MULTILINE)
        
        # Add note about extracted PDF
        if 'extracted from larger document' not in text.lower():
            text += f"\n[Note: This is page {current_page} of {total_pages} in extracted PDF]"
        
        return text
    
    def _extract_pymupdf_tables(self, page, page_num: int) -> List[ExtractedTable]:
        """Extract basic table structures using PyMuPDF"""
        tables = []
        
        try:
            # Find tables using text layout analysis
            blocks = page.get_text("dict")
            
            # Simple table detection based on aligned text blocks
            # This is a basic implementation - PyMuPDF's strength is in text, not tables
            table_candidates = self._detect_table_patterns(blocks, page_num)
            
            for i, candidate in enumerate(table_candidates):
                table = ExtractedTable(
                    id=f"pymupdf_p{page_num}_t{i}",
                    extraction_method="pymupdf",
                    confidence_score=0.6,  # Lower confidence for basic detection
                    headers=candidate.get('headers', []),
                    data=candidate.get('data', []),
                    metadata={'detection_method': 'text_alignment'},
                    page_number=page_num,
                    bbox=candidate.get('bbox')
                )
                tables.append(table)
        
        except Exception as e:
            logger.warning("PyMuPDF table extraction failed", page=page_num, error=str(e))
        
        return tables
    
    def _detect_table_patterns(self, blocks: Dict, page_num: int) -> List[Dict[str, Any]]:
        """Detect table-like patterns in text blocks"""
        # Basic implementation - looks for aligned text that might be tabular
        # This is intentionally simple as we rely on Camelot/Tabula for tables
        candidates = []
        
        # Extract lines with consistent spacing patterns
        lines = []
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text and len(text.split()) > 2:  # Multi-column potential
                            lines.append({
                                'text': text,
                                'bbox': span["bbox"],
                                'y': span["bbox"][1]
                            })
        
        # Group lines by vertical position (simple table row detection)
        lines.sort(key=lambda x: x['y'])
        
        # Look for patterns with multiple columns (simplified)
        table_rows = []
        for line in lines:
            # Check if line has tab-separated or multiple space-separated values
            if '\t' in line['text'] or len(re.split(r'\s{2,}', line['text'])) > 2:
                table_rows.append(line)
        
        if len(table_rows) >= 3:  # Minimum table size
            # Create a basic table candidate
            data_rows = []
            for row in table_rows[:10]:  # Limit to first 10 rows
                cells = re.split(r'\s{2,}|\t', row['text'])
                if len(cells) > 1:
                    data_rows.append(cells)
            
            if data_rows:
                candidates.append({
                    'headers': data_rows[0] if data_rows else [],
                    'data': data_rows[1:] if len(data_rows) > 1 else [],
                    'bbox': (
                        min(row['bbox'][0] for row in table_rows),
                        min(row['bbox'][1] for row in table_rows),
                        max(row['bbox'][2] for row in table_rows),
                        max(row['bbox'][3] for row in table_rows)
                    )
                })
        
        return candidates
    
    def _extract_tables_with_camelot(self, pdf_path: str) -> List[ExtractedTable]:
        """Layer 2: Camelot for precise table extraction"""
        tables = []
        
        # First, determine actual page count
        try:
            import fitz
            temp_doc = fitz.open(pdf_path)
            actual_pages = len(temp_doc)
            temp_doc.close()
            
            if actual_pages == 0:
                logger.info("No pages found for Camelot extraction")
                return tables
            
            logger.info("Camelot processing pages", actual_pages=actual_pages)
            
        except Exception as e:
            logger.warning("Could not determine page count, using 'all'", error=str(e))
            actual_pages = None
        
        # Build page specification - use actual pages or 'all'
        page_spec = 'all' if actual_pages is None else f'1-{actual_pages}'
        
        try:
            # Use lattice method for tables with clear borders
            logger.info("Camelot lattice extraction", pages=page_spec)
            lattice_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages=page_spec)
            
            for i, table in enumerate(lattice_tables):
                # Validate table has content
                if table.df.empty:
                    continue
                    
                # Map page number to actual page (Camelot uses 1-based indexing)
                actual_page_num = table.page if hasattr(table, 'page') else i + 1
                
                extracted_table = ExtractedTable(
                    id=f"camelot_lattice_{i}",
                    extraction_method="camelot_lattice",
                    confidence_score=float(table.accuracy) / 100.0 if hasattr(table, 'accuracy') else 0.8,
                    headers=table.df.columns.tolist(),
                    data=table.df.values.tolist(),
                    metadata={
                        'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else {},
                        'shape': table.df.shape,
                        'actual_pages_context': actual_pages,
                        'extraction_note': f'Extracted from page {actual_page_num} of {actual_pages} actual pages'
                    },
                    page_number=actual_page_num
                )
                tables.append(extracted_table)
        
        except Exception as e:
            logger.warning("Camelot lattice extraction failed", error=str(e), pages=page_spec)
        
        try:
            # Use stream method for tables without clear borders
            logger.info("Camelot stream extraction", pages=page_spec)
            stream_tables = camelot.read_pdf(pdf_path, flavor='stream', pages=page_spec)
            
            for i, table in enumerate(stream_tables):
                # Validate table has content
                if table.df.empty:
                    continue
                    
                # Map page number to actual page
                actual_page_num = table.page if hasattr(table, 'page') else i + 1
                
                extracted_table = ExtractedTable(
                    id=f"camelot_stream_{i}",
                    extraction_method="camelot_stream",
                    confidence_score=float(table.accuracy) / 100.0 if hasattr(table, 'accuracy') else 0.7,
                    headers=table.df.columns.tolist(),
                    data=table.df.values.tolist(),
                    metadata={
                        'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else {},
                        'shape': table.df.shape,
                        'actual_pages_context': actual_pages,
                        'extraction_note': f'Extracted from page {actual_page_num} of {actual_pages} actual pages'
                    },
                    page_number=actual_page_num
                )
                tables.append(extracted_table)
        
        except Exception as e:
            logger.warning("Camelot stream extraction failed", error=str(e), pages=page_spec)
        
        logger.info("Camelot extraction completed", 
                   tables_found=len(tables),
                   actual_pages=actual_pages,
                   page_spec=page_spec)
        return tables
    
    def _extract_tables_with_tabula(self, pdf_path: str) -> List[ExtractedTable]:
        """Layer 3: Tabula as fallback for complex tables"""
        tables = []
        
        # First, determine actual page count for context
        try:
            import fitz
            temp_doc = fitz.open(pdf_path)
            actual_pages = len(temp_doc)
            temp_doc.close()
            
            if actual_pages == 0:
                logger.info("No pages found for Tabula extraction")
                return tables
            
            logger.info("Tabula processing pages", actual_pages=actual_pages)
            
        except Exception as e:
            logger.warning("Could not determine page count for Tabula", error=str(e))
            actual_pages = None
        
        try:
            # Extract all tables from all available pages
            logger.info("Tabula extraction starting", pages="all")
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
            
            for i, df in enumerate(dfs):
                if df.empty:
                    continue
                
                # Calculate confidence based on data quality
                total_cells = df.size
                null_cells = df.isnull().sum().sum()
                data_completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
                confidence = 0.75 * data_completeness  # Adjust confidence based on completeness
                
                # Estimate page number (Tabula doesn't provide exact page mapping)
                estimated_page = min(i + 1, actual_pages or 999)
                
                extracted_table = ExtractedTable(
                    id=f"tabula_{i}",
                    extraction_method="tabula",
                    confidence_score=confidence,
                    headers=df.columns.tolist(),
                    data=df.values.tolist(),
                    metadata={
                        'shape': df.shape,
                        'null_count': null_cells,
                        'data_completeness': data_completeness,
                        'actual_pages_context': actual_pages,
                        'extraction_note': f'Extracted table {i+1}, estimated from page {estimated_page} of {actual_pages or "unknown"} actual pages'
                    },
                    page_number=estimated_page
                )
                tables.append(extracted_table)
        
        except Exception as e:
            logger.warning("Tabula extraction failed", error=str(e))
        
        logger.info("Tabula extraction completed", 
                   tables_found=len(tables),
                   actual_pages=actual_pages)
        return tables
    
    def _merge_table_extractions(self, camelot_tables: List[ExtractedTable], 
                                tabula_tables: List[ExtractedTable], 
                                pymupdf_tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Merge and deduplicate table extractions from all three methods"""
        
        all_tables = camelot_tables + tabula_tables + pymupdf_tables
        
        if not all_tables:
            return []
        
        # Sort by confidence score (highest first)
        all_tables.sort(key=lambda t: t.confidence_score, reverse=True)
        
        # Simple deduplication based on table similarity
        unique_tables = []
        
        for table in all_tables:
            is_duplicate = False
            
            for existing_table in unique_tables:
                if self._are_tables_similar(table, existing_table):
                    is_duplicate = True
                    # Keep the higher quality table
                    if table.confidence_score > existing_table.confidence_score:
                        unique_tables.remove(existing_table)
                        unique_tables.append(table)
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        logger.info("Table deduplication completed", 
                   original=len(all_tables), 
                   unique=len(unique_tables))
        
        return unique_tables
    
    def _are_tables_similar(self, table1: ExtractedTable, table2: ExtractedTable) -> bool:
        """Check if two tables are similar (potential duplicates)"""
        
        # Compare dimensions
        shape1 = (len(table1.data), len(table1.headers))
        shape2 = (len(table2.data), len(table2.headers))
        
        if shape1 != shape2:
            return False
        
        # Compare headers
        if len(table1.headers) > 0 and len(table2.headers) > 0:
            header_similarity = len(set(table1.headers) & set(table2.headers)) / max(len(table1.headers), len(table2.headers))
            if header_similarity > 0.8:
                return True
        
        # Compare first few data rows
        if len(table1.data) > 0 and len(table2.data) > 0:
            sample_size = min(3, len(table1.data), len(table2.data))
            
            similar_rows = 0
            for i in range(sample_size):
                if len(table1.data[i]) == len(table2.data[i]):
                    cell_matches = sum(1 for c1, c2 in zip(table1.data[i], table2.data[i]) 
                                     if str(c1).strip() == str(c2).strip())
                    if cell_matches / len(table1.data[i]) > 0.8:
                        similar_rows += 1
            
            if similar_rows / sample_size > 0.7:
                return True
        
        return False
    
    def _extract_enhanced_entities(self, content: str, pdf_path: str) -> List[ExtractedEntity]:
        """Enhanced entity extraction using spaCy + custom patterns"""
        entities = []
        
        if not content or len(content.strip()) == 0:
            logger.warning("No content provided for entity extraction")
            return entities
        
        # Clean content to remove misleading page references
        cleaned_content = self._clean_content_for_entity_extraction(content)
        
        # Use spaCy for basic NER - limit content size for processing
        content_to_process = cleaned_content[:1000000]
        
        try:
            doc = nlp(content_to_process)
            
            # Extract standard entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'MONEY', 'PRODUCT', 'WORK_OF_ART', 'GPE']:
                    # Estimate page number from position in content
                    estimated_page = self._estimate_page_from_position(ent.start_char, cleaned_content)
                    
                    entity = ExtractedEntity(
                        name=ent.text.strip(),
                        type=ent.label_.lower(),
                        confidence=0.8,
                        context=ent.sent.text[:200] if ent.sent else "",
                        page_number=estimated_page,
                        embedding=embedding_model.encode(ent.text).tolist()
                    )
                    entities.append(entity)
        
        except Exception as e:
            logger.warning("spaCy entity extraction failed", error=str(e))
        
        # Custom patterns for fire alarm domain
        try:
            custom_entities = self._extract_domain_entities(cleaned_content)
            entities.extend(custom_entities)
        except Exception as e:
            logger.warning("Custom entity extraction failed", error=str(e))
        
        # Extract relationships between entities
        try:
            entities = self._extract_entity_relationships(entities, cleaned_content)
        except Exception as e:
            logger.warning("Entity relationship extraction failed", error=str(e))
        
        # Remove duplicate entities
        unique_entities = self._deduplicate_entities(entities)
        
        logger.info("Enhanced entity extraction completed", 
                   spacy_entities=len([e for e in unique_entities if e.type in ['org', 'person', 'money', 'product', 'gpe']]),
                   custom_entities=len([e for e in unique_entities if e.type not in ['org', 'person', 'money', 'product', 'gpe']]),
                   total=len(unique_entities))
        
        return unique_entities
    
    def _clean_content_for_entity_extraction(self, content: str) -> str:
        """Clean content to improve entity extraction accuracy"""
        if not content:
            return content
        
        # Remove page markers that might confuse entity extraction
        cleaned = re.sub(r'\n--- (?:Actual )?Page \d+ (?:of \d+ )?---\n', '\n', content)
        
        # Remove extraction notes
        cleaned = re.sub(r'\[Note: This is page \d+ of \d+ in extracted PDF\]', '', cleaned)
        cleaned = re.sub(r'\[Actual Page \d+ of \d+\]', '', cleaned)
        cleaned = re.sub(r'\[\d+/\d+\]', '', cleaned)
        
        # Remove standalone page numbers at line start/end
        cleaned = re.sub(r'^\d+\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def _estimate_page_from_position(self, char_position: int, content: str) -> int:
        """Estimate page number from character position in content"""
        if not content:
            return 1
        
        # Count page markers before this position
        content_before = content[:char_position]
        page_markers = re.findall(r'--- (?:Actual )?Page (\d+)', content_before)
        
        if page_markers:
            return int(page_markers[-1])  # Return the last page marker found
        
        # Fallback: estimate based on content length
        # Assume roughly 2000 characters per page
        estimated_page = max(1, char_position // 2000 + 1)
        return estimated_page
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on name and type"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key for deduplication
            key = (entity.name.lower().strip(), entity.type.lower())
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # If duplicate, keep the one with higher confidence
                existing_idx = None
                for i, existing in enumerate(unique_entities):
                    if (existing.name.lower().strip(), existing.type.lower()) == key:
                        existing_idx = i
                        break
                
                if existing_idx is not None and entity.confidence > unique_entities[existing_idx].confidence:
                    unique_entities[existing_idx] = entity
        
        return unique_entities
    
    def _extract_domain_entities(self, content: str) -> List[ExtractedEntity]:
        """Extract domain-specific entities for fire alarm systems"""
        entities = []
        
        # Define patterns for fire alarm domain
        patterns = {
            'product_code': r'\b[A-Z]{2,}\-?[0-9]{2,}[A-Z]?\b',
            'standard': r'\b(?:BS|EN|ISO|NFPA|AS)\s*[0-9]{3,}(?:\-[0-9]+)?\b',
            'cable_spec': r'\b[0-9]+(?:\.[0-9]+)?\s*mm²?\s*(?:cable|wire)\b',
            'voltage': r'\b[0-9]+(?:\.[0-9]+)?\s*(?:V|volt|voltage)\b',
            'detection_zone': r'\b(?:zone|area|sector)\s*[0-9]+\b',
            'quantity': r'\b[0-9]+(?:\.[0-9]+)?\s*(?:pcs?|units?|nos?|pieces?|items?)\b',
            'price': r'[\$£€]\s*[0-9,]+(?:\.[0-9]+)?|[0-9,]+(?:\.[0-9]+)?\s*(?:USD|GBP|EUR|AED)',
            'fire_device': r'\b(?:detector|sensor|alarm|panel|sounder|beacon|call point|break glass)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                entity_text = match.group().strip()
                
                # Get context around the match
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                entity = ExtractedEntity(
                    name=entity_text,
                    type=entity_type,
                    confidence=0.85,
                    context=context,
                    page_number=0,  # Would need more complex mapping
                    embedding=embedding_model.encode(entity_text).tolist()
                )
                entities.append(entity)
        
        return entities
    
    def _extract_entity_relationships(self, entities: List[ExtractedEntity], content: str) -> List[ExtractedEntity]:
        """Extract relationships between entities"""
        
        # Simple relationship extraction based on co-occurrence
        for i, entity1 in enumerate(entities):
            relationships = []
            
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Check if entities appear in same context
                    if entity1.context and entity2.context:
                        # Simple overlap check
                        context1_words = set(entity1.context.lower().split())
                        context2_words = set(entity2.context.lower().split())
                        
                        overlap = len(context1_words & context2_words)
                        if overlap > 5:  # Arbitrary threshold
                            relationship = {
                                'target_entity': entity2.name,
                                'relationship_type': 'co_occurs',
                                'confidence': min(0.9, overlap / 10.0)
                            }
                            relationships.append(relationship)
            
            entities[i].relationships = relationships[:5]  # Limit to top 5 relationships
        
        return entities
    
    def _generate_embeddings(self, content: str, tables: List[ExtractedTable]) -> Dict[str, List[float]]:
        """Generate embeddings for content sections"""
        embeddings = {}
        
        # Split content into sections
        sections = content.split('\n--- Page')
        
        section_embeddings = []
        for i, section in enumerate(sections[:10]):  # Limit to first 10 sections
            if section.strip():
                section_embedding = embedding_model.encode(section[:1000]).tolist()  # Limit section size
                section_embeddings.append(section_embedding)
                embeddings[f'section_{i}'] = section_embedding
        
        # Create document-level embedding (average of sections)
        if section_embeddings:
            doc_embedding = np.mean(section_embeddings, axis=0).tolist()
            embeddings['document'] = doc_embedding
        
        # Generate table embeddings
        for table in tables:
            if table.data:
                # Create table text representation
                table_text = ' '.join([' '.join(table.headers), 
                                     ' '.join([' '.join(map(str, row)) for row in table.data[:5]])])
                table_embedding = embedding_model.encode(table_text[:1000]).tolist()
                embeddings[f'table_{table.id}'] = table_embedding
        
        return embeddings
    
    def _calculate_quality_score(self, tables: List[ExtractedTable], entities: List[ExtractedEntity]) -> float:
        """Calculate overall extraction quality score"""
        
        # Base score
        score = 0.5
        
        # Table quality contribution
        if tables:
            avg_table_confidence = sum(t.confidence_score for t in tables) / len(tables)
            score += avg_table_confidence * 0.3
        
        # Entity quality contribution
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            score += avg_entity_confidence * 0.2
        
        # Bonus for finding multiple extraction types
        extraction_methods = set(t.extraction_method for t in tables)
        if len(extraction_methods) > 1:
            score += 0.1
        
        # Normalize to 0-1 range
        return min(1.0, score)

class EnhancedDatasetProcessor:
    """Enhanced dataset processor using triple-layer extraction"""
    
    def __init__(self):
        self.extractor = TripleLayerExtractor()
        
    async def process_document(self, file_path: str, document_type: str = None) -> Optional[EnhancedDocument]:
        """Process a single document with enhanced extraction"""
        
        try:
            logger.info("Processing document with enhanced extraction", 
                       file=file_path, 
                       type=document_type)
            
            # Check if file is PDF
            if not file_path.lower().endswith('.pdf'):
                logger.warning("Enhanced extraction only supports PDFs currently", file=file_path)
                return None
            
            # Check if file exists and has content
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error("File does not exist", file=file_path)
                return None
            
            file_size = file_path_obj.stat().st_size
            if file_size == 0:
                logger.error("File is empty", file=file_path)
                return None
            
            logger.info("File validation passed", file=file_path, size_mb=file_size/(1024*1024))
            
            # Extract with triple-layer approach (not async)
            logger.info("Starting triple-layer extraction", file=file_path)
            document = self.extractor.extract_from_pdf(file_path)
            
            if not document:
                logger.error("Triple-layer extractor returned None", file=file_path)
                return None
            
            logger.info("Triple-layer extraction completed", 
                       file=file_path,
                       pages=document.page_count,
                       content_length=len(document.content),
                       tables=len(document.tables),
                       entities=len(document.entities))
            
            # Add document type classification if not provided
            if document_type:
                document.metadata['document_type'] = document_type
            else:
                document.metadata['document_type'] = self._classify_document_type(document)
            
            logger.info("Enhanced document processing completed",
                       file=document.filename,
                       tables=len(document.tables),
                       entities=len(document.entities),
                       quality=document.extraction_summary['quality_score'])
            
            return document
            
        except Exception as e:
            logger.error("Enhanced document processing failed", 
                        file=file_path, 
                        error=str(e))
            import traceback
            logger.error("Full traceback", traceback=traceback.format_exc())
            return None
    
    def _classify_document_type(self, document: EnhancedDocument) -> str:
        """Classify document type based on content and entities"""
        
        content_lower = document.content.lower()
        
        # Look for specific keywords
        if any(keyword in content_lower for keyword in ['specification', 'compliance', 'requirement', 'standard']):
            return 'specifications'
        elif any(keyword in content_lower for keyword in ['boq', 'bill of quantities', 'pricing', 'quantity']):
            return 'boq'
        elif any(keyword in content_lower for keyword in ['offer', 'proposal', 'quote', 'quotation']):
            return 'offer'
        else:
            return 'unknown'
    
    def export_enhanced_document(self, document: EnhancedDocument, output_path: str):
        """Export enhanced document to JSON"""
        
        # Convert to serializable format
        doc_dict = {
            'filename': document.filename,
            'content': document.content[:10000],  # Truncate for export
            'page_count': document.page_count,
            'tables': [
                {
                    'id': table.id,
                    'method': table.extraction_method,
                    'confidence': table.confidence_score,
                    'headers': table.headers,
                    'data': table.data[:20],  # Limit data size
                    'page': table.page_number,
                    'metadata': table.metadata
                }
                for table in document.tables
            ],
            'entities': [
                {
                    'name': entity.name,
                    'type': entity.type,
                    'confidence': entity.confidence,
                    'context': entity.context[:200],  # Truncate context
                    'page': entity.page_number,
                    'relationships': entity.relationships[:3] if entity.relationships else []
                }
                for entity in document.entities
            ],
            'metadata': document.metadata,
            'extraction_summary': document.extraction_summary,
            'embeddings_count': len(document.embeddings)
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Enhanced document exported", 
                   file=document.filename, 
                   output=output_path)