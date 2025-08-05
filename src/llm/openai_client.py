#!/usr/bin/env python3
"""
OpenAI Client for LLM-assisted extraction and generation
Uses GPT-4.1-mini for ingestion and GPT-4o for generation
"""

import openai
from typing import List, Dict, Any, Optional
import json
import asyncio
from ..utils import settings, logger

class OpenAIClient:
    """OpenAI client for LLM-assisted processing"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.ingestion_model = "gpt-4-1106-preview"  # GPT-4.1-mini for ingestion
        self.generation_model = "gpt-4o"  # GPT-4o for generation
        self.embedding_model = "text-embedding-3-small"
        
    async def extract_entities_with_llm(self, content: str, document_type: str = "technical_document") -> Dict[str, Any]:
        """Use GPT-4.1-mini for enhanced entity extraction during ingestion"""
        
        prompt = self._get_entity_extraction_prompt(document_type)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.ingestion_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Extract entities from this fire alarm system document:\n\n{content[:8000]}"}  # Limit content
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content_text = response.choices[0].message.content
            
            # Parse structured response
            entities = self._parse_entity_response(content_text)
            
            logger.info("LLM entity extraction completed", 
                       entities_found=len(entities.get('entities', [])),
                       model=self.ingestion_model)
            
            return entities
            
        except Exception as e:
            logger.error("LLM entity extraction failed", error=str(e))
            return {'entities': [], 'relationships': [], 'metadata': {'error': str(e)}}
    
    async def extract_tables_with_llm(self, table_data: List[Dict], document_context: str) -> Dict[str, Any]:
        """Use GPT-4.1-mini to understand and structure table data"""
        
        prompt = self._get_table_analysis_prompt()
        
        # Prepare table data for LLM
        table_summary = []
        for i, table in enumerate(table_data[:3]):  # Limit to first 3 tables
            table_info = {
                'table_id': i + 1,
                'headers': table.get('headers', []),
                'sample_rows': table.get('data', [])[:5],  # First 5 rows
                'extraction_method': table.get('extraction_method', 'unknown'),
                'confidence': table.get('confidence_score', 0)
            }
            table_summary.append(table_info)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.ingestion_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Document context: {document_context[:2000]}\n\nAnalyze these tables:\n{json.dumps(table_summary, indent=2)}"}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            content_text = response.choices[0].message.content
            analysis = self._parse_table_analysis(content_text)
            
            logger.info("LLM table analysis completed", 
                       tables_analyzed=len(table_summary),
                       model=self.ingestion_model)
            
            return analysis
            
        except Exception as e:
            logger.error("LLM table analysis failed", error=str(e))
            return {'table_analysis': [], 'extracted_entities': [], 'metadata': {'error': str(e)}}
    
    async def enhance_relationships_with_llm(self, entities: List[Dict], content_context: str) -> List[Dict[str, Any]]:
        """Use GPT-4.1-mini to identify relationships between entities"""
        
        prompt = self._get_relationship_extraction_prompt()
        
        # Prepare entity data
        entity_summary = []
        for entity in entities[:20]:  # Limit to first 20 entities
            entity_summary.append({
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'context': entity.get('context', '')[:200]  # Limit context
            })
        
        try:
            response = await self.client.chat.completions.create(
                model=self.ingestion_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Document context: {content_context[:3000]}\n\nIdentify relationships between these entities:\n{json.dumps(entity_summary, indent=2)}"}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            content_text = response.choices[0].message.content
            relationships = self._parse_relationship_response(content_text)
            
            logger.info("LLM relationship extraction completed", 
                       relationships_found=len(relationships),
                       model=self.ingestion_model)
            
            return relationships
            
        except Exception as e:
            logger.error("LLM relationship extraction failed", error=str(e))
            return []
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's text-embedding-3-small"""
        
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            logger.info("Embedding generation completed", 
                       texts_processed=len(texts),
                       embeddings_generated=len(all_embeddings),
                       model=self.embedding_model)
            
            return all_embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            return []
    
    async def generate_response_with_gpt4o(self, query: str, context: Dict[str, Any]) -> str:
        """Use GPT-4o for final response generation"""
        
        prompt = self._get_generation_prompt()
        
        # Prepare context
        context_str = self._format_context_for_generation(context)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str}"}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            content_text = response.choices[0].message.content
            
            logger.info("Response generation completed", 
                       query_length=len(query),
                       response_length=len(content_text),
                       model=self.generation_model)
            
            return content_text
            
        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return f"Error generating response: {str(e)}"
    
    def _get_entity_extraction_prompt(self, document_type: str) -> str:
        """Get prompt for entity extraction based on document type"""
        
        base_prompt = """You are an expert in fire alarm system documentation analysis. Extract relevant entities from technical documents.

FIRE ALARM DOMAIN ENTITIES TO EXTRACT:
- Products: Fire alarm panels, detectors, sounders, beacons, call points, cables
- Standards: BS 5839, BS EN 54, EN 54, NFPA 72, ISO 7240, IEC standards
- Specifications: Voltage ratings, current ratings, IP ratings, temperature ranges
- Components: Modules, interfaces, power supplies, batteries, mounting hardware
- Locations: Zones, areas, buildings, floors, rooms
- Quantities: Numbers, measurements, distances
- Manufacturers: Company names, brand names
- Model Numbers: Product codes, part numbers, SKUs

RELATIONSHIPS TO IDENTIFY:
- COMPATIBLE_WITH: Product A works with Product B
- HAS_MODULE: Panel has specific modules
- REQUIRES: Product A needs Product B to function
- ALTERNATIVE_TO: Product A can replace Product B
- PART_OF: Component is part of larger system
- CONNECTS_TO: Physical or logical connections
- COMPLIES_WITH: Meets specific standards

Output format:
<entities>
[
  {"name": "entity_name", "type": "entity_type", "confidence": 0.9, "context": "surrounding_text"},
  ...
]
</entities>

<relationships>
[
  {"source": "entity1", "target": "entity2", "type": "RELATIONSHIP_TYPE", "confidence": 0.8, "evidence": "supporting_text"},
  ...
]
</relationships>"""
        
        if document_type == "specifications":
            base_prompt += "\n\nFocus especially on technical requirements, standards compliance, and product specifications."
        elif document_type == "boq":
            base_prompt += "\n\nFocus especially on quantities, part numbers, pricing, and product relationships."
        elif document_type == "offer":
            base_prompt += "\n\nFocus especially on proposed products, alternatives, and technical solutions."
        
        return base_prompt
    
    def _get_table_analysis_prompt(self) -> str:
        """Get prompt for table analysis"""
        
        return """You are an expert at analyzing fire alarm system tables and extracting structured information.

ANALYZE THE TABLES FOR:
- Product specifications (model numbers, ratings, features)
- Bill of quantities (items, quantities, prices)
- Compatibility matrices (which products work together)
- Technical parameters (voltage, current, temperature, etc.)
- Standards compliance information

OUTPUT FORMAT:
<table_analysis>
[
  {
    "table_id": 1,
    "table_type": "product_specifications|boq|compatibility|technical_params",
    "key_columns": ["column1", "column2"],
    "extracted_products": ["product1", "product2"],
    "extracted_specs": {"spec_name": "value"},
    "relationships_found": [{"source": "A", "target": "B", "type": "COMPATIBLE_WITH"}]
  }
]
</table_analysis>

<extracted_entities>
[
  {"name": "entity_name", "type": "product|specification|standard", "source": "table_1", "confidence": 0.9}
]
</extracted_entities>"""
    
    def _get_relationship_extraction_prompt(self) -> str:
        """Get prompt for relationship extraction"""
        
        return """You are an expert at identifying relationships between fire alarm system components.

RELATIONSHIP TYPES FOR FIRE ALARM SYSTEMS:
- COMPATIBLE_WITH: Products that can work together
- HAS_MODULE: Panels/systems that contain specific modules
- REQUIRES: Dependencies (Product A needs Product B)
- ALTERNATIVE_TO: Products that can substitute for each other
- PART_OF: Components that belong to larger assemblies
- CONNECTS_TO: Physical or logical connections
- COMPLIES_WITH: Standards compliance relationships
- POWERS: Power supply relationships
- MONITORS: Monitoring/supervision relationships

OUTPUT FORMAT:
<relationships>
[
  {
    "source": "source_entity_name",
    "target": "target_entity_name", 
    "type": "RELATIONSHIP_TYPE",
    "confidence": 0.8,
    "evidence": "text that supports this relationship",
    "bidirectional": true/false
  }
]
</relationships>"""
    
    def _get_generation_prompt(self) -> str:
        """Get prompt for final response generation with GPT-4o"""
        
        return """You are an expert fire alarm system consultant. Generate comprehensive, accurate responses based on the provided technical documentation and knowledge graph context.

CAPABILITIES:
- Analyze fire alarm system requirements
- Recommend appropriate products and solutions
- Explain technical specifications and compliance
- Provide installation and configuration guidance
- Compare different products and alternatives

RESPONSE GUIDELINES:
- Be technically accurate and specific
- Reference relevant standards (BS 5839, EN 54, NFPA 72, etc.)
- Include product specifications when relevant
- Provide practical implementation advice
- Cite sources from the knowledge graph when available
- Structure responses clearly with headings and bullet points

Always base your response on the provided context from the knowledge graph and document excerpts."""
    
    def _parse_entity_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response for entity extraction"""
        
        result = {'entities': [], 'relationships': [], 'metadata': {}}
        
        try:
            # Extract entities section
            if '<entities>' in content and '</entities>' in content:
                entities_text = content.split('<entities>')[1].split('</entities>')[0]
                entities = json.loads(entities_text.strip())
                result['entities'] = entities
            
            # Extract relationships section
            if '<relationships>' in content and '</relationships>' in content:
                relationships_text = content.split('<relationships>')[1].split('</relationships>')[0]
                relationships = json.loads(relationships_text.strip())
                result['relationships'] = relationships
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM entity response", error=str(e))
            # Fallback: try to extract from text
            result = self._fallback_entity_parsing(content)
        
        return result
    
    def _parse_table_analysis(self, content: str) -> Dict[str, Any]:
        """Parse LLM response for table analysis"""
        
        result = {'table_analysis': [], 'extracted_entities': [], 'metadata': {}}
        
        try:
            # Extract table analysis
            if '<table_analysis>' in content and '</table_analysis>' in content:
                analysis_text = content.split('<table_analysis>')[1].split('</table_analysis>')[0]
                analysis = json.loads(analysis_text.strip())
                result['table_analysis'] = analysis
            
            # Extract entities from tables
            if '<extracted_entities>' in content and '</extracted_entities>' in content:
                entities_text = content.split('<extracted_entities>')[1].split('</extracted_entities>')[0]
                entities = json.loads(entities_text.strip())
                result['extracted_entities'] = entities
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM table analysis", error=str(e))
        
        return result
    
    def _parse_relationship_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM response for relationship extraction"""
        
        relationships = []
        
        try:
            if '<relationships>' in content and '</relationships>' in content:
                relationships_text = content.split('<relationships>')[1].split('</relationships>')[0]
                relationships = json.loads(relationships_text.strip())
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM relationship response", error=str(e))
        
        return relationships
    
    def _fallback_entity_parsing(self, content: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        
        # Simple text-based extraction as fallback
        entities = []
        relationships = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') and ':' in line:
                # Try to extract entity-like information
                parts = line[1:].strip().split(':', 1)
                if len(parts) == 2:
                    entities.append({
                        'name': parts[0].strip(),
                        'type': 'extracted',
                        'confidence': 0.7,
                        'context': parts[1].strip()[:200]
                    })
        
        return {'entities': entities, 'relationships': relationships, 'metadata': {'fallback_parsing': True}}
    
    def _format_context_for_generation(self, context: Dict[str, Any]) -> str:
        """Format context for GPT-4o generation"""
        
        formatted_parts = []
        
        # Add graph context
        if 'graph_context' in context:
            formatted_parts.append("=== KNOWLEDGE GRAPH CONTEXT ===")
            formatted_parts.append(json.dumps(context['graph_context'], indent=2))
        
        # Add document excerpts
        if 'document_excerpts' in context:
            formatted_parts.append("=== RELEVANT DOCUMENT EXCERPTS ===")
            for excerpt in context['document_excerpts']:
                formatted_parts.append(f"Source: {excerpt.get('source', 'Unknown')}")
                formatted_parts.append(excerpt.get('content', ''))
                formatted_parts.append("---")
        
        # Add vector search results
        if 'vector_results' in context:
            formatted_parts.append("=== SIMILAR CONTENT ===")
            for result in context['vector_results']:
                formatted_parts.append(f"Similarity: {result.get('score', 0):.3f}")
                formatted_parts.append(result.get('content', ''))
                formatted_parts.append("---")
        
        return '\n'.join(formatted_parts)