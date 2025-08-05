#!/usr/bin/env python3
"""
BYOKG-RAG Engine with KG-Linker
Complete implementation with LLM-assisted processing using GPT-4.1-mini for ingestion
and GPT-4o for generation
"""

import yaml
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils import Neo4jClient, logger, settings
from ..llm.openai_client import OpenAIClient
from ..vector.faiss_store import FAISSVectorStore, VectorDocument

@dataclass
class KGLinkerPrompts:
    """Container for KG-Linker prompts"""
    entity_extraction: Dict[str, Any]
    path_generation: Dict[str, Any]  
    query_generation: Dict[str, Any]
    answer_generation: Dict[str, Any]

class BYOKGRAGEngine:
    """Complete BYOKG-RAG engine with KG-Linker and LLM assistance"""
    
    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.openai_client = OpenAIClient()
        self.vector_store = FAISSVectorStore()
        
        # Load existing vector store data
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, schedule for later
                asyncio.create_task(self._load_vector_store())
            else:
                # If not in async context, run directly
                asyncio.run(self._load_vector_store())
        except:
            # Fallback - will be loaded on first query
            pass
            
        self.prompts = self._load_prompts()
        
        # Fire alarm domain schema
        self.schema = self._get_fire_alarm_schema()
        
        logger.info("BYOKG-RAG engine initialized")
    
    async def _load_vector_store(self):
        """Load vector store from disk"""
        try:
            loaded = await self.vector_store.load_from_disk()
            if loaded:
                logger.info(f"Vector store loaded with {len(self.vector_store.documents)} documents")
            else:
                logger.warning("Failed to load vector store from disk")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
    
    def _load_prompts(self) -> KGLinkerPrompts:
        """Load KG-Linker prompts from YAML files"""
        
        prompts_dir = Path("./prompts")
        
        try:
            # Load all prompt files
            with open(prompts_dir / "entity_extraction.yaml", 'r') as f:
                entity_extraction = yaml.safe_load(f)
            
            with open(prompts_dir / "path_generation.yaml", 'r') as f:
                path_generation = yaml.safe_load(f)
            
            with open(prompts_dir / "query_generation.yaml", 'r') as f:
                query_generation = yaml.safe_load(f)
            
            with open(prompts_dir / "answer_generation.yaml", 'r') as f:
                answer_generation = yaml.safe_load(f)
            
            logger.info("KG-Linker prompts loaded successfully")
            
            return KGLinkerPrompts(
                entity_extraction=entity_extraction,
                path_generation=path_generation,
                query_generation=query_generation,
                answer_generation=answer_generation
            )
            
        except Exception as e:
            logger.error("Failed to load prompts", error=str(e))
            raise
    
    def _get_fire_alarm_schema(self) -> Dict[str, Any]:
        """Get fire alarm domain schema for the knowledge graph"""
        
        return {
            "node_types": [
                "Panel",           # Fire alarm control panels
                "Module",          # Interface and communication modules
                "Device",          # Detectors, sounders, call points
                "Cable",           # Wiring and cable assemblies
                "Standard",        # Compliance standards and regulations
                "Specification",   # Technical parameters and requirements
                "Zone",            # Detection zones and system areas
                "Manufacturer"     # Companies and brands
            ],
            "relationship_types": [
                "COMPATIBLE_WITH",    # Products that can work together
                "HAS_MODULE",         # Panels that contain specific modules
                "REQUIRES",           # Dependencies between components
                "ALTERNATIVE_TO",     # Products that can substitute
                "PART_OF",            # Components belonging to assemblies
                "CONNECTS_TO",        # Physical or logical connections
                "COMPLIES_WITH",      # Standards compliance
                "POWERS",             # Power supply relationships
                "MONITORS",           # Monitoring and supervision
                "HAS_SPECIFICATION",  # Technical specification links
                "MANUFACTURED_BY",    # Product manufacturer links
                "INSTALLED_IN"        # Location and zone relationships
            ],
            "node_properties": {
                "common": ["name", "type", "model", "confidence", "category", "manufacturer"],
                "Panel": ["capacity", "loop_count", "protocol", "power_supply"],
                "Device": ["detection_type", "sensitivity", "operating_range", "current_draw"],
                "Specification": ["value", "unit", "tolerance", "conditions"],
                "Standard": ["version", "authority", "scope", "effective_date"]
            },
            "relationship_properties": [
                "confidence", "evidence", "strength", "created_at"
            ]
        }
    
    async def ingest_document_with_llm(self, content: str, source: str, 
                                     document_type: str = "technical_document",
                                     tables: List[Dict] = None) -> Dict[str, Any]:
        """Ingest document using LLM-assisted extraction (GPT-4.1-mini)"""
        
        logger.info("Starting LLM-assisted document ingestion", 
                   source=source, 
                   document_type=document_type,
                   content_length=len(content))
        
        results = {
            'source': source,
            'document_type': document_type,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'vector_chunks_added': 0,
            'graph_nodes_created': 0,
            'graph_edges_created': 0
        }
        
        try:
            # Step 1: LLM Entity Extraction with GPT-4.1-mini
            logger.info("Step 1: LLM entity extraction")
            entity_data = await self.openai_client.extract_entities_with_llm(content, document_type)
            
            entities = entity_data.get('entities', [])
            relationships = entity_data.get('relationships', [])
            
            results['entities_extracted'] = len(entities)
            results['relationships_extracted'] = len(relationships)
            
            # Step 2: LLM Table Analysis (if tables provided)
            if tables:
                logger.info("Step 2: LLM table analysis", tables=len(tables))
                table_analysis = await self.openai_client.extract_tables_with_llm(tables, content[:2000])
                
                # Add table-derived entities
                table_entities = table_analysis.get('extracted_entities', [])
                entities.extend(table_entities)
                results['entities_extracted'] += len(table_entities)
            
            # Step 3: Enhanced Relationship Extraction
            if entities:
                logger.info("Step 3: Enhanced relationship extraction")
                enhanced_relationships = await self.openai_client.enhance_relationships_with_llm(
                    entities, content[:3000]
                )
                relationships.extend(enhanced_relationships)
                results['relationships_extracted'] = len(relationships)
            
            # Step 4: Add to Knowledge Graph
            logger.info("Step 4: Adding to knowledge graph")
            graph_stats = await self._add_to_knowledge_graph(entities, relationships, source)
            results.update(graph_stats)
            
            # Step 5: Add to Vector Store
            logger.info("Step 5: Adding to vector store")
            vector_stats = await self._add_to_vector_store(content, source, {
                'document_type': document_type,
                'entities_count': len(entities),
                'relationships_count': len(relationships)
            })
            results.update(vector_stats)
            
            # Step 6: BYOKG-RAG Iterative Refinement Process
            logger.info("Step 6: Starting BYOKG-RAG iterative refinement")
            refinement_stats = await self._byokg_iterative_refinement(entities, relationships, source)
            results.update(refinement_stats)
            
            logger.info("LLM-assisted ingestion completed successfully", **results)
            
            return results
            
        except Exception as e:
            logger.error("LLM-assisted ingestion failed", error=str(e))
            results['error'] = str(e)
            return results
    
    async def _add_to_knowledge_graph(self, entities: List[Dict], relationships: List[Dict], 
                                    source: str) -> Dict[str, Any]:
        """Add extracted entities and relationships to knowledge graph"""
        
        nodes_created = 0
        edges_created = 0
        
        try:
            # Create nodes for entities
            for entity in entities:
                node_data = {
                    'name': entity.get('name', ''),
                    'type': entity.get('type', 'unknown'),
                    'confidence': entity.get('confidence', 0.5),
                    'source': source,
                    'context': entity.get('context', '')[:500]  # Limit context size
                }
                
                # Add domain-specific properties based on type
                entity_type = entity.get('type', '').lower()
                if entity_type in ['panel', 'device', 'module', 'detector', 'sensor', 'base', 'cable']:
                    node_data['category'] = 'hardware'
                elif entity_type in ['standard', 'specification', 'requirement', 'compliance']:
                    node_data['category'] = 'technical'
                elif entity_type in ['manufacturer', 'brand', 'company']:
                    node_data['category'] = 'commercial'
                else:
                    # Default category for all other entity types
                    node_data['category'] = 'general'
                
                # Create node in Neo4j
                await self._create_enhanced_node(node_data)
                nodes_created += 1
            
            # Create relationships
            for relationship in relationships:
                edge_data = {
                    'source_name': relationship.get('source', ''),
                    'target_name': relationship.get('target', ''),
                    'relationship_type': relationship.get('type', 'RELATED_TO'),
                    'confidence': relationship.get('confidence', 0.5),
                    'evidence': relationship.get('evidence', '')[:500],
                    'source_document': source,
                    'created_at': 'datetime()'
                }
                
                # Create relationship in Neo4j
                await self._create_enhanced_relationship(edge_data)
                edges_created += 1
            
            return {
                'graph_nodes_created': nodes_created,
                'graph_edges_created': edges_created
            }
            
        except Exception as e:
            logger.error("Failed to add to knowledge graph", error=str(e))
            return {
                'graph_nodes_created': 0,
                'graph_edges_created': 0,
                'graph_error': str(e)
            }
    
    async def _create_enhanced_node(self, node_data: Dict[str, Any]):
        """Create enhanced node in Neo4j"""
        
        # Determine node label based on type
        node_type = node_data.get('type', 'unknown').lower()
        
        label_mapping = {
            'panel': 'Panel',
            'control_panel': 'Panel',
            'device': 'Device',
            'detector': 'Device',
            'sounder': 'Device',
            'module': 'Module',
            'cable': 'Cable',
            'standard': 'Standard',
            'specification': 'Specification',
            'manufacturer': 'Manufacturer',
            'zone': 'Zone'
        }
        
        label = label_mapping.get(node_type, 'Entity')
        
        query = f"""
        MERGE (n:{label} {{name: $name}})
        ON CREATE SET 
            n.type = $type,
            n.confidence = $confidence,
            n.source = $source,
            n.context = $context,
            n.category = $category,
            n.created_at = datetime()
        ON MATCH SET 
            n.confidence = CASE 
                WHEN $confidence > n.confidence THEN $confidence 
                ELSE n.confidence 
            END,
            n.sources = CASE 
                WHEN n.sources IS NULL THEN [$source]
                WHEN NOT $source IN n.sources THEN n.sources + [$source]
                ELSE n.sources
            END
        RETURN n.name as name
        """
        
        with self.neo4j_client.driver.session() as session:
            session.run(query, **node_data)
    
    async def _create_enhanced_relationship(self, edge_data: Dict[str, Any]):
        """Create enhanced relationship in Neo4j"""
        
        query = """
        MATCH (source) WHERE toLower(source.name) = toLower($source_name)
        MATCH (target) WHERE toLower(target.name) = toLower($target_name)
        WITH source, target, $relationship_type as rel_type
        CALL apoc.create.relationship(source, rel_type, {
            confidence: $confidence,
            evidence: $evidence,
            source_document: $source_document,
            created_at: datetime()
        }, target) YIELD rel
        RETURN rel
        """
        
        # Fallback if APOC is not available
        fallback_query = f"""
        MATCH (source) WHERE toLower(source.name) = toLower($source_name)
        MATCH (target) WHERE toLower(target.name) = toLower($target_name)
        MERGE (source)-[r:ENHANCED_RELATIONSHIP {{type: $relationship_type}}]->(target)
        ON CREATE SET 
            r.confidence = $confidence,
            r.evidence = $evidence,
            r.source_document = $source_document,
            r.created_at = datetime()
        ON MATCH SET 
            r.confidence = CASE 
                WHEN $confidence > r.confidence THEN $confidence 
                ELSE r.confidence 
            END
        RETURN r
        """
        
        with self.neo4j_client.driver.session() as session:
            try:
                session.run(query, **edge_data)
            except:
                # Use fallback if APOC is not available
                session.run(fallback_query, **edge_data)
    
    async def _add_to_vector_store(self, content: str, source: str, 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add document content to vector store"""
        
        try:
            chunks_added = await self.vector_store.add_document_chunks(
                content=content,
                source=source,
                metadata=metadata,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            return {'vector_chunks_added': chunks_added}
            
        except Exception as e:
            logger.error("Failed to add to vector store", error=str(e))
            return {'vector_chunks_added': 0, 'vector_error': str(e)}
    
    async def query_with_rag(self, user_query: str, k_vector: int = 5, 
                           k_graph: int = 10) -> Dict[str, Any]:
        """Process query using complete BYOKG-RAG pipeline"""
        
        logger.info("Processing RAG query", query=user_query)
        
        try:
            # Step 1: Generate knowledge graph paths using GPT-4.1-mini
            logger.info("Step 1: Generating graph paths")
            paths = await self._generate_graph_paths(user_query)
            
            # Step 2: Execute graph queries
            logger.info("Step 2: Executing graph queries")
            graph_results = await self._execute_graph_queries(paths, user_query)
            
            # Step 3: Vector similarity search
            logger.info("Step 3: Vector similarity search")
            
            # Ensure vector store is loaded
            if not self.vector_store.documents:
                logger.info("Vector store empty, attempting to load from disk")
                await self.vector_store.load_from_disk()
            
            vector_results = await self.vector_store.search(user_query, k=k_vector)
            
            # Step 4: Combine contexts
            logger.info("Step 4: Combining contexts")
            combined_context = {
                'graph_context': graph_results,
                'vector_results': vector_results,
                'query': user_query,
                'schema': self.schema
            }
            
            # Step 5: Generate answer using GPT-4o
            logger.info("Step 5: Generating final answer with GPT-4o")
            final_answer = await self._generate_final_answer(user_query, combined_context)
            
            result = {
                'query': user_query,
                'answer': final_answer,
                'graph_paths': paths,
                'graph_results_count': len(graph_results),
                'vector_results_count': len(vector_results),
                'sources': self._extract_sources(graph_results, vector_results)
            }
            
            logger.info("RAG query completed successfully", 
                       answer_length=len(final_answer),
                       sources_count=len(result['sources']))
            
            return result
            
        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            return {
                'query': user_query,
                'answer': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'error': str(e)
            }
    
    async def _generate_graph_paths(self, query: str) -> List[Dict[str, Any]]:
        """Generate knowledge graph paths using GPT-4.1-mini"""
        
        prompt = self.prompts.path_generation['system_prompt']
        user_prompt = self.prompts.path_generation['user_prompt'].format(
            query=query,
            schema=json.dumps(self.schema, indent=2)
        )
        
        try:
            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.ingestion_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse paths from response
            paths = self._parse_paths_response(content)
            
            logger.info("Graph paths generated", paths_count=len(paths))
            return paths
            
        except Exception as e:
            logger.error("Failed to generate graph paths", error=str(e))
            return []
    
    def _parse_paths_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse paths from LLM response"""
        
        paths = []
        
        try:
            if '<paths>' in content and '</paths>' in content:
                paths_text = content.split('<paths>')[1].split('</paths>')[0]
                paths_data = json.loads(paths_text.strip())
                
                if isinstance(paths_data, list):
                    paths = paths_data
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse paths JSON", error=str(e))
            # Fallback: extract simple paths from text
            paths = self._extract_paths_from_text(content)
        
        return paths
    
    def _extract_paths_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Fallback path extraction from text"""
        
        paths = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if '->' in line and ('Panel' in line or 'Device' in line or 'Standard' in line):
                paths.append({
                    'path': line,
                    'description': f'Extracted path: {line}',
                    'priority': 1
                })
        
        return paths[:5]  # Limit to 5 paths
    
    async def _execute_graph_queries(self, paths: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """Execute graph queries based on generated paths"""
        
        results = []
        
        for path_info in paths[:3]:  # Limit to top 3 paths
            try:
                # Convert path to Cypher query
                cypher_query = self._path_to_cypher(path_info, query)
                
                if cypher_query:
                    # Execute query with parameter
                    with self.neo4j_client.driver.session() as session:
                        result = session.run(cypher_query, query_term=query)
                        records = [record.data() for record in result]
                        
                        if records:
                            results.append({
                                'path': path_info.get('path', ''),
                                'description': path_info.get('description', ''),
                                'results': records[:10],  # Limit results
                                'count': len(records)
                            })
            
            except Exception as e:
                logger.warning("Graph query execution failed", 
                             path=path_info.get('path', ''),
                             error=str(e))
                continue
        
        return results
    
    def _path_to_cypher(self, path_info: Dict[str, Any], query: str) -> Optional[str]:
        """Convert path description to Cypher query"""
        
        path = path_info.get('path', '')
        
        # Simple path to Cypher conversion
        # This is a basic implementation - in production, use more sophisticated parsing
        
        if 'Panel' in path and 'COMPATIBLE_WITH' in path and 'Device' in path:
            return """
            MATCH (p:Panel)-[r:COMPATIBLE_WITH|:ENHANCED_RELATIONSHIP {type: 'COMPATIBLE_WITH'}]->(d:Device)
            WHERE toLower(p.name) CONTAINS toLower($query_term) OR toLower(d.name) CONTAINS toLower($query_term)
            RETURN p.name as panel, d.name as device, d.type, r.confidence
            ORDER BY r.confidence DESC LIMIT 10
            """
        elif 'Standard' in path and 'COMPLIES_WITH' in path:
            return """
            MATCH (n)-[r:COMPLIES_WITH|:ENHANCED_RELATIONSHIP {type: 'COMPLIES_WITH'}]->(s:Standard)
            WHERE toLower(n.name) CONTAINS toLower($query_term) OR toLower(s.name) CONTAINS toLower($query_term)
            RETURN n.name as entity, s.name as standard, n.type, r.confidence
            ORDER BY r.confidence DESC LIMIT 10
            """
        elif 'REQUIRES' in path:
            return """
            MATCH (n1)-[r:REQUIRES|:ENHANCED_RELATIONSHIP {type: 'REQUIRES'}]->(n2)
            WHERE toLower(n1.name) CONTAINS toLower($query_term) OR toLower(n2.name) CONTAINS toLower($query_term)
            RETURN n1.name as source, n2.name as target, n1.type, n2.type, r.confidence
            ORDER BY r.confidence DESC LIMIT 10
            """
        
        # Generic fallback query
        return """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($query_term)
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n.name, n.type, collect(DISTINCT related.name)[0..5] as related_entities
        LIMIT 10
        """
    
    async def _generate_final_answer(self, query: str, context: Dict[str, Any]) -> str:
        """Generate final answer using GPT-4o"""
        
        prompt = self.prompts.answer_generation['system_prompt']
        
        # Format context for the prompt
        formatted_context = self._format_context_for_answer(context)
        
        user_prompt = self.prompts.answer_generation['user_prompt'].format(
            query=query,
            graph_context=json.dumps(context.get('graph_context', []), indent=2),
            document_excerpts="",  # Could be added from vector results
            vector_results=json.dumps(context.get('vector_results', []), indent=2)
        )
        
        try:
            response = await self.openai_client.client.chat.completions.create(
                model=self.openai_client.generation_model,  # GPT-4o for generation
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error("Failed to generate final answer", error=str(e))
            return f"I apologize, but I encountered an error generating the response: {str(e)}"
    
    def _format_context_for_answer(self, context: Dict[str, Any]) -> str:
        """Format context for answer generation"""
        
        parts = []
        
        # Add graph results
        graph_results = context.get('graph_context', [])
        if graph_results:
            parts.append("=== KNOWLEDGE GRAPH RESULTS ===")
            for result in graph_results:
                parts.append(f"Path: {result.get('path', '')}")
                parts.append(f"Results: {result.get('count', 0)} items")
                for item in result.get('results', [])[:3]:
                    parts.append(f"  - {json.dumps(item)}")
                parts.append("")
        
        # Add vector results  
        vector_results = context.get('vector_results', [])
        if vector_results:
            parts.append("=== SIMILAR DOCUMENTS ===")
            for result in vector_results[:3]:
                parts.append(f"Source: {result.get('source', '')}")
                parts.append(f"Score: {result.get('score', 0):.3f}")
                parts.append(f"Content: {result.get('content', '')[:200]}...")
                parts.append("")
        
        return '\n'.join(parts)
    
    def _extract_sources(self, graph_results: List[Dict], vector_results: List[Dict]) -> List[str]:
        """Extract unique sources from results"""
        
        sources = set()
        
        # Extract from graph results
        for result in graph_results:
            for item in result.get('results', []):
                if 'source' in item:
                    sources.add(item['source'])
        
        # Extract from vector results
        for result in vector_results:
            sources.add(result.get('source', 'unknown'))
        
        return list(sources)
    
    async def _byokg_iterative_refinement(self, entities: List[Dict], relationships: List[Dict], 
                                        source: str) -> Dict[str, Any]:
        """
        Implement BYOKG-RAG 5-phase iterative refinement process:
        Phase 1: Candidate Extraction (already done)
        Phase 2: Refinement - Entity deduplication and normalization
        Phase 3: Clustering - Domain-specific grouping
        Phase 4: Validation - Quality assessment and filtering
        Phase 5: Persistence - Enhanced storage with confidence metrics
        """
        
        logger.info("Starting BYOKG-RAG iterative refinement", 
                   entities=len(entities), relationships=len(relationships))
        
        refinement_stats = {
            'entities_before_refinement': len(entities),
            'relationships_before_refinement': len(relationships),
            'entities_merged': 0,
            'relationships_enhanced': 0,
            'quality_score': 0.0
        }
        
        try:
            # Phase 2: Entity Refinement and Deduplication
            logger.info("Phase 2: Entity refinement and deduplication")
            refined_entities = await self._refine_entities(entities, source)
            refinement_stats['entities_merged'] = len(entities) - len(refined_entities)
            
            # Phase 3: Domain-Specific Clustering
            logger.info("Phase 3: Domain-specific clustering")
            clustered_entities = await self._cluster_fire_alarm_entities(refined_entities)
            
            # Phase 4: Quality Validation and Enhancement
            logger.info("Phase 4: Quality validation and enhancement")
            validated_entities, quality_score = await self._validate_and_enhance_quality(
                clustered_entities, relationships, source
            )
            refinement_stats['quality_score'] = quality_score
            
            # Phase 5: Enhanced Relationship Inference
            logger.info("Phase 5: Enhanced relationship inference")
            enhanced_relationships = await self._infer_enhanced_relationships(
                validated_entities, relationships
            )
            refinement_stats['relationships_enhanced'] = len(enhanced_relationships) - len(relationships)
            
            # Update final counts
            refinement_stats['entities_after_refinement'] = len(validated_entities)
            refinement_stats['relationships_after_refinement'] = len(enhanced_relationships)
            
            logger.info("BYOKG-RAG iterative refinement completed", **refinement_stats)
            return refinement_stats
            
        except Exception as e:
            logger.error("BYOKG-RAG refinement failed", error=str(e))
            refinement_stats['refinement_error'] = str(e)
            return refinement_stats
    
    async def _refine_entities(self, entities: List[Dict], source: str) -> List[Dict]:
        """Phase 2: Entity refinement with deduplication and normalization"""
        
        refined_entities = []
        entity_groups = {}
        
        for entity in entities:
            name = entity.get('name', '').strip().lower()
            entity_type = entity.get('type', '').lower()
            
            # Normalize fire alarm model numbers
            normalized_name = self._normalize_fire_alarm_entity(name, entity_type)
            
            # Group similar entities
            group_key = f"{normalized_name}_{entity_type}"
            if group_key not in entity_groups:
                entity_groups[group_key] = []
            entity_groups[group_key].append(entity)
        
        # Merge entities in each group
        for group_key, group_entities in entity_groups.items():
            if len(group_entities) == 1:
                refined_entities.append(group_entities[0])
            else:
                # Merge multiple entities into one with higher confidence
                merged_entity = self._merge_similar_entities(group_entities)
                refined_entities.append(merged_entity)
        
        return refined_entities
    
    def _normalize_fire_alarm_entity(self, name: str, entity_type: str) -> str:
        """Normalize fire alarm specific entity names"""
        
        # Normalize model numbers (e.g., "4098-9714" stays as is, "4098 9714" becomes "4098-9714")
        if entity_type in ['device', 'detector', 'panel', 'module']:
            # Standardize model number format
            import re
            model_pattern = r'(\d{4})\s*[-\s]*(\d{4})'
            match = re.search(model_pattern, name)
            if match:
                return f"{match.group(1)}-{match.group(2)}"
        
        # Normalize manufacturer names
        manufacturer_aliases = {
            'simplex': 'tyco safety products',
            'tyco': 'tyco safety products',
            'johnson controls': 'tyco safety products'
        }
        
        for alias, canonical in manufacturer_aliases.items():
            if alias in name.lower():
                return canonical
        
        return name.lower().strip()
    
    def _merge_similar_entities(self, entities: List[Dict]) -> Dict:
        """Merge similar entities with confidence aggregation"""
        
        # Start with the highest confidence entity
        base_entity = max(entities, key=lambda e: e.get('confidence', 0))
        
        # Aggregate contexts and increase confidence
        contexts = []
        total_confidence = 0
        
        for entity in entities:
            if entity.get('context'):
                contexts.append(entity['context'])
            total_confidence += entity.get('confidence', 0.5)
        
        # Create merged entity
        merged_entity = base_entity.copy()
        merged_entity['context'] = ' | '.join(contexts)
        merged_entity['confidence'] = min(0.95, total_confidence / len(entities) + 0.1)  # Cap at 0.95
        merged_entity['merged_from'] = len(entities)
        
        return merged_entity
    
    async def _cluster_fire_alarm_entities(self, entities: List[Dict]) -> List[Dict]:
        """Phase 3: Fire alarm domain-specific clustering"""
        
        # Group entities by fire alarm categories
        clusters = {
            'control_panels': [],
            'detection_devices': [],
            'notification_devices': [],
            'modules_interfaces': [],
            'wiring_components': [],
            'standards_compliance': [],
            'manufacturers': []
        }
        
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            name = entity.get('name', '').lower()
            
            # Classify into fire alarm categories
            if entity_type in ['panel', 'control_panel'] or 'panel' in name:
                clusters['control_panels'].append(entity)
            elif entity_type in ['detector', 'sensor'] or any(term in name for term in ['detector', 'sensor', 'smoke', 'heat']):
                clusters['detection_devices'].append(entity)
            elif entity_type in ['sounder', 'beacon', 'horn'] or any(term in name for term in ['sounder', 'alarm', 'beacon']):
                clusters['notification_devices'].append(entity)
            elif entity_type in ['module', 'interface'] or 'module' in name:
                clusters['modules_interfaces'].append(entity)
            elif entity_type in ['cable', 'wire'] or any(term in name for term in ['cable', 'wire']):
                clusters['wiring_components'].append(entity)
            elif entity_type in ['standard', 'specification'] or any(term in name for term in ['bs', 'en', 'nfpa']):
                clusters['standards_compliance'].append(entity)
            elif entity_type in ['manufacturer', 'brand']:
                clusters['manufacturers'].append(entity)
        
        # Add cluster information to entities
        clustered_entities = []
        for cluster_name, cluster_entities in clusters.items():
            for entity in cluster_entities:
                entity['fire_alarm_cluster'] = cluster_name
                clustered_entities.append(entity)
        
        return clustered_entities
    
    async def _validate_and_enhance_quality(self, entities: List[Dict], relationships: List[Dict], 
                                          source: str) -> Tuple[List[Dict], float]:
        """Phase 4: Quality validation and enhancement"""
        
        validated_entities = []
        quality_scores = []
        
        for entity in entities:
            # Calculate quality score based on multiple factors
            name_quality = self._assess_name_quality(entity.get('name', ''))
            context_quality = self._assess_context_quality(entity.get('context', ''))
            type_quality = self._assess_type_quality(entity.get('type', ''))
            
            overall_quality = (name_quality + context_quality + type_quality) / 3
            quality_scores.append(overall_quality)
            
            # Only keep entities with quality score > 0.3
            if overall_quality > 0.3:
                entity['quality_score'] = overall_quality
                validated_entities.append(entity)
        
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return validated_entities, average_quality
    
    def _assess_name_quality(self, name: str) -> float:
        """Assess entity name quality"""
        if not name or len(name.strip()) < 2:
            return 0.0
        
        # Higher quality for structured names (model numbers, proper nouns)
        import re
        if re.match(r'\d{4}-\d{4}', name):  # Model number pattern
            return 0.9
        elif re.match(r'[A-Z][a-z]+', name):  # Proper noun
            return 0.8
        elif len(name.split()) > 1:  # Multi-word names
            return 0.7
        else:
            return 0.5
    
    def _assess_context_quality(self, context: str) -> float:
        """Assess context quality"""
        if not context:
            return 0.3
        
        # Higher quality for longer, more descriptive contexts
        word_count = len(context.split())
        if word_count > 20:
            return 0.9
        elif word_count > 10:
            return 0.7
        elif word_count > 5:
            return 0.5
        else:
            return 0.3
    
    def _assess_type_quality(self, entity_type: str) -> float:
        """Assess entity type quality"""
        if not entity_type or entity_type == 'unknown':
            return 0.2
        
        # Higher quality for specific fire alarm types
        specific_types = ['panel', 'detector', 'sounder', 'module', 'standard', 'manufacturer']
        if entity_type.lower() in specific_types:
            return 0.9
        else:
            return 0.6
    
    async def _infer_enhanced_relationships(self, entities: List[Dict], 
                                         relationships: List[Dict]) -> List[Dict]:
        """Phase 5: Enhanced relationship inference"""
        
        enhanced_relationships = relationships.copy()
        
        # Infer compatibility relationships for fire alarm components
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                
                # Infer detector-base relationships
                if self._could_be_compatible(entity1, entity2):
                    inferred_rel = {
                        'source': entity1.get('name', ''),
                        'target': entity2.get('name', ''),
                        'type': 'COMPATIBLE_WITH',
                        'confidence': 0.7,
                        'evidence': 'Inferred from fire alarm compatibility rules',
                        'inferred': True
                    }
                    enhanced_relationships.append(inferred_rel)
        
        return enhanced_relationships
    
    def _could_be_compatible(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities could be compatible based on fire alarm rules"""
        
        name1 = entity1.get('name', '').lower()
        name2 = entity2.get('name', '').lower()
        type1 = entity1.get('type', '').lower()
        type2 = entity2.get('type', '').lower()
        
        # Detector and base compatibility rules
        if ('detector' in type1 and 'base' in type2) or ('base' in type1 and 'detector' in type2):
            # Same manufacturer series (e.g., 4098-xxxx with 4098-yyyy)
            import re
            series1 = re.match(r'(\d{4})-', name1)
            series2 = re.match(r'(\d{4})-', name2)
            
            if series1 and series2 and series1.group(1) == series2.group(1):
                return True
        
        return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        try:
            # Neo4j stats
            neo4j_stats = self.neo4j_client.get_document_stats()
            
            # Vector store stats
            vector_stats = self.vector_store.get_stats()
            
            # Combined stats
            return {
                'knowledge_graph': neo4j_stats,
                'vector_store': vector_stats,
                'schema': self.schema,
                'prompts_loaded': {
                    'entity_extraction': bool(self.prompts.entity_extraction),
                    'path_generation': bool(self.prompts.path_generation),
                    'query_generation': bool(self.prompts.query_generation),
                    'answer_generation': bool(self.prompts.answer_generation)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system stats", error=str(e))
            return {'error': str(e)}