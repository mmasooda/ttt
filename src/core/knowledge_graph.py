from typing import List, Dict, Any, Optional
from ..utils import Neo4jClient, logger
from ..ingestion import ProcessedDocument

class KnowledgeGraphBuilder:
    """Build and manage knowledge graph from processed documents"""
    
    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.setup_database()
    
    def setup_database(self):
        """Initialize database with indexes"""
        try:
            self.neo4j_client.create_indexes()
            logger.info("Knowledge graph database initialized")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    def ingest_document(self, doc: ProcessedDocument) -> bool:
        """Ingest a processed document into the knowledge graph"""
        try:
            logger.info("Ingesting document into knowledge graph", doc_id=doc.id)
            
            # Create document node
            doc_data = {
                'id': doc.id,
                'filename': doc.filename,
                'filepath': doc.filepath,
                's3_key': doc.s3_key,
                'content_type': doc.content_type,
                'size': doc.size,
                'title': doc.title,
                'content_preview': doc.content_preview,
                'page_count': doc.page_count,
                'metadata': doc.metadata
            }
            
            doc_id = self.neo4j_client.create_document_node(doc_data)
            
            # Create entity nodes and link to document
            entity_ids = []
            for entity in doc.entities:
                entity_data = {
                    'name': entity['name'],
                    'type': entity['type'],
                    'confidence': entity['confidence'],
                    'description': entity.get('description', '')
                }
                entity_id = self.neo4j_client.create_entity_node(entity_data)
                entity_ids.append(entity_id)
            
            # Link document to entities
            if entity_ids:
                self.neo4j_client.link_document_to_entities(doc_id, entity_ids)
            
            # Create relationships between entities
            for rel in doc.relationships:
                source_entity = next((e for e in doc.entities if e['name'] == rel['source']), None)
                target_entity = next((e for e in doc.entities if e['name'] == rel['target']), None)
                
                if source_entity and target_entity:
                    # Find entity IDs
                    source_results = self.neo4j_client.search_entities(source_entity['name'], limit=1)
                    target_results = self.neo4j_client.search_entities(target_entity['name'], limit=1)
                    
                    if source_results and target_results:
                        self.neo4j_client.create_relationship(
                            source_results[0]['id'],
                            target_results[0]['id'],
                            rel['type'],
                            {
                                'confidence': rel['confidence'],
                                'context': rel.get('context', '')
                            }
                        )
            
            logger.info("Document ingested successfully", 
                       doc_id=doc_id, 
                       entities=len(entity_ids),
                       relationships=len(doc.relationships))
            
            return True
            
        except Exception as e:
            logger.error("Failed to ingest document", doc_id=doc.id, error=str(e))
            return False
    
    def ingest_documents_batch(self, documents: List[ProcessedDocument]) -> Dict[str, Any]:
        """Ingest multiple documents"""
        results = {
            'total': len(documents),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for doc in documents:
            try:
                if self.ingest_document(doc):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'doc_id': doc.id,
                    'error': str(e)
                })
        
        logger.info("Batch ingestion completed", **results)
        return results
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return self.neo4j_client.get_document_stats()
    
    def search_knowledge(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search the knowledge graph"""
        entities = self.neo4j_client.search_entities(query, limit)
        
        # Get relationships for top entities
        results = []
        for entity in entities[:5]:  # Top 5 entities
            relationships = self.neo4j_client.get_entity_relationships(entity['id'])
            results.append({
                'entity': entity,
                'relationships': relationships
            })
        
        return {
            'query': query,
            'entities': entities,
            'detailed_results': results
        }
    
    def clear_all_data(self):
        """Clear all data from knowledge graph"""
        self.neo4j_client.clear_database()
        logger.info("Knowledge graph cleared")
    
    def close(self):
        """Close knowledge graph connections"""
        self.neo4j_client.close()