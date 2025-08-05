from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import json
from .config import settings
from .logging import logger

class Neo4jClient:
    """Neo4j client for knowledge graph operations"""
    
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.driver = None
        self.connect()
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] == 1:
                    logger.info("Connected to Neo4j", uri=self.uri)
                else:
                    raise Exception("Connection test failed")
                    
        except Exception as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        indexes = [
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATED_TO]->() ON (r.type)"
        ]
        
        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info("Created index", query=index_query)
                except Exception as e:
                    logger.warning("Index creation failed or already exists", 
                                 query=index_query, error=str(e))
    
    def create_document_node(self, doc_data: Dict[str, Any]) -> str:
        """Create a document node"""
        query = """
        CREATE (d:Document {
            id: $id,
            filename: $filename,
            filepath: $filepath,
            s3_key: $s3_key,
            content_type: $content_type,
            size: $size,
            created_at: datetime(),
            title: $title,
            content_preview: $content_preview,
            page_count: $page_count,
            metadata: $metadata
        })
        RETURN d.id as doc_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, **doc_data)
            doc_id = result.single()["doc_id"]
            logger.info("Created document node", doc_id=doc_id, filename=doc_data.get('filename'))
            return doc_id
    
    def create_entity_node(self, entity_data: Dict[str, Any]) -> str:
        """Create an entity node"""
        query = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET 
            e.id = randomUUID(),
            e.created_at = datetime(),
            e.confidence = $confidence,
            e.description = $description,
            e.mentions = 1
        ON MATCH SET 
            e.mentions = e.mentions + 1,
            e.confidence = CASE 
                WHEN $confidence > e.confidence THEN $confidence 
                ELSE e.confidence 
            END
        RETURN e.id as entity_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, **entity_data)
            entity_id = result.single()["entity_id"]
            return entity_id
    
    def create_relationship(self, source_id: str, target_id: str, 
                          rel_type: str, properties: Dict[str, Any] = None):
        """Create a relationship between two nodes"""
        if properties is None:
            properties = {}
        
        query = """
        MATCH (s) WHERE s.id = $source_id
        MATCH (t) WHERE t.id = $target_id
        MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
        ON CREATE SET 
            r.created_at = datetime(),
            r.confidence = $confidence,
            r.context = $context,
            r.mentions = 1
        ON MATCH SET 
            r.mentions = r.mentions + 1,
            r.confidence = CASE 
                WHEN $confidence > r.confidence THEN $confidence 
                ELSE r.confidence 
            END
        RETURN r
        """
        
        with self.driver.session() as session:
            session.run(query, 
                       source_id=source_id, 
                       target_id=target_id, 
                       rel_type=rel_type,
                       **properties)
    
    def link_document_to_entities(self, doc_id: str, entity_ids: List[str]):
        """Link document to extracted entities"""
        query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (e:Entity) WHERE e.id IN $entity_ids
        MERGE (d)-[r:CONTAINS]->(e)
        ON CREATE SET r.created_at = datetime()
        """
        
        with self.driver.session() as session:
            session.run(query, doc_id=doc_id, entity_ids=entity_ids)
            logger.info("Linked document to entities", 
                       doc_id=doc_id, entity_count=len(entity_ids))
    
    def get_document_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        queries = {
            'documents': "MATCH (d:Document) RETURN count(d) as count",
            'entities': "MATCH (e:Entity) RETURN count(e) as count",
            'relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        stats = {}
        with self.driver.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                stats[key] = result.single()["count"]
        
        return stats
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name"""
        search_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        RETURN e.id as id, e.name as name, e.type as type, 
               e.confidence as confidence, e.mentions as mentions
        ORDER BY e.mentions DESC, e.confidence DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(search_query, query=query, limit=limit)
            return [record.data() for record in result]
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity"""
        query = """
        MATCH (e:Entity {id: $entity_id})-[r]-(other)
        RETURN other.id as related_id, other.name as related_name, 
               other.type as related_type, type(r) as relationship_type,
               r.type as relationship_subtype, r.confidence as confidence
        ORDER BY r.confidence DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_id=entity_id)
            return [record.data() for record in result]