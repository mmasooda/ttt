#!/usr/bin/env python3
"""
Enhanced Graph Builder with Iterative Node/Edge/Property Extraction
Follows BYOKG-RAG structure for perfect data-graph and sub-graph construction
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils import Neo4jClient, logger, settings
from ..ingestion.enhanced_document_processor import EnhancedDocument, ExtractedEntity, ExtractedTable

# Load models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class GraphNode:
    """Enhanced graph node representation"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float
    source_documents: List[str]
    embeddings: List[float]
    aliases: Set[str]
    category: str
    sub_category: Optional[str] = None
    domain_specific_attrs: Dict[str, Any] = None

@dataclass 
class GraphEdge:
    """Enhanced graph edge representation"""
    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float
    evidence: List[str]
    source_documents: List[str]
    weight: float
    directional: bool = True

@dataclass
class GraphCluster:
    """Sub-graph cluster for domain-specific grouping"""
    id: str
    name: str
    cluster_type: str
    nodes: List[str]
    internal_edges: List[str]
    external_edges: List[str]
    properties: Dict[str, Any]
    confidence: float

class EnhancedGraphBuilder:
    """Enhanced graph builder with iterative extraction following BYOKG-RAG structure"""
    
    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Domain knowledge for fire alarm systems
        self.domain_taxonomy = self._load_domain_taxonomy()
        self.relation_patterns = self._load_relation_patterns()
        
        # Graph structures for iterative building
        self.nodes_registry: Dict[str, GraphNode] = {}
        self.edges_registry: Dict[str, GraphEdge] = {}
        self.clusters_registry: Dict[str, GraphCluster] = {}
        
        # Temporary storage for iterative processing
        self.candidate_nodes: List[GraphNode] = []
        self.candidate_edges: List[GraphEdge] = []
        
        logger.info("Enhanced Graph Builder initialized")
    
    def _load_domain_taxonomy(self) -> Dict[str, Any]:
        """Load fire alarm system domain taxonomy"""
        return {
            'devices': {
                'detection': ['detector', 'sensor', 'smoke detector', 'heat detector', 'flame detector'],
                'notification': ['sounder', 'beacon', 'horn', 'strobe', 'speaker'],
                'control': ['panel', 'repeater', 'interface', 'controller'],
                'initiation': ['call point', 'break glass', 'manual call point', 'push button']
            },
            'components': {
                'wiring': ['cable', 'wire', 'conductor', 'armoured cable'],
                'power': ['battery', 'charger', 'psu', 'power supply'],
                'mounting': ['bracket', 'back box', 'mounting plate', 'enclosure']
            },
            'standards': {
                'british': ['BS 5839', 'BS EN 54', 'BS 7671'],
                'european': ['EN 54', 'EN 12094'],
                'international': ['ISO 7240', 'IEC 60849'],
                'american': ['NFPA 72', 'UL 268']
            },
            'specifications': {
                'electrical': ['voltage', 'current', 'power', 'impedance'],
                'environmental': ['temperature', 'humidity', 'ip rating'],
                'performance': ['sensitivity', 'response time', 'coverage']
            }
        }
    
    def _load_relation_patterns(self) -> Dict[str, List[str]]:
        """Load relationship extraction patterns"""
        return {
            'part_of': [
                r'(\w+)\s+(?:is\s+)?(?:part\s+of|component\s+of|belongs\s+to)\s+(\w+)',
                r'(\w+)\s+(?:comprises?|contains?|includes?)\s+(\w+)',
                r'(\w+)\s+with\s+(\w+)'
            ],
            'connects_to': [
                r'(\w+)\s+(?:connects?\s+to|wired\s+to|linked\s+to)\s+(\w+)',
                r'(\w+)\s+(?:and|with)\s+(\w+)\s+(?:are\s+)?connected',
                r'connect\s+(\w+)\s+to\s+(\w+)'
            ],
            'compatible_with': [
                r'(\w+)\s+(?:compatible\s+with|works\s+with|suitable\s+for)\s+(\w+)',
                r'(\w+)\s+(?:and|with)\s+(\w+)\s+compatibility'
            ],
            'requires': [
                r'(\w+)\s+(?:requires?|needs?)\s+(\w+)',
                r'(\w+)\s+(?:must\s+have|should\s+have)\s+(\w+)'
            ],
            'specifies': [
                r'(\w+)\s+(?:specifies?|defines?|states?)\s+(\w+)',
                r'(?:according\s+to|as\s+per)\s+(\w+),?\s+(\w+)'
            ]
        }
    
    async def build_enhanced_graph(self, documents: List[EnhancedDocument]) -> Dict[str, Any]:
        """Main method to build enhanced graph with iterative extraction"""
        logger.info("Starting enhanced graph construction", documents=len(documents))
        
        # Phase 1: Extract candidate nodes and edges
        await self._extract_candidates(documents)
        
        # Phase 2: Iterative refinement
        await self._iterative_refinement()
        
        # Phase 3: Cluster formation (sub-graphs)
        await self._form_clusters()
        
        # Phase 4: Graph validation and optimization
        await self._validate_and_optimize()
        
        # Phase 5: Persist to Neo4j
        await self._persist_to_neo4j()
        
        # Return construction statistics
        return {
            'nodes_created': len(self.nodes_registry),
            'edges_created': len(self.edges_registry),
            'clusters_formed': len(self.clusters_registry),
            'documents_processed': len(documents),
            'quality_metrics': await self._calculate_quality_metrics()
        }
    
    async def _extract_candidates(self, documents: List[EnhancedDocument]):
        """Phase 1: Extract candidate nodes and edges from documents"""
        logger.info("Phase 1: Extracting candidate nodes and edges")
        
        for doc in documents:
            # Extract nodes from entities
            await self._extract_nodes_from_entities(doc)
            
            # Extract nodes from tables
            await self._extract_nodes_from_tables(doc)
            
            # Extract edges from relationships
            await self._extract_edges_from_relationships(doc)
            
            # Extract edges from co-occurrence patterns
            await self._extract_edges_from_cooccurrence(doc)
            
            # Extract edges from text patterns
            await self._extract_edges_from_patterns(doc)
        
        logger.info("Candidate extraction completed", 
                   nodes=len(self.candidate_nodes),
                   edges=len(self.candidate_edges))
    
    async def _extract_nodes_from_entities(self, doc: EnhancedDocument):
        """Extract graph nodes from document entities"""
        for entity in doc.entities:
            # Determine node category and properties
            category, sub_category = self._classify_entity(entity.name, entity.type)
            
            # Create enhanced properties
            properties = {
                'original_type': entity.type,
                'confidence_score': entity.confidence,
                'context_snippet': entity.context[:200] if entity.context else "",
                'page_number': entity.page_number,
                'extraction_method': 'entity_recognition'
            }
            
            # Add domain-specific attributes
            domain_attrs = self._extract_domain_attributes(entity.name, entity.context)
            
            # Create graph node
            node = GraphNode(
                id=f"entity_{len(self.candidate_nodes)}",
                name=entity.name,
                type=entity.type,
                properties=properties,
                confidence=entity.confidence,
                source_documents=[doc.filename],
                embeddings=entity.embedding if entity.embedding else [],
                aliases=self._extract_aliases(entity.name, entity.context),
                category=category,
                sub_category=sub_category,
                domain_specific_attrs=domain_attrs
            )
            
            self.candidate_nodes.append(node)
    
    async def _extract_nodes_from_tables(self, doc: EnhancedDocument):
        """Extract graph nodes from table data"""
        for table in doc.tables:
            if not table.data or not table.headers:
                continue
            
            # Extract product/component nodes from table rows
            for row_idx, row in enumerate(table.data):
                for col_idx, cell_value in enumerate(row):
                    if not cell_value or len(str(cell_value).strip()) < 2:
                        continue
                    
                    cell_str = str(cell_value).strip()
                    
                    # Check if cell contains product codes, part numbers, etc.
                    if self._is_significant_table_entity(cell_str, table.headers):
                        category, sub_category = self._classify_table_entity(
                            cell_str, table.headers[col_idx] if col_idx < len(table.headers) else ""
                        )
                        
                        properties = {
                            'table_context': f"Table {table.id}",
                            'column_header': table.headers[col_idx] if col_idx < len(table.headers) else "",
                            'row_number': row_idx,
                            'extraction_method': f'table_{table.extraction_method}',
                            'table_confidence': table.confidence_score
                        }
                        
                        node = GraphNode(
                            id=f"table_{table.id}_{row_idx}_{col_idx}",
                            name=cell_str,
                            type='table_entity',
                            properties=properties,
                            confidence=min(0.8, table.confidence_score),
                            source_documents=[doc.filename],
                            embeddings=embedding_model.encode(cell_str).tolist(),
                            aliases=set(),
                            category=category,
                            sub_category=sub_category
                        )
                        
                        self.candidate_nodes.append(node)
    
    async def _extract_edges_from_relationships(self, doc: EnhancedDocument):
        """Extract edges from existing entity relationships"""
        for entity in doc.entities:
            if not entity.relationships:
                continue
                
            for rel in entity.relationships:
                # Find source and target nodes
                source_candidates = [n for n in self.candidate_nodes 
                                   if n.name == entity.name and doc.filename in n.source_documents]
                target_candidates = [n for n in self.candidate_nodes 
                                   if n.name == rel['target_entity'] and doc.filename in n.source_documents]
                
                if source_candidates and target_candidates:
                    source_node = source_candidates[0]
                    target_node = target_candidates[0]
                    
                    edge = GraphEdge(
                        id=f"rel_{len(self.candidate_edges)}",
                        source_node_id=source_node.id,
                        target_node_id=target_node.id,
                        relationship_type=rel['relationship_type'],
                        properties={'original_confidence': rel['confidence']},
                        confidence=rel['confidence'],
                        evidence=[entity.context[:500] if entity.context else ""],
                        source_documents=[doc.filename],
                        weight=rel['confidence'],
                        directional=True
                    )
                    
                    self.candidate_edges.append(edge)
    
    async def _extract_edges_from_cooccurrence(self, doc: EnhancedDocument):
        """Extract edges based on entity co-occurrence in context"""
        # Group entities by context similarity
        entity_contexts = []
        for entity in doc.entities:
            if entity.context:
                entity_contexts.append((entity, entity.context))
        
        # Find co-occurring entities (within same context window)
        for i, (entity1, context1) in enumerate(entity_contexts):
            for j, (entity2, context2) in enumerate(entity_contexts[i+1:], i+1):
                
                # Calculate context overlap
                context1_words = set(context1.lower().split())
                context2_words = set(context2.lower().split())
                overlap = len(context1_words & context2_words)
                
                if overlap > 5:  # Significant context overlap
                    # Find corresponding nodes
                    source_candidates = [n for n in self.candidate_nodes 
                                       if n.name == entity1.name and doc.filename in n.source_documents]
                    target_candidates = [n for n in self.candidate_nodes 
                                       if n.name == entity2.name and doc.filename in n.source_documents]
                    
                    if source_candidates and target_candidates:
                        confidence = min(0.9, overlap / 20.0)
                        
                        edge = GraphEdge(
                            id=f"cooc_{len(self.candidate_edges)}",
                            source_node_id=source_candidates[0].id,
                            target_node_id=target_candidates[0].id,
                            relationship_type='co_occurs_with',
                            properties={'context_overlap': overlap},
                            confidence=confidence,
                            evidence=[f"Context overlap: {overlap} words"],
                            source_documents=[doc.filename],
                            weight=confidence,
                            directional=False
                        )
                        
                        self.candidate_edges.append(edge)
    
    async def _extract_edges_from_patterns(self, doc: EnhancedDocument):
        """Extract edges using regex patterns"""
        content = doc.content
        
        for rel_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1_name = match.group(1).strip()
                        entity2_name = match.group(2).strip()
                        
                        # Find corresponding nodes
                        source_candidates = [n for n in self.candidate_nodes 
                                           if entity1_name.lower() in n.name.lower() 
                                           and doc.filename in n.source_documents]
                        target_candidates = [n for n in self.candidate_nodes 
                                           if entity2_name.lower() in n.name.lower() 
                                           and doc.filename in n.source_documents]
                        
                        if source_candidates and target_candidates:
                            edge = GraphEdge(
                                id=f"pattern_{len(self.candidate_edges)}",
                                source_node_id=source_candidates[0].id,
                                target_node_id=target_candidates[0].id,
                                relationship_type=rel_type,
                                properties={'pattern_matched': pattern},
                                confidence=0.75,
                                evidence=[match.group(0)],
                                source_documents=[doc.filename],
                                weight=0.75,
                                directional=True
                            )
                            
                            self.candidate_edges.append(edge)
    
    async def _iterative_refinement(self):
        """Phase 2: Iterative refinement of nodes and edges"""
        logger.info("Phase 2: Starting iterative refinement")
        
        # Iteration 1: Merge similar nodes
        await self._merge_similar_nodes()
        
        # Iteration 2: Validate and filter edges
        await self._validate_edges()
        
        # Iteration 3: Enhance node properties
        await self._enhance_node_properties()
        
        # Iteration 4: Calculate edge weights
        await self._calculate_edge_weights()
        
        logger.info("Iterative refinement completed",
                   final_nodes=len(self.nodes_registry),
                   final_edges=len(self.edges_registry))
    
    async def _merge_similar_nodes(self):
        """Merge nodes that represent the same entity"""
        # Group nodes by similarity
        node_embeddings = []
        node_names = []
        
        for node in self.candidate_nodes:
            if node.embeddings:
                node_embeddings.append(node.embeddings)
                node_names.append(node.name)
        
        if not node_embeddings:
            # Fallback: merge by exact name match
            await self._merge_by_name()
            return
        
        # Use DBSCAN clustering to find similar nodes
        similarity_matrix = cosine_similarity(node_embeddings)
        clustering = DBSCAN(eps=0.15, min_samples=2, metric='precomputed')
        
        # Convert similarity to distance matrix
        distance_matrix = 1 - similarity_matrix
        clusters = clustering.fit_predict(distance_matrix)
        
        # Merge nodes in same cluster
        cluster_groups = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # Not noise
                cluster_groups[cluster_id].append(self.candidate_nodes[idx])
        
        # Create merged nodes
        for cluster_id, nodes_in_cluster in cluster_groups.items():
            merged_node = await self._merge_node_group(nodes_in_cluster)
            self.nodes_registry[merged_node.id] = merged_node
        
        # Add non-clustered nodes as-is
        for idx, cluster_id in enumerate(clusters):
            if cluster_id == -1:  # Noise/singleton
                node = self.candidate_nodes[idx]
                self.nodes_registry[node.id] = node
    
    async def _merge_node_group(self, nodes: List[GraphNode]) -> GraphNode:
        """Merge a group of similar nodes into one"""
        if len(nodes) == 1:
            return nodes[0]
        
        # Select primary node (highest confidence)
        primary_node = max(nodes, key=lambda n: n.confidence)
        
        # Merge properties
        merged_properties = primary_node.properties.copy()
        merged_source_docs = set(primary_node.source_documents)
        merged_aliases = primary_node.aliases.copy()
        
        for node in nodes:
            if node.id != primary_node.id:
                merged_properties.update(node.properties)
                merged_source_docs.update(node.source_documents)
                merged_aliases.update(node.aliases)
                merged_aliases.add(node.name)  # Add variant names as aliases
        
        # Calculate merged confidence (weighted average)
        total_confidence = sum(n.confidence for n in nodes)
        merged_confidence = total_confidence / len(nodes)
        
        # Create merged node
        merged_node = GraphNode(
            id=f"merged_{primary_node.id}",
            name=primary_node.name,
            type=primary_node.type,
            properties=merged_properties,
            confidence=merged_confidence,
            source_documents=list(merged_source_docs),
            embeddings=primary_node.embeddings,
            aliases=merged_aliases,
            category=primary_node.category,
            sub_category=primary_node.sub_category,
            domain_specific_attrs=primary_node.domain_specific_attrs
        )
        
        return merged_node
    
    async def _merge_by_name(self):
        """Fallback merge method using exact name matching"""
        name_groups = defaultdict(list)
        
        for node in self.candidate_nodes:
            name_groups[node.name.lower()].append(node)
        
        for name, nodes_with_name in name_groups.items():
            if len(nodes_with_name) > 1:
                merged_node = await self._merge_node_group(nodes_with_name)
                self.nodes_registry[merged_node.id] = merged_node
            else:
                node = nodes_with_name[0]
                self.nodes_registry[node.id] = node
    
    async def _validate_edges(self):
        """Validate and filter candidate edges"""
        valid_node_ids = set(self.nodes_registry.keys())
        
        for edge in self.candidate_edges:
            # Update node IDs if nodes were merged
            updated_source_id = self._find_updated_node_id(edge.source_node_id)
            updated_target_id = self._find_updated_node_id(edge.target_node_id)
            
            if updated_source_id in valid_node_ids and updated_target_id in valid_node_ids:
                # Avoid self-loops
                if updated_source_id != updated_target_id:
                    edge.source_node_id = updated_source_id
                    edge.target_node_id = updated_target_id
                    self.edges_registry[edge.id] = edge
    
    def _find_updated_node_id(self, original_id: str) -> str:
        """Find the updated node ID after merging"""
        # Check if original ID still exists
        if original_id in self.nodes_registry:
            return original_id
        
        # Look for merged node that might contain this original ID
        for node_id, node in self.nodes_registry.items():
            if original_id in node.id or node_id.startswith('merged_'):
                # This is a heuristic - in a production system, you'd maintain a mapping
                return node_id
        
        return original_id  # Fallback
    
    async def _enhance_node_properties(self):
        """Enhance node properties with additional information"""
        for node_id, node in self.nodes_registry.items():
            # Add centrality measures (will be calculated after graph construction)
            node.properties['degree_centrality'] = 0
            node.properties['betweenness_centrality'] = 0
            
            # Add domain classification confidence
            domain_confidence = self._calculate_domain_confidence(node)
            node.properties['domain_confidence'] = domain_confidence
            
            # Add entity importance score
            importance_score = self._calculate_importance_score(node)
            node.properties['importance_score'] = importance_score
    
    async def _calculate_edge_weights(self):
        """Calculate sophisticated edge weights"""
        for edge_id, edge in self.edges_registry.items():
            source_node = self.nodes_registry.get(edge.source_node_id)
            target_node = self.nodes_registry.get(edge.target_node_id)
            
            if source_node and target_node:
                # Base weight from confidence
                weight = edge.confidence
                
                # Boost weight for same-category connections
                if source_node.category == target_node.category:
                    weight *= 1.2
                
                # Boost weight for frequent co-occurrence
                if edge.relationship_type == 'co_occurs_with':
                    co_occurrence_count = len(set(source_node.source_documents) & 
                                            set(target_node.source_documents))
                    weight *= (1 + co_occurrence_count * 0.1)
                
                # Domain-specific relationship boosts
                if edge.relationship_type in ['part_of', 'requires', 'connects_to']:
                    weight *= 1.3
                
                edge.weight = min(1.0, weight)
    
    async def _form_clusters(self):
        """Phase 3: Form domain-specific sub-graph clusters"""
        logger.info("Phase 3: Forming sub-graph clusters")
        
        # Create NetworkX graph for cluster analysis
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes_registry.items():
            G.add_node(node_id, 
                      name=node.name, 
                      category=node.category,
                      type=node.type)
        
        # Add edges
        for edge_id, edge in self.edges_registry.items():
            G.add_edge(edge.source_node_id, edge.target_node_id, 
                      weight=edge.weight, 
                      type=edge.relationship_type)
        
        # Detect communities/clusters
        clusters_by_category = self._cluster_by_category(G)
        clusters_by_connectivity = self._cluster_by_connectivity(G)
        
        # Combine clustering approaches
        final_clusters = self._combine_clustering_results(
            clusters_by_category, clusters_by_connectivity
        )
        
        # Create cluster objects
        for cluster_id, node_ids in final_clusters.items():
            cluster = self._create_cluster(cluster_id, node_ids, G)
            self.clusters_registry[cluster.id] = cluster
        
        logger.info("Cluster formation completed", clusters=len(self.clusters_registry))
    
    def _cluster_by_category(self, G: nx.Graph) -> Dict[str, List[str]]:
        """Cluster nodes by domain category"""
        category_clusters = defaultdict(list)
        
        for node_id in G.nodes():
            node = self.nodes_registry[node_id]
            cluster_key = f"{node.category}_{node.sub_category}" if node.sub_category else node.category
            category_clusters[cluster_key].append(node_id)
        
        return dict(category_clusters)
    
    def _cluster_by_connectivity(self, G: nx.Graph) -> Dict[str, List[str]]:
        """Cluster nodes by connectivity patterns"""
        # Use community detection algorithm
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G)
            
            connectivity_clusters = {}
            for i, community in enumerate(communities):
                connectivity_clusters[f"connectivity_{i}"] = list(community)
            
            return connectivity_clusters
            
        except ImportError:
            # Fallback: simple connected components
            connected_components = nx.connected_components(G)
            connectivity_clusters = {}
            
            for i, component in enumerate(connected_components):
                if len(component) > 1:  # Only clusters with multiple nodes
                    connectivity_clusters[f"component_{i}"] = list(component)
            
            return connectivity_clusters
    
    def _combine_clustering_results(self, category_clusters: Dict[str, List[str]], 
                                  connectivity_clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Combine different clustering approaches"""
        # For now, use category-based clustering as primary
        # In production, you'd implement more sophisticated combination logic
        combined_clusters = category_clusters.copy()
        
        # Add small connectivity clusters that don't overlap with category clusters
        category_nodes = set()
        for nodes in category_clusters.values():
            category_nodes.update(nodes)
        
        for cluster_id, nodes in connectivity_clusters.items():
            nodes_set = set(nodes)
            if len(nodes_set - category_nodes) > 1:  # New nodes not in category clusters
                combined_clusters[f"conn_{cluster_id}"] = list(nodes_set - category_nodes)
        
        return combined_clusters
    
    def _create_cluster(self, cluster_id: str, node_ids: List[str], G: nx.Graph) -> GraphCluster:
        """Create a cluster object"""
        # Find internal and external edges
        internal_edges = []
        external_edges = []
        
        node_set = set(node_ids)
        
        for edge_id, edge in self.edges_registry.items():
            source_in_cluster = edge.source_node_id in node_set
            target_in_cluster = edge.target_node_id in node_set
            
            if source_in_cluster and target_in_cluster:
                internal_edges.append(edge_id)
            elif source_in_cluster or target_in_cluster:
                external_edges.append(edge_id)
        
        # Calculate cluster properties
        if node_ids:
            representative_node = self.nodes_registry[node_ids[0]]
            cluster_name = f"{representative_node.category}_cluster"
            cluster_type = representative_node.category
        else:
            cluster_name = f"cluster_{cluster_id}"
            cluster_type = "unknown"
        
        # Calculate cluster confidence (average of node confidences)
        node_confidences = [self.nodes_registry[nid].confidence for nid in node_ids]
        cluster_confidence = sum(node_confidences) / len(node_confidences) if node_confidences else 0.0
        
        cluster = GraphCluster(
            id=cluster_id,
            name=cluster_name,
            cluster_type=cluster_type,
            nodes=node_ids,
            internal_edges=internal_edges,
            external_edges=external_edges,
            properties={
                'node_count': len(node_ids),
                'internal_edge_count': len(internal_edges),
                'external_edge_count': len(external_edges),
                'density': len(internal_edges) / (len(node_ids) * (len(node_ids) - 1) / 2) if len(node_ids) > 1 else 0
            },
            confidence=cluster_confidence
        )
        
        return cluster
    
    async def _validate_and_optimize(self):
        """Phase 4: Validate and optimize the graph"""
        logger.info("Phase 4: Validation and optimization")
        
        # Remove low-confidence nodes and edges
        await self._prune_low_confidence_elements()
        
        # Calculate graph metrics
        await self._calculate_graph_metrics()
        
        # Optimize edge weights
        await self._optimize_edge_weights()
        
        logger.info("Validation and optimization completed")
    
    async def _prune_low_confidence_elements(self):
        """Remove elements with very low confidence"""
        min_node_confidence = 0.3
        min_edge_confidence = 0.2
        
        # Prune nodes
        nodes_to_remove = []
        for node_id, node in self.nodes_registry.items():
            if node.confidence < min_node_confidence:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.nodes_registry[node_id]
        
        # Prune edges (and edges connected to removed nodes)
        edges_to_remove = []
        valid_node_ids = set(self.nodes_registry.keys())
        
        for edge_id, edge in self.edges_registry.items():
            if (edge.confidence < min_edge_confidence or 
                edge.source_node_id not in valid_node_ids or 
                edge.target_node_id not in valid_node_ids):
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.edges_registry[edge_id]
        
        logger.info("Pruning completed", 
                   nodes_removed=len(nodes_to_remove),
                   edges_removed=len(edges_to_remove))
    
    async def _calculate_graph_metrics(self):
        """Calculate graph-level metrics for nodes"""
        # Create NetworkX graph
        G = nx.Graph()
        
        for node_id in self.nodes_registry.keys():
            G.add_node(node_id)
        
        for edge in self.edges_registry.values():
            G.add_edge(edge.source_node_id, edge.target_node_id, weight=edge.weight)
        
        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Update node properties
            for node_id, node in self.nodes_registry.items():
                if node_id in degree_centrality:
                    node.properties['degree_centrality'] = degree_centrality[node_id]
                    node.properties['betweenness_centrality'] = betweenness_centrality[node_id]
        
        except Exception as e:
            logger.warning("Failed to calculate centrality measures", error=str(e))
    
    async def _optimize_edge_weights(self):
        """Optimize edge weights based on graph structure"""
        # Apply PageRank-like algorithm to boost important connections
        # This is a simplified version - production would use more sophisticated methods
        
        for edge_id, edge in self.edges_registry.items():
            source_node = self.nodes_registry.get(edge.source_node_id)
            target_node = self.edges_registry.get(edge.target_node_id)
            
            if source_node and target_node:
                # Boost weight based on node importance
                source_importance = source_node.properties.get('importance_score', 0.5)
                target_importance = target_node.properties.get('importance_score', 0.5)
                
                importance_boost = (source_importance + target_importance) / 2
                edge.weight = min(1.0, edge.weight * (1 + importance_boost * 0.2))
    
    async def _persist_to_neo4j(self):
        """Phase 5: Persist enhanced graph to Neo4j"""
        logger.info("Phase 5: Persisting to Neo4j")
        
        # Create enhanced indexes
        await self._create_enhanced_indexes()
        
        # Create nodes with enhanced properties
        for node_id, node in self.nodes_registry.items():
            await self._create_enhanced_node(node)
        
        # Create edges with enhanced properties
        for edge_id, edge in self.edges_registry.items():
            await self._create_enhanced_edge(edge)
        
        # Create cluster nodes
        for cluster_id, cluster in self.clusters_registry.items():
            await self._create_cluster_node(cluster)
        
        logger.info("Neo4j persistence completed")
    
    async def _create_enhanced_indexes(self):
        """Create enhanced indexes for better performance"""
        enhanced_indexes = [
            "CREATE INDEX enhanced_node_category IF NOT EXISTS FOR (n:EnhancedNode) ON (n.category)",
            "CREATE INDEX enhanced_node_confidence IF NOT EXISTS FOR (n:EnhancedNode) ON (n.confidence)",
            "CREATE INDEX enhanced_edge_type IF NOT EXISTS FOR ()-[r:ENHANCED_RELATIONSHIP]->() ON (r.relationship_type)",
            "CREATE INDEX enhanced_edge_weight IF NOT EXISTS FOR ()-[r:ENHANCED_RELATIONSHIP]->() ON (r.weight)",
            "CREATE INDEX cluster_type IF NOT EXISTS FOR (c:Cluster) ON (c.cluster_type)"
        ]
        
        with self.neo4j_client.driver.session() as session:
            for index_query in enhanced_indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.warning("Enhanced index creation failed", query=index_query, error=str(e))
    
    async def _create_enhanced_node(self, node: GraphNode):
        """Create enhanced node in Neo4j"""
        query = """
        CREATE (n:EnhancedNode {
            id: $id,
            name: $name,
            type: $type,
            category: $category,
            sub_category: $sub_category,
            confidence: $confidence,
            source_documents: $source_documents,
            aliases: $aliases,
            properties: $properties,
            domain_specific_attrs: $domain_specific_attrs,
            created_at: datetime()
        })
        RETURN n.id as node_id
        """
        
        with self.neo4j_client.driver.session() as session:
            session.run(query,
                       id=node.id,
                       name=node.name,
                       type=node.type,
                       category=node.category,
                       sub_category=node.sub_category,
                       confidence=node.confidence,
                       source_documents=node.source_documents,
                       aliases=list(node.aliases),
                       properties=json.dumps(node.properties),
                       domain_specific_attrs=json.dumps(node.domain_specific_attrs or {}))
    
    async def _create_enhanced_edge(self, edge: GraphEdge):
        """Create enhanced edge in Neo4j"""
        query = """
        MATCH (source:EnhancedNode {id: $source_id})
        MATCH (target:EnhancedNode {id: $target_id})
        CREATE (source)-[r:ENHANCED_RELATIONSHIP {
            id: $edge_id,
            relationship_type: $relationship_type,
            confidence: $confidence,
            weight: $weight,
            directional: $directional,
            evidence: $evidence,
            source_documents: $source_documents,
            properties: $properties,
            created_at: datetime()
        }]->(target)
        RETURN r.id as edge_id
        """
        
        with self.neo4j_client.driver.session() as session:
            session.run(query,
                       edge_id=edge.id,
                       source_id=edge.source_node_id,
                       target_id=edge.target_node_id,
                       relationship_type=edge.relationship_type,
                       confidence=edge.confidence,
                       weight=edge.weight,
                       directional=edge.directional,
                       evidence=edge.evidence,
                       source_documents=edge.source_documents,
                       properties=json.dumps(edge.properties))
    
    async def _create_cluster_node(self, cluster: GraphCluster):
        """Create cluster node in Neo4j"""
        query = """
        CREATE (c:Cluster {
            id: $id,
            name: $name,
            cluster_type: $cluster_type,
            node_count: $node_count,
            confidence: $confidence,
            properties: $properties,
            created_at: datetime()
        })
        RETURN c.id as cluster_id
        """
        
        with self.neo4j_client.driver.session() as session:
            cluster_id = session.run(query,
                                   id=cluster.id,
                                   name=cluster.name,
                                   cluster_type=cluster.cluster_type,
                                   node_count=len(cluster.nodes),
                                   confidence=cluster.confidence,
                                   properties=json.dumps(cluster.properties)).single()["cluster_id"]
            
            # Link cluster to its nodes
            link_query = """
            MATCH (c:Cluster {id: $cluster_id})
            MATCH (n:EnhancedNode) WHERE n.id IN $node_ids
            CREATE (c)-[:CONTAINS]->(n)
            """
            
            session.run(link_query, cluster_id=cluster.id, node_ids=cluster.nodes)
    
    async def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics for the constructed graph"""
        if not self.nodes_registry or not self.edges_registry:
            return {'overall_quality': 0.0}
        
        # Node quality metrics
        avg_node_confidence = sum(n.confidence for n in self.nodes_registry.values()) / len(self.nodes_registry)
        
        # Edge quality metrics
        avg_edge_confidence = sum(e.confidence for e in self.edges_registry.values()) / len(self.edges_registry)
        avg_edge_weight = sum(e.weight for e in self.edges_registry.values()) / len(self.edges_registry)
        
        # Graph structure metrics
        total_nodes = len(self.nodes_registry)
        total_edges = len(self.edges_registry)
        graph_density = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        # Domain coverage
        categories = set(n.category for n in self.nodes_registry.values())
        domain_coverage = len(categories) / len(self.domain_taxonomy) if self.domain_taxonomy else 1.0
        
        # Overall quality score
        overall_quality = (
            avg_node_confidence * 0.3 +
            avg_edge_confidence * 0.3 +
            min(1.0, graph_density * 10) * 0.2 +  # Normalize density
            domain_coverage * 0.2
        )
        
        return {
            'overall_quality': overall_quality,
            'avg_node_confidence': avg_node_confidence,
            'avg_edge_confidence': avg_edge_confidence,
            'avg_edge_weight': avg_edge_weight,
            'graph_density': graph_density,
            'domain_coverage': domain_coverage,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_clusters': len(self.clusters_registry)
        }
    
    # Helper methods for entity classification and processing
    
    def _classify_entity(self, entity_name: str, entity_type: str) -> Tuple[str, Optional[str]]:
        """Classify entity into domain categories"""
        entity_lower = entity_name.lower()
        
        for category, subcategories in self.domain_taxonomy.items():
            for sub_category, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword.lower() in entity_lower:
                        return category, sub_category
        
        # Fallback based on entity type
        type_mapping = {
            'product_code': ('devices', 'control'),
            'standard': ('standards', 'british'),
            'cable_spec': ('components', 'wiring'),
            'voltage': ('specifications', 'electrical'),
            'fire_device': ('devices', 'detection')
        }
        
        return type_mapping.get(entity_type, ('unknown', None))
    
    def _classify_table_entity(self, cell_value: str, column_header: str) -> Tuple[str, Optional[str]]:
        """Classify table entity based on cell value and column context"""
        cell_lower = cell_value.lower()
        header_lower = column_header.lower()
        
        # Check column header for context
        if any(keyword in header_lower for keyword in ['part', 'product', 'item', 'code']):
            return 'devices', 'control'
        elif any(keyword in header_lower for keyword in ['qty', 'quantity', 'number']):
            return 'specifications', 'performance'
        elif any(keyword in header_lower for keyword in ['price', 'cost', 'value']):
            return 'specifications', 'performance'
        
        # Check cell content
        if re.match(r'^[A-Z]{2,}\d+', cell_value):  # Product code pattern
            return 'devices', 'control'
        elif any(keyword in cell_lower for keyword in ['detector', 'sensor', 'alarm']):
            return 'devices', 'detection'
        
        return 'unknown', None
    
    def _is_significant_table_entity(self, cell_value: str, headers: List[str]) -> bool:
        """Check if table cell contains significant entity"""
        if len(cell_value.strip()) < 2:
            return False
        
        # Skip common non-entities
        skip_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[\d.]+$',  # Decimal numbers
            r'^[.\-_]+$',  # Just punctuation
            r'^n/?a$',  # N/A variations
            r'^tbc$',  # TBC
            r'^tbd$'   # TBD
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, cell_value.lower()):
                return False
        
        # Check for product codes, part numbers, etc.
        significant_patterns = [
            r'^[A-Z]{2,}\d+',  # Product codes
            r'.*detector.*',   # Devices
            r'.*cable.*',      # Components
            r'.*panel.*',      # Equipment
            r'BS\s*\d+',       # Standards
            r'EN\s*\d+'        # Standards
        ]
        
        for pattern in significant_patterns:
            if re.match(pattern, cell_value, re.IGNORECASE):
                return True
        
        return len(cell_value.split()) <= 4  # Not too long to be a sentence
    
    def _extract_aliases(self, entity_name: str, context: str) -> Set[str]:
        """Extract potential aliases for an entity"""
        aliases = set()
        
        if not context:
            return aliases
        
        # Look for parenthetical aliases
        paren_matches = re.findall(r'\(([^)]+)\)', context)
        for match in paren_matches:
            if len(match.split()) <= 3:  # Not too long
                aliases.add(match.strip())
        
        # Look for "also known as" patterns
        aka_patterns = [
            r'also known as ([^,.]+)',
            r'or ([A-Z][^,.]+)',
            r'aka ([^,.]+)'
        ]
        
        for pattern in aka_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                aliases.add(match.strip())
        
        return aliases
    
    def _extract_domain_attributes(self, entity_name: str, context: str) -> Dict[str, Any]:
        """Extract domain-specific attributes"""
        attrs = {}
        
        if not context:
            return attrs
        
        # Extract electrical specifications
        voltage_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:V|volt)', context, re.IGNORECASE)
        if voltage_match:
            attrs['voltage'] = float(voltage_match.group(1))
        
        current_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:A|amp|mA)', context, re.IGNORECASE)
        if current_match:
            attrs['current'] = current_match.group(1)
        
        # Extract temperature ratings
        temp_match = re.search(r'(-?\d+(?:\.\d+)?)\s*°?C', context)
        if temp_match:
            attrs['temperature_rating'] = float(temp_match.group(1))
        
        # Extract IP ratings
        ip_match = re.search(r'IP\s*(\d{2})', context, re.IGNORECASE)
        if ip_match:
            attrs['ip_rating'] = f"IP{ip_match.group(1)}"
        
        return attrs
    
    def _calculate_domain_confidence(self, node: GraphNode) -> float:
        """Calculate confidence that node belongs to the fire alarm domain"""
        confidence = 0.5  # Base confidence
        
        # Boost for domain keywords in name
        domain_keywords = [
            'fire', 'alarm', 'smoke', 'heat', 'detector', 'sensor',
            'panel', 'sounder', 'beacon', 'call point', 'break glass'
        ]
        
        name_lower = node.name.lower()
        keyword_matches = sum(1 for keyword in domain_keywords if keyword in name_lower)
        confidence += keyword_matches * 0.1
        
        # Boost for standards
        if any(std in name_lower for std in ['bs 5839', 'bs en 54', 'en 54', 'nfpa 72']):
            confidence += 0.3
        
        # Boost for category classification
        if node.category != 'unknown':
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_importance_score(self, node: GraphNode) -> float:
        """Calculate importance score for a node"""
        score = node.confidence
        
        # Boost for multiple source documents
        score += len(node.source_documents) * 0.1
        
        # Boost for having aliases
        score += len(node.aliases) * 0.05
        
        # Boost for domain-specific attributes
        if node.domain_specific_attrs:
            score += len(node.domain_specific_attrs) * 0.1
        
        return min(1.0, score)