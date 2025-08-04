# TTT-Enhanced BYOKG-RAG System - Complete Implementation
# This is a comprehensive implementation guide. Save each section as a separate file.

# =====================================
# File: requirements.txt
# =====================================
"""
# Core dependencies
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0

# LLM and ML
openai==1.3.5
transformers==4.36.0
torch==2.1.1
peft==0.7.0  # For LoRA
accelerate==0.25.0

# Graph database
neo4j==5.14.0
py2neo==2021.2.3

# Document processing
PyMuPDF==1.23.8
pandas==2.1.3
numpy==1.25.2

# AWS
boto3==1.29.7
aioboto3==11.3.0

# API and async
aiofiles==23.2.1
httpx==0.25.2
python-multipart==0.0.6

# Utilities
tqdm==4.66.1
structlog==23.2.0
prometheus-client==0.19.0
"""

# =====================================
# File: .env.example
# =====================================
"""
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name

# TTT Configuration
TTT_CACHE_DIR=./data/adapters
TTT_EXAMPLES_DIR=./data/examples
TTT_LEARNING_RATE=5e-5
TTT_NUM_EPOCHS=2
TTT_LORA_RANK=128
TTT_BATCH_SIZE=2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
"""

# =====================================
# File: src/utils/config.py
# =====================================
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    
    # Neo4j
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    # AWS S3
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    s3_bucket_name: str = Field(..., env="S3_BUCKET_NAME")
    
    # TTT Configuration
    ttt_cache_dir: Path = Field(Path("./data/adapters"), env="TTT_CACHE_DIR")
    ttt_examples_dir: Path = Field(Path("./data/examples"), env="TTT_EXAMPLES_DIR")
    ttt_learning_rate: float = Field(5e-5, env="TTT_LEARNING_RATE")
    ttt_num_epochs: int = Field(2, env="TTT_NUM_EPOCHS")
    ttt_lora_rank: int = Field(128, env="TTT_LORA_RANK")
    ttt_batch_size: int = Field(2, env="TTT_BATCH_SIZE")
    ttt_use_gpu: bool = Field(True, env="TTT_USE_GPU")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def setup_directories(self):
        """Create necessary directories"""
        self.ttt_cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttt_examples_dir.mkdir(parents=True, exist_ok=True)
        (Path("./data/cache")).mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.setup_directories()

# =====================================
# File: src/utils/logger.py
# =====================================
import structlog
import logging
from typing import Any, Dict

def setup_logger() -> structlog.BoundLogger:
    """Configure structured logging"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
    
    return structlog.get_logger()

logger = setup_logger()

# =====================================
# File: src/core/ttt_adapter.py
# =====================================
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from src.utils.config import settings
from src.utils.logger import logger

@dataclass
class TTTExample:
    """Training example for TTT"""
    query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    boq_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

class TTTAdapter:
    """Test-Time Training adapter for BYOKG-RAG"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.ttt_use_gpu else "cpu")
        self.model = None
        self.tokenizer = None
        self.base_model_name = "microsoft/phi-2"  # Lightweight model for TTT
        self.adapter_cache = {}
        
        logger.info("Initializing TTT Adapter", device=str(self.device))
        
    def load_base_model(self):
        """Load the base language model"""
        if self.model is None:
            logger.info("Loading base model", model=self.base_model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
                
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for efficient fine-tuning"""
        return LoraConfig(
            r=settings.ttt_lora_rank,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
    def prepare_training_data(self, examples: List[TTTExample], 
                            leave_one_out: bool = True) -> List[Dict[str, str]]:
        """Prepare training data with leave-one-out strategy"""
        training_data = []
        
        if leave_one_out:
            for i, test_example in enumerate(examples):
                # Create context from other examples
                context_examples = examples[:i] + examples[i+1:]
                
                # Format as few-shot prompt
                prompt = self._format_few_shot_prompt(context_examples, test_example.query)
                target = self._format_target(test_example)
                
                training_data.append({
                    "input": prompt,
                    "output": target,
                    "metadata": test_example.metadata
                })
        else:
            # Direct I/O training
            for example in examples:
                prompt = f"Query: {example.query}\nExtract entities and relationships:"
                target = self._format_target(example)
                
                training_data.append({
                    "input": prompt,
                    "output": target,
                    "metadata": example.metadata
                })
                
        return training_data
    
    def _format_few_shot_prompt(self, context_examples: List[TTTExample], 
                               query: str) -> str:
        """Format examples as few-shot prompt"""
        prompt_parts = ["Given these examples of fire alarm system queries and their extracted information:\n"]
        
        for i, example in enumerate(context_examples[:3]):  # Use top 3 examples
            prompt_parts.append(f"\nExample {i+1}:")
            prompt_parts.append(f"Query: {example.query}")
            prompt_parts.append(f"Entities: {json.dumps(example.entities, indent=2)}")
            prompt_parts.append(f"Relationships: {json.dumps(example.relationships, indent=2)}")
            
        prompt_parts.append(f"\nNow extract entities and relationships for:")
        prompt_parts.append(f"Query: {query}")
        
        return "\n".join(prompt_parts)
    
    def _format_target(self, example: TTTExample) -> str:
        """Format target output"""
        return json.dumps({
            "entities": example.entities,
            "relationships": example.relationships,
            "boq_relevant_items": [item["product_code"] for item in example.boq_items]
        }, indent=2)
    
    def train_adapter(self, examples: List[TTTExample], 
                     cache_key: Optional[str] = None) -> PeftModel:
        """Train LoRA adapter on examples using TTT approach"""
        
        # Check cache first
        if cache_key and cache_key in self.adapter_cache:
            logger.info("Loading cached adapter", cache_key=cache_key)
            return self.adapter_cache[cache_key]
        
        # Load base model if needed
        self.load_base_model()
        
        # Prepare training data
        training_data = self.prepare_training_data(examples)
        logger.info("Prepared training data", num_examples=len(training_data))
        
        # Create LoRA model
        lora_config = self.create_lora_config()
        peft_model = get_peft_model(self.model, lora_config)
        
        # Training setup
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=settings.ttt_learning_rate)
        peft_model.train()
        
        # Training loop
        for epoch in range(settings.ttt_num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(training_data, desc=f"TTT Epoch {epoch+1}")
            
            for batch_idx in range(0, len(training_data), settings.ttt_batch_size):
                batch = training_data[batch_idx:batch_idx + settings.ttt_batch_size]
                
                # Tokenize batch
                inputs = [item["input"] for item in batch]
                targets = [item["output"] for item in batch]
                
                encodings = self.tokenizer(
                    [f"{inp}\n{tgt}" for inp, tgt in zip(inputs, targets)],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = peft_model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.update(len(batch))
                
            avg_loss = epoch_loss / len(training_data)
            logger.info(f"TTT Epoch {epoch+1} completed", avg_loss=avg_loss)
        
        # Cache the adapter
        if cache_key:
            self.adapter_cache[cache_key] = peft_model
            
            # Save to disk
            adapter_path = settings.ttt_cache_dir / f"{cache_key}.pth"
            peft_model.save_pretrained(str(adapter_path))
            
        return peft_model
    
    def generate_with_adapter(self, query: str, peft_model: PeftModel, 
                            max_length: int = 512) -> Dict[str, Any]:
        """Generate response using trained adapter"""
        peft_model.eval()
        
        # Prepare input
        input_text = f"Query: {query}\nExtract entities and relationships:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and parse
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"entities": [], "relationships": []}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
            result = {"entities": [], "relationships": []}
            
        return result
    
    def compute_cache_key(self, examples: List[TTTExample]) -> str:
        """Compute cache key for examples"""
        # Create a hash of example queries
        example_str = "|".join(sorted([ex.query for ex in examples]))
        return hashlib.md5(example_str.encode()).hexdigest()[:16]

# =====================================
# File: src/core/kg_linker.py
# =====================================
import re
from typing import List, Dict, Any, Tuple, Optional
import openai
from dataclasses import dataclass
import json

from src.utils.config import settings
from src.utils.logger import logger
from src.core.ttt_adapter import TTTAdapter, TTTExample

@dataclass
class KGLinkingResult:
    """Result of knowledge graph linking"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query_type: str
    constraints: Dict[str, Any]
    confidence: float

class EnhancedKGLinker:
    """Knowledge Graph Linker enhanced with TTT capabilities"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.ttt_adapter = TTTAdapter()
        self.entity_patterns = self._load_entity_patterns()
        
    def _load_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Load regex patterns for entity recognition"""
        return {
            "room_count": re.compile(r'(\d+)\s*(?:rooms?|spaces?)', re.I),
            "floor_count": re.compile(r'(\d+)\s*(?:floors?|stories?|levels?)', re.I),
            "building_type": re.compile(r'(office|hospital|school|warehouse|retail)', re.I),
            "product_code": re.compile(r'[A-Z]{2,4}-?\d{3,4}[A-Z]?', re.I),
            "compliance": re.compile(r'(NFPA|UL|FM|EN)\s*\d+', re.I)
        }
    
    async def link_with_ttt(self, query: str, examples: Optional[List[TTTExample]] = None) -> KGLinkingResult:
        """Link query to KG with TTT enhancement"""
        
        # First try rule-based extraction
        rule_based_entities = self._extract_entities_rule_based(query)
        
        # If examples provided, use TTT
        if examples and len(examples) >= 5:
            logger.info("Using TTT-enhanced linking", num_examples=len(examples))
            
            # Train adapter
            cache_key = self.ttt_adapter.compute_cache_key(examples)
            peft_model = self.ttt_adapter.train_adapter(examples, cache_key)
            
            # Generate with adapter
            ttt_result = self.ttt_adapter.generate_with_adapter(query, peft_model)
            
            # Merge results
            entities = self._merge_entities(rule_based_entities, ttt_result.get("entities", []))
            relationships = ttt_result.get("relationships", [])
        else:
            # Fallback to LLM-based extraction
            entities, relationships = await self._extract_with_llm(query)
            entities = self._merge_entities(rule_based_entities, entities)
        
        # Determine query type and constraints
        query_type = self._classify_query(query)
        constraints = self._extract_constraints(query)
        
        # Calculate confidence
        confidence = self._calculate_confidence(entities, relationships, query_type)
        
        return KGLinkingResult(
            entities=entities,
            relationships=relationships,
            query_type=query_type,
            constraints=constraints,
            confidence=confidence
        )
    
    def _extract_entities_rule_based(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(query)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "value": match,
                    "source": "rule_based",
                    "position": query.find(match)
                })
                
        return entities
    
    async def _extract_with_llm(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships using LLM"""
        
        prompt = f"""
        Extract entities and relationships from this fire alarm system query:
        
        Query: {query}
        
        Return JSON with:
        - entities: List of {{type, value, properties}}
        - relationships: List of {{source, relation, target}}
        
        Entity types: product, location, requirement, compliance, feature
        Relation types: requires, compatible_with, installed_in, complies_with
        """
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a fire alarm system expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("entities", []), result.get("relationships", [])
        except:
            return [], []
    
    def _merge_entities(self, rule_based: List[Dict], llm_based: List[Dict]) -> List[Dict]:
        """Merge entities from different sources"""
        merged = {f"{e['type']}:{e['value']}": e for e in rule_based}
        
        for entity in llm_based:
            key = f"{entity['type']}:{entity['value']}"
            if key not in merged:
                entity["source"] = "llm"
                merged[key] = entity
            else:
                # Merge properties
                if "properties" in entity:
                    merged[key].setdefault("properties", {}).update(entity["properties"])
                    
        return list(merged.values())
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if "bill of" in query_lower or "boq" in query_lower:
            return "boq_generation"
        elif "compatible" in query_lower:
            return "compatibility_check"
        elif "comply" in query_lower or "compliance" in query_lower:
            return "compliance_check"
        elif "replace" in query_lower or "upgrade" in query_lower:
            return "system_upgrade"
        else:
            return "general_query"
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints from query"""
        constraints = {}
        
        # Budget constraints
        budget_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)\s*(?:budget|max|limit)', query, re.I)
        if budget_match:
            constraints["max_budget"] = float(budget_match.group(1).replace(",", ""))
        
        # Timeline constraints
        timeline_match = re.search(r'(\d+)\s*(?:days?|weeks?|months?)', query, re.I)
        if timeline_match:
            constraints["timeline"] = timeline_match.group(0)
        
        # Compliance requirements
        compliance_matches = self.entity_patterns["compliance"].findall(query)
        if compliance_matches:
            constraints["compliance_required"] = compliance_matches
            
        return constraints
    
    def _calculate_confidence(self, entities: List[Dict], relationships: List[Dict], 
                            query_type: str) -> float:
        """Calculate confidence score for the linking result"""
        confidence = 0.5  # Base confidence
        
        # Boost for entities found
        confidence += min(len(entities) * 0.05, 0.25)
        
        # Boost for relationships
        confidence += min(len(relationships) * 0.03, 0.15)
        
        # Boost for specific query types
        if query_type != "general_query":
            confidence += 0.1
            
        return min(confidence, 1.0)

# =====================================
# File: src/core/graph_retriever.py
# =====================================
from typing import List, Dict, Any, Optional, Set
from neo4j import GraphDatabase, Driver
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.config import settings
from src.utils.logger import logger
from src.core.kg_linker import KGLinkingResult

class MultiStrategyGraphRetriever:
    """Graph retriever with multiple traversal strategies"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def retrieve(self, linking_result: KGLinkingResult, 
                      strategies: List[str] = None) -> Dict[str, Any]:
        """Retrieve relevant information using multiple strategies"""
        
        if strategies is None:
            strategies = self._select_strategies(linking_result)
            
        logger.info("Retrieving with strategies", strategies=strategies)
        
        # Execute strategies in parallel
        tasks = []
        for strategy in strategies:
            if strategy == "entity_expansion":
                tasks.append(self._entity_expansion(linking_result))
            elif strategy == "path_traversal":
                tasks.append(self._path_traversal(linking_result))
            elif strategy == "pattern_matching":
                tasks.append(self._pattern_matching(linking_result))
            elif strategy == "neighborhood_search":
                tasks.append(self._neighborhood_search(linking_result))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        merged_result = self._merge_results(results)
        
        # Rank and filter
        ranked_result = self._rank_results(merged_result, linking_result)
        
        return ranked_result
    
    def _select_strategies(self, linking_result: KGLinkingResult) -> List[str]:
        """Select appropriate strategies based on query type"""
        strategies = ["entity_expansion"]  # Always use entity expansion
        
        if linking_result.query_type == "boq_generation":
            strategies.extend(["pattern_matching", "neighborhood_search"])
        elif linking_result.query_type == "compatibility_check":
            strategies.append("path_traversal")
        elif linking_result.query_type == "compliance_check":
            strategies.append("pattern_matching")
            
        return strategies
    
    async def _entity_expansion(self, linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Expand entities to find related products"""
        
        def run_query():
            with self.driver.session() as session:
                results = {"products": [], "components": [], "relationships": []}
                
                # Find products matching entities
                for entity in linking_result.entities:
                    if entity["type"] == "product_code":
                        query = """
                        MATCH (p:Product {code: $code})
                        OPTIONAL MATCH (p)-[r]->(related)
                        RETURN p, r, related
                        """
                        result = session.run(query, code=entity["value"])
                        
                        for record in result:
                            if record["p"]:
                                results["products"].append(dict(record["p"]))
                            if record["r"]:
                                results["relationships"].append({
                                    "type": record["r"].type,
                                    "properties": dict(record["r"])
                                })
                            if record["related"]:
                                results["components"].append(dict(record["related"]))
                                
                return results
                
        return await asyncio.get_event_loop().run_in_executor(self.executor, run_query)
    
    async def _path_traversal(self, linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Find paths between entities"""
        
        def run_query():
            with self.driver.session() as session:
                results = {"paths": [], "nodes": [], "relationships": []}
                
                # Find paths between product entities
                product_entities = [e for e in linking_result.entities if e["type"] == "product_code"]
                
                if len(product_entities) >= 2:
                    for i in range(len(product_entities)):
                        for j in range(i + 1, len(product_entities)):
                            query = """
                            MATCH path = shortestPath(
                                (p1:Product {code: $code1})-[*..5]-(p2:Product {code: $code2})
                            )
                            RETURN path
                            """
                            result = session.run(
                                query,
                                code1=product_entities[i]["value"],
                                code2=product_entities[j]["value"]
                            )
                            
                            for record in result:
                                path = record["path"]
                                results["paths"].append({
                                    "start": product_entities[i]["value"],
                                    "end": product_entities[j]["value"],
                                    "length": len(path.relationships)
                                })
                                
                                for node in path.nodes:
                                    results["nodes"].append(dict(node))
                                for rel in path.relationships:
                                    results["relationships"].append({
                                        "type": rel.type,
                                        "properties": dict(rel)
                                    })
                                    
                return results
                
        return await asyncio.get_event_loop().run_in_executor(self.executor, run_query)
    
    async def _pattern_matching(self, linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Match graph patterns based on query requirements"""
        
        def run_query():
            with self.driver.session() as session:
                results = {"patterns": [], "products": []}
                
                # Define patterns based on query type
                if linking_result.query_type == "boq_generation":
                    # Find complete system patterns
                    query = """
                    MATCH (panel:ControlPanel)-[:SUPPORTS]->(device:Device)
                    WHERE device.type IN $device_types
                    OPTIONAL MATCH (device)-[:REQUIRES]->(accessory:Accessory)
                    RETURN panel, collect(DISTINCT device) as devices, 
                           collect(DISTINCT accessory) as accessories
                    """
                    
                    device_types = self._extract_device_types(linking_result)
                    result = session.run(query, device_types=device_types)
                    
                    for record in result:
                        pattern = {
                            "panel": dict(record["panel"]),
                            "devices": [dict(d) for d in record["devices"]],
                            "accessories": [dict(a) for a in record["accessories"] if a]
                        }
                        results["patterns"].append(pattern)
                        
                return results
                
        return await asyncio.get_event_loop().run_in_executor(self.executor, run_query)
    
    async def _neighborhood_search(self, linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Search neighborhood of identified entities"""
        
        def run_query():
            with self.driver.session() as session:
                results = {"neighbors": {}, "clusters": []}
                
                for entity in linking_result.entities[:5]:  # Limit to prevent explosion
                    if entity["type"] in ["product_code", "building_type"]:
                        query = """
                        MATCH (start {$prop: $value})-[r*1..2]-(neighbor)
                        RETURN start, neighbor, r
                        LIMIT 50
                        """
                        
                        prop = "code" if entity["type"] == "product_code" else "type"
                        result = session.run(query, prop=prop, value=entity["value"])
                        
                        neighbors = []
                        for record in result:
                            neighbors.append({
                                "node": dict(record["neighbor"]),
                                "distance": len(record["r"])
                            })
                            
                        results["neighbors"][entity["value"]] = neighbors
                        
                return results
                
        return await asyncio.get_event_loop().run_in_executor(self.executor, run_query)
    
    def _extract_device_types(self, linking_result: KGLinkingResult) -> List[str]:
        """Extract device types from entities"""
        device_types = ["smoke_detector", "heat_detector", "manual_station"]  # Defaults
        
        for entity in linking_result.entities:
            if entity["type"] == "product_code":
                # Infer device type from product code
                code = entity["value"].upper()
                if "SD" in code or "SMOKE" in code:
                    device_types.append("smoke_detector")
                elif "HD" in code or "HEAT" in code:
                    device_types.append("heat_detector")
                elif "MS" in code or "PULL" in code:
                    device_types.append("manual_station")
                    
        return list(set(device_types))
    
    def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple strategies"""
        merged = {
            "products": [],
            "components": [],
            "relationships": [],
            "patterns": [],
            "paths": []
        }
        
        seen_products = set()
        seen_relationships = set()
        
        for result in results:
            if isinstance(result, Exception):
                logger.error("Strategy failed", error=str(result))
                continue
                
            # Merge products
            for product in result.get("products", []):
                product_id = product.get("code", product.get("id"))
                if product_id and product_id not in seen_products:
                    seen_products.add(product_id)
                    merged["products"].append(product)
                    
            # Merge relationships
            for rel in result.get("relationships", []):
                rel_key = f"{rel.get('type')}:{rel.get('properties', {})}"
                if rel_key not in seen_relationships:
                    seen_relationships.add(rel_key)
                    merged["relationships"].append(rel)
                    
            # Add other results
            merged["patterns"].extend(result.get("patterns", []))
            merged["paths"].extend(result.get("paths", []))
            
        return merged
    
    def _rank_results(self, results: Dict[str, Any], 
                     linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Rank and filter results based on relevance"""
        
        # Score products based on relevance
        scored_products = []
        for product in results["products"]:
            score = self._calculate_product_score(product, linking_result)
            scored_products.append((score, product))
            
        # Sort by score and take top results
        scored_products.sort(key=lambda x: x[0], reverse=True)
        results["products"] = [p[1] for p in scored_products[:50]]
        
        # Add relevance scores
        results["relevance_scores"] = {
            p["code"]: score for score, p in scored_products[:50]
        }
        
        return results
    
    def _calculate_product_score(self, product: Dict[str, Any], 
                                linking_result: KGLinkingResult) -> float:
        """Calculate relevance score for a product"""
        score = 0.0
        
        # Match with entities
        for entity in linking_result.entities:
            if entity["value"].lower() in str(product).lower():
                score += 0.3
                
        # Check constraints
        if "price" in product and "max_budget" in linking_result.constraints:
            if product["price"] <= linking_result.constraints["max_budget"]:
                score += 0.2
            else:
                score -= 0.3
                
        # Compliance match
        if "compliance" in product:
            for req in linking_result.constraints.get("compliance_required", []):
                if req in product["compliance"]:
                    score += 0.2
                    
        return score
    
    def close(self):
        """Close driver connection"""
        self.driver.close()
        self.executor.shutdown()

# =====================================
# File: src/core/orchestrator.py
# =====================================
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import json

from src.utils.config import settings
from src.utils.logger import logger
from src.core.kg_linker import EnhancedKGLinker, KGLinkingResult
from src.core.graph_retriever import MultiStrategyGraphRetriever
from src.core.ttt_adapter import TTTExample

class TTTOrchestrator:
    """Orchestrator for TTT-enhanced BYOKG-RAG pipeline"""
    
    def __init__(self):
        self.kg_linker = EnhancedKGLinker()
        self.graph_retriever = MultiStrategyGraphRetriever()
        self.iteration_history = []
        
    async def generate_boq(self, query: str, examples: Optional[List[TTTExample]] = None,
                          max_iterations: int = 3) -> Dict[str, Any]:
        """Generate BOQ with iterative refinement"""
        
        logger.info("Starting BOQ generation", query=query, max_iterations=max_iterations)
        
        # Initialize result
        result = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "iterations": [],
            "final_boq": None,
            "confidence": 0.0
        }
        
        current_query = query
        accumulated_context = {}
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: KG Linking with TTT
            linking_result = await self.kg_linker.link_with_ttt(current_query, examples)
            
            # Step 2: Graph Retrieval
            retrieval_result = await self.graph_retriever.retrieve(linking_result)
            
            # Step 3: Generate BOQ items
            boq_items = await self._generate_boq_items(
                linking_result, 
                retrieval_result,
                accumulated_context
            )
            
            # Step 4: Validate and refine
            validation_result = await self._validate_boq(boq_items, linking_result)
            
            # Record iteration
            iteration_data = {
                "iteration": iteration + 1,
                "linking_result": self._serialize_linking_result(linking_result),
                "retrieval_stats": self._get_retrieval_stats(retrieval_result),
                "boq_items_count": len(boq_items),
                "validation": validation_result,
                "confidence": linking_result.confidence
            }
            result["iterations"].append(iteration_data)
            
            # Check if we should continue
            if validation_result["is_complete"] or iteration == max_iterations - 1:
                result["final_boq"] = {
                    "items": boq_items,
                    "total_cost": sum(item.get("total_price", 0) for item in boq_items),
                    "compliance": validation_result.get("compliance_status", {}),
                    "warnings": validation_result.get("warnings", [])
                }
                result["confidence"] = linking_result.confidence
                break
            else:
                # Prepare for next iteration
                current_query = self._generate_refinement_query(
                    current_query,
                    validation_result["missing_components"]
                )
                accumulated_context.update(retrieval_result)
                
        return result
    
    async def _generate_boq_items(self, linking_result: KGLinkingResult,
                                 retrieval_result: Dict[str, Any],
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate BOQ items from retrieval results"""
        
        boq_items = []
        
        # Process products
        for product in retrieval_result.get("products", []):
            quantity = self._calculate_quantity(product, linking_result)
            
            boq_item = {
                "product_code": product.get("code"),
                "description": product.get("description", ""),
                "category": product.get("category", "Unknown"),
                "quantity": quantity,
                "unit_price": product.get("price", 0),
                "total_price": quantity * product.get("price", 0),
                "compliance": product.get("compliance", []),
                "compatibility": product.get("compatible_with", []),
                "lead_time": product.get("lead_time", "TBD"),
                "notes": []
            }
            
            # Add installation requirements
            if "installation_requirements" in product:
                boq_item["installation"] = product["installation_requirements"]
                
            boq_items.append(boq_item)
            
        # Process patterns for system components
        for pattern in retrieval_result.get("patterns", []):
            system_items = self._process_system_pattern(pattern, linking_result)
            boq_items.extend(system_items)
            
        # Remove duplicates and merge quantities
        boq_items = self._merge_duplicate_items(boq_items)
        
        return boq_items
    
    def _calculate_quantity(self, product: Dict[str, Any], 
                          linking_result: KGLinkingResult) -> int:
        """Calculate quantity based on project requirements"""
        
        base_quantity = 1
        
        # Extract room/floor counts from entities
        room_count = 0
        floor_count = 0
        
        for entity in linking_result.entities:
            if entity["type"] == "room_count":
                room_count = int(entity["value"])
            elif entity["type"] == "floor_count":
                floor_count = int(entity["value"])
                
        # Calculate based on product type
        product_type = product.get("type", "").lower()
        
        if "detector" in product_type:
            # Typical: 1 detector per room
            base_quantity = room_count if room_count > 0 else 10
        elif "panel" in product_type:
            # Usually 1 panel per building
            base_quantity = 1
        elif "station" in product_type:
            # Manual stations: 1-2 per floor
            base_quantity = floor_count * 2 if floor_count > 0 else 4
            
        # Apply any multipliers from product specs
        if "coverage_area" in product:
            # Adjust based on coverage
            pass
            
        return max(base_quantity, 1)
    
    def _process_system_pattern(self, pattern: Dict[str, Any],
                               linking_result: KGLinkingResult) -> List[Dict[str, Any]]:
        """Process a system pattern to generate BOQ items"""
        
        items = []
        
        # Add control panel
        if "panel" in pattern:
            panel = pattern["panel"]
            items.append({
                "product_code": panel.get("code"),
                "description": f"Control Panel - {panel.get('model', '')}",
                "category": "Control Equipment",
                "quantity": 1,
                "unit_price": panel.get("price", 0),
                "total_price": panel.get("price", 0)
            })
            
        # Add devices
        for device in pattern.get("devices", []):
            quantity = self._calculate_quantity(device, linking_result)
            items.append({
                "product_code": device.get("code"),
                "description": device.get("description", ""),
                "category": device.get("category", "Detection Device"),
                "quantity": quantity,
                "unit_price": device.get("price", 0),
                "total_price": quantity * device.get("price", 0)
            })
            
        # Add accessories
        for accessory in pattern.get("accessories", []):
            # Accessories typically match device quantities
            device_quantity = sum(item["quantity"] for item in items if item["category"] == "Detection Device")
            items.append({
                "product_code": accessory.get("code"),
                "description": accessory.get("description", ""),
                "category": "Accessory",
                "quantity": device_quantity,
                "unit_price": accessory.get("price", 0),
                "total_price": device_quantity * accessory.get("price", 0)
            })
            
        return items
    
    def _merge_duplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate items and sum quantities"""
        
        merged = {}
        
        for item in items:
            key = item["product_code"]
            
            if key in merged:
                # Sum quantities
                merged[key]["quantity"] += item["quantity"]
                merged[key]["total_price"] = merged[key]["quantity"] * merged[key]["unit_price"]
                
                # Merge notes
                if "notes" in item:
                    merged[key].setdefault("notes", []).extend(item["notes"])
            else:
                merged[key] = item.copy()
                
        return list(merged.values())
    
    async def _validate_boq(self, boq_items: List[Dict[str, Any]],
                           linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Validate BOQ completeness and compliance"""
        
        validation_result = {
            "is_complete": True,
            "missing_components": [],
            "compliance_status": {},
            "warnings": [],
            "suggestions": []
        }
        
        # Check for essential components
        categories = {item["category"] for item in boq_items}
        
        if "Control Equipment" not in categories:
            validation_result["is_complete"] = False
            validation_result["missing_components"].append("Control Panel")
            
        if "Detection Device" not in categories:
            validation_result["is_complete"] = False
            validation_result["missing_components"].append("Detection Devices")
            
        # Check compliance requirements
        required_compliance = linking_result.constraints.get("compliance_required", [])
        for req in required_compliance:
            compliant_items = [
                item for item in boq_items 
                if req in item.get("compliance", [])
            ]
            validation_result["compliance_status"][req] = len(compliant_items) > 0
            
        # Check budget constraints
        if "max_budget" in linking_result.constraints:
            total_cost = sum(item["total_price"] for item in boq_items)
            if total_cost > linking_result.constraints["max_budget"]:
                validation_result["warnings"].append(
                    f"Total cost ${total_cost:,.2f} exceeds budget ${linking_result.constraints['max_budget']:,.2f}"
                )
                
        # Compatibility checks
        self._check_compatibility(boq_items, validation_result)
        
        return validation_result
    
    def _check_compatibility(self, boq_items: List[Dict[str, Any]], 
                           validation_result: Dict[str, Any]):
        """Check compatibility between BOQ items"""
        
        # Group items by category
        panels = [item for item in boq_items if item["category"] == "Control Equipment"]
        devices = [item for item in boq_items if item["category"] == "Detection Device"]
        
        # Check panel-device compatibility
        if panels and devices:
            panel_compatible = set()
            for panel in panels:
                panel_compatible.update(panel.get("compatibility", []))
                
            for device in devices:
                if device["product_code"] not in panel_compatible:
                    validation_result["warnings"].append(
                        f"Device {device['product_code']} may not be compatible with selected panel"
                    )
    
    def _generate_refinement_query(self, original_query: str, 
                                  missing_components: List[str]) -> str:
        """Generate refined query for next iteration"""
        
        refinement_parts = [original_query]
        
        if missing_components:
            refinement_parts.append(
                f"Also include: {', '.join(missing_components)}"
            )
            
        return " ".join(refinement_parts)
    
    def _serialize_linking_result(self, linking_result: KGLinkingResult) -> Dict[str, Any]:
        """Serialize linking result for storage"""
        return {
            "entities": linking_result.entities,
            "relationships": linking_result.relationships,
            "query_type": linking_result.query_type,
            "constraints": linking_result.constraints,
            "confidence": linking_result.confidence
        }
    
    def _get_retrieval_stats(self, retrieval_result: Dict[str, Any]) -> Dict[str, int]:
        """Get statistics from retrieval result"""
        return {
            "products_retrieved": len(retrieval_result.get("products", [])),
            "patterns_found": len(retrieval_result.get("patterns", [])),
            "relationships": len(retrieval_result.get("relationships", [])),
            "paths": len(retrieval_result.get("paths", []))
        }

# =====================================
# File: src/api/models.py
# =====================================
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class BOQRequest(BaseModel):
    """Request model for BOQ generation"""
    project_description: str = Field(..., description="Description of the fire alarm project")
    max_iterations: int = Field(3, ge=1, le=5, description="Maximum refinement iterations")
    use_ttt: bool = Field(True, description="Whether to use TTT enhancement")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Additional constraints")

class BOQItem(BaseModel):
    """Individual BOQ line item"""
    product_code: str
    description: str
    category: str
    quantity: int
    unit_price: float
    total_price: float
    compliance: List[str] = Field(default_factory=list)
    compatibility: List[str] = Field(default_factory=list)
    lead_time: str = "TBD"
    notes: List[str] = Field(default_factory=list)

class BOQResponse(BaseModel):
    """Response model for BOQ generation"""
    query: str
    timestamp: str
    iterations: List[Dict[str, Any]]
    final_boq: Optional[Dict[str, Any]]
    confidence: float
    processing_time: float

class GraphSearchRequest(BaseModel):
    """Request model for graph search"""
    query: str
    entity_types: List[str] = Field(default_factory=lambda: ["Product", "Component"])
    max_results: int = Field(50, ge=1, le=200)

class GraphStatsResponse(BaseModel):
    """Response model for graph statistics"""
    node_count: int
    relationship_count: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    last_updated: Optional[str]

class ExampleRequest(BaseModel):
    """Request model for submitting TTT examples"""
    examples: List[Dict[str, Any]] = Field(..., min_items=10, max_items=50)
    validate: bool = Field(True, description="Whether to validate examples")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: str

# =====================================
# File: src/api/endpoints.py
# =====================================
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import time

from src.api.models import (
    BOQRequest, BOQResponse, GraphSearchRequest, 
    GraphStatsResponse, ExampleRequest, HealthResponse
)
from src.core.orchestrator import TTTOrchestrator
from src.core.ttt_adapter import TTTExample
from src.utils.logger import logger

router = APIRouter()

# Initialize orchestrator
orchestrator = TTTOrchestrator()

# Cache for TTT examples
ttt_examples_cache: List[TTTExample] = []

@router.post("/generate_boq", response_model=BOQResponse)
async def generate_boq(request: BOQRequest):
    """Generate Bill of Quantities for a fire alarm project"""
    
    start_time = time.time()
    
    try:
        # Use cached examples if TTT is enabled
        examples = ttt_examples_cache if request.use_ttt else None
        
        # Generate BOQ
        result = await orchestrator.generate_boq(
            query=request.project_description,
            examples=examples,
            max_iterations=request.max_iterations
        )
        
        # Add processing time
        result["processing_time"] = time.time() - start_time
        
        return BOQResponse(**result)
        
    except Exception as e:
        logger.error("BOQ generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit_examples")
async def submit_examples(request: ExampleRequest):
    """Submit examples for TTT training"""
    
    try:
        # Convert to TTTExample objects
        new_examples = []
        
        for ex in request.examples:
            example = TTTExample(
                query=ex["query"],
                entities=ex.get("entities", []),
                relationships=ex.get("relationships", []),
                boq_items=ex.get("boq_items", []),
                metadata=ex.get("metadata", {})
            )
            new_examples.append(example)
            
        # Validate if requested
        if request.validate:
            # TODO: Add validation logic
            pass
            
        # Update cache
        ttt_examples_cache.clear()
        ttt_examples_cache.extend(new_examples)
        
        logger.info("Updated TTT examples", count=len(new_examples))
        
        return {
            "status": "success",
            "examples_received": len(new_examples),
            "cache_size": len(ttt_examples_cache)
        }
        
    except Exception as e:
        logger.error("Failed to submit examples", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Get statistics about the knowledge graph"""
    
    try:
        # Query Neo4j for stats
        with orchestrator.graph_retriever.driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            # Count by type
            node_types_result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as type, count(n) as count
            """)
            node_types = {record["type"]: record["count"] for record in node_types_result}
            
            rel_types_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count
            """)
            rel_types = {record["type"]: record["count"] for record in rel_types_result}
            
        return GraphStatsResponse(
            node_count=node_count,
            relationship_count=rel_count,
            node_types=node_types,
            relationship_types=rel_types,
            last_updated=None
        )
        
    except Exception as e:
        logger.error("Failed to get graph stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/search")
async def search_graph(request: GraphSearchRequest):
    """Search the knowledge graph"""
    
    try:
        results = []
        
        with orchestrator.graph_retriever.driver.session() as session:
            for entity_type in request.entity_types:
                query = f"""
                MATCH (n:{entity_type})
                WHERE toLower(n.name) CONTAINS toLower($search_term)
                   OR toLower(n.description) CONTAINS toLower($search_term)
                   OR toLower(n.code) CONTAINS toLower($search_term)
                RETURN n
                LIMIT $limit
                """
                
                result = session.run(
                    query,
                    search_term=request.query,
                    limit=request.max_results
                )
                
                for record in result:
                    node = dict(record["n"])
                    node["_type"] = entity_type
                    results.append(node)
                    
        return {
            "query": request.query,
            "results": results[:request.max_results],
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("Graph search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    services = {}
    
    # Check Neo4j
    try:
        with orchestrator.graph_retriever.driver.session() as session:
            session.run("RETURN 1")
        services["neo4j"] = "healthy"
    except:
        services["neo4j"] = "unhealthy"
        
    # Check OpenAI
    try:
        # Simple check - could enhance with actual API call
        services["openai"] = "healthy" if orchestrator.kg_linker.client else "unhealthy"
    except:
        services["openai"] = "unhealthy"
        
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        version="1.0.0",
        services=services,
        timestamp=datetime.utcnow().isoformat()
    )

# =====================================
# File: src/api/main.py
# =====================================
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from contextlib import asynccontextmanager

from src.api.endpoints import router
from src.utils.config import settings
from src.utils.logger import logger
from src.scripts.collect_examples import collect_initial_examples

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting TTT-Enhanced BYOKG-RAG API")
    
    # Check if we need to collect examples
    if len(router.ttt_examples_cache) == 0:
        logger.info("No TTT examples found. Initiating collection...")
        await collect_initial_examples()
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    router.orchestrator.graph_retriever.close()

# Create FastAPI app
app = FastAPI(
    title="TTT-Enhanced BYOKG-RAG API",
    description="Knowledge Graph-based RAG system with Test-Time Training for Simplex fire alarm products",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "TTT-Enhanced BYOKG-RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# =====================================
# File: src/utils/s3_manager.py
# =====================================
import boto3
from botocore.exceptions import ClientError
import aiofiles
import asyncio
from pathlib import Path
from typing import List, Optional, BinaryIO
import io

from src.utils.config import settings
from src.utils.logger import logger

class S3Manager:
    """Manager for AWS S3 operations"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3_bucket_name
        
    async def list_documents(self, prefix: str = "documents/") -> List[str]:
        """List all documents in S3 bucket"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
            
        except ClientError as e:
            logger.error("Failed to list S3 objects", error=str(e))
            return []
    
    async def download_document(self, key: str, local_path: Optional[Path] = None) -> Optional[bytes]:
        """Download document from S3"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            )
            
            content = response['Body'].read()
            
            if local_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(local_path, 'wb') as f:
                    await f.write(content)
                    
            return content
            
        except ClientError as e:
            logger.error("Failed to download from S3", key=key, error=str(e))
            return None
    
    async def upload_document(self, key: str, content: BinaryIO, 
                            metadata: Optional[dict] = None) -> bool:
        """Upload document to S3"""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
                
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=content,
                    **extra_args
                )
            )
            
            logger.info("Uploaded to S3", key=key)
            return True
            
        except ClientError as e:
            logger.error("Failed to upload to S3", key=key, error=str(e))
            return False
    
    async def get_document_metadata(self, key: str) -> Optional[dict]:
        """Get document metadata from S3"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            )
            
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'metadata': response.get('Metadata', {})
            }
            
        except ClientError as e:
            logger.error("Failed to get S3 metadata", key=key, error=str(e))
            return None

# =====================================
# File: src/utils/neo4j_client.py
# =====================================
from neo4j import GraphDatabase, Driver, AsyncGraphDatabase
from typing import List, Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

from src.utils.config import settings
from src.utils.logger import logger

class Neo4jClient:
    """Client for Neo4j operations"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self.async_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
    def close(self):
        """Close database connections"""
        self.driver.close()
        self.async_driver.close()
        
    @asynccontextmanager
    async def async_session(self):
        """Create async session context"""
        async with self.async_driver.session() as session:
            yield session
            
    def create_indexes(self):
        """Create necessary indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.code)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Component) ON (c.code)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Compliance) ON (c.standard)",
            "CREATE FULLTEXT INDEX product_search IF NOT EXISTS FOR (n:Product) ON EACH [n.name, n.description]"
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.info("Created index", query=index[:50])
                except Exception as e:
                    logger.warning("Index creation failed", error=str(e))
                    
    async def bulk_create_nodes(self, nodes: List[Dict[str, Any]], label: str):
        """Bulk create nodes"""
        async with self.async_session() as session:
            query = f"""
            UNWIND $nodes as node
            CREATE (n:{label})
            SET n = node
            """
            
            await session.run(query, nodes=nodes)
            logger.info(f"Created {len(nodes)} {label} nodes")
            
    async def bulk_create_relationships(self, relationships: List[Dict[str, Any]]):
        """Bulk create relationships"""
        async with self.async_session() as session:
            for rel in relationships:
                query = f"""
                MATCH (a:{rel['from_label']} {{code: $from_code}})
                MATCH (b:{rel['to_label']} {{code: $to_code}})
                CREATE (a)-[r:{rel['type']}]->(b)
                SET r = $properties
                """
                
                await session.run(
                    query,
                    from_code=rel['from_code'],
                    to_code=rel['to_code'],
                    properties=rel.get('properties', {})
                )
                
    async def find_shortest_path(self, start_code: str, end_code: str, 
                               max_length: int = 5) -> Optional[Dict[str, Any]]:
        """Find shortest path between two nodes"""
        async with self.async_session() as session:
            query = """
            MATCH path = shortestPath(
                (start {code: $start_code})-[*..%d]-(end {code: $end_code})
            )
            RETURN path
            """ % max_length
            
            result = await session.run(query, start_code=start_code, end_code=end_code)
            record = await result.single()
            
            if record:
                path = record["path"]
                return {
                    "length": len(path.relationships),
                    "nodes": [dict(node) for node in path.nodes],
                    "relationships": [
                        {"type": rel.type, "properties": dict(rel)}
                        for rel in path.relationships
                    ]
                }
            return None

# =====================================
# File: src/ingestion/pdf_parser.py
# =====================================
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import pandas as pd

from src.utils.logger import logger

class PDFParser:
    """Parser for Simplex product PDF documents"""
    
    def __init__(self):
        self.table_patterns = {
            "product_table": re.compile(r'product\s*code|part\s*number', re.I),
            "specification": re.compile(r'specification|technical\s*data', re.I),
            "compatibility": re.compile(r'compatible|compatibility', re.I)
        }
        
    async def parse_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse PDF document and extract structured information"""
        
        logger.info("Parsing PDF", path=str(pdf_path))
        
        try:
            doc = fitz.open(pdf_path)
            
            result = {
                "metadata": self._extract_metadata(doc),
                "text_content": [],
                "tables": [],
                "images": [],
                "sections": {}
            }
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                result["text_content"].append({
                    "page": page_num + 1,
                    "text": text
                })
                
                # Extract tables
                tables = self._extract_tables(page, text)
                result["tables"].extend(tables)
                
                # Extract images (if needed)
                # images = self._extract_images(page)
                # result["images"].extend(images)
                
            # Identify sections
            result["sections"] = self._identify_sections(result["text_content"])
            
            # Extract product information
            result["products"] = self._extract_products(result)
            
            doc.close()
            
            return result
            
        except Exception as e:
            logger.error("PDF parsing failed", path=str(pdf_path), error=str(e))
            raise
            
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = doc.metadata or {}
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "pages": doc.page_count,
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", "")
        }
        
    def _extract_tables(self, page: fitz.Page, text: str) -> List[Dict[str, Any]]:
        """Extract tables from page"""
        tables = []
        
        # Check if page likely contains a table
        for table_type, pattern in self.table_patterns.items():
            if pattern.search(text):
                # Use PyMuPDF's table extraction
                tabs = page.find_tables()
                
                for tab in tabs:
                    # Convert to pandas DataFrame for easier processing
                    df = pd.DataFrame(tab.extract())
                    
                    # Clean up the dataframe
                    df = self._clean_table(df)
                    
                    if not df.empty:
                        tables.append({
                            "page": page.number + 1,
                            "type": table_type,
                            "data": df.to_dict(orient="records"),
                            "headers": df.columns.tolist()
                        })
                        
        return tables
        
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted table data"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return df
            
        # Try to identify headers
        if len(df) > 1:
            # Check if first row looks like headers
            first_row = df.iloc[0]
            if all(isinstance(val, str) and val for val in first_row):
                # Use first row as headers
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
                
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else '')
            
        return df
        
    def _identify_sections(self, text_content: List[Dict[str, str]]) -> Dict[str, List[int]]:
        """Identify document sections based on headers"""
        sections = {
            "overview": [],
            "specifications": [],
            "installation": [],
            "compatibility": [],
            "compliance": [],
            "ordering": []
        }
        
        section_patterns = {
            "overview": re.compile(r'overview|introduction|description', re.I),
            "specifications": re.compile(r'specification|technical|features', re.I),
            "installation": re.compile(r'installation|mounting|wiring', re.I),
            "compatibility": re.compile(r'compatible|compatibility', re.I),
            "compliance": re.compile(r'compliance|approval|listing|standard', re.I),
            "ordering": re.compile(r'ordering|part\s*number|model', re.I)
        }
        
        for page_data in text_content:
            page_num = page_data["page"]
            text = page_data["text"]
            
            # Check each line for section headers
            lines = text.split('\n')
            for line in lines[:20]:  # Check first 20 lines
                for section, pattern in section_patterns.items():
                    if pattern.search(line):
                        sections[section].append(page_num)
                        
        return sections
        
    def _extract_products(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract product information from parsed data"""
        products = []
        
        # Extract from tables
        for table in parsed_data["tables"]:
            if table["type"] == "product_table":
                for row in table["data"]:
                    product = self._parse_product_row(row)
                    if product:
                        products.append(product)
                        
        # Extract from text using patterns
        product_pattern = re.compile(r'([A-Z]{2,4}-?\d{3,4}[A-Z]?)\s*[-–]\s*([^,\n]+)')
        
        for page_data in parsed_data["text_content"]:
            matches = product_pattern.findall(page_data["text"])
            for code, description in matches:
                products.append({
                    "code": code.strip(),
                    "description": description.strip(),
                    "source_page": page_data["page"]
                })
                
        # Deduplicate
        seen = set()
        unique_products = []
        for product in products:
            if product["code"] not in seen:
                seen.add(product["code"])
                unique_products.append(product)
                
        return unique_products
        
    def _parse_product_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse a product row from table"""
        
        # Common column name variations
        code_columns = ["Product Code", "Part Number", "Model", "Code"]
        desc_columns = ["Description", "Product Description", "Name"]
        price_columns = ["Price", "List Price", "MSRP"]
        
        product = {}
        
        # Find product code
        for col in code_columns:
            if col in row and row[col]:
                product["code"] = row[col].strip()
                break
                
        # Find description
        for col in desc_columns:
            if col in row and row[col]:
                product["description"] = row[col].strip()
                break
                
        # Find price if available
        for col in price_columns:
            if col in row and row[col]:
                # Extract numeric price
                price_str = row[col].replace(', '').replace(',', '').strip()
                try:
                    product["price"] = float(price_str)
                except:
                    pass
                break
                
        # Only return if we have at least a code
        if "code" in product:
            return product
            
        return None

# =====================================
# File: src/ingestion/knowledge_extractor.py
# =====================================
import openai
from typing import List, Dict, Any, Tuple
import json
import re

from src.utils.config import settings
from src.utils.logger import logger

class KnowledgeExtractor:
    """Extract structured knowledge from documents using LLM"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.extraction_schemas = self._load_extraction_schemas()
        
    def _load_extraction_schemas(self) -> Dict[str, Any]:
        """Load schemas for different entity types"""
        return {
            "product": {
                "code": "Product code/model number",
                "name": "Product name",
                "description": "Product description",
                "category": "Product category (e.g., detector, panel, accessory)",
                "price": "List price if available",
                "specifications": "Technical specifications dict",
                "compliance": "List of compliance standards",
                "compatible_with": "List of compatible product codes"
            },
            "compliance": {
                "standard": "Standard name (e.g., NFPA 72)",
                "version": "Standard version/year",
                "description": "What the standard covers"
            },
            "compatibility": {
                "product_a": "First product code",
                "product_b": "Second product code",
                "relationship": "Type of compatibility",
                "notes": "Any restrictions or notes"
            }
        }
        
    async def extract_knowledge(self, parsed_document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured knowledge from parsed document"""
        
        logger.info("Extracting knowledge from document")
        
        # Combine relevant text
        relevant_text = self._prepare_text_for_extraction(parsed_document)
        
        # Extract different types of entities
        entities = {
            "products": await self._extract_products(relevant_text, parsed_document),
            "compliance": await self._extract_compliance(relevant_text),
            "relationships": await self._extract_relationships(relevant_text, parsed_document)
        }
        
        # Post-process and validate
        entities = self._validate_entities(entities)
        
        return entities
        
    def _prepare_text_for_extraction(self, parsed_document: Dict[str, Any]) -> str:
        """Prepare text for LLM extraction"""
        
        text_parts = []
        
        # Add text from relevant sections
        relevant_sections = ["overview", "specifications", "compatibility", "compliance"]
        
        for section in relevant_sections:
            if section in parsed_document["sections"]:
                for page_num in parsed_document["sections"][section][:3]:  # Limit pages
                    for page_data in parsed_document["text_content"]:
                        if page_data["page"] == page_num:
                            text_parts.append(f"\n[{section.upper()}]\n{page_data['text'][:2000]}")
                            
        # Add table data
        for table in parsed_document["tables"][:5]:  # Limit tables
            text_parts.append(f"\n[TABLE: {table['type']}]\n")
            for row in table["data"][:20]:  # Limit rows
                text_parts.append(str(row))
                
        return "\n".join(text_parts)[:8000]  # Limit total length
        
    async def _extract_products(self, text: str, parsed_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract product entities"""
        
        # Start with products found by parser
        products = parsed_document.get("products", [])
        
        # Enhance with LLM extraction
        prompt = f"""
        Extract all fire alarm products from this document.
        
        For each product, extract:
        {json.dumps(self.extraction_schemas["product"], indent=2)}
        
        Document text:
        {text}
        
        Return as JSON array of products.
        """
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a fire alarm systems expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            llm_products = json.loads(response.choices[0].message.content).get("products", [])
            
            # Merge with parsed products
            product_map = {p["code"]: p for p in products}
            
            for llm_product in llm_products:
                code = llm_product.get("code")
                if code:
                    if code in product_map:
                        # Merge information
                        product_map[code].update(llm_product)
                    else:
                        product_map[code] = llm_product
                        
            return list(product_map.values())
            
        except Exception as e:
            logger.error("Failed to extract products", error=str(e))
            return products
            
    async def _extract_compliance(self, text: str) -> List[Dict[str, Any]]:
        """Extract compliance standards"""
        
        # First try regex patterns
        compliance_list = []
        patterns = [
            r'(UL\s*\d+[A-Z]?)',
            r'(NFPA\s*\d+)',
            r'(EN\s*\d+)',
            r'(FM\s*\d+)',
            r'(CSFM\s*\d+-\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.I)
            compliance_list.extend(matches)
            
        # Enhance with LLM
        prompt = f"""
        Extract all compliance standards and certifications mentioned.
        
        For each standard, extract:
        {json.dumps(self.extraction_schemas["compliance"], indent=2)}
        
        Text:
        {text[:3000]}
        
        Return as JSON array.
        """
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Extract compliance standards."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            llm_compliance = json.loads(response.choices[0].message.content).get("standards", [])
            
            # Merge and deduplicate
            compliance_map = {}
            
            for standard in compliance_list:
                compliance_map[standard] = {"standard": standard}
                
            for llm_std in llm_compliance:
                std_name = llm_std.get("standard", "")
                if std_name:
                    compliance_map[std_name] = llm_std
                    
            return list(compliance_map.values())
            
        except Exception as e:
            logger.error("Failed to extract compliance", error=str(e))
            return [{"standard": std} for std in set(compliance_list)]
            
    async def _extract_relationships(self, text: str, parsed_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships between products"""
        
        relationships = []
        
        # Extract from compatibility tables
        for table in parsed_document["tables"]:
            if table["type"] == "compatibility":
                relationships.extend(self._parse_compatibility_table(table))
                
        # Extract using LLM
        prompt = f"""
        Extract compatibility relationships between products.
        
        For each relationship, extract:
        {json.dumps(self.extraction_schemas["compatibility"], indent=2)}
        
        Focus on:
        - Which products are compatible with each other
        - Which components are required together
        - Any restrictions or limitations
        
        Text:
        {text[:3000]}
        
        Return as JSON array.
        """
        
        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Extract product compatibility relationships."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            llm_relationships = json.loads(response.choices[0].message.content).get("relationships", [])
            relationships.extend(llm_relationships)
            
        except Exception as e:
            logger.error("Failed to extract relationships", error=str(e))
            
        return relationships
        
    def _parse_compatibility_table(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse compatibility information from table"""
        
        relationships = []
        data = table["data"]
        
        if not data:
            return relationships
            
        # Try to identify product columns
        for i, row in enumerate(data):
            for j, col in enumerate(row.keys()):
                if i == j:
                    continue
                    
                value = str(row[col]).strip().upper()
                if value in ["Y", "YES", "COMPATIBLE", "✓"]:
                    relationships.append({
                        "product_a": list(row.values())[0],  # Assuming first col is product
                        "product_b": col,
                        "relationship": "compatible",
                        "source": "compatibility_table"
                    })
                    
        return relationships
        
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted entities"""
        
        # Validate product codes
        valid_products = []
        for product in entities.get("products", []):
            if self._is_valid_product_code(product.get("code", "")):
                valid_products.append(product)
                
        entities["products"] = valid_products
        
        # Validate relationships
        valid_rels = []
        valid_codes = {p["code"] for p in valid_products}
        
        for rel in entities.get("relationships", []):
            if (rel.get("product_a") in valid_codes or 
                rel.get("product_b") in valid_codes):
                valid_rels.append(rel)
                
        entities["relationships"] = valid_rels
        
        return entities
        
    def _is_valid_product_code(self, code: str) -> bool:
        """Check if product code matches expected pattern"""
        if not code:
            return False
            
        # Simplex product codes typically follow patterns like:
        # ABC-123, AB-1234, ABC-123D
        pattern = re.compile(r'^[A-Z]{2,4}-?\d{3,4}[A-Z]?)
        return bool(pattern.match(code.strip().upper()))

# =====================================
# File: src/ingestion/graph_loader.py
# =====================================
from typing import List, Dict, Any
import asyncio
from datetime import datetime

from src.utils.neo4j_client import Neo4jClient
from src.utils.logger import logger

class GraphLoader:
    """Load extracted knowledge into Neo4j graph database"""
    
    def __init__(self):
        self.neo4j_client = Neo4jClient()
        
    async def load_knowledge(self, knowledge: Dict[str, Any], source_document: str):
        """Load extracted knowledge into graph"""
        
        logger.info("Loading knowledge into graph", source=source_document)
        
        # Create indexes first
        self.neo4j_client.create_indexes()
        
        # Load entities
        await self._load_products(knowledge.get("products", []), source_document)
        await self._load_compliance(knowledge.get("compliance", []))
        
        # Load relationships
        await self._load_relationships(knowledge.get("relationships", []))
        
        # Create document node
        await self._create_document_node(source_document, knowledge)
        
        logger.info("Knowledge loading completed")
        
    async def _load_products(self, products: List[Dict[str, Any]], source: str):
        """Load product nodes"""
        
        nodes = []
        for product in products:
            node = {
                "code": product["code"],
                "name": product.get("name", product["code"]),
                "description": product.get("description", ""),
                "category": product.get("category", "Unknown"),
                "price": product.get("price", 0),
                "source_document": source,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Add specifications as properties
            if "specifications" in product:
                for key, value in product["specifications"].items():
                    node[f"spec_{key}"] = value
                    
            # Add compliance as array property
            if "compliance" in product:
                node["compliance_standards"] = product["compliance"]
                
            nodes.append(node)
            
        if nodes:
            await self.neo4j_client.bulk_create_nodes(nodes, "Product")
            
    async def _load_compliance(self, compliance_standards: List[Dict[str, Any]]):
        """Load compliance standard nodes"""
        
        nodes = []
        for standard in compliance_standards:
            node = {
                "standard": standard["standard"],
                "version": standard.get("version", ""),
                "description": standard.get("description", ""),
                "created_at": datetime.utcnow().isoformat()
            }
            nodes.append(node)
            
        if nodes:
            await self.neo4j_client.bulk_create_nodes(nodes, "ComplianceStandard")
            
    async def _load_relationships(self, relationships: List[Dict[str, Any]]):
        """Load relationships between entities"""
        
        # Group by relationship type
        compatibility_rels = []
        requires_rels = []
        complies_with_rels = []
        
        for rel in relationships:
            rel_type = rel.get("relationship", "").lower()
            
            if "compat" in rel_type:
                compatibility_rels.append({
                    "from_label": "Product",
                    "to_label": "Product",
                    "from_code": rel["product_a"],
                    "to_code": rel["product_b"],
                    "type": "COMPATIBLE_WITH",
                    "properties": {"notes": rel.get("notes", "")}
                })
            elif "require" in rel_type:
                requires_rels.append({
                    "from_label": "Product",
                    "to_label": "Product",
                    "from_code": rel["product_a"],
                    "to_code": rel["product_b"],
                    "type": "REQUIRES",
                    "properties": {"notes": rel.get("notes", "")}
                })
                
        # Load relationships
        if compatibility_rels:
            await self.neo4j_client.bulk_create_relationships(compatibility_rels)
        if requires_rels:
            await self.neo4j_client.bulk_create_relationships(requires_rels)
            
    async def _create_document_node(self, source_document: str, knowledge: Dict[str, Any]):
        """Create a document node to track source"""
        
        async with self.neo4j_client.async_session() as session:
            query = """
            CREATE (d:Document {
                name: $name,
                processed_at: $processed_at,
                product_count: $product_count,
                compliance_count: $compliance_count,
                relationship_count: $relationship_count
            })
            """
            
            await session.run(
                query,
                name=source_document,
                processed_at=datetime.utcnow().isoformat(),
                product_count=len(knowledge.get("products", [])),
                compliance_count=len(knowledge.get("compliance", [])),
                relationship_count=len(knowledge.get("relationships", []))
            )

# =====================================
# File: src/scripts/collect_examples.py
# =====================================
import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path

from src.core.ttt_adapter import TTTExample
from src.utils.config import settings
from src.utils.logger import logger

async def collect_initial_examples():
    """Collect initial TTT examples from user"""
    
    logger.info("Starting TTT example collection")
    
    print("\n" + "="*60)
    print("TTT-Enhanced BYOKG-RAG System - Initial Setup")
    print("="*60)
    print("\nThe system needs example BOQs to train its Test-Time Training adapter.")
    print("Please provide 10 example queries with their expected outputs.\n")
    
    examples = []
    
    # Check if examples file exists
    examples_file = settings.ttt_examples_dir / "initial_examples.json"
    if examples_file.exists():
        print(f"Found existing examples at {examples_file}")
        use_existing = input("Use existing examples? (y/n): ").lower() == 'y'
        
        if use_existing:
            with open(examples_file, 'r') as f:
                loaded_examples = json.load(f)
                for ex in loaded_examples:
                    examples.append(TTTExample(**ex))
                print(f"Loaded {len(examples)} examples")
                return examples
    
    # Collect new examples
    print("\nPlease provide examples in the following format:")
    print("- Query: The project description/requirements")
    print("- Entities: Key entities mentioned (products, locations, requirements)")
    print("- BOQ Items: Expected products in the bill of quantities\n")
    
    for i in range(10):
        print(f"\n--- Example {i+1}/10 ---")
        
        # Get query
        query = input("Query: ").strip()
        if not query:
            print("Query cannot be empty. Please try again.")
            i -= 1
            continue
        
        # Get entities (simplified for user input)
        entities_str = input("Key entities (comma-separated): ").strip()
        entities = []
        if entities_str:
            for entity in entities_str.split(','):
                entity = entity.strip()
                # Determine entity type
                entity_type = "product_code" if re.match(r'[A-Z]{2,4}-?\d{3,4}', entity) else "requirement"
                entities.append({
                    "type": entity_type,
                    "value": entity,
                    "source": "user_input"
                })
        
        # Get expected products
        products_str = input("Expected product codes (comma-separated): ").strip()
        boq_items = []
        if products_str:
            for product in products_str.split(','):
                product = product.strip()
                boq_items.append({
                    "product_code": product,
                    "quantity": 1  # Default quantity
                })
        
        # Create example
        example = TTTExample(
            query=query,
            entities=entities,
            relationships=[],  # Simplified for initial collection
            boq_items=boq_items,
            metadata={"source": "user_input", "index": i}
        )
        
        examples.append(example)
        
        # Save progress
        if (i + 1) % 3 == 0:
            save_examples(examples, examples_file)
            print(f"\nProgress saved. {i+1} examples collected.")
    
    # Final save
    save_examples(examples, examples_file)
    print(f"\n✅ Successfully collected {len(examples)} examples!")
    print(f"Examples saved to: {examples_file}")
    
    return examples

def save_examples(examples: List[TTTExample], filepath: Path):
    """Save examples to file"""
    
    examples_data = []
    for ex in examples:
        examples_data.append({
            "query": ex.query,
            "entities": ex.entities,
            "relationships": ex.relationships,
            "boq_items": ex.boq_items,
            "metadata": ex.metadata
        })
    
    with open(filepath, 'w') as f:
        json.dump(examples_data, f, indent=2)

# Default examples if user doesn't want to provide
DEFAULT_EXAMPLES = [
    {
        "query": "I need a fire alarm system for a 3-story office building with 50 rooms",
        "entities": [
            {"type": "floor_count", "value": "3", "source": "default"},
            {"type": "room_count", "value": "50", "source": "default"},
            {"type": "building_type", "value": "office", "source": "default"}
        ],
        "relationships": [],
        "boq_items": [
            {"product_code": "CP-100", "quantity": 1},
            {"product_code": "SD-200", "quantity": 50},
            {"product_code": "MS-300", "quantity": 6}
        ],
        "metadata": {"source": "default", "index": 0}
    },
    # Add more default examples as needed
]

if __name__ == "__main__":
    asyncio.run(collect_initial_examples())

# =====================================
# File: src/scripts/run_ingestion.py
# =====================================
import asyncio
from pathlib import Path
import sys

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.knowledge_extractor import KnowledgeExtractor
from src.ingestion.graph_loader import GraphLoader
from src.utils.s3_manager import S3Manager
from src.utils.config import settings
from src.utils.logger import logger

async def run_ingestion_pipeline():
    """Run the complete document ingestion pipeline"""
    
    logger.info("Starting document ingestion pipeline")
    
    # Initialize components
    s3_manager = S3Manager()
    pdf_parser = PDFParser()
    knowledge_extractor = KnowledgeExtractor()
    graph_loader = GraphLoader()
    
    # Get list of documents from S3
    document_keys = await s3_manager.list_documents()
    logger.info(f"Found {len(document_keys)} documents in S3")
    
    # Process each document
    for doc_key in document_keys:
        if not doc_key.endswith('.pdf'):
            continue
            
        try:
            logger.info(f"Processing document: {doc_key}")
            
            # Download document
            local_path = Path(f"./data/downloads/{Path(doc_key).name}")
            content = await s3_manager.download_document(doc_key, local_path)
            
            if not content:
                logger.error(f"Failed to download {doc_key}")
                continue
            
            # Parse PDF
            parsed_doc = await pdf_parser.parse_document(local_path)
            
            # Extract knowledge
            knowledge = await knowledge_extractor.extract_knowledge(parsed_doc)
            
            # Load into graph
            await graph_loader.load_knowledge(knowledge, doc_key)
            
            logger.info(f"Successfully processed {doc_key}")
            
        except Exception as e:
            logger.error(f"Failed to process {doc_key}", error=str(e))
            continue
    
    logger.info("Ingestion pipeline completed")

if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())

# =====================================
# File: src/scripts/run_api.py
# =====================================
import uvicorn
from src.utils.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True
    )

# =====================================
# File: docker-compose.yml
# =====================================
"""
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV_FILE=.env
    volumes:
      - ./data:/app/data
    depends_on:
      - neo4j
    networks:
      - ttt-network

  neo4j:
    image: neo4j:5.14
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your-password
      - NEO4J_dbms_memory_pagecache_size=1G
      - NEO4J_dbms_memory_heap_max__size=1G
    volumes:
      - neo4j_data:/data
    networks:
      - ttt-network

volumes:
  neo4j_data:

networks:
  ttt-network:
    driver: bridge
"""

# =====================================
# File: Dockerfile
# =====================================
"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/adapters data/examples data/cache data/downloads

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "scripts/run_api.py"]
"""

# =====================================
# File: README.md
# =====================================
"""
# TTT-Enhanced BYOKG-RAG System

A Knowledge Graph-based Retrieval Augmented Generation (RAG) system enhanced with Test-Time Training (TTT) for Simplex fire alarm products.

## Features

- **Test-Time Training (TTT)**: Adapts to new queries by fine-tuning on task-specific examples
- **Multi-Strategy Graph Retrieval**: Multiple traversal strategies for comprehensive information retrieval
- **Microservices Architecture**: Clean separation of concerns with modular components
- **Production Ready**: Docker support, health checks, and monitoring

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ttt-enhanced-byokg-rag
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run ingestion**
   ```bash
   python scripts/run_ingestion.py
   ```

5. **Start API**
   ```bash
   python scripts/run_api.py
   ```

## API Endpoints

- `POST /api/v1/generate_boq` - Generate Bill of Quantities
- `POST /api/v1/submit_examples` - Submit TTT training examples
- `GET /api/v1/graph/stats` - Get graph statistics
- `POST /api/v1/graph/search` - Search the knowledge graph
- `GET /api/v1/health` - Health check

## Docker Deployment

```bash
docker-compose up --build
```

## TTT Integration

The system will prompt for 10 example BOQs on first run. These examples are used to train task-specific LoRA adapters that improve entity recognition and relationship extraction.

## Architecture

- **Core**: TTT adapter, KG linker, graph retriever, orchestrator
- **Ingestion**: PDF parser, knowledge extractor, graph loader
- **API**: FastAPI application with async endpoints
- **Storage**: Neo4j graph database, AWS S3 for documents

## License

This project implements the BYOKG-RAG framework for educational and research purposes.
"""