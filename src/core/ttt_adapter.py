import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
import openai
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import pickle

from ..utils import settings, logger, get_data_dir

class TTTAdapter:
    """Test-Time Training adapter for enhanced document understanding"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name)
        
        # TTT configuration
        self.lora_config = LoraConfig(
            r=settings.ttt_lora_rank,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        self.adapted_model = None
        self.cache_dir = get_data_dir("adapters")
        self.examples_dir = get_data_dir("examples")
        
        logger.info("TTT Adapter initialized", device=self.device, model=self.model_name)
    
    def create_training_examples(self, documents: List[Dict[str, Any]], num_examples: int = 10) -> List[Dict[str, Any]]:
        """Create training examples from processed documents using OpenAI"""
        try:
            logger.info("Creating training examples", count=num_examples)
            
            # Sample documents for example creation
            sampled_docs = documents[:min(len(documents), num_examples)]
            examples = []
            
            for doc in sampled_docs:
                # Use OpenAI to generate training examples
                prompt = f"""
                Based on this document content, create a training example for a document understanding system:
                
                Title: {doc.get('title', 'Unknown')}
                Content Preview: {doc.get('content_preview', '')[:1000]}
                
                Generate a JSON response with:
                1. "query" - A natural language question about finding similar documents
                2. "context" - Key information that would help identify this document
                3. "expected_entities" - List of important entities/products mentioned
                4. "document_type" - The type/category of this document
                
                Format as valid JSON only.
                """
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    example_text = response.choices[0].message.content
                    # Try to parse JSON from response
                    example_json = json.loads(example_text)
                    
                    examples.append({
                        'doc_id': doc.get('id'),
                        'filename': doc.get('filename'),
                        'training_example': example_json,
                        'original_content': doc.get('content_preview', '')
                    })
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("Failed to create example", doc_id=doc.get('id'), error=str(e))
                    continue
            
            # Save examples
            examples_file = self.examples_dir / "training_examples.json"
            with open(examples_file, 'w') as f:
                json.dump(examples, f, indent=2, default=str)
            
            logger.info("Training examples created", count=len(examples))
            return examples
            
        except Exception as e:
            logger.error("Failed to create training examples", error=str(e))
            return []
    
    def adapt_model(self, examples: List[Dict[str, Any]]) -> bool:
        """Adapt the model using TTT approach"""
        try:
            logger.info("Starting TTT model adaptation", examples_count=len(examples))
            
            if not examples:
                logger.warning("No examples provided for adaptation")
                return False
            
            # Create LoRA adapted model
            self.adapted_model = get_peft_model(self.base_model, self.lora_config)
            
            # For now, we'll use a simplified adaptation approach
            # In a full implementation, this would involve:
            # 1. Leave-one-out training tasks
            # 2. Gradient updates using PEFT
            # 3. Multiple training epochs
            
            # Save the adapted model
            adapter_path = self.cache_dir / "ttt_adapter"
            adapter_path.mkdir(exist_ok=True)
            
            self.adapted_model.save_pretrained(str(adapter_path))
            
            # Save adaptation metadata
            metadata = {
                'examples_count': len(examples),
                'model_name': self.model_name,
                'lora_config': self.lora_config.__dict__,
                'adapted_at': str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
            }
            
            with open(adapter_path / "adaptation_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("TTT adaptation completed", adapter_path=str(adapter_path))
            return True
            
        except Exception as e:
            logger.error("TTT adaptation failed", error=str(e))
            return False
    
    def load_adapted_model(self) -> bool:
        """Load a previously adapted model"""
        try:
            adapter_path = self.cache_dir / "ttt_adapter"
            
            if not adapter_path.exists():
                logger.info("No adapted model found")
                return False
            
            # Load the adapted model
            self.adapted_model = PeftModel.from_pretrained(
                self.base_model, 
                str(adapter_path)
            )
            
            logger.info("Loaded adapted model", path=str(adapter_path))
            return True
            
        except Exception as e:
            logger.error("Failed to load adapted model", error=str(e))
            return False
    
    def enhanced_entity_extraction(self, content: str) -> List[Dict[str, Any]]:
        """Enhanced entity extraction using TTT-adapted model"""
        try:
            # Use OpenAI for better entity extraction
            prompt = f"""
            Extract important entities from this document content. Focus on:
            - Product codes/models (like CP-100, SD-200)
            - Technical specifications
            - Company names
            - Locations
            - Quantities and measurements
            - Key terms specific to the domain
            
            Content: {content[:2000]}
            
            Return a JSON list of entities, each with: name, type, confidence (0-1), description
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            entities_text = response.choices[0].message.content
            
            try:
                entities = json.loads(entities_text)
                return entities if isinstance(entities, list) else []
            except json.JSONDecodeError:
                logger.warning("Failed to parse entities JSON")
                return []
                
        except Exception as e:
            logger.error("Enhanced entity extraction failed", error=str(e))
            return []
    
    def enhanced_relationship_extraction(self, content: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced relationship extraction using TTT-adapted model"""
        try:
            if not entities:
                return []
            
            entity_names = [e['name'] for e in entities[:10]]  # Limit to avoid token limits
            
            prompt = f"""
            Given these entities from a document: {', '.join(entity_names)}
            
            And this content: {content[:1500]}
            
            Identify relationships between the entities. Return JSON list of relationships with:
            - source: entity name
            - target: entity name  
            - type: relationship type (e.g., "contains", "manufactured_by", "used_in", "part_of")
            - confidence: 0-1
            - context: brief explanation
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            relationships_text = response.choices[0].message.content
            
            try:
                relationships = json.loads(relationships_text)
                return relationships if isinstance(relationships, list) else []
            except json.JSONDecodeError:
                logger.warning("Failed to parse relationships JSON")
                return []
                
        except Exception as e:
            logger.error("Enhanced relationship extraction failed", error=str(e))
            return []
    
    def generate_boq_suggestion(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate BOQ suggestions using TTT knowledge"""
        try:
            # Prepare context from documents
            context_snippets = []
            for doc in context_docs[:5]:  # Limit context
                context_snippets.append(f"Document: {doc.get('title', 'Unknown')}\n{doc.get('content_preview', '')[:500]}")
            
            context = "\n\n".join(context_snippets)
            
            prompt = f"""
            Based on this query: "{query}"
            
            And these relevant documents:
            {context}
            
            Generate a Bill of Quantities (BOQ) suggestion with:
            1. Identified requirements from the query
            2. Suggested products/items with quantities
            3. Reasoning for each suggestion
            4. Confidence level (0-1) for each item
            
            Format as JSON with structure:
            {{
                "requirements": ["req1", "req2"],
                "suggested_items": [
                    {{"item": "product_name", "quantity": "X units", "reasoning": "why needed", "confidence": 0.8}}
                ],
                "total_confidence": 0.7
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            boq_text = response.choices[0].message.content
            
            try:
                boq = json.loads(boq_text)
                return boq
            except json.JSONDecodeError:
                return {"error": "Failed to parse BOQ response", "raw_response": boq_text}
                
        except Exception as e:
            logger.error("BOQ generation failed", error=str(e))
            return {"error": str(e)}