#!/usr/bin/env python3
"""
TTT Dataset Processor
Processes dataset directory from S3 and converts project files to JSON for TTT inference
"""

import asyncio
from pathlib import Path
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import settings, logger, ensure_data_dirs
from src.ingestion.dataset_processor import DatasetProcessor, ProjectSet

class TTTInferenceSampleGenerator:
    """Generate TTT inference samples from processed projects"""
    
    def __init__(self):
        self.dataset_processor = DatasetProcessor()
    
    def generate_inference_samples(self, project_sets: List[ProjectSet]) -> List[Dict[str, Any]]:
        """Generate TTT inference samples from project sets"""
        samples = []
        
        for project in project_sets:
            if not project:
                continue
                
            # Generate different types of inference samples
            
            # 1. Specification-based query samples
            if project.specs_doc:
                spec_samples = self._generate_spec_samples(project)
                samples.extend(spec_samples)
            
            # 2. BOQ-based query samples  
            if project.boq_doc:
                boq_samples = self._generate_boq_samples(project)
                samples.extend(boq_samples)
            
            # 3. Offer-based query samples
            if project.offer_doc:
                offer_samples = self._generate_offer_samples(project)
                samples.extend(offer_samples)
            
            # 4. Cross-document correlation samples
            if project.specs_doc and project.boq_doc:
                correlation_samples = self._generate_correlation_samples(project)
                samples.extend(correlation_samples)
        
        logger.info("Generated TTT inference samples", total=len(samples))
        return samples
    
    def _generate_spec_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate samples based on specifications"""
        samples = []
        specs_doc = project.specs_doc
        
        # Extract key requirements and standards
        requirements = [e for e in specs_doc.entities if e['type'] in ['requirement', 'standard', 'compliance']]
        products = [e for e in specs_doc.entities if e['type'] == 'product_code']
        
        # Sample 1: Requirements-based query
        if requirements:
            sample = {
                'query_type': 'specification_requirements',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"What are the key requirements for a fire alarm system similar to {project.project_name}?",
                'context': {
                    'document_type': 'specifications',
                    'key_requirements': [r['name'] for r in requirements[:5]],
                    'content_preview': specs_doc.content[:1000]
                },
                'expected_entities': [r['name'] for r in requirements],
                'expected_products': [p['name'] for p in products[:10]],
                'metadata': {
                    'source_file': specs_doc.filename,
                    'entity_count': len(specs_doc.entities),
                    'confidence_level': 'high'
                }
            }
            samples.append(sample)
        
        # Sample 2: Standards compliance query
        standards = [e for e in specs_doc.entities if e['type'] == 'standard']
        if standards:
            sample = {
                'query_type': 'standards_compliance',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"Which standards and codes need to be followed for {project.project_name}?",
                'context': {
                    'document_type': 'specifications',
                    'standards_mentioned': [s['name'] for s in standards],
                    'content_preview': specs_doc.content[:1000]
                },
                'expected_entities': [s['name'] for s in standards],
                'metadata': {
                    'source_file': specs_doc.filename,
                    'standards_count': len(standards)
                }
            }
            samples.append(sample)
        
        return samples
    
    def _generate_boq_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate samples based on BOQ"""
        samples = []
        boq_doc = project.boq_doc
        
        quantities = [e for e in boq_doc.entities if e['type'] == 'quantity']
        money_values = [e for e in boq_doc.entities if e['type'] == 'money']
        products = [e for e in boq_doc.entities if e['type'] == 'product_code']
        
        # Sample 1: Quantity estimation query
        if quantities and products:
            sample = {
                'query_type': 'quantity_estimation',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"What quantities of fire alarm components are needed for a project like {project.project_name}?",
                'context': {
                    'document_type': 'boq',
                    'key_products': [p['name'] for p in products[:10]],
                    'quantities_found': [q['name'] for q in quantities[:5]],
                    'content_preview': boq_doc.content[:1000]
                },
                'expected_entities': [p['name'] for p in products] + [q['name'] for q in quantities],
                'expected_output': {
                    'type': 'boq_suggestion',
                    'products_with_quantities': [(p['name'], 'estimated_quantity') for p in products[:5]]
                },
                'metadata': {
                    'source_file': boq_doc.filename,
                    'product_count': len(products),
                    'quantity_references': len(quantities)
                }
            }
            samples.append(sample)
        
        # Sample 2: Cost estimation query
        if money_values:
            sample = {
                'query_type': 'cost_estimation',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"What is the estimated cost breakdown for {project.project_name}?",
                'context': {
                    'document_type': 'boq',
                    'cost_values': [m['name'] for m in money_values[:5]],
                    'content_preview': boq_doc.content[:1000]
                },
                'expected_entities': [m['name'] for m in money_values],
                'metadata': {
                    'source_file': boq_doc.filename,
                    'cost_references': len(money_values)
                }
            }
            samples.append(sample)
        
        return samples
    
    def _generate_offer_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate samples based on offers/proposals"""
        samples = []
        offer_doc = project.offer_doc
        
        products = [e for e in offer_doc.entities if e['type'] == 'product_code']
        money_values = [e for e in offer_doc.entities if e['type'] == 'money']
        
        if products and money_values:
            sample = {
                'query_type': 'offer_analysis',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"Analyze the technical offer for {project.project_name} and suggest similar solutions",
                'context': {
                    'document_type': 'offer',
                    'offered_products': [p['name'] for p in products[:10]],
                    'pricing_info': [m['name'] for m in money_values[:3]],
                    'content_preview': offer_doc.content[:1000]
                },
                'expected_entities': [p['name'] for p in products],
                'expected_output': {
                    'type': 'technical_offer',
                    'recommended_products': [p['name'] for p in products[:5]],
                    'price_range': money_values[0]['name'] if money_values else 'TBD'
                },
                'metadata': {
                    'source_file': offer_doc.filename,
                    'product_count': len(products)
                }
            }
            samples.append(sample)
        
        return samples
    
    def _generate_correlation_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate samples that correlate specs with BOQ"""
        samples = []
        
        specs_products = [e for e in project.specs_doc.entities if e['type'] == 'product_code']
        boq_products = [e for e in project.boq_doc.entities if e['type'] == 'product_code']
        
        # Find common products
        specs_product_names = {p['name'] for p in specs_products}
        boq_product_names = {p['name'] for p in boq_products}
        common_products = list(specs_product_names & boq_product_names)
        
        if common_products:
            sample = {
                'query_type': 'spec_boq_correlation',
                'project_id': project.project_id,
                'project_name': project.project_name,
                'query': f"Generate a BOQ that matches the specifications for {project.project_name}",
                'context': {
                    'document_types': ['specifications', 'boq'],
                    'spec_requirements': project.specs_doc.content[:500],
                    'boq_structure': project.boq_doc.content[:500],
                    'common_products': common_products[:5]
                },
                'expected_entities': common_products,
                'expected_output': {
                    'type': 'compliant_boq',
                    'products': common_products,
                    'compliance_notes': 'Generated BOQ should match specification requirements'
                },
                'metadata': {
                    'source_files': [project.specs_doc.filename, project.boq_doc.filename],
                    'correlation_strength': len(common_products) / max(len(specs_products), len(boq_products), 1)
                }
            }
            samples.append(sample)
        
        return samples
    
    def export_inference_samples(self, samples: List[Dict[str, Any]], output_file: Path) -> bool:
        """Export inference samples to JSON file"""
        try:
            inference_data = {
                'ttt_inference_samples': {
                    'version': '1.0',
                    'total_samples': len(samples),
                    'sample_types': list(set(s['query_type'] for s in samples)),
                    'generated_at': pd.Timestamp.now().isoformat()
                },
                'samples': samples
            }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("TTT inference samples exported", 
                       file=str(output_file), 
                       samples=len(samples))
            
            return True
            
        except Exception as e:
            logger.error("Failed to export inference samples", error=str(e))
            return False

async def main():
    """Main dataset processing execution"""
    print("=== TTT Dataset Processor ===")
    print(f"S3 Bucket: {settings.s3_bucket_name}")
    print()
    
    # Ensure data directories
    ensure_data_dirs()
    
    # Initialize processors
    dataset_processor = DatasetProcessor()
    sample_generator = TTTInferenceSampleGenerator()
    
    try:
        # Step 1: Discover dataset structure
        print("Step 1: Discovering dataset structure...")
        projects_structure = dataset_processor.discover_dataset_structure("dataset/")
        
        if not projects_structure:
            print("No dataset directory found or no files in dataset/")
            return
        
        print(f"Found {len(projects_structure)} project directories")
        
        # Step 2: Process each project
        print("\nStep 2: Processing project files...")
        project_sets = []
        
        for project_name, file_objects in projects_structure.items():
            print(f"Processing project: {project_name} ({len(file_objects)} files)")
            
            project_set = await dataset_processor.process_project_files(project_name, file_objects)
            if project_set:
                project_sets.append(project_set)
        
        print(f"Successfully processed {len(project_sets)} projects")
        
        # Step 3: Export project data to JSON
        print("\nStep 3: Exporting project data...")
        projects_json_file = Path("./data/processed/projects_dataset.json")
        dataset_processor.export_to_json(project_sets, projects_json_file)
        
        # Step 4: Generate TTT inference samples
        print("\nStep 4: Generating TTT inference samples...")
        inference_samples = sample_generator.generate_inference_samples(project_sets)
        
        # Step 5: Export inference samples
        print("\nStep 5: Exporting TTT inference samples...")
        samples_json_file = Path("./data/processed/ttt_inference_samples.json")
        sample_generator.export_inference_samples(inference_samples, samples_json_file)
        
        # Summary
        print("\n=== Processing Complete ===")
        print(f"Projects processed: {len(project_sets)}")
        print(f"TTT inference samples: {len(inference_samples)}")
        print(f"Project data: {projects_json_file}")
        print(f"Inference samples: {samples_json_file}")
        
        # Show sample statistics
        sample_types = {}
        for sample in inference_samples:
            query_type = sample['query_type']
            sample_types[query_type] = sample_types.get(query_type, 0) + 1
        
        print("\nSample types distribution:")
        for sample_type, count in sample_types.items():
            print(f"  {sample_type}: {count}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        logger.error("Dataset processing failed", error=str(e))

if __name__ == "__main__":
    asyncio.run(main())