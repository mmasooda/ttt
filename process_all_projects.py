#!/usr/bin/env python3
"""
Process all project directories in S3 bucket for TTT inference samples
"""

import asyncio
from pathlib import Path
import sys
import json
import pandas as pd
from typing import List, Dict, Any
import re

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import settings, logger, ensure_data_dirs, S3Client
from src.ingestion.dataset_processor import DatasetProcessor, ProjectSet

class ProjectDatasetProcessor:
    """Process all project data from S3 bucket"""
    
    def __init__(self):
        self.s3_client = S3Client()
        self.dataset_processor = DatasetProcessor()
    
    def discover_all_projects(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all project data across different directories"""
        all_projects = {}
        
        # 1. Process root dataset directory
        dataset_files = self.s3_client.list_all_objects("dataset/")
        if dataset_files:
            # Group dataset files by naming pattern
            dataset_projects = self._group_dataset_files(dataset_files)
            all_projects.update(dataset_projects)
        
        # 2. Process enquiries directory
        enquiry_files = self.s3_client.list_all_objects("enquiries/")
        if enquiry_files:
            enquiry_projects = self._group_enquiry_files(enquiry_files, "enquiries/")
            all_projects.update(enquiry_projects)
        
        # 3. Process enq 11 to 20 directory
        enq1120_files = self.s3_client.list_all_objects("enq 11 to 20/")
        if enq1120_files:
            enq1120_projects = self._group_enquiry_files(enq1120_files, "enq 11 to 20/")
            all_projects.update(enq1120_projects)
        
        logger.info("Discovered all projects", total_projects=len(all_projects))
        return all_projects
    
    def _group_dataset_files(self, files: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group dataset files by project number/name"""
        projects = {}
        
        for file_obj in files:
            key = file_obj['key']
            filename = Path(key).name
            
            # Extract project identifier from filename
            # Look for patterns like "10th BOQ.xlsx", "1st Specs.pdf", etc.
            match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+', filename, re.IGNORECASE)
            if match:
                project_num = match.group(1)
                project_id = f"dataset_project_{project_num}"
            else:
                # Fallback: use first word or full filename
                project_id = f"dataset_{filename.split('.')[0]}"
            
            if project_id not in projects:
                projects[project_id] = []
            projects[project_id].append(file_obj)
        
        return projects
    
    def _group_enquiry_files(self, files: List[Dict[str, Any]], base_prefix: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group enquiry files by subdirectory"""
        projects = {}
        
        for file_obj in files:
            key = file_obj['key']
            # Remove base prefix and get directory structure
            relative_path = key.replace(base_prefix, '')
            path_parts = relative_path.split('/')
            
            if len(path_parts) > 1:
                # File is in subdirectory
                subdir = path_parts[0]
                project_id = f"{base_prefix.rstrip('/')}/{subdir}"
                
                if project_id not in projects:
                    projects[project_id] = []
                projects[project_id].append(file_obj)
        
        return projects
    
    async def process_all_projects(self) -> List[ProjectSet]:
        """Process all discovered projects"""
        # Discover all projects
        all_projects = self.discover_all_projects()
        
        project_sets = []
        total_projects = len(all_projects)
        
        print(f"Processing {total_projects} projects...")
        
        for i, (project_name, file_objects) in enumerate(all_projects.items(), 1):
            print(f"[{i}/{total_projects}] Processing: {project_name} ({len(file_objects)} files)")
            
            try:
                project_set = await self.dataset_processor.process_project_files(project_name, file_objects)
                if project_set:
                    project_sets.append(project_set)
                    print(f"  ✓ Success - Specs: {'✓' if project_set.specs_doc else '✗'}, BOQ: {'✓' if project_set.boq_doc else '✗'}, Offer: {'✓' if project_set.offer_doc else '✗'}")
                else:
                    print(f"  ✗ Failed to process project")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                logger.error("Project processing failed", project=project_name, error=str(e))
        
        print(f"\nSuccessfully processed {len(project_sets)} out of {total_projects} projects")
        return project_sets
    
    def generate_comprehensive_samples(self, project_sets: List[ProjectSet]) -> List[Dict[str, Any]]:
        """Generate comprehensive TTT inference samples"""
        samples = []
        
        for project in project_sets:
            if not project:
                continue
            
            # Generate various sample types for each project
            project_samples = []
            
            # 1. Document-specific samples
            if project.specs_doc:
                project_samples.extend(self._generate_spec_samples(project))
            
            if project.boq_doc:
                project_samples.extend(self._generate_boq_samples(project))
            
            if project.offer_doc:
                project_samples.extend(self._generate_offer_samples(project))
            
            # 2. Cross-document samples
            if len([d for d in [project.specs_doc, project.boq_doc, project.offer_doc] if d]) >= 2:
                project_samples.extend(self._generate_cross_document_samples(project))
            
            # 3. Project summary samples
            project_samples.extend(self._generate_project_summary_samples(project))
            
            samples.extend(project_samples)
        
        logger.info("Generated comprehensive TTT samples", total=len(samples))
        return samples
    
    def _generate_spec_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate specification-focused samples"""
        samples = []
        specs_doc = project.specs_doc
        
        # Sample 1: Requirements extraction
        standards = [e for e in specs_doc.entities if e['type'] == 'standard']
        requirements = [e for e in specs_doc.entities if 'requirement' in e['type']]
        
        sample = {
            'id': f"{project.project_id}_specs_requirements",
            'query_type': 'specification_analysis',
            'project_context': {
                'project_name': project.project_name,
                'project_id': project.project_id
            },
            'input_query': f"What are the key technical requirements and standards for a fire alarm system project like {project.project_name}?",
            'context_documents': {
                'specifications': {
                    'content_preview': specs_doc.content[:1500],
                    'key_entities': [e['name'] for e in specs_doc.entities[:10]],
                    'standards_found': [s['name'] for s in standards],
                    'document_metadata': specs_doc.metadata
                }
            },
            'expected_output': {
                'type': 'technical_requirements',
                'standards': [s['name'] for s in standards],
                'key_requirements': [r['name'] for r in requirements[:5]],
                'compliance_notes': 'System must comply with all mentioned standards'
            },
            'training_metadata': {
                'source_file': specs_doc.filename,
                'confidence_level': 0.9,
                'complexity': 'medium'
            }
        }
        samples.append(sample)
        
        return samples
    
    def _generate_boq_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate BOQ-focused samples"""
        samples = []
        boq_doc = project.boq_doc
        
        products = [e for e in boq_doc.entities if e['type'] == 'product_code']
        quantities = [e for e in boq_doc.entities if e['type'] == 'quantity']
        costs = [e for e in boq_doc.entities if e['type'] == 'money']
        
        # Extract table data if available
        boq_tables = []
        for table in boq_doc.tables:
            if table.get('data'):
                boq_tables.append({
                    'headers': table['data'][0] if table['data'] else [],
                    'sample_rows': table['data'][1:6] if len(table['data']) > 1 else []
                })
        
        sample = {
            'id': f"{project.project_id}_boq_generation",
            'query_type': 'boq_generation',
            'project_context': {
                'project_name': project.project_name,
                'project_id': project.project_id
            },
            'input_query': f"Generate a detailed Bill of Quantities for a fire alarm system project similar to {project.project_name}",
            'context_documents': {
                'boq': {
                    'content_preview': boq_doc.content[:1500],
                    'products_identified': [p['name'] for p in products[:15]],
                    'quantities_found': [q['name'] for q in quantities[:10]],
                    'cost_references': [c['name'] for c in costs[:5]],
                    'tables_extracted': boq_tables[:2]  # First 2 tables
                }
            },
            'expected_output': {
                'type': 'detailed_boq',
                'product_list': [p['name'] for p in products[:10]],
                'quantity_estimates': 'Based on project scope and requirements',
                'cost_structure': 'Itemized pricing with totals'
            },
            'training_metadata': {
                'source_file': boq_doc.filename,
                'confidence_level': 0.85,
                'complexity': 'high'
            }
        }
        samples.append(sample)
        
        return samples
    
    def _generate_offer_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate offer/proposal-focused samples"""
        samples = []
        offer_doc = project.offer_doc
        
        products = [e for e in offer_doc.entities if e['type'] == 'product_code']
        costs = [e for e in offer_doc.entities if e['type'] == 'money']
        
        sample = {
            'id': f"{project.project_id}_technical_offer",
            'query_type': 'technical_proposal',
            'project_context': {
                'project_name': project.project_name,
                'project_id': project.project_id
            },
            'input_query': f"Create a technical proposal and offer for a fire alarm system project like {project.project_name}",
            'context_documents': {
                'offer': {
                    'content_preview': offer_doc.content[:1500],
                    'proposed_products': [p['name'] for p in products[:10]],
                    'pricing_information': [c['name'] for c in costs[:3]],
                    'technical_details': 'Extracted from offer document'
                }
            },
            'expected_output': {
                'type': 'technical_offer',
                'recommended_solution': [p['name'] for p in products[:5]],
                'pricing_structure': costs[0]['name'] if costs else 'Competitive pricing',
                'technical_specifications': 'Detailed technical compliance'
            },
            'training_metadata': {
                'source_file': offer_doc.filename,
                'confidence_level': 0.8,
                'complexity': 'medium'
            }
        }
        samples.append(sample)
        
        return samples
    
    def _generate_cross_document_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate samples that use multiple documents"""
        samples = []
        
        available_docs = []
        if project.specs_doc:
            available_docs.append(('specs', project.specs_doc))
        if project.boq_doc:
            available_docs.append(('boq', project.boq_doc))
        if project.offer_doc:
            available_docs.append(('offer', project.offer_doc))
        
        if len(available_docs) >= 2:
            sample = {
                'id': f"{project.project_id}_integrated_analysis",
                'query_type': 'multi_document_analysis',
                'project_context': {
                    'project_name': project.project_name,
                    'project_id': project.project_id
                },
                'input_query': f"Analyze all project documents for {project.project_name} and provide integrated recommendations",
                'context_documents': {},
                'expected_output': {
                    'type': 'integrated_recommendation',
                    'compliance_check': 'Verify BOQ matches specifications',
                    'cost_optimization': 'Suggest cost-effective alternatives',
                    'technical_validation': 'Ensure technical compatibility'
                },
                'training_metadata': {
                    'source_files': [doc[1].filename for doc in available_docs],
                    'confidence_level': 0.95,
                    'complexity': 'high'
                }
            }
            
            # Add context from each document
            for doc_type, doc in available_docs:
                sample['context_documents'][doc_type] = {
                    'content_preview': doc.content[:1000],
                    'key_entities': [e['name'] for e in doc.entities[:8]],
                    'metadata': doc.metadata
                }
            
            samples.append(sample)
        
        return samples
    
    def _generate_project_summary_samples(self, project: ProjectSet) -> List[Dict[str, Any]]:
        """Generate project summary samples"""
        samples = []
        
        sample = {
            'id': f"{project.project_id}_project_summary",
            'query_type': 'project_overview',
            'project_context': {
                'project_name': project.project_name,
                'project_id': project.project_id
            },
            'input_query': f"Provide a comprehensive overview of the fire alarm system project: {project.project_name}",
            'context_documents': {
                'project_summary': project.summary
            },
            'expected_output': {
                'type': 'project_overview',
                'scope': 'Fire alarm system implementation',
                'key_products': project.summary.get('key_products', [])[:5],
                'estimated_value': project.summary.get('estimated_value'),
                'standards_compliance': project.summary.get('standards_mentioned', [])
            },
            'training_metadata': {
                'source_files': [doc.filename for doc in [project.specs_doc, project.boq_doc, project.offer_doc] if doc],
                'confidence_level': 0.7,
                'complexity': 'low'
            }
        }
        samples.append(sample)
        
        return samples

async def main():
    """Main execution"""
    print("=== Comprehensive Project Dataset Processor ===")
    print(f"S3 Bucket: {settings.s3_bucket_name}")
    print()
    
    ensure_data_dirs()
    
    processor = ProjectDatasetProcessor()
    
    try:
        # Process all projects
        project_sets = await processor.process_all_projects()
        
        if not project_sets:
            print("No projects were successfully processed")
            return
        
        # Generate comprehensive inference samples
        print(f"\nGenerating TTT inference samples...")
        inference_samples = processor.generate_comprehensive_samples(project_sets)
        
        # Export results
        print(f"\nExporting results...")
        
        # Export project data
        projects_file = Path("./data/processed/all_projects_dataset.json")
        processor.dataset_processor.export_to_json(project_sets, projects_file)
        
        # Export inference samples
        samples_file = Path("./data/processed/comprehensive_ttt_samples.json")
        inference_data = {
            'metadata': {
                'version': '1.0',
                'total_samples': len(inference_samples),
                'total_projects': len(project_sets),
                'generated_at': pd.Timestamp.now().isoformat(),
                'sample_types': list(set(s['query_type'] for s in inference_samples))
            },
            'inference_samples': inference_samples
        }
        
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Summary
        print(f"\n=== Processing Complete ===")
        print(f"Projects processed: {len(project_sets)}")
        print(f"TTT inference samples: {len(inference_samples)}")
        print(f"Projects data: {projects_file}")
        print(f"Inference samples: {samples_file}")
        
        # Statistics
        sample_types = {}
        for sample in inference_samples:
            query_type = sample['query_type']
            sample_types[query_type] = sample_types.get(query_type, 0) + 1
        
        print(f"\nSample distribution:")
        for sample_type, count in sorted(sample_types.items()):
            print(f"  {sample_type}: {count}")
        
        # Project statistics
        doc_stats = {'specs': 0, 'boq': 0, 'offer': 0}
        for project in project_sets:
            if project.specs_doc:
                doc_stats['specs'] += 1
            if project.boq_doc:
                doc_stats['boq'] += 1
            if project.offer_doc:
                doc_stats['offer'] += 1
        
        print(f"\nDocument availability:")
        for doc_type, count in doc_stats.items():
            print(f"  {doc_type}: {count} projects")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        logger.error("Comprehensive processing failed", error=str(e))

if __name__ == "__main__":
    asyncio.run(main())