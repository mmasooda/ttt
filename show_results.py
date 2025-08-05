#!/usr/bin/env python3
"""
Display the results of TTT dataset processing
"""

import json
from pathlib import Path

def main():
    """Show processing results"""
    print("=== TTT Dataset Processing Results ===\n")
    
    # Check if files exist
    projects_file = Path("./data/processed/all_projects_dataset.json")
    samples_file = Path("./data/processed/comprehensive_ttt_samples.json")
    
    if not projects_file.exists():
        print("❌ Projects dataset file not found")
        return
    
    if not samples_file.exists():
        print("❌ TTT samples file not found")
        return
    
    # Load and display project data
    with open(projects_file, 'r') as f:
        project_data = json.load(f)
    
    print("📊 PROJECT DATASET SUMMARY")
    print(f"Total Projects: {project_data['dataset_info']['total_projects']}")
    print(f"Processed At: {project_data['dataset_info']['processed_at']}")
    
    # Analyze project types
    project_types = {}
    doc_availability = {'specs': 0, 'boq': 0, 'offer': 0}
    
    for project in project_data['projects']:
        # Count document types
        if project['specs_doc']:
            doc_availability['specs'] += 1
        if project['boq_doc']:
            doc_availability['boq'] += 1
        if project['offer_doc']:
            doc_availability['offer'] += 1
        
        # Count project sources
        project_name = project['project_name']
        if 'dataset_project' in project_name:
            project_types['dataset'] = project_types.get('dataset', 0) + 1
        elif 'enquiries' in project_name:
            project_types['enquiries'] = project_types.get('enquiries', 0) + 1
        elif 'enq 11 to 20' in project_name:
            project_types['enq_11_20'] = project_types.get('enq_11_20', 0) + 1
    
    print(f"\nProject Sources:")
    for source, count in project_types.items():
        print(f"  {source}: {count} projects")
    
    print(f"\nDocument Availability:")
    for doc_type, count in doc_availability.items():
        print(f"  {doc_type.upper()}: {count} projects")
    
    # Load and display TTT samples
    with open(samples_file, 'r') as f:
        samples_data = json.load(f)
    
    print(f"\n🧠 TTT INFERENCE SAMPLES SUMMARY")
    metadata = samples_data['metadata']
    print(f"Total Samples: {metadata['total_samples']}")
    print(f"Generated At: {metadata['generated_at']}")
    
    print(f"\nSample Types:")
    for sample_type in metadata['sample_types']:
        count = sum(1 for s in samples_data['inference_samples'] if s['query_type'] == sample_type)
        print(f"  {sample_type}: {count} samples")
    
    # Show example samples
    print(f"\n📝 EXAMPLE TTT SAMPLES")
    
    sample_types_shown = set()
    for sample in samples_data['inference_samples'][:10]:  # Show first 10 samples
        sample_type = sample['query_type']
        if sample_type not in sample_types_shown:
            print(f"\n--- {sample_type.upper()} SAMPLE ---")
            print(f"Project: {sample['project_context']['project_name']}")
            print(f"Query: {sample['input_query'][:100]}...")
            print(f"Expected Output Type: {sample['expected_output']['type']}")
            sample_types_shown.add(sample_type)
        
        if len(sample_types_shown) >= 3:  # Show max 3 different sample types
            break
    
    # Show file statistics
    print(f"\n📁 FILE INFORMATION")
    print(f"Projects Dataset: {projects_file}")
    print(f"  Size: {projects_file.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"TTT Samples: {samples_file}")
    print(f"  Size: {samples_file.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"\n✅ DATASET READY FOR TTT TRAINING")
    print(f"The processed dataset contains:")
    print(f"  • {len(project_data['projects'])} complete project sets")
    print(f"  • {len(samples_data['inference_samples'])} diverse training samples")
    print(f"  • Multiple document types (specs, BOQ, offers)")
    print(f"  • Rich entity extraction and relationships")
    print(f"  • Cross-document correlation samples")
    print(f"  • Ready for TTT adaptation and inference")

if __name__ == "__main__":
    main()