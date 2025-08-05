#!/usr/bin/env python3
"""
Explore S3 bucket structure to understand dataset organization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import S3Client, logger

def main():
    """Explore bucket structure"""
    print("=== S3 Bucket Explorer ===")
    
    s3_client = S3Client()
    
    # Get all objects
    all_objects = s3_client.list_all_objects()
    
    print(f"Total objects: {len(all_objects)}")
    
    # Analyze directory structure
    directories = {}
    
    for obj in all_objects:
        key = obj['key']
        parts = key.split('/')
        
        if len(parts) > 1:
            # Has directory structure
            top_dir = parts[0]
            if top_dir not in directories:
                directories[top_dir] = []
            directories[top_dir].append(obj)
        else:
            # Root level file
            if 'root' not in directories:
                directories['root'] = []
            directories['root'].append(obj)
    
    print(f"\nDirectory structure:")
    for dir_name, files in directories.items():
        print(f"\n📁 {dir_name}/ ({len(files)} files)")
        
        # Show file types
        extensions = {}
        for file_obj in files:
            ext = Path(file_obj['key']).suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        for ext, count in sorted(extensions.items()):
            print(f"   {ext}: {count} files")
        
        # Show first few files as examples
        print("   Examples:")
        for file_obj in files[:3]:
            size_mb = file_obj['size'] / (1024 * 1024)
            print(f"   - {file_obj['key']} ({size_mb:.2f} MB)")
        
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more files")
    
    # Look for dataset-like directories
    print(f"\n=== Analysis ===")
    project_dirs = []
    
    for dir_name in directories.keys():
        if any(keyword in dir_name.lower() for keyword in ['enq', 'enquir', 'project', 'dataset']):
            project_dirs.append(dir_name)
    
    print(f"Potential project directories: {project_dirs}")
    
    # Look for patterns in enquiry directories
    if 'enquiries' in directories:
        print(f"\n=== Enquiries Directory Analysis ===")
        enq_files = directories['enquiries']
        
        # Group by subdirectory
        subdirs = {}
        for file_obj in enq_files:
            key_parts = file_obj['key'].split('/')
            if len(key_parts) > 2:  # enquiries/subdir/file
                subdir = key_parts[1]
                if subdir not in subdirs:
                    subdirs[subdir] = []
                subdirs[subdir].append(file_obj)
        
        print(f"Enquiry subdirectories: {len(subdirs)}")
        for subdir, files in list(subdirs.items())[:5]:  # Show first 5
            print(f"  {subdir}: {len(files)} files")
            file_types = {}
            for f in files:
                name = Path(f['key']).name.lower()
                if 'spec' in name or 'compliance' in name:
                    file_types['specs'] = file_types.get('specs', 0) + 1
                elif 'boq' in name or 'quantity' in name or 'pricing' in name:
                    file_types['boq'] = file_types.get('boq', 0) + 1
                elif 'offer' in name or 'proposal' in name or 'quote' in name:
                    file_types['offer'] = file_types.get('offer', 0) + 1
                else:
                    file_types['other'] = file_types.get('other', 0) + 1
            
            print(f"    Types: {file_types}")

if __name__ == "__main__":
    main()