import boto3
import asyncio
import aioboto3
from typing import List, Dict, Any, AsyncGenerator, Optional
from pathlib import Path
import os
from .config import settings
from .logging import logger

class S3Client:
    """S3 client for listing and downloading files"""
    
    def __init__(self):
        self.bucket_name = settings.s3_bucket_name
        self.region = settings.aws_region
        
        # Sync client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=self.region
        )
        
        # Async session
        self.session = aioboto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=self.region
        )
    
    def list_all_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List all objects in the S3 bucket"""
        try:
            logger.info("Listing S3 objects", bucket=self.bucket_name, prefix=prefix)
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Filter out directories (keys ending with /)
                        if not obj['Key'].endswith('/'):
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'etag': obj['ETag'].strip('"')
                            })
            
            logger.info("Found S3 objects", count=len(objects))
            return objects
            
        except Exception as e:
            logger.error("Failed to list S3 objects", error=str(e))
            raise
    
    def filter_supported_files(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter objects to only supported file types"""
        supported_exts = settings.supported_extensions
        max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        filtered = []
        for obj in objects:
            key = obj['key']
            size = obj['size']
            
            # Check file extension
            ext = Path(key).suffix.lower()
            if ext not in supported_exts:
                continue
            
            # Check file size
            if size > max_size:
                logger.warning("Skipping large file", key=key, size_mb=size/1024/1024)
                continue
            
            filtered.append(obj)
        
        logger.info("Filtered supported files", 
                   total=len(objects), 
                   supported=len(filtered),
                   extensions=supported_exts)
        return filtered
    
    async def download_file(self, key: str, local_path: Path) -> bool:
        """Download a single file from S3"""
        try:
            # Ensure directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with self.session.client('s3') as s3:
                await s3.download_file(
                    self.bucket_name, 
                    key, 
                    str(local_path)
                )
            
            logger.info("Downloaded file", key=key, local_path=str(local_path))
            return True
            
        except Exception as e:
            logger.error("Failed to download file", key=key, error=str(e))
            return False
    
    async def download_files_batch(self, objects: List[Dict[str, Any]], 
                                   download_dir: Path, 
                                   max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Download multiple files concurrently"""
        
        download_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_single(obj: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                key = obj['key']
                # Preserve directory structure
                local_path = download_dir / key
                
                success = await self.download_file(key, local_path)
                
                return {
                    **obj,
                    'local_path': str(local_path) if success else None,
                    'download_success': success
                }
        
        logger.info("Starting batch download", 
                   count=len(objects), 
                   max_concurrent=max_concurrent)
        
        results = await asyncio.gather(*[download_single(obj) for obj in objects])
        
        successful = sum(1 for r in results if r['download_success'])
        logger.info("Batch download completed", 
                   total=len(objects), 
                   successful=successful, 
                   failed=len(objects)-successful)
        
        return results
    
    def get_bucket_structure(self) -> Dict[str, Any]:
        """Get a hierarchical view of the bucket structure"""
        objects = self.list_all_objects()
        
        structure = {}
        for obj in objects:
            parts = obj['key'].split('/')
            current = structure
            
            for part in parts[:-1]:  # directories
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # File
            filename = parts[-1]
            current[filename] = {
                'size': obj['size'],
                'last_modified': obj['last_modified'],
                'type': 'file'
            }
        
        return structure
    
    def print_bucket_structure(self, structure: Optional[Dict] = None, indent: int = 0):
        """Print bucket structure in tree format"""
        if structure is None:
            structure = self.get_bucket_structure()
        
        for name, content in structure.items():
            prefix = "  " * indent
            if isinstance(content, dict) and 'type' in content:
                # File
                size_mb = content['size'] / (1024 * 1024)
                print(f"{prefix}📄 {name} ({size_mb:.2f} MB)")
            else:
                # Directory
                print(f"{prefix}📁 {name}/")
                self.print_bucket_structure(content, indent + 1)