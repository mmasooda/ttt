#!/usr/bin/env python3
"""
Test improved extraction system with extracted PDFs handling
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from process_enhanced_dataset import ComprehensiveDatasetProcessor

async def test_improved_system():
    """Test the improved extraction system"""
    print("=== Testing Improved Triple-Layer Extraction System ===")
    print("Processing extracted PDFs with dynamic page handling...\n")
    
    processor = ComprehensiveDatasetProcessor()
    
    # Process first 5 files to test improvements
    result = await processor.process_complete_dataset(max_files=5)
    
    if result['success']:
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Processing stats:")
        stats = result['processing_stats']
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Quality assessment:")
        quality = result['quality_assessment']
        for key, value in quality.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n📁 Results exported to: data/enhanced_results/")
        
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_improved_system())