#!/usr/bin/env python3
"""
Reload vector store with processed data to fix empty vector store issue
"""

import sys
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vector.faiss_store import FAISSVectorStore, VectorDocument
from src.llm.openai_client import OpenAIClient

async def main():
    """Reload vector store with actual data"""
    print("🔄 Reloading Vector Store with Processed Data")
    
    try:
        # Initialize components
        vector_store = FAISSVectorStore()
        openai_client = OpenAIClient()
        
        # Load processed data
        data_file = Path("data/processed/all_projects_dataset.json")
        print(f"📂 Loading data from: {data_file}")
        
        with open(data_file, 'r') as f:
            projects_data = json.load(f)
        
        documents = []
        
        # Process each project's data
        for project_name, project_data in projects_data.items():
            print(f"📄 Processing project: {project_name}")
            
            if isinstance(project_data, dict) and 'extracted_data' in project_data:
                for doc_name, doc_data in project_data['extracted_data'].items():
                    if isinstance(doc_data, dict) and 'chunks' in doc_data:
                        # Add document chunks
                        for i, chunk in enumerate(doc_data['chunks']):
                            if isinstance(chunk, dict) and 'content' in chunk:
                                content = chunk['content']
                                if content and len(content.strip()) > 50:  # Only meaningful chunks
                                    
                                    # Create vector document
                                    doc = VectorDocument(
                                        content=content,
                                        metadata={
                                            'source': f"{project_name}/{doc_name}",
                                            'chunk_id': i,
                                            'project': project_name,
                                            'document': doc_name
                                        }
                                    )
                                    documents.append(doc)
        
        print(f"📊 Prepared {len(documents)} documents for vector store")
        
        if documents:
            # Generate embeddings and add to vector store
            print("🔄 Generating embeddings...")
            for i, doc in enumerate(documents):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(documents)}")
                
                # Generate embedding
                embedding = await openai_client.get_embedding(doc.content)
                doc.embedding = embedding
            
            # Add documents to vector store
            print("💾 Adding documents to vector store...")
            await vector_store.add_documents(documents)
            
            # Save vector store
            vector_store.save()
            print(f"✅ Vector store reloaded with {len(documents)} documents")
            
            # Test search
            test_query = "4098-5288 base detector"
            print(f"\n🔍 Testing search: '{test_query}'")
            results = await vector_store.search(test_query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.4f}")
                print(f"     Source: {result.metadata.get('source', 'Unknown')}")
                print(f"     Content: {result.content[:200]}...")
                print()
        
        else:
            print("❌ No documents found to add to vector store")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())