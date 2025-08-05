#!/bin/bash

# TTT-Enhanced BYOKG-RAG Setup and Run Script

echo "=== TTT-Enhanced BYOKG-RAG System ==="
echo "Starting setup and verification..."

# Activate virtual environment
source venv/bin/activate

# Run setup test
echo "Running setup verification..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Configure your .env file with:"
    echo "   - OpenAI API key"
    echo "   - AWS credentials (if using S3)"
    echo "   - Neo4j password (default: password123)"
    echo ""
    echo "2. Set Neo4j password manually:"
    echo "   cypher-shell -u neo4j -p neo4j"
    echo "   Then run: ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'password123'"
    echo ""
    echo "3. To start development:"
    echo "   source venv/bin/activate"
    echo "   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "4. Available files and structure:"
    echo "   - requirements.txt: All Python dependencies"
    echo "   - .env: Environment configuration"
    echo "   - src/: Source code directory structure"
    echo "   - data/: Data storage for examples, adapters, downloads"
    echo "   - test_setup.py: Setup verification script"
    echo ""
    echo "=== Important Notes ==="
    echo "• Python virtual environment: ./venv/"
    echo "• Neo4j running on: localhost:7687 (bolt) and localhost:7474 (http)"
    echo "• All ML dependencies installed (PyTorch, Transformers, etc.)"
    echo "• AWS CLI installed and ready for configuration"
    echo ""
else
    echo "❌ Setup verification failed. Please check the error messages above."
    exit 1
fi