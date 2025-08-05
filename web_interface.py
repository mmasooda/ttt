#!/usr/bin/env python3
"""
TTT-Enhanced BYOKG-RAG Web Interface
Provides browser-based chat interface for querying the system
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import threading
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import logger, settings
from src.core.byokg_rag_engine import BYOKGRAGEngine

app = Flask(__name__)
CORS(app)

# Global RAG engine instance
rag_engine = None
processing_status = {
    'is_processing': True,
    'files_processed': 0,
    'total_files': 139,
    'current_file': 'Initializing...',
    'start_time': datetime.now(),
    'errors': [],
    'stage': 'Starting background processing...'
}

def get_server_ip():
    """Get server IP address"""
    import socket
    try:
        # Get the IP address by connecting to a remote server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

async def initialize_rag_engine():
    """Initialize the RAG engine"""
    global rag_engine
    try:
        logger.info("Initializing RAG engine for web interface")
        rag_engine = BYOKGRAGEngine()
        await rag_engine.vector_store.load_from_disk()
        logger.info("RAG engine initialized successfully")
        return True
    except Exception as e:
        logger.error("Failed to initialize RAG engine", error=str(e))
        return False

def monitor_processing():
    """Monitor the background processing"""
    global processing_status
    
    log_file = Path("/root/ttt/processing.log")
    
    while processing_status['is_processing']:
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Parse log for progress
                lines = content.split('\n')
                for line in reversed(lines[-50:]):  # Check last 50 lines
                    if '[' in line and ']' in line and 'Processing:' in line:
                        # Extract current file being processed
                        try:
                            if 'Processing:' in line:
                                file_part = line.split('Processing:')[1].strip()
                                if file_part:
                                    processing_status['current_file'] = file_part.split()[0]
                        except:
                            pass
                    
                    if 'Successfully processed' in line and 'files' in line:
                        try:
                            # Extract number of processed files
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'processed' and i+1 < len(parts):
                                    processing_status['files_processed'] = int(parts[i+1])
                                    break
                        except:
                            pass
                
                # Check if processing completed
                if 'Processing Complete' in content or 'Successfully processed' in content:
                    processing_status['is_processing'] = False
                    processing_status['stage'] = 'Processing completed - System ready!'
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error("Error monitoring processing", error=str(e))
            time.sleep(10)

# Start monitoring thread
monitoring_thread = threading.Thread(target=monitor_processing, daemon=True)
monitoring_thread.start()

@app.route('/')
def index():
    """Main chat interface"""
    server_ip = get_server_ip()
    return render_template('chat.html', server_ip=server_ip)

@app.route('/status')
def status():
    """Get current processing status"""
    global processing_status
    
    # Calculate elapsed time
    elapsed = datetime.now() - processing_status['start_time']
    elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
    
    # Calculate progress percentage
    progress = 0
    if processing_status['total_files'] > 0:
        progress = (processing_status['files_processed'] / processing_status['total_files']) * 100
    
    # Estimate remaining time
    remaining_str = "Calculating..."
    if processing_status['files_processed'] > 0:
        avg_time_per_file = elapsed.total_seconds() / processing_status['files_processed']
        remaining_files = processing_status['total_files'] - processing_status['files_processed']
        remaining_seconds = remaining_files * avg_time_per_file
        remaining_str = f"{int(remaining_seconds // 3600)}h {int((remaining_seconds % 3600) // 60)}m"
    
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'files_processed': processing_status['files_processed'],
        'total_files': processing_status['total_files'],
        'current_file': processing_status['current_file'],
        'progress_percent': round(progress, 1),
        'elapsed_time': elapsed_str,
        'estimated_remaining': remaining_str,
        'stage': processing_status['stage']
    })

@app.route('/query', methods=['POST'])
async def query():
    """Handle RAG queries"""
    global rag_engine
    
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        # Initialize RAG engine if not already done
        if rag_engine is None:
            success = await initialize_rag_engine()
            if not success:
                return jsonify({'error': 'RAG engine not available. Please wait for system initialization.'}), 503
        
        # Process the query
        logger.info("Processing web query", query=user_query)
        
        result = await rag_engine.query_with_rag(
            user_query=user_query,
            k_vector=5,
            k_graph=10
        )
        
        response = {
            'answer': result.get('answer', 'No answer generated'),
            'sources': result.get('sources', []),
            'graph_results': result.get('graph_results_count', 0),
            'vector_results': result.get('vector_results_count', 0),
            'query': user_query,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Web query processed successfully", 
                   answer_length=len(response['answer']),
                   sources_count=len(response['sources']))
        
        return jsonify(response)
        
    except Exception as e:
        logger.error("Web query failed", error=str(e))
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/system-info')
def system_info():
    """Get system information"""
    return jsonify({
        'system_name': 'TTT-Enhanced BYOKG-RAG',
        'version': '1.0',
        'models': {
            'ingestion': 'GPT-4.1-mini',
            'generation': 'GPT-4o',
            'embeddings': 'text-embedding-3-small'
        },
        'components': {
            'pdf_extraction': 'Triple-layer (PyMuPDF + Camelot + Tabula)',
            'knowledge_graph': 'Neo4j with enhanced relationships',
            'vector_database': 'FAISS',
            'domain': 'Fire Alarm Systems'
        },
        's3_bucket': settings.s3_bucket_name
    })

if __name__ == '__main__':
    # Create templates directory and HTML file
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Get server info
    server_ip = get_server_ip()
    port = 8080
    
    print(f"\n🌐 TTT-Enhanced BYOKG-RAG Web Interface")
    print(f"=" * 50)
    print(f"🖥️  Server IP: {server_ip}")
    print(f"🌐 Access URL: http://{server_ip}:{port}")
    print(f"📊 Status URL: http://{server_ip}:{port}/status")
    print(f"ℹ️  System Info: http://{server_ip}:{port}/system-info")
    print(f"\n🚀 Starting Flask server...")
    print(f"📝 Background S3 processing is running...")
    print(f"💬 Chat interface will be available in your browser!")
    
    # Log the information
    with open('/root/ttt/COMPLETE_SYSTEM_LOG.md', 'a') as f:
        f.write(f"\n\n### [TIMESTAMP: {datetime.now().isoformat()}] - Web Interface Started\n")
        f.write(f"**Action**: Started Flask web server for browser access\n")
        f.write(f"**Server IP**: {server_ip}\n") 
        f.write(f"**Access URL**: http://{server_ip}:{port}\n")
        f.write(f"**Status**: Background S3 processing running\n")
        f.write(f"**Components**: Chat interface, status dashboard, RAG queries\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)