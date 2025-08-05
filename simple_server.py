#!/usr/bin/env python3
"""
Simple HTTP Server for TTT-Enhanced BYOKG-RAG System
Direct HTTP interface without Flask dependencies
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import logger, settings
from src.core.byokg_rag_engine import BYOKGRAGEngine

# Global RAG engine
rag_engine = None
processing_status = {
    'is_processing': True,
    'files_processed': 0,
    'total_files': 139,
    'current_file': 'Initializing...',
    'start_time': datetime.now(),
    'stage': 'Starting background processing...'
}

class TTTRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_main_page()
        elif parsed_path.path == '/status':
            self.serve_status()
        elif parsed_path.path == '/system-info':
            self.serve_system_info()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/query':
            self.handle_query()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main chat interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTT-Enhanced BYOKG-RAG Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .status-bar { background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .chat-container { border: 1px solid #ddd; border-radius: 5px; height: 400px; overflow-y: auto; padding: 15px; margin-bottom: 20px; background: #fafafa; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #007bff; color: white; text-align: right; }
        .assistant-message { background: #f8f9fa; border: 1px solid #dee2e6; }
        .input-container { display: flex; gap: 10px; }
        .query-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .send-button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .send-button:hover { background: #0056b3; }
        .sample-queries { margin: 20px 0; }
        .sample-query { display: inline-block; margin: 5px; padding: 8px 15px; background: #e9ecef; border-radius: 15px; cursor: pointer; font-size: 14px; }
        .sample-query:hover { background: #dee2e6; }
        .loading { text-align: center; padding: 20px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TTT-Enhanced BYOKG-RAG System</h1>
            <p>Fire Alarm Systems Knowledge Assistant</p>
        </div>
        
        <div class="status-bar" id="statusBar">
            <div class="loading">Loading system status...</div>
        </div>
        
        <div class="sample-queries">
            <strong>Sample Questions:</strong><br>
            <span class="sample-query" onclick="askQuery('What fire alarm panels are available?')">🔥 Fire alarm panels</span>
            <span class="sample-query" onclick="askQuery('Power requirements for smoke detectors?')">⚡ Power requirements</span>
            <span class="sample-query" onclick="askQuery('Compliance standards for fire alarms?')">📋 Compliance standards</span>
            <span class="sample-query" onclick="askQuery('Compatible devices for panels?')">🔗 Device compatibility</span>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                <strong>Assistant:</strong> Hello! I'm your Fire Alarm Systems Knowledge Assistant. 
                The system is processing your S3 data in the background. You can start asking questions now!
                <br><br>Try clicking one of the sample questions above or type your own query below.
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="queryInput" class="query-input" placeholder="Ask about fire alarm systems..." onkeypress="if(event.key==='Enter') sendQuery()">
            <button class="send-button" onclick="sendQuery()">Send</button>
        </div>
    </div>

    <script>
        // Update status every 10 seconds
        setInterval(updateStatus, 10000);
        updateStatus();
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusBar = document.getElementById('statusBar');
                    const progressPercent = ((data.files_processed / data.total_files) * 100).toFixed(1);
                    
                    statusBar.innerHTML = `
                        <strong>System Status:</strong> ${data.is_processing ? 'Processing S3 Data' : 'Ready'} | 
                        <strong>Progress:</strong> ${data.files_processed}/${data.total_files} files (${progressPercent}%) | 
                        <strong>Current:</strong> ${data.current_file}
                    `;
                })
                .catch(error => {
                    console.error('Status update failed:', error);
                });
        }
        
        function askQuery(query) {
            document.getElementById('queryInput').value = query;
            sendQuery();
        }
        
        function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            
            if (!query) return;
            
            // Add user message
            addMessage('user', query);
            input.value = '';
            
            // Add loading message
            addMessage('assistant', '<div class="loading">Processing your query with GPT-4o...</div>', 'loading');
            
            // Send query
            fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                removeMessage('loading');
                
                if (data.error) {
                    addMessage('assistant', `❌ Error: ${data.error}`);
                } else {
                    let response = data.answer;
                    if (data.sources && data.sources.length > 0) {
                        response += `<br><br><small>📚 Sources: ${data.sources.length} documents</small>`;
                    }
                    addMessage('assistant', response);
                }
            })
            .catch(error => {
                removeMessage('loading');
                addMessage('assistant', `❌ Network error: ${error.message}`);
            });
        }
        
        function addMessage(type, content, id = null) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            if (id) messageDiv.id = id;
            
            const time = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `<strong>${type === 'user' ? 'You' : 'Assistant'}:</strong> ${content} <small>(${time})</small>`;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function removeMessage(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_status(self):
        """Serve processing status"""
        # Try to read from JSON status file first
        status_file = Path("/root/ttt/data/processing_status.json")
        
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    file_status = json.load(f)
                
                # Update global status with file data
                processing_status.update(file_status)
            except:
                pass
        
        elapsed = datetime.now() - processing_status['start_time']
        progress = (processing_status['files_processed'] / processing_status['total_files']) * 100
        
        status_data = {
            'is_processing': processing_status['is_processing'],
            'files_processed': processing_status['files_processed'],
            'total_files': processing_status['total_files'],
            'current_file': processing_status['current_file'],
            'progress_percent': round(progress, 1),
            'elapsed_time': str(elapsed).split('.')[0]
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status_data).encode())
    
    def serve_system_info(self):
        """Serve system information"""
        info = {
            'system_name': 'TTT-Enhanced BYOKG-RAG',
            'version': '1.0',
            'status': 'Operational',
            's3_bucket': settings.s3_bucket_name
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(info).encode())
    
    def handle_query(self):
        """Handle RAG queries"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query', '').strip()
            if not query:
                self.send_error(400, "Empty query")
                return
            
            # Initialize RAG engine if not already done
            global rag_engine
            if rag_engine is None:
                try:
                    rag_engine = BYOKGRAGEngine()
                    # Force load vector store
                    asyncio.run(rag_engine.vector_store.load_from_disk())
                    print(f"[{datetime.now()}] RAG engine initialized with {len(rag_engine.vector_store.documents)} documents")
                except Exception as e:
                    print(f"[{datetime.now()}] Failed to initialize RAG engine: {e}")
                    response = {
                        'answer': f"System initialization error: {str(e)}. Please try again in a moment.",
                        'sources': [],
                        'query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    return
            
            # Process query with RAG engine
            try:
                print(f"[{datetime.now()}] Processing query: {query}")
                rag_result = asyncio.run(rag_engine.query_with_rag(query))
                
                response = {
                    'answer': rag_result.get('answer', 'No answer generated'),
                    'sources': rag_result.get('sources', []),
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'graph_results': len(rag_result.get('graph_results', [])),
                        'vector_results': len(rag_result.get('vector_results', [])),
                        'confidence': rag_result.get('confidence', 0.0)
                    }
                }
                print(f"[{datetime.now()}] Query processed successfully")
                
            except Exception as e:
                print(f"[{datetime.now()}] RAG query failed: {e}")
                response = {
                    'answer': f"Query processing error: {str(e)}. The system is operational but may need more processed data for better answers.",
                    'sources': [],
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {'error': str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

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
                for line in reversed(lines[-100:]):
                    # Look for pattern like [89/139] Processing: filename.pdf
                    if 'Processing:' in line and '[' in line and '/' in line:
                        try:
                            # Extract [X/Y] pattern
                            bracket_part = line[line.index('['):line.index(']')+1]
                            numbers = bracket_part.strip('[]').split('/')
                            if len(numbers) == 2:
                                processing_status['files_processed'] = int(numbers[0])
                                processing_status['total_files'] = int(numbers[1])
                            
                            # Extract filename
                            file_part = line.split('Processing:')[1].strip()
                            if file_part:
                                processing_status['current_file'] = file_part
                        except:
                            pass
                    
                    # Check for successful completion
                    if 'Successfully processed:' in line and 'files' in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'processed:' and i+1 < len(parts):
                                    processing_status['files_processed'] = int(parts[i+1])
                                    break
                        except:
                            pass
                
                if 'Processing Complete' in content:
                    processing_status['is_processing'] = False
                    processing_status['stage'] = 'Processing completed - System ready!'
            
            time.sleep(15)
            
        except Exception as e:
            time.sleep(15)

if __name__ == '__main__':
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitor_processing, daemon=True)
    monitoring_thread.start()
    
    server_ip = '0.0.0.0'
    port = 8080  # Try port 8080 which is commonly allowed
    
    print(f"\n🌐 TTT-Enhanced BYOKG-RAG Simple Server")
    print(f"=" * 50)
    print(f"🖥️  Server IP: 217.154.45.86")
    print(f"🌐 Access URL: http://217.154.45.86:{port}")
    print(f"📊 Status URL: http://217.154.45.86:{port}/status")
    print(f"\n🚀 Starting HTTP server on port {port}...")
    print(f"📝 Background S3 processing is running...")
    print(f"💬 Simple chat interface will be available!")
    
    # Start HTTP server
    server = HTTPServer((server_ip, port), TTTRequestHandler)
    print(f"✅ Server started successfully!")
    server.serve_forever()