#!/usr/bin/env python3
"""
Progress updater script that monitors processing.log and updates a JSON status file
"""

import json
import time
import re
from pathlib import Path

def update_progress():
    log_file = Path("/root/ttt/processing.log")
    status_file = Path("/root/ttt/data/processing_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    last_position = 0
    
    while True:
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    last_position = f.tell()
                
                if new_content:
                    # Find all processing patterns
                    pattern = r'\[(\d+)/(\d+)\] Processing: (.+)'
                    matches = re.findall(pattern, new_content)
                    
                    if matches:
                        # Get the latest match
                        current, total, filename = matches[-1]
                        
                        status = {
                            'is_processing': True,
                            'files_processed': int(current) - 1,  # Current file is being processed
                            'current_file_number': int(current),
                            'total_files': int(total),
                            'current_file': filename,
                            'progress_percent': ((int(current) - 1) / int(total)) * 100,
                            'timestamp': time.time()
                        }
                        
                        # Check for completion
                        if 'Processing Complete' in new_content:
                            status['is_processing'] = False
                            status['files_processed'] = status['total_files']
                            status['current_file'] = 'Processing completed!'
                            status['progress_percent'] = 100.0
                        
                        # Write status
                        with open(status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                        
                        print(f"Updated: {current}/{total} - {filename}")
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("Starting progress monitor...")
    update_progress()