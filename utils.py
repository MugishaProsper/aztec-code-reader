"""Utility functions for Aztec Code Reader"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

def get_image_files(input_path: Path, supported_formats: List[str]) -> List[Path]:
    """Get list of image files from input path"""
    if input_path.is_dir():
        images = []
        for ext in supported_formats:
            images.extend(input_path.glob(ext))
        return sorted(images)
    else:
        return [input_path] if input_path.exists() else []

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def calculate_processing_stats(records: List[Dict]) -> Dict:
    """Calculate processing statistics from records"""
    if not records:
        return {}
    
    successful_codes = [r for r in records if r['aztec_index'] > 0 and not str(r['data']).startswith('ERROR')]
    error_records = [r for r in records if str(r['data']).startswith('ERROR')]
    
    total_images = len(set(r['filename'] for r in records))
    total_processing_time = sum(r.get('processing_time_ms', 0) for r in records)
    
    return {
        'total_images': total_images,
        'successful_codes': len(successful_codes),
        'error_count': len(error_records),
        'total_processing_time_ms': total_processing_time,
        'avg_processing_time_ms': total_processing_time / len(records) if records else 0,
        'success_rate': len(successful_codes) / len(records) if records else 0
    }

def create_summary_report(records: List[Dict], output_path: Optional[Path] = None) -> str:
    """Create a summary report of processing results"""
    stats = calculate_processing_stats(records)
    
    report = f"""
Aztec Code Reader - Processing Summary
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
-----------
Total Images Processed: {stats.get('total_images', 0)}
Aztec Codes Found: {stats.get('successful_codes', 0)}
Processing Errors: {stats.get('error_count', 0)}
Success Rate: {stats.get('success_rate', 0):.1%}

Performance:
------------
Total Processing Time: {stats.get('total_processing_time_ms', 0):.1f}ms
Average Time per Image: {stats.get('avg_processing_time_ms', 0):.1f}ms

Files with Codes:
-----------------
"""
    
    # Group by filename
    files_with_codes = {}
    for record in records:
        if record['aztec_index'] > 0 and not str(record['data']).startswith('ERROR'):
            filename = record['filename']
            if filename not in files_with_codes:
                files_with_codes[filename] = []
            files_with_codes[filename].append(record)
    
    for filename, codes in files_with_codes.items():
        report += f"\n{filename}: {len(codes)} code(s)\n"
        for i, code in enumerate(codes, 1):
            preview = str(code['data'])[:60] + "..." if len(str(code['data'])) > 60 else str(code['data'])
            confidence = code.get('confidence', 0)
            report += f"  [{i}] (Q:{confidence}) {preview}\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.time() - self.start_time) * 1000
        self.logger.debug(f"{self.operation_name} completed in {elapsed:.1f}ms")

def validate_output_paths(csv_path: Path, json_path: Optional[Path], output_dir: Path) -> bool:
    """Validate that output paths are writable"""
    try:
        # Check if we can write to the CSV path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check JSON path if provided
        if json_path:
            json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logging.error(f"Cannot create output paths: {e}")
        return False