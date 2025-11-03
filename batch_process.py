#!/usr/bin/env python3
"""Batch processing script for multiple directories of images"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

from aztec_reader import AztecProcessor, process_image, setup_logging, save_results
from config import ProcessingConfig
from utils import get_image_files, create_summary_report, calculate_processing_stats

def process_batch_directories(directories: List[Path], output_base: Path, 
                            config: ProcessingConfig, draw: bool = False) -> Dict:
    """Process multiple directories in batch"""
    
    all_results = []
    directory_stats = {}
    
    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            logging.warning(f"Skipping non-existent directory: {directory}")
            continue
            
        logging.info(f"Processing directory: {directory}")
        
        # Create output subdirectory
        dir_output = output_base / directory.name
        dir_output.mkdir(exist_ok=True)
        
        # Get images
        images = get_image_files(directory, config.supported_formats)
        if not images:
            logging.warning(f"No images found in {directory}")
            continue
        
        # Process images
        processor = AztecProcessor(config.clahe_clip_limit, config.clahe_grid_size)
        dir_records = []
        
        for img in images:
            records = process_image(str(img), str(dir_output), draw, processor)
            dir_records.extend(records)
        
        # Save directory-specific results
        dir_csv = dir_output / f"{directory.name}_results.csv"
        dir_json = dir_output / f"{directory.name}_results.json"
        save_results(dir_records, dir_csv, dir_json)
        
        # Generate directory report
        report_path = dir_output / f"{directory.name}_summary.txt"
        create_summary_report(dir_records, report_path)
        
        # Store stats
        directory_stats[directory.name] = calculate_processing_stats(dir_records)
        all_results.extend(dir_records)
        
        logging.info(f"Completed {directory.name}: {len(images)} images, "
                    f"{directory_stats[directory.name]['successful_codes']} codes found")
    
    return {
        'all_results': all_results,
        'directory_stats': directory_stats
    }

def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple directories of images for Aztec codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py dir1 dir2 dir3 --output batch_results
  python batch_process.py /path/to/images/* --draw --config custom_config.json
        """
    )
    
    parser.add_argument("directories", nargs='+', help="Directories to process")
    parser.add_argument("-o", "--output", default="batch_output", help="Base output directory")
    parser.add_argument("--draw", action="store_true", help="Generate annotated images")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--parallel", type=int, help="Number of parallel processes (future feature)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = ProcessingConfig.from_file(Path(args.config))
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = ProcessingConfig()
    
    # Prepare directories
    directories = [Path(d) for d in args.directories]
    output_base = Path(args.output)
    output_base.mkdir(exist_ok=True)
    
    logger.info(f"Starting batch processing of {len(directories)} directories")
    start_time = datetime.now()
    
    # Process all directories
    results = process_batch_directories(directories, output_base, config, args.draw)
    
    # Generate master summary
    master_csv = output_base / "master_results.csv"
    master_json = output_base / "master_results.json"
    save_results(results['all_results'], master_csv, master_json)
    
    # Create comprehensive report
    master_report = output_base / "batch_summary.txt"
    total_time = (datetime.now() - start_time).total_seconds()
    
    report_content = f"""
Batch Processing Summary
========================
Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Time: {total_time:.1f} seconds

Configuration:
--------------
CLAHE Clip Limit: {config.clahe_clip_limit}
CLAHE Grid Size: {config.clahe_grid_size}
Min Data Length: {config.min_data_length}

Directory Results:
------------------
"""
    
    total_images = 0
    total_codes = 0
    
    for dir_name, stats in results['directory_stats'].items():
        report_content += f"\n{dir_name}:\n"
        report_content += f"  Images: {stats['total_images']}\n"
        report_content += f"  Codes Found: {stats['successful_codes']}\n"
        report_content += f"  Success Rate: {stats['success_rate']:.1%}\n"
        report_content += f"  Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms\n"
        
        total_images += stats['total_images']
        total_codes += stats['successful_codes']
    
    report_content += f"\nOverall Totals:\n"
    report_content += f"  Total Images: {total_images}\n"
    report_content += f"  Total Codes: {total_codes}\n"
    report_content += f"  Overall Success Rate: {total_codes/len(results['all_results']):.1%}\n"
    
    with open(master_report, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Print summary
    print(f"\nBatch Processing Complete!")
    print(f"Processed {len(directories)} directories in {total_time:.1f}s")
    print(f"Total: {total_images} images, {total_codes} Aztec codes found")
    print(f"Results saved to: {output_base.resolve()}")
    
    logger.info("Batch processing completed successfully")

if __name__ == "__main__":
    main()