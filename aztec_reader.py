import os
import cv2
import csv
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from pyzbar import pyzbar

class AztecProcessor:
    """Enhanced Aztec code processor with configurable parameters"""
    
    def __init__(self, clahe_clip_limit: float = 4.0, clahe_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.logger = logging.getLogger(__name__)
    
    def enhance_for_aztec(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocessing optimized for Aztec codes with enhanced techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        versions = [gray]

        # 1. CLAHE - boosts local contrast
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        versions.append(clahe.apply(gray))

        # 2. Otsu binarization
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(thresh)

        # 3. Inverted threshold
        _, inv_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        versions.append(inv_thresh)

        # 4. Morphological closing (fill gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        versions.append(closed)

        # 5. Sharpening
        blur = cv2.GaussianBlur(gray, (0,0), 2)
        sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
        versions.append(sharp)

        # 6. High-pass filter (edge enhancement)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        versions.append(laplacian)

        # 7. Adaptive threshold (new enhancement)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        versions.append(adaptive)

        # 8. Bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        versions.append(bilateral)

        return versions

    def validate_aztec_data(self, data: str) -> bool:
        """Validate if decoded data looks like valid Aztec content"""
        if not data or len(data) < 3:
            return False
        
        # Check for common patterns in valid Aztec codes
        valid_patterns = [
            data.isalnum(),  # Alphanumeric
            any(char in data for char in ['/', ':', '=', '+', '-']),  # URL/Base64 patterns
            data.startswith(('http', 'https', 'ftp')),  # URLs
        ]
        
        return any(valid_patterns)

    def decode_aztec_codes(self, image: np.ndarray) -> List[Dict]:
        """Decode only Aztec codes with enhanced validation"""
        enhanced = self.enhance_for_aztec(image)
        all_aztec = []

        for i, proc in enumerate(enhanced):
            try:
                # Force Aztec symbol only
                decoded = pyzbar.decode(proc, symbols=[pyzbar.ZBarSymbol.AZTEC])
                for az in decoded:
                    try:
                        data = az.data.decode('utf-8', errors='replace')
                        
                        # Validate data quality
                        if not self.validate_aztec_data(data):
                            self.logger.debug(f"Skipping invalid data: {data[:50]}...")
                            continue
                            
                        x, y, w, h = az.rect.left, az.rect.top, az.rect.width, az.rect.height

                        # Get polygon for rotated codes
                        points = az.polygon
                        if len(points) < 3:
                            points = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]

                        all_aztec.append({
                            'data': data,
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'points': points,
                            'confidence': getattr(az, 'quality', 0),
                            'processing_method': i
                        })
                    except Exception as e:
                        self.logger.warning(f"Error decoding Aztec data: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Error in processing method {i}: {e}")

        # Remove duplicates with improved logic
        seen = set()
        unique = []
        for a in all_aztec:
            # Use data and approximate position for deduplication
            key = (a['data'], a['x'] // 10, a['y'] // 10)  # Group nearby positions
            if key not in seen:
                seen.add(key)
                unique.append(a)
        
        return sorted(unique, key=lambda x: x.get('confidence', 0), reverse=True)

    def draw_aztec(self, image: np.ndarray, aztec_list: List[Dict], output_path: str):
        """Draw Aztec codes with enhanced visualization"""
        img = image.copy()
        colors = [(0, 165, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]  # Multiple colors
        
        for i, az in enumerate(aztec_list):
            color = colors[i % len(colors)]
            
            # Draw accurate polygon
            pts = np.array(az['points'], np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
            
            # Fill polygon with semi-transparent overlay
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            # Enhanced label with confidence
            preview = az['data'][:40] + "..." if len(az['data']) > 40 else az['data']
            confidence = az.get('confidence', 0)
            label = f"Aztec {i+1} (Q:{confidence}): {preview}"
            
            # Multi-line text for long labels
            label_lines = [label[j:j+60] for j in range(0, len(label), 60)]
            for line_idx, line in enumerate(label_lines):
                y_offset = az['y'] - 15 - (len(label_lines) - line_idx - 1) * 20
                cv2.putText(img, line, (az['x'], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(output_path, img)

def process_image(img_path: str, output_dir: str, draw: bool, processor: AztecProcessor) -> List[Dict]:
    """Process single image with enhanced error handling and metadata"""
    start_time = datetime.now()
    
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Cannot load image - unsupported format or corrupted file")

        # Get image metadata
        height, width = image.shape[:2]
        file_size = os.path.getsize(img_path)
        
        aztec_codes = processor.decode_aztec_codes(image)
        records = []

        print(f"\n{Path(img_path).name}: {len(aztec_codes)} Aztec code(s) "
              f"({width}x{height}, {file_size/1024:.1f}KB)")

        for i, az in enumerate(aztec_codes):
            record = {
                'filename': Path(img_path).name,
                'aztec_index': i + 1,
                'data': az['data'],
                'x': az['x'], 'y': az['y'],
                'width': az['w'], 'height': az['h'],
                'confidence': az.get('confidence', 0),
                'processing_method': az.get('processing_method', 0),
                'image_width': width,
                'image_height': height,
                'file_size_kb': round(file_size/1024, 1),
                'processing_time_ms': 0  # Will be updated below
            }
            records.append(record)
            
            # Enhanced output with confidence
            confidence_str = f" (Q:{az.get('confidence', 0)})" if az.get('confidence') else ""
            print(f"  [Aztec{i+1}]{confidence_str} {az['data']}")

        if draw and aztec_codes:
            out_path = os.path.join(output_dir, f"aztec_{Path(img_path).name}")
            processor.draw_aztec(image, aztec_codes, out_path)
            print(f"  Saved: {out_path}")

        if not aztec_codes:
            records.append({
                'filename': Path(img_path).name,
                'aztec_index': 0,
                'data': 'NO AZTEC CODE',
                'x': '', 'y': '', 'width': '', 'height': '',
                'confidence': 0,
                'processing_method': -1,
                'image_width': width,
                'image_height': height,
                'file_size_kb': round(file_size/1024, 1),
                'processing_time_ms': 0
            })

        # Update processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        for record in records:
            record['processing_time_ms'] = round(processing_time, 1)

        return records

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logging.error(f"Error processing {img_path}: {e}")
        print(f"Error: {img_path} -> {e}")
        
        return [{
            'filename': Path(img_path).name,
            'aztec_index': 0,
            'data': f'ERROR: {e}',
            'x': '', 'y': '', 'width': '', 'height': '',
            'confidence': 0,
            'processing_method': -1,
            'image_width': 0,
            'image_height': 0,
            'file_size_kb': 0,
            'processing_time_ms': round(processing_time, 1)
        }]

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('aztec_reader.log'),
            logging.StreamHandler()
        ]
    )

def save_results(all_records: List[Dict], csv_path: Path, json_path: Optional[Path] = None) -> None:
    """Save results in multiple formats"""
    # Enhanced CSV with all fields
    fieldnames = [
        'filename', 'aztec_index', 'data', 'x', 'y', 'width', 'height',
        'confidence', 'processing_method', 'image_width', 'image_height', 
        'file_size_kb', 'processing_time_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    
    # Optional JSON export
    if json_path:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_images': len(set(r['filename'] for r in all_records)),
                    'total_codes': len([r for r in all_records if r['aztec_index'] > 0])
                },
                'results': all_records
            }, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Aztec Code Reader",
        epilog="""
Examples:
  python aztec_reader.py boarding_pass.jpg --draw
  python aztec_reader.py "C:\\Tickets\\" -o results --csv data.csv --json data.json
  python aztec_reader.py images/ --verbose --clahe-clip 3.0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("-o", "--output", default="aztec_output", help="Output folder")
    parser.add_argument("--csv", default="aztec_codes.csv", help="CSV output file")
    parser.add_argument("--json", help="Optional JSON output file")
    parser.add_argument("--draw", action="store_true", help="Save images with Aztec boxes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--clahe-clip", type=float, default=4.0, help="CLAHE clip limit (default: 4.0)")
    parser.add_argument("--clahe-grid", type=int, nargs=2, default=[8, 8], help="CLAHE grid size (default: 8 8)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Initialize processor with custom parameters
    processor = AztecProcessor(
        clahe_clip_limit=args.clahe_clip,
        clahe_grid_size=tuple(args.clahe_grid)
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect images with more formats
    supported_formats = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif", 
                        "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIFF", "*.TIF"]
    
    if input_path.is_dir():
        images = []
        for ext in supported_formats:
            images.extend(input_path.glob(ext))
        images.sort()  # Process in consistent order
    else:
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return
        images = [input_path]

    if not images:
        logger.error("No supported images found!")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF")
        return

    logger.info(f"Processing {len(images)} image(s)")
    print(f"Processing {len(images)} image(s)...")

    # Process images
    all_records = []
    start_time = datetime.now()
    
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img.name}")
        records = process_image(str(img), str(output_dir), args.draw, processor)
        all_records.extend(records)

    # Save results
    csv_path = Path(args.csv)
    json_path = Path(args.json) if args.json else None
    
    save_results(all_records, csv_path, json_path)

    # Summary statistics
    total_time = (datetime.now() - start_time).total_seconds()
    successful_codes = len([r for r in all_records if r['aztec_index'] > 0 and not r['data'].startswith('ERROR')])
    total_images = len(set(r['filename'] for r in all_records))
    
    print(f"\n{'='*50}")
    print(f"Aztec Reading Complete!")
    print(f"  Processed: {total_images} images in {total_time:.1f}s")
    print(f"  Found: {successful_codes} Aztec codes")
    print(f"  CSV: {csv_path.resolve()}")
    if json_path:
        print(f"  JSON: {json_path.resolve()}")
    if args.draw:
        print(f"  Images: {output_dir.resolve()}")
    print(f"  Log: aztec_reader.log")

if __name__ == "__main__":
    main()