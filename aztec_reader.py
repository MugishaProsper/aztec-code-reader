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

def decode_aztec_codes(image: np.ndarray) -> List[Dict]:
    """Decode only Aztec codes"""
    enhanced = enhance_for_aztec(image)
    all_aztec = []

    for proc in enhanced:
        # Force Aztec symbol only
        decoded = pyzbar.decode(proc, symbols=[pyzbar.ZBarSymbol.AZTEC])
        for az in decoded:
            data = az.data.decode('utf-8', errors='replace')
            x, y, w, h = az.rect.left, az.rect.top, az.rect.width, az.rect.height

            # Get polygon for rotated codes
            points = az.polygon
            if len(points) < 3:
                points = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]

            all_aztec.append({
                'data': data,
                'x': x, 'y': y, 'w': w, 'h': h,
                'points': points
            })

    # Remove duplicates
    seen = set()
    unique = []
    for a in all_aztec:
        key = (a['data'], a['x'], a['y'])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique

def draw_aztec(image: np.ndarray, aztec_list: List[Dict], output_path: str):
    """Draw Aztec codes with orange boxes"""
    img = image.copy()
    for i, az in enumerate(aztec_list):
        # Draw accurate polygon
        pts = np.array(az['points'], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 165, 255), thickness=3)  # Orange

        # Label
        preview = az['data'][:50] + "..." if len(az['data']) > 50 else az['data']
        label = f"Aztec {i+1}: {preview}"
        cv2.putText(img, label, (az['x'], az['y'] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imwrite(output_path, img)

def process_image(img_path: str, output_dir: str, draw: bool) -> List[Dict]:
    """Process single image"""
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Cannot load image")

        aztec_codes = decode_aztec_codes(image)
        records = []

        print(f"\n{Path(img_path).name}: {len(aztec_codes)} Aztec code(s)")

        for i, az in enumerate(aztec_codes):
            record = {
                'filename': Path(img_path).name,
                'aztec_index': i + 1,
                'data': az['data'],
                'x': az['x'], 'y': az['y'],
                'width': az['w'], 'height': az['h']
            }
            records.append(record)
            print(f"  [Aztec{i+1}] {az['data']}")

        if draw and aztec_codes:
            out_path = os.path.join(output_dir, f"aztec_{Path(img_path).name}")
            draw_aztec(image, aztec_codes, out_path)
            print(f"  Saved: {out_path}")

        if not aztec_codes:
            records.append({
                'filename': Path(img_path).name,
                'aztec_index': 0,
                'data': 'NO AZTEC CODE',
                'x': '', 'y': '', 'width': '', 'height': ''
            })

        return records

    except Exception as e:
        print(f"Error: {img_path} -> {e}")
        return [{
            'filename': Path(img_path).name,
            'aztec_index': 0,
            'data': f'ERROR: {e}',
            'x': '', 'y': '', 'width': '', 'height': ''
        }]

def main():
    parser = argparse.ArgumentParser(
        description="Aztec Code Reader - Windows",
        epilog="""
Examples:
  python read_aztec.py boarding_pass.jpg --draw
  python read_aztec.py "C:\\Tickets\\" -o aztec_results --csv aztec_data.csv
        """
    )
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("-o", "--output", default="aztec_output", help="Output folder")
    parser.add_argument("--csv", default="aztec_codes.csv", help="CSV file")
    parser.add_argument("--draw", action="store_true", help="Save images with Aztec boxes")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect images
    if input_path.is_dir():
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.PNG", "*.JPG"]:
            images.extend(input_path.glob(ext))
    else:
        images = [input_path]

    if not images:
        print("No images found!")
        return

    all_records = []
    for img in images:
        records = process_image(str(img), str(output_dir), args.draw)
        all_records.extend(records)

    # Save CSV
    csv_path = Path(args.csv)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'aztec_index', 'data', 'x', 'y', 'width', 'height'
        ])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nAztec Reading Complete!")
    print(f"  CSV: {csv_path.resolve()}")
    if args.draw:
        print(f"  Images: {output_dir.resolve()}")

if __name__ == "__main__":
    main()