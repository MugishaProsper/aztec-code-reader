# Enhanced Aztec Code Reader

A robust Python application for detecting and decoding Aztec barcodes from images with advanced preprocessing and comprehensive output options.

## Features

- **Multi-format Support**: PNG, JPG, JPEG, BMP, TIFF
- **Advanced Image Processing**: 8 different preprocessing techniques including CLAHE, morphological operations, and adaptive thresholding
- **Batch Processing**: Process single images or entire directories
- **Multiple Output Formats**: CSV and JSON export with detailed metadata
- **Visual Debugging**: Generate annotated images showing detected codes
- **Configurable Parameters**: Adjust CLAHE settings for different image types
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Quality Validation**: Automatic filtering of low-quality detections

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
# Process single image
python aztec_reader.py boarding_pass.jpg --draw

# Process directory
python aztec_reader.py images/ --csv results.csv --json results.json
```

### Advanced Options
```bash
# Custom CLAHE settings for low-contrast images
python aztec_reader.py images/ --clahe-clip 3.0 --clahe-grid 16 16

# Verbose logging
python aztec_reader.py images/ --verbose

# Custom output locations
python aztec_reader.py images/ -o output_folder --csv data.csv --json data.json
```

## Output Formats

### CSV Output
Contains detailed information for each detected code:
- Basic info: filename, position, dimensions
- Quality metrics: confidence score, processing method
- Image metadata: dimensions, file size
- Performance: processing time

### JSON Output
Structured format with metadata and results array, ideal for programmatic processing.

### Annotated Images
When using `--draw`, creates images with:
- Color-coded bounding polygons
- Confidence scores
- Multi-line labels for long data
- Semi-transparent overlays

## Configuration

The processor supports several configurable parameters:

- `--clahe-clip`: CLAHE clip limit (default: 4.0)
- `--clahe-grid`: CLAHE grid size (default: 8x8)
- `--verbose`: Enable debug logging

## Processing Methods

The application uses 8 different image preprocessing techniques:

1. **Original Grayscale**: Base conversion
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Otsu Thresholding**: Automatic binary thresholding
4. **Inverted Otsu**: For white-on-black codes
5. **Morphological Closing**: Fill gaps in patterns
6. **Sharpening**: Enhance edge definition
7. **Laplacian Filter**: High-pass edge enhancement
8. **Adaptive Threshold**: Local thresholding
9. **Bilateral Filter**: Noise reduction with edge preservation

## Troubleshooting

- **No codes detected**: Try adjusting `--clahe-clip` parameter (lower for high-contrast, higher for low-contrast images)
- **False positives**: Check the confidence scores in output
- **Performance issues**: Use `--verbose` to identify bottlenecks

## Requirements

- Python 3.7+
- OpenCV 4.8+
- pyzbar 0.1.9+
- NumPy 1.24+
- Pillow 10.0+