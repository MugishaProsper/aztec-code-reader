"""Simple tests for Aztec Code Reader"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import os
from aztec_reader import AztecProcessor
from config import ProcessingConfig
from utils import get_image_files, calculate_processing_stats, format_file_size

class TestAztecProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = AztecProcessor()
        # Create a simple test image (black square)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_enhance_for_aztec(self):
        """Test image enhancement pipeline"""
        enhanced = self.processor.enhance_for_aztec(self.test_image)
        
        # Should return 8 different processed versions
        self.assertEqual(len(enhanced), 8)
        
        # All should be grayscale
        for img in enhanced:
            self.assertEqual(len(img.shape), 2)
    
    def test_validate_aztec_data(self):
        """Test data validation"""
        # Valid data
        self.assertTrue(self.processor.validate_aztec_data("ABC123"))
        self.assertTrue(self.processor.validate_aztec_data("https://example.com"))
        self.assertTrue(self.processor.validate_aztec_data("data/with/slashes"))
        
        # Invalid data
        self.assertFalse(self.processor.validate_aztec_data(""))
        self.assertFalse(self.processor.validate_aztec_data("ab"))  # Too short
        self.assertFalse(self.processor.validate_aztec_data(None))

class TestUtils(unittest.TestCase):
    
    def test_format_file_size(self):
        """Test file size formatting"""
        self.assertEqual(format_file_size(512), "512.0B")
        self.assertEqual(format_file_size(1024), "1.0KB")
        self.assertEqual(format_file_size(1536), "1.5KB")
        self.assertEqual(format_file_size(1048576), "1.0MB")
    
    def test_get_image_files(self):
        """Test image file collection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.jpg").touch()
            (temp_path / "test2.png").touch()
            (temp_path / "test3.txt").touch()  # Should be ignored
            
            config = ProcessingConfig()
            files = get_image_files(temp_path, config.supported_formats)
            
            # Should find 2 image files
            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.suffix.lower() in ['.jpg', '.png'] for f in files))
    
    def test_calculate_processing_stats(self):
        """Test statistics calculation"""
        records = [
            {'filename': 'test1.jpg', 'aztec_index': 1, 'data': 'ABC123', 'processing_time_ms': 100},
            {'filename': 'test1.jpg', 'aztec_index': 2, 'data': 'DEF456', 'processing_time_ms': 150},
            {'filename': 'test2.jpg', 'aztec_index': 0, 'data': 'NO AZTEC CODE', 'processing_time_ms': 50},
            {'filename': 'test3.jpg', 'aztec_index': 0, 'data': 'ERROR: Cannot load', 'processing_time_ms': 25},
        ]
        
        stats = calculate_processing_stats(records)
        
        self.assertEqual(stats['total_images'], 3)
        self.assertEqual(stats['successful_codes'], 2)
        self.assertEqual(stats['error_count'], 1)
        self.assertEqual(stats['total_processing_time_ms'], 325)

class TestConfig(unittest.TestCase):
    
    def test_processing_config_defaults(self):
        """Test default configuration values"""
        config = ProcessingConfig()
        
        self.assertEqual(config.clahe_clip_limit, 4.0)
        self.assertEqual(config.clahe_grid_size, (8, 8))
        self.assertEqual(config.min_data_length, 3)
        self.assertIsNotNone(config.supported_formats)
    
    def test_config_serialization(self):
        """Test config save/load"""
        config = ProcessingConfig(clahe_clip_limit=3.0, clahe_grid_size=(16, 16))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save_to_file(temp_path)
            loaded_config = ProcessingConfig.from_file(temp_path)
            
            self.assertEqual(loaded_config.clahe_clip_limit, 3.0)
            self.assertEqual(loaded_config.clahe_grid_size, (16, 16))
        finally:
            temp_path.unlink()

if __name__ == '__main__':
    unittest.main()