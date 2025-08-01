#!/usr/bin/env python3
"""
Local test script for Stable-Makeup model
Run this to test the model before deploying to Replicate
"""

import os
import sys
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.append(os.getcwd())

def create_test_images():
    """Create simple test images for testing"""
    
    # Create a simple source image (face)
    source_img = Image.new('RGB', (512, 512), color='peachpuff')
    source_img.save('test_source.jpg')
    
    # Create a simple reference image (makeup)
    reference_img = Image.new('RGB', (512, 512), color='lightpink')
    reference_img.save('test_reference.jpg')
    
    print("✅ Created test images")

def test_predictor():
    """Test the Predictor class"""
    
    try:
        from predict import Predictor
        
        print("🚀 Testing Predictor setup...")
        predictor = Predictor()
        predictor.setup()
        
        print("🎨 Testing makeup transfer...")
        result_path = predictor.predict(
            source_image="test_source.jpg",
            reference_image="test_reference.jpg",
            makeup_intensity=1.0
        )
        
        print(f"✅ Test completed! Result saved to: {result_path}")
        
        # Load and display result info
        result_img = Image.open(result_path)
        print(f"📊 Result image size: {result_img.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Starting local test...")
    
    # Create test images
    create_test_images()
    
    # Test the predictor
    success = test_predictor()
    
    if success:
        print("🎉 All tests passed! Ready for Replicate deployment.")
    else:
        print("⚠️ Tests failed. Check the setup before deploying.") 