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
    source_path = os.path.join(os.getcwd(), 'test_source.jpg')
    source_img.save(source_path)
    
    # Create a simple reference image (makeup)
    reference_img = Image.new('RGB', (512, 512), color='lightpink')
    reference_path = os.path.join(os.getcwd(), 'test_reference.jpg')
    reference_img.save(reference_path)
    
    print("✅ Created test images")
    return source_path, reference_path

def test_predictor():
    """Test the Predictor class"""
    
    try:
        from predict import Predictor
        
        print("🚀 Testing Predictor setup...")
        predictor = Predictor()
        predictor.setup()
        
        print("🎨 Testing makeup transfer...")
        source_path, reference_path = create_test_images()
        
        result_path = predictor.predict(
            source_image=source_path,
            reference_image=reference_path,
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
    
    # Test the predictor
    success = test_predictor()
    
    if success:
        print("🎉 All tests passed! Ready for Replicate deployment.")
    else:
        print("⚠️ Tests failed. Check the setup before deploying.") 