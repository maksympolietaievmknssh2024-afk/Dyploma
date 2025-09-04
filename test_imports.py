#!/usr/bin/env python3
"""
Test script to check if imports work correctly.
"""

try:
    print("Testing basic imports...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    from models.diffusion_model import DiffusionModel
    print("DiffusionModel imported successfully")
    
    from utils.text_processing import TextProcessor
    print("TextProcessor imported successfully")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()