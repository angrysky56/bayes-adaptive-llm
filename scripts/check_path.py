#!/usr/bin/env python
"""
Check Python's path for debugging import issues
"""

import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nCurrent directory:")
print(f"  {os.getcwd()}")

print("\nParent directory:")
print(f"  {os.path.dirname(os.getcwd())}")

print("\nChecking if utils module exists:")
try:
    import utils
    print("  utils module found")
    print(f"  utils module path: {utils.__file__}")
    
    try:
        from utils import svd
        print("  svd module found")
        print(f"  svd module path: {svd.__file__}")
    except ImportError as e:
        print(f"  Error importing svd: {e}")
        
except ImportError as e:
    print(f"  Error importing utils: {e}")
