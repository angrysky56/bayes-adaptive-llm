#!/usr/bin/env python
"""
Check imports to verify that our fixes worked correctly.
"""

import sys
from pathlib import Path

# Add the repo root to the path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def check_import(module_name, required_items=None):
    """
    Check if module can be imported and, optionally, if required items are present.
    
    Args:
        module_name: Name of the module to import
        required_items: Optional list of items that should be in the module
        
    Returns:
        True if import succeeded, False otherwise
    """
    try:
        module = __import__(module_name)
        
        if required_items:
            # Check if all required items are present
            for item in required_items:
                if not hasattr(module, item):
                    print(f"ERROR: {module_name} does not have {item}")
                    return False
                
        print(f"SUCCESS: {module_name} imported successfully")
        if required_items:
            print(f"  Contains: {', '.join(required_items)}")
            
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error importing {module_name}: {e}")
        return False

def main():
    """
    Main function to check imports.
    """
    # Check utils module
    utils_success = check_import("utils", ["apply_svd", "update_with_svf", "setup_logging", "get_logger"])
    
    # Check svf module
    svf_success = check_import("svf", ["SVF", "apply_expert_vector", "ExpertVector", "ExpertManager"])
    
    # Check if can import from bayesian_adaptive_inference
    try:
        from scripts.bayesian_adaptive_inference import run_inference
        print("SUCCESS: run_inference from scripts.bayesian_adaptive_inference imported successfully")
        inference_success = True
    except ImportError as e:
        print(f"ERROR: Failed to import run_inference from scripts.bayesian_adaptive_inference: {e}")
        inference_success = False
    except Exception as e:
        print(f"ERROR: Unexpected error importing run_inference: {e}")
        inference_success = False
    
    # Check overall status
    if utils_success and svf_success and inference_success:
        print("\nAll imports successful! The fixes worked.")
        return 0
    else:
        print("\nSome imports failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
