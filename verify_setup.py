#!/usr/bin/env python3
"""
Verification script for DD-Checklist setup
Tests that all dependencies are correctly installed and configured
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✅ Sentence Transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence Transformers: {e}")
        return False
    
    try:
        import PyPDF2
        print(f"✅ PyPDF2 {PyPDF2.__version__}")
    except ImportError as e:
        print(f"❌ PyPDF2: {e}")
        return False
    
    try:
        import docx
        print(f"✅ python-docx")
    except ImportError as e:
        print(f"❌ python-docx: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import faiss
        print(f"✅ FAISS")
    except ImportError as e:
        print(f"❌ FAISS: {e}")
        return False
    
    try:
        import langchain_anthropic
        print(f"✅ LangChain Anthropic")
    except ImportError as e:
        print(f"❌ LangChain Anthropic: {e}")
        return False
    
    try:
        import langgraph
        print(f"✅ LangGraph")
    except ImportError as e:
        print(f"❌ LangGraph: {e}")
        return False
    
    return True

def test_data_structure():
    """Test that the data directory structure is correct"""
    print("\n📁 Testing data structure...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data/ directory not found")
        return False
    
    required_dirs = ["checklist", "questions", "strategy", "vdrs"]
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"❌ data/{dir_name}/ directory not found")
            return False
        print(f"✅ data/{dir_name}/ exists")
    
    # Check for some content
    vdrs_dir = data_dir / "vdrs"
    project_dirs = [d for d in vdrs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"✅ Found {len(project_dirs)} projects in VDRs")
    
    return True

def test_app_loading():
    """Test that the main app can be imported"""
    print("\n🚀 Testing app loading...")
    
    try:
        # Test if we can import the main app components
        import app
        print("✅ Main app module imported")
        
        # Test model loading (this will cache the model)
        model = app.load_model()
        print(f"✅ AI model loaded: {type(model).__name__}")
        
        # Test config imports
        from langgraph_config import LANGGRAPH_AVAILABLE, DDChecklistAgent
        print(f"✅ LangGraph config imported (available: {LANGGRAPH_AVAILABLE})")
        
        from vector_store_config import get_vector_store
        print("✅ Vector store config imported")
        
        return True
    except Exception as e:
        print(f"❌ App loading failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🔧 DD-Checklist Setup Verification")
    print("=" * 40)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python {python_version}")
    
    if sys.version_info < (3, 11):
        print("⚠️  Python 3.11+ recommended")
    else:
        print("✅ Python version OK")
    
    # Run tests
    tests = [
        ("Dependencies", test_imports),
        ("Data Structure", test_data_structure), 
        ("App Loading", test_app_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To start the app, run:")
        print("   ./run.sh")
        print("   or")
        print("   uv run streamlit run app.py")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
