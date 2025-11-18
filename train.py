# train.py
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ml_pipeline.train_models import main
    print("✅ Successfully imported training module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📁 Current directory:", os.getcwd())
    print("📁 Project root:", project_root)
    print("🔍 Checking if files exist...")
    
    # Check if required files exist
    files_to_check = [
        "ml_pipeline/__init__.py",
        "ml_pipeline/train_models.py", 
        "src/__init__.py",
        "src/config.py"
    ]
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        exists = full_path.exists()
        print(f"   {file_path}: {'✅ EXISTS' if exists else '❌ MISSING'}")
    
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Model Training...")
    main()