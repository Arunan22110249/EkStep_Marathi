#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Python version:", sys.version)
print("Importing marathi_tts...")

try:
    from marathi_tts.config import TrainingConfig
    print("✓ config imported")
except Exception as e:
    print(f"✗ config import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from marathi_tts.training import create_training_dataset
    print("✓ training imported")
except Exception as e:
    print(f"✗ training import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All imports successful!")
