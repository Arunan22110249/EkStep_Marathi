#!/usr/bin/env python3
import sys
import traceback
from datasets import load_dataset, Audio

print("Loading dataset...")
ds = load_dataset('SPRINGLab/IndicTTS_Marathi', split='train', streaming=False)
print(f"Loaded {len(ds)} examples")

# Don't decode audio - keep as paths
# Instead, use sampling_rate only without decoding
try:
    print("Accessing first example (raw, no decoding)...")
    # Get raw features without casting to Audio
    print(f"Dataset features: {ds.features}")
    
    # Access raw row
    ex_raw = ds.data.to_pylist()[0]
    print(f"Raw example keys: {list(ex_raw.keys())}")
    print(f"Audio field: {ex_raw.get('audio')}")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
