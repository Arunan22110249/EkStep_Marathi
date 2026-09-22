#!/usr/bin/env python3
"""Debug script to inspect dataset examples and trace where processing fails."""

import sys
import traceback
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Open log file with UTF-8 encoding
log_file = Path(__file__).parent.parent / "debug_output.log"
log_fp = open(log_file, 'w', encoding='utf-8')

def log(msg):
    """Write to console and log file, handling encoding."""
    print(msg, flush=True)
    try:
        log_fp.write(msg + '\n')
        log_fp.flush()
    except UnicodeEncodeError:
        # Fall back to ASCII-safe version
        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        log_fp.write(safe_msg + '\n')
        log_fp.flush()

from marathi_tts.preprocess import DataPipelineProcessor

def debug_pipeline():
    """Load 10 examples and trace where processing fails."""
    
    log("\n" + "="*80)
    log("DEBUG: Dataset Inspection and Pipeline Tracing")
    log("="*80)
    
    # Load dataset
    log("\n[Step 1] Loading dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "SPRINGLab/IndicTTS_Marathi",
            split="train",
            streaming=False,
        )
        
        if len(ds) > 10:
            ds = ds.select(range(10))
        
        log(f"✓ Loaded {len(ds)} examples")
    except Exception as e:
        log(f"✗ Failed to load dataset: {e}")
        traceback.print_exc(file=log_fp)
        return
    
    # Initialize processor
    log("\n[Step 2] Initializing processor...")
    try:
        processor = DataPipelineProcessor(device="cpu")
        log("✓ Processor initialized")
    except Exception as e:
        log(f"✗ Failed to init processor: {e}")
        traceback.print_exc(file=log_fp)
        return
    
    # Inspect first example
    log("\n[Step 3] Inspecting first example structure...")
    try:
        ex0 = ds[0]
        log(f"  Keys: {list(ex0.keys())}")
        log(f"  text type: {type(ex0.get('text'))}")
        log(f"  audio type: {type(ex0.get('audio'))}")
        
        if isinstance(ex0.get('audio'), dict):
            log(f"    audio keys: {list(ex0['audio'].keys())}")
            audio_array = ex0['audio'].get('array')
            if audio_array is not None:
                log(f"    array type: {type(audio_array)}, dtype: {getattr(audio_array, 'dtype', 'N/A')}")
                if hasattr(audio_array, 'shape'):
                    log(f"    array shape: {audio_array.shape}")
            sr = ex0['audio'].get('sampling_rate')
            log(f"    sampling_rate: {sr}")
    except Exception as e:
        log(f"  ERROR accessing first example: {e}")
        traceback.print_exc(file=log_fp)
    
    # Process each example with full traceback
    log("\n[Step 4] Processing examples with error tracing...")
    log("-" * 80)
    
    for idx in range(len(ds)):
        ex = ds[idx]
        log(f"\nExample {idx+1}:")
        log(f"  Text preview: {ex.get('text', '')[:60]}")
        
        try:
            result = processor.process(ex)
            log(f"  ✓ Processed: duration={result['audio_duration_sec']:.2f}s, "
                  f"frames={result['n_frames']}, tokens={result['n_tokens']}")
        except Exception as e:
            log(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:100]}")
            log(f"  Full traceback:")
            traceback.print_exc(file=log_fp)
            log("")
            # Stop after first error
            break
    
    log("\n" + "="*80)

if __name__ == "__main__":
    try:
        debug_pipeline()
    finally:
        log_fp.close()
        print(f"\n✓ Debug log written to: {log_file}")

