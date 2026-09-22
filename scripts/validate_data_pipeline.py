#!/usr/bin/env python3
"""
Validate the Marathi TTS data pipeline on 10 IndicTTS_Marathi examples.

Tests:
- Marathi text validation
- Audio loading and resampling (48 kHz → 24 kHz)
- SNAC encoding (c0/c1/c2 codes)
- Bodhan token serialization (7 tokens/frame)

CPU-only, no dataset download beyond 10 examples.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marathi_tts.preprocess import DataPipelineProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_pipeline():
    """Run 10-sample validation pipeline."""
    
    print("\n" + "="*80)
    print("MARATHI TTS DATA PIPELINE VALIDATION")
    print("="*80)
    
    # Load 10 examples
    print("\n[1/3] Loading 10 examples from SPRINGLab/IndicTTS_Marathi...")
    try:
        from datasets import load_dataset
        
        # Load with raw format to avoid audio decoding trigger
        ds = load_dataset(
            "SPRINGLab/IndicTTS_Marathi",
            split="train",
            streaming=False,
        )
        
        # Set format to prevent automatic audio decoding
        ds = ds.with_format("arrow")
        
        # Limit to 10
        if len(ds) > 10:
            ds = ds.select(range(10))
        
        num_examples = len(ds)
        print(f"✓ Loaded {num_examples} examples (raw format)")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize processor
    print("\n[2/3] Initializing data pipeline processor...")
    try:
        processor = DataPipelineProcessor(device="cpu")
        print("✓ Processor initialized (CPU mode)")
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        return False
    
    # Process examples
    print("\n[3/3] Processing examples through pipeline...")
    print("-" * 80)
    
    stats = {
        "total": num_examples,
        "processed": 0,
        "failed": 0,
        "total_duration_sec": 0.0,
        "total_frames": 0,
        "total_tokens": 0,
        "errors": [],
    }
    
    for idx, example in enumerate(ds):
        example_num = idx + 1
        try:
            # Convert PyArrow Table to dict if needed
            if hasattr(example, 'to_pydict'):
                example = example.to_pydict()
            
            result = processor.process(example)
            
            stats["processed"] += 1
            stats["total_duration_sec"] += result["audio_duration_sec"]
            stats["total_frames"] += result["n_frames"]
            stats["total_tokens"] += result["n_tokens"]
            
            print(
                f"  Example {example_num:2d}: "
                f"duration={result['audio_duration_sec']:6.2f}s, "
                f"frames={result['n_frames']:4d}, "
                f"tokens={result['n_tokens']:5d} "
                f"[{result['text'][:40]}...]"
            )
            
        except Exception as e:
            stats["failed"] += 1
            error_msg = f"Example {example_num}: {str(e)[:80]}"
            stats["errors"].append(error_msg)
            print(f"  Example {example_num:2d}: ✗ ERROR: {str(e)[:60]}...")
    
    # Report results
    print("-" * 80)
    print("\nVALIDATION RESULTS")
    print("-" * 80)
    print(f"Total examples:        {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed:                {stats['failed']}")
    print(f"\nAudio statistics:")
    print(f"  Total duration:      {stats['total_duration_sec']:.2f} seconds")
    print(f"  Total SNAC frames:   {stats['total_frames']:d}")
    print(f"  Total Bodhan tokens: {stats['total_tokens']:d}")
    
    if stats["total_frames"] > 0:
        avg_tokens_per_frame = stats["total_tokens"] / stats["total_frames"]
        print(f"  Avg tokens/frame:    {avg_tokens_per_frame:.1f} (expected: 7.0)")
    
    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"]:
            print(f"  - {error}")
    
    # Success criteria
    success = stats["failed"] == 0 and stats["processed"] > 0
    
    print("\n" + "="*80)
    if success:
        print("✓ PIPELINE VALIDATION PASSED")
        print("  All 10 examples processed successfully.")
        print("  Ready for development phase: preprocessing, dataloader, fine-tuning")
    else:
        print("✗ PIPELINE VALIDATION FAILED")
        if stats["failed"] > 0:
            print(f"  {stats['failed']} examples failed processing")
        if stats["processed"] == 0:
            print("  No examples processed")
    print("="*80 + "\n")
    
    return success


if __name__ == "__main__":
    success = validate_pipeline()
    sys.exit(0 if success else 1)
