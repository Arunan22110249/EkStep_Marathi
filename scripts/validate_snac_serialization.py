"""
Validate Bodhan SNAC audio-to-token serialization pipeline.

Uses a deterministic synthetic waveform to verify:
1. SNAC encoding produces expected codebook structure
2. Bodhan token serialization matches verified offset mapping
3. Round-trip: audio → SNAC codes → Bodhan tokens → SNAC codes
"""

import sys
import warnings

import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore")


def create_test_waveform(sr: int = 24000, duration_sec: float = 0.2) -> np.ndarray:
    """
    Create a deterministic test waveform: mix of sine waves.
    
    Args:
        sr: Sample rate (24000 Hz)
        duration_sec: Duration in seconds
    
    Returns:
        float32 mono waveform, normalized to [-1, 1]
    """
    n_samples = int(sr * duration_sec)
    t = np.arange(n_samples, dtype=np.float32) / sr
    
    # Mix of frequencies to create an interesting signal
    # 440 Hz (A4) + 880 Hz (A5) + 1320 Hz
    waveform = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.sin(2 * np.pi * 1320 * t)
    )
    
    # Normalize to [-1, 1]
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    return waveform.astype(np.float32)


def inspect_snac_api():
    """Inspect what methods are available in the installed SNAC."""
    try:
        from snac import SNAC
        
        # Try to see what's available
        print("\n=== SNAC API Inspection ===")
        print(f"SNAC class: {SNAC}")
        print(f"Available methods: {[m for m in dir(SNAC) if not m.startswith('_')]}")
        return SNAC
    except Exception as exc:
        print(f"ERROR: Failed to import SNAC: {exc}")
        sys.exit(1)


def encode_with_snac(waveform: np.ndarray, device: str = "cpu") -> dict:
    """
    Encode waveform with SNAC using the full encode pipeline.
    
    Args:
        waveform: float32 mono waveform at 24 kHz
        device: torch device
    
    Returns:
        Dictionary with codes and metadata
    """
    from snac import SNAC
    
    # Load SNAC model
    print(f"\nLoading SNAC from hubertsiuzdak/snac_24khz on {device}...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    
    # Prepare input: convert numpy waveform to torch tensor
    # SNAC expects shape (batch, channels, time)
    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    wav_tensor = wav_tensor.to(device)
    
    print(f"Input waveform shape: {wav_tensor.shape}")
    
    # Use the full encode pipeline
    with torch.no_grad():
        # The encode() method should handle preprocessing, encoding, and quantization
        codes = snac.encode(wav_tensor)
    
    print(f"Encode output type: {type(codes)}")
    
    # Inspect structure
    if isinstance(codes, (list, tuple)):
        print(f"Codes is a sequence of {len(codes)} items:")
        for i, code in enumerate(codes):
            if hasattr(code, 'shape'):
                print(f"  codes[{i}] shape: {code.shape}, dtype: {code.dtype}, "
                      f"min={code.min()}, max={code.max()}")
            else:
                print(f"  codes[{i}] type: {type(code)}")
    elif isinstance(codes, torch.Tensor):
        print(f"Codes tensor: shape={codes.shape}, dtype={codes.dtype}, "
              f"min={codes.min()}, max={codes.max()}")
    
    return {
        "snac": snac,
        "codes": codes,
        "waveform_shape": wav_tensor.shape,
        "device": device,
    }


def extract_hierarchical_codes(encoder_output) -> dict:
    """
    Extract c0, c1, c2 from SNAC encoder output.
    
    SNAC.encode() returns a list of 3 hierarchical codebook levels:
    - Level 0 (c0): 1 code per frame at 12.5 Hz (for 4800 samples at 24 kHz)
    - Level 1 (c1): 2 codes per frame at 25 Hz
    - Level 2 (c2): 4 codes per frame at 50 Hz
    
    For 4800 samples: 3 frames × (1 + 2 + 4) codes/frame = 21 total codes
    """
    codes = encoder_output["codes"]
    
    print(f"\n=== Inspecting Encoder Output ===")
    
    if not isinstance(codes, (tuple, list)):
        raise ValueError(f"Expected list/tuple, got {type(codes)}")
    
    if len(codes) != 3:
        raise ValueError(f"Expected 3 hierarchical levels, got {len(codes)}")
    
    c0_raw, c1_raw, c2_raw = codes
    
    # Convert to numpy and squeeze batch dimension
    c0 = c0_raw.squeeze(0).cpu().numpy() if isinstance(c0_raw, torch.Tensor) else np.squeeze(c0_raw)
    c1 = c1_raw.squeeze(0).cpu().numpy() if isinstance(c1_raw, torch.Tensor) else np.squeeze(c1_raw)
    c2 = c2_raw.squeeze(0).cpu().numpy() if isinstance(c2_raw, torch.Tensor) else np.squeeze(c2_raw)
    
    print(f"c0 (Level 0, c0): shape {c0.shape}, dtype {c0.dtype}, "
          f"min={c0.min()}, max={c0.max()}")
    print(f"c1 (Level 1, 2× rate): shape {c1.shape}, dtype {c1.dtype}, "
          f"min={c1.min()}, max={c1.max()}")
    print(f"c2 (Level 2, 4× rate): shape {c2.shape}, dtype {c2.dtype}, "
          f"min={c2.min()}, max={c2.max()}")
    
    # Verify the frame relationship
    n_frames = c0.shape[0]
    expected_c1_codes = n_frames * 2
    expected_c2_codes = n_frames * 4
    
    assert c1.shape[0] == expected_c1_codes, f"c1 shape mismatch: {c1.shape[0]} vs {expected_c1_codes}"
    assert c2.shape[0] == expected_c2_codes, f"c2 shape mismatch: {c2.shape[0]} vs {expected_c2_codes}"
    
    print(f"\n✓ Frame structure verified:")
    print(f"  Frames: {n_frames}")
    print(f"  Codes per frame: 1 (c0) + 2 (c1) + 4 (c2) = 7 total")
    
    return {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "n_frames": n_frames,
        "num_codebooks": 3,  # 3 hierarchical levels
        "structure": "3_hierarchical_levels",
    }


def serialize_bodhan_tokens(codebooks: dict, base: int = 128266) -> tuple:
    """
    Serialize SNAC codebooks to Bodhan token IDs.
    
    Verified serialization per frame i:
    [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
    
    Token IDs (position-based offsets):
    pos 0: base + 0*4096 + c0[i]
    pos 1: base + 1*4096 + c1[2i]
    pos 2: base + 2*4096 + c2[4i]
    pos 3: base + 3*4096 + c2[4i+1]
    pos 4: base + 4*4096 + c1[2i+1]
    pos 5: base + 5*4096 + c2[4i+2]
    pos 6: base + 6*4096 + c2[4i+3]
    
    Args:
        codebooks: Dictionary with c0, c1, c2 numpy arrays
        base: Base token ID for audio (128266)
    
    Returns:
        (tokens, c0, c1, c2) tuple
    """
    print(f"\n=== Bodhan Token Serialization ===")
    
    c0 = codebooks["c0"]
    c1 = codebooks["c1"]
    c2 = codebooks["c2"]
    n_frames = codebooks["n_frames"]
    
    print(f"Input codebook shapes:")
    print(f"  c0: {c0.shape} (1 code per frame)")
    print(f"  c1: {c1.shape} ({c1.shape[0] // n_frames} codes per frame)")
    print(f"  c2: {c2.shape} ({c2.shape[0] // n_frames} codes per frame)")
    
    # Serialize: for each frame i, generate 7 tokens
    tokens = []
    for i in range(n_frames):
        frame_tokens = []
        
        # Position 0: c0[i]
        frame_tokens.append(int(base + 0 * 4096 + c0[i]))
        
        # Position 1: c1[2i]
        frame_tokens.append(int(base + 1 * 4096 + c1[2*i]))
        
        # Position 2: c2[4i]
        frame_tokens.append(int(base + 2 * 4096 + c2[4*i]))
        
        # Position 3: c2[4i+1]
        frame_tokens.append(int(base + 3 * 4096 + c2[4*i+1]))
        
        # Position 4: c1[2i+1]
        frame_tokens.append(int(base + 4 * 4096 + c1[2*i+1]))
        
        # Position 5: c2[4i+2]
        frame_tokens.append(int(base + 5 * 4096 + c2[4*i+2]))
        
        # Position 6: c2[4i+3]
        frame_tokens.append(int(base + 6 * 4096 + c2[4*i+3]))
        
        tokens.extend(frame_tokens)
    
    tokens = np.array(tokens, dtype=np.int32)
    print(f"\nSerialized token count: {len(tokens)} ({len(tokens) // 7} frames × 7)")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    
    return tokens, c0, c1, c2


def validate_token_ranges(tokens: np.ndarray, base: int = 128266) -> bool:
    """
    Validate that all tokens fall into the expected position-specific ranges.
    
    Args:
        tokens: Array of serialized token IDs
        base: Base token ID (128266)
    
    Returns:
        True if all tokens are valid, False otherwise
    """
    print(f"\n=== Token Range Validation ===")
    
    range_specs = [
        (0, base + 0 * 4096, base + 0 * 4096 + 4096),
        (1, base + 1 * 4096, base + 1 * 4096 + 4096),
        (2, base + 2 * 4096, base + 2 * 4096 + 4096),
        (3, base + 3 * 4096, base + 3 * 4096 + 4096),
        (4, base + 4 * 4096, base + 4 * 4096 + 4096),
        (5, base + 5 * 4096, base + 5 * 4096 + 4096),
        (6, base + 6 * 4096, base + 6 * 4096 + 4096),
    ]
    
    n_frames = len(tokens) // 7
    for frame_idx in range(min(n_frames, 5)):  # Check first 5 frames
        frame_tokens = tokens[frame_idx * 7 : (frame_idx + 1) * 7]
        frame_valid = True
        
        for pos, (range_idx, range_min, range_max) in enumerate(range_specs):
            tok = frame_tokens[pos]
            in_range = range_min <= tok < range_max
            
            if not in_range:
                print(f"  Frame {frame_idx}, pos {pos}: token {tok} NOT in range [{range_min}, {range_max})")
                frame_valid = False
        
        if frame_valid:
            print(f"  Frame {frame_idx}: ✓ all 7 tokens in valid ranges")
    
    # Check all frames
    all_valid = True
    for frame_idx in range(n_frames):
        frame_tokens = tokens[frame_idx * 7 : (frame_idx + 1) * 7]
        for pos, tok in enumerate(frame_tokens):
            range_idx, range_min, range_max = range_specs[pos]
            if not (range_min <= tok < range_max):
                all_valid = False
                break
    
    if all_valid:
        print(f"\n✓ All {n_frames} frames have tokens in valid position-specific ranges")
    else:
        print(f"\n✗ Some tokens out of range")
    
    return all_valid


def deserialize_snac_codes(tokens: np.ndarray, base: int = 128266) -> tuple:
    """
    Deserialize Bodhan tokens back to SNAC codes.
    
    Args:
        tokens: Array of serialized token IDs
        base: Base token ID (128266)
    
    Returns:
        (c0, c1, c2) as numpy arrays
    """
    print(f"\n=== Deserialization (Tokens → SNAC Codes) ===")
    
    n_frames = len(tokens) // 7
    
    # Reconstruct codebooks by extracting codes from positions
    c0 = np.empty(n_frames, dtype=np.int32)
    c1 = np.empty(n_frames * 2, dtype=np.int32)
    c2 = np.empty(n_frames * 4, dtype=np.int32)
    
    for frame_idx in range(n_frames):
        frame_tokens = tokens[frame_idx * 7 : (frame_idx + 1) * 7]
        
        # Extract code from each position by removing offset
        offsets = [0, 1, 2, 3, 4, 5, 6]
        codes = []
        for pos, offset in enumerate(offsets):
            tok = frame_tokens[pos]
            code = tok - (base + offset * 4096)
            codes.append(int(code))
        
        # Position mapping
        c0[frame_idx] = codes[0]
        c1[2*frame_idx] = codes[1]
        c1[2*frame_idx + 1] = codes[4]
        c2[4*frame_idx] = codes[2]
        c2[4*frame_idx + 1] = codes[3]
        c2[4*frame_idx + 2] = codes[5]
        c2[4*frame_idx + 3] = codes[6]
    
    print(f"Deserialized c0 shape: {c0.shape}, range [{c0.min()}, {c0.max()}]")
    print(f"Deserialized c1 shape: {c1.shape}, range [{c1.min()}, {c1.max()}]")
    print(f"Deserialized c2 shape: {c2.shape}, range [{c2.min()}, {c2.max()}]")
    
    return c0, c1, c2


def compare_codes(original: tuple, deserialized: tuple) -> bool:
    """
    Compare original SNAC codes with deserialized codes.
    
    Args:
        original: Tuple of (tokens, c0, c1, c2) from serialize_bodhan_tokens
        deserialized: Tuple of (c0, c1, c2) from deserialize_snac_codes
    
    Returns:
        True if they match, False otherwise
    """
    print(f"\n=== Code Round-Trip Validation ===")
    
    _, orig_c0, orig_c1, orig_c2 = original
    deser_c0, deser_c1, deser_c2 = deserialized
    
    # Compare shapes
    shapes_match = (
        orig_c0.shape == deser_c0.shape and
        orig_c1.shape == deser_c1.shape and
        orig_c2.shape == deser_c2.shape
    )
    
    if not shapes_match:
        print(f"✗ Shape mismatch:")
        print(f"  Original: c0={orig_c0.shape}, c1={orig_c1.shape}, c2={orig_c2.shape}")
        print(f"  Deserialized: c0={deser_c0.shape}, c1={deser_c1.shape}, c2={deser_c2.shape}")
        return False
    
    # Compare values
    c0_match = np.allclose(orig_c0, deser_c0)
    c1_match = np.allclose(orig_c1, deser_c1)
    c2_match = np.allclose(orig_c2, deser_c2)
    
    print(f"c0 match: {'✓' if c0_match else '✗'} (max diff: {np.abs(orig_c0 - deser_c0).max()})")
    print(f"c1 match: {'✓' if c1_match else '✗'} (max diff: {np.abs(orig_c1 - deser_c1).max()})")
    print(f"c2 match: {'✓' if c2_match else '✗'} (max diff: {np.abs(orig_c2 - deser_c2).max()})")
    
    if c0_match and c1_match and c2_match:
        print(f"\n✓ Code round-trip PASSED")
        return True
    else:
        print(f"\n✗ Code round-trip FAILED")
        return False


def main():
    print("\n" + "="*70)
    print("SNAC Serialization Validation")
    print("="*70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create test waveform
    sr = 24000
    duration = 0.2  # 200 ms
    waveform = create_test_waveform(sr, duration)
    print(f"\n=== Test Waveform ===")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration} sec")
    print(f"Samples: {len(waveform)}")
    print(f"Range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    
    # Inspect SNAC API
    SNAC = inspect_snac_api()
    
    # Encode with SNAC
    try:
        encoder_output = encode_with_snac(waveform, device)
    except Exception as exc:
        print(f"ERROR during SNAC encoding: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Extract hierarchical codes
    try:
        codebooks = extract_hierarchical_codes(encoder_output)
    except Exception as exc:
        print(f"ERROR extracting hierarchical codes: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Serialize to Bodhan tokens
    try:
        result = serialize_bodhan_tokens(codebooks)
        if result is None:
            print("ERROR: Serialization returned None")
            return 1
        tokens, c0, c1, c2 = result
    except Exception as exc:
        print(f"ERROR during serialization: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Validate token ranges
    try:
        ranges_valid = validate_token_ranges(tokens)
    except Exception as exc:
        print(f"ERROR during token range validation: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Deserialize back to codes
    try:
        deser_codes = deserialize_snac_codes(tokens)
    except Exception as exc:
        print(f"ERROR during deserialization: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Compare original and round-trip codes
    try:
        roundtrip_ok = compare_codes((tokens, c0, c1, c2), deser_codes)
    except Exception as exc:
        print(f"ERROR during code comparison: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print final report
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Sample rate: {sr} Hz")
    print(f"Input samples: {len(waveform)}")
    print(f"SNAC codebooks: {codebooks['num_codebooks']}")
    print(f"c0 shape: {c0.shape}")
    print(f"c1 shape: {c1.shape}")
    print(f"c2 shape: {c2.shape}")
    print(f"Number of frames: {len(c0)}")
    print(f"Serialized token count: {len(tokens)}")
    print(f"Tokens per frame: {len(tokens) // len(c0)}")
    print(f"Token range validation: {'PASS' if ranges_valid else 'FAIL'}")
    print(f"Code round-trip: {'PASS' if roundtrip_ok else 'FAIL'}")
    print(f"Waveform round-trip: NOT PERFORMED (no decoder in this validation)")
    print(f"Maximum waveform error: N/A")
    print(f"Result: {'SUCCESS' if (ranges_valid and roundtrip_ok) else 'FAILURE'}")
    print("="*70)
    
    return 0 if (ranges_valid and roundtrip_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
