#!/usr/bin/env python3
"""
Inspect SPRINGLab/IndicTTS_Marathi dataset schema and samples.
Purpose: Verify fields, audio format, sample rate, text, speaker metadata.
"""

from datasets import load_dataset

print('=== Loading SPRINGLab/IndicTTS_Marathi dataset ===')

# Load dataset (streaming=False to download, but we'll only access limited samples)
ds = load_dataset('SPRINGLab/IndicTTS_Marathi', split='train', streaming=False)

print(f'\n=== Dataset Features ===')
for field_name, field_type in ds.features.items():
    print(f'{field_name}: {field_type}')

print(f'\n=== Dataset Summary ===')
print(f'Total samples: {len(ds)}')

print(f'\n=== First 3 Examples ===')
for i in range(min(3, len(ds))):
    example = ds[i]
    print(f'\n--- Sample {i} ---')
    for key in sorted(example.keys()):
        val = example[key]
        
        if key == 'audio':
            if isinstance(val, dict) and 'array' in val:
                sr = val.get('sampling_rate', 'N/A')
                num_samples = len(val['array']) if hasattr(val['array'], '__len__') else 'unknown'
                duration_sec = num_samples / sr if isinstance(sr, int) and isinstance(num_samples, int) else 'N/A'
                print(f'  {key}:')
                print(f'    - sample_rate: {sr}')
                print(f'    - num_samples: {num_samples}')
                if isinstance(duration_sec, float):
                    print(f'    - duration_sec: {duration_sec:.3f}')
                else:
                    print(f'    - duration_sec: {duration_sec}')
            else:
                print(f'  {key}: {type(val).__name__}')
        elif isinstance(val, str) and len(val) > 100:
            print(f'  {key}: "{val[:100]}..."')
        elif isinstance(val, (int, float, bool)):
            print(f'  {key}: {val}')
        elif isinstance(val, str):
            print(f'  {key}: "{val}"')
        else:
            print(f'  {key}: {type(val).__name__}')

print('\n=== Dataset Inspection Complete ===')
