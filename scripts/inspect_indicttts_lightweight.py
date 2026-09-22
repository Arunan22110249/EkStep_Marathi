#!/usr/bin/env python3
"""
Lightweight inspection of SPRINGLab/IndicTTS_Marathi dataset metadata.
Uses huggingface_hub to get dataset info without full download.
"""

from huggingface_hub import dataset_info
import json

print('=== Fetching Dataset Info from Hugging Face Hub ===\n')

try:
    # Get dataset info
    info = dataset_info('SPRINGLab/IndicTTS_Marathi')
    
    print(f'Dataset ID: {info.id}')
    print(f'Repo Type: {info.repo_type}')
    print(f'Siblings: {len(info.siblings)} files')
    print()
    
    # Print file list
    print('=== Dataset Files ===')
    for sibling in info.siblings:
        print(f'  {sibling.rfilename} ({sibling.size / (1024**2):.1f} MB)')
    
except Exception as e:
    print(f'Error fetching dataset info: {e}')
    print('Trying alternative approach...')
    
    # Try loading with streaming
    from datasets import load_dataset
    
    print('\nLoading with streaming=True (minimal download)...')
    try:
        ds = load_dataset('SPRINGLab/IndicTTS_Marathi', split='train', streaming=True)
        
        print(f'\n=== Dataset Schema ===')
        print(ds.features)
        
        print(f'\n=== First Example ===')
        example = next(iter(ds))
        for key, val in example.items():
            if key == 'audio' and isinstance(val, dict):
                if 'array' in val:
                    sr = val.get('sampling_rate', '?')
                    samples = len(val['array'])
                    duration = samples / sr if isinstance(sr, int) else '?'
                    print(f'{key}: sample_rate={sr}, num_samples={samples}, duration≈{duration:.2f}s')
                else:
                    print(f'{key}: {type(val).__name__}')
            elif isinstance(val, str) and len(val) > 80:
                print(f'{key}: "{val[:80]}..."')
            else:
                print(f'{key}: {val}')
                
    except Exception as e2:
        print(f'Error with streaming: {e2}')
