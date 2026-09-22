#!/usr/bin/env python3
"""Quick check of audio dict structure."""

from datasets import load_dataset

ds = load_dataset(
    "SPRINGLab/IndicTTS_Marathi",
    split="train",
    streaming=False,
)

ds = ds.with_format("arrow")
ds = ds.select(range(1))

ex = ds[0]
ex_dict = ex.to_pydict()

print("Example 0 keys:", list(ex_dict.keys()))
print("\nAudio type:", type(ex_dict.get('audio')))
print("Audio value:", ex_dict.get('audio'))

if isinstance(ex_dict.get('audio'), list) and len(ex_dict['audio']) > 0:
    audio0 = ex_dict['audio'][0]
    print("\nAudio[0] type:", type(audio0))
    if isinstance(audio0, dict):
        print("Audio[0] keys:", list(audio0.keys()))
        for k, v in audio0.items():
            if k == 'bytes':
                print(f"  {k}: <binary data, {len(v) if hasattr(v, '__len__') else 'N/A'} bytes>")
            else:
                print(f"  {k}: {v}")
