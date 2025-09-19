
# Stego AI Hybrid

LSB steganography + tamper-detection AI.
This repository contains a single-file Python tool that:
- Embeds/extracts payloads using LSB in PNG images.
- Simulates tampering (JPEG compression, noise, resize, crop).
- Generates a synthetic dataset of stego images and tampered variants.
- Trains a small CNN classifier to detect tampered images.

## Features
- CLI utilities: `embed`, `extract`, `tamper`, `createdata`, `train`, `detect`.
- Simple end-to-end pipeline for research and learning.
- Example usage provided below.

## Requirements
See `requirements.txt`. Tested with Python 3.9+.

## Quickstart

1. Install deps:
```bash
python -m pip install -r requirements.txt
```

2. Embed a small payload:
```bash
python stego_ai_hybrid.py embed --cover examples/covers/cover1.png --payload examples/payloads/secret.txt --out stego.png
```

3. Extract:
```bash
python stego_ai_hybrid.py extract --stego stego.png --out extracted.bin
```

4. Create a tiny dataset (use your covers & payloads):
```bash
python stego_ai_hybrid.py createdata --covers-dir examples/covers --payloads-dir examples/payloads --outdir data --num 100
```

5. Train the tamper detector:
```bash
python stego_ai_hybrid.py train --datadir data --model tamper_model.h5 --epochs 5
```

6. Detect tampering:
```bash
python stego_ai_hybrid.py detect --model tamper_model.h5 --image some_image.png
```

## Notes & Limitations
- The LSB routine uses 1–4 LSBs per channel; using more increases capacity but is easier to detect.
- The tamper detector is a proof-of-concept CNN; it requires a sufficiently large and diverse dataset for good accuracy.
- Not production-grade for secure steganography or forensics.

## License
MIT — see LICENSE.

## Contact
Vinod — add email or GitHub handle here.
