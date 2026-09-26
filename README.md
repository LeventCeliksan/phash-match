# phash-match

Find **visually similar images**: resized, recompressed, brightened or recolored copies, with perceptual hashes (**pHash**, **dHash**, **aHash**) and Hamming distance. Pure Python on Pillow + NumPy.

It illustrates the first, cheapest stage of the image-matching idea I use at [Sealify](https://sealify.io) to find copies of registered content across the web: a compact 64-bit fingerprint per image, compared in microseconds. (Sealify's production engine adds learned embeddings, face matching and more; this repo is an independent, from-scratch implementation of the fingerprinting idea only and does not contain Sealify code.)

## Install
```bash
pip install git+https://github.com/LeventCeliksan/phash-match
```

## Usage
```bash
phash-match compare original.png repost.jpg
# distance 2/64  similarity 97%  MATCH

phash-match scan ./downloads --threshold 10     # near-duplicate pairs in a folder
phash-match --method dhash hash *.jpg           # print fingerprints
```

```python
from pathlib import Path
from phash_match import phash, find_duplicates
distance = phash("a.png") - phash("b.jpg")      # 0 = identical fingerprint
for m in find_duplicates(Path("images"), threshold=10):
    print(m.distance, m.a, m.b)
```

## Measured behaviour (pHash, 64 bits, default threshold 10)
Test images are synthetic photo-like scenes (gradients + shapes); distances are the worst case over 20 images.

| Edit | Max distance | Result |
|---|---:|---|
| Resized to 25% | 2 | match |
| JPEG quality 20 | 2 | match |
| Resized + recompressed | 2 | match |
| Gaussian blur | 2 | match |
| Grayscale | 0 | match |
| Contrast +40% | 6 | match |
| Brightness +30% | 8 | match |
| **Different images** (4,950 pairs) | min 14, mean 30.5 | no false match |

Known limits, by design of global perceptual hashes: **cropping** (median distance 22), **rotation** (34) and **mirroring** (32) are not caught. Catching those needs local features or learned embeddings.

## Tests
```bash
pip install -e ".[test]"
pytest
```
41 tests: every edit above on 5 images, no false matches among 40 different images, folder scanning (skips non-images and broken files), the CLI, and equality with the widely used [`imagehash`](https://pypi.org/project/ImageHash/) library's pHash/aHash/dHash on 20 images.

## License
MIT
