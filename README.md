# Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family (COLM 2026)

Datasets
---
The datasets used in this paper are available at [oscarchew/clip-cross-eyed-data](https://huggingface.co/datasets/oscarchew/clip-cross-eyed-data). To download them:
```
hf download oscarchew/clip-cross-eyed-data --repo-type dataset --local-dir ./data
```
This dataset is derived from [whatsup_vlms](https://github.com/amitakamath/whatsup_vlms). We thank its authors for the inspiring research and making their work publicly available.

Usage
---
To reproduce the results on What'sUp:
```
python3 src/whatsup_inference.py
```

To reproduce the results on GRID:
```
python3 src/grid_inference.py
```

