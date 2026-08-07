# Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family (COLM 2026)

Datasets
---
The datasets used in this paper are available at [oscarchew/clip-cross-eyed-data](https://huggingface.co/datasets/oscarchew/clip-cross-eyed-data). To download them:
```
hf download oscarchew/clip-cross-eyed-data --repo-type dataset --local-dir ./data
```
This dataset is derived from [whatsup_vlms](https://github.com/amitakamath/whatsup_vlms). We thank its authors for the inspiring research and making their work publicly available.

Models
---
All OpenCLIP/OpenAI CLIP models are directly supported. For [NegCLIP](https://github.com/mertyg/vision-language-models-are-bows), [DetailCLIP](https://github.com/KishoreP1/DetailCLIP), [SuperCLIP](https://github.com/hustvl/SuperCLIP), please download the weights from their original sources and place them at `checkpoints`.

Usage
---
To reproduce the results on What'sUp:
```
python3 src/inference_whatsup.py
```

To reproduce the results on GRID:
```
python3 src/inference_grid.py
```

Citation
---
Please consider citing our paper if you find our work helpful. Thank you!
```
@inproceedings{
    chew2026clip,
    title={Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family},
    author={Chew, Oscar and Huang, Hsiao-Ying and Jain, Kunal and Chen, Tai-I and Doan, Khoa D and Huang, Kuan-Hao},
    booktitle={Third Conference on Language Modeling},
    year={2026},
    url={https://openreview.net/forum?id=ieFdrMYzog}
}
```