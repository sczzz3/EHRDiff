# EHRDiff

This is the official code base for paper: ["EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models
"](https://arxiv.org/abs/2303.05656).

Currently the repo contains the code for the experiment of binary EHR data (MIMIC). Codes for other types of EHR data will be released soon.

# Requirements
- Install the dependencies by:

```bash
conda create -n ehrdiff python=3.8
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

# Training

First, you need to preprocess the EHR data into a binary matrix, which serves as the input of the diffusion model. 
Then start training by:

```bash
python main.py --data_file "path to the preprocessed file" --ehr_dim 1782 --mlp_dims 1024 384 384 384 1024
```

The `figs` directory contatins plots of dimension-wise probability and `logs` directory contatins training logs, both of which help to moniter the training process.

# Citation

```bibtex
@article{ehrdiff,
  doi = {10.48550/ARXIV.2303.05656},
  url = {https://arxiv.org/abs/2303.05656},
  author = {Yuan, Hongyi and Zhou, Songchi and Yu, Sheng},
  title = {EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models},
  publisher = {arXiv},
  year = {2023}
}
```

# Acknowledgements
Parts of our codes are modified from [lucidrains/denoising-diffusion-pytorch repo](https://github.com/lucidrains/denoising-diffusion-pytorch).
