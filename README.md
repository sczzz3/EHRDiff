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

For the MIMIC data, we put our processed data to this [link](https://drive.google.com/file/d/1A0E2-JU7KKb7jkMJNtZqikGyHO37-45R/view?usp=share_link) for open access and reproducing our results. Please note that the data is generated from the original MIMIC data, and if you use our processed data, do follow MIMIC's license and cite the original MIMIC source. We truncate the original ICD9 code in MIMIC following this Python code snippet:
```python
def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr
```

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
