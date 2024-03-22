# EHRDiff [TMLR]

This is the official code base for paper: ["EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models"](https://arxiv.org/abs/2303.05656).


# Requirements
- Install the dependencies by:

```bash
conda create -n ehrdiff python=3.9
pip install -r requirements.txt
```

# Usage

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


Adjust the data path and parameter setting in the corresponding config file and then start training by:
```bash
python main.py --mode train --workdir <new_directory> --config <config_file>
```
Adjust the checkpoint path and parameter setting and then you can do sampling by running:
```bash
python main.py --mode eval --workdir <new_directory> --config <config_file>
```

Note that we modify the code from the DPDM repo, with which we attempt to equip the model with the ability of differential privacy. In our preliminary experiments, we found it to be effective for the CinC dataset, while it is hard to adjust for MIMIC data which may be due to the high-dimension problem commonly concerned with differential privacy.

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
Parts of our codes are modified from [DPDM](https://github.com/nv-tlabs/DPDM) and [edm](https://github.com/NVlabs/edm).

