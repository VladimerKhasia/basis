<div align="center">

# BASIS: Balanced Activation Sketching with Invariant Scalars for "Ghost Backpropagation"


</div>

<p align="center">
  <strong>Implementation of the paper -> BASIS: Balanced Activation Sketching with Invariant Scalars for "Ghost Backpropagation"</strong>
</p>



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18881670.svg)](https://doi.org/10.5281/zenodo.18881670)



## Abstract

The activation memory required for exact backpropagation scales linearly with network depth,
context length, and feature dimensionality, forming an O(L · BN ) spatial bottleneck (where B is the
sequence-batch cardinality and N is the feature dimension). This constraint historically throttles the
scaling of deep neural networks. While randomized automatic differentiation attempts to mitigate
this, it historically suffers from catastrophic variance. In this paper, we introduce BASIS (Balanced
Activation Sketching with Invariant Scalars), an efficient backpropagation algorithm that fully
decouples activation memory from the batch and sequence dimensions. BASIS propagates the exact
error signal (dX) to preserve flawless gradient flow, but computes the weight updates (dW) using
massively compressed rank-R tensors. To solve the foundational instability of sketched gradients, we
propose two novel mechanisms: Balanced Hashing, which strictly eliminates off-diagonal collision
variance, and Invariant Scalars, a principled bias-variance tradeoff that deterministically preserves
the exact continuous energy norm of the spatial geometry. Theoretically, BASIS reduces activation
memory to O(L · RN ) and heavily decreases the backward pass matrix-multiplication footprint.
Empirically, training a GPT architecture for 50,000 steps validates our theoretical guarantees: at
R = 32, BASIS achieves parity with (and marginally outperforms) exact backpropagation validation
loss (6.575 vs. 6.616), acting as an implicit regularizer. Remarkably, the stabilized magnitude
trajectory allows the model to converge smoothly even under extreme spatial compression (R = 1),
proving the extreme robustness of the estimator.

---

## Installation

1. For quick experimentation with Jupyter Notebook or Google Colab turn `.py` file into `.ipynb`. 

```bash
pip install numpy jax jaxlib flax optax torch torchvision datasets tiktoken matplotlib pandas tqdm
```

2. Or clone the repository to your local machine and install the required dependencies using pip:

```bash
# cd basis
pip install -r requirements.txt 
# Once the dependencies are installed, you can execute the script using:
python basis.py 
```

## Citation

If you utilize this code or the concepts presented in **BASIS** for your research, please cite the following paper:



```bibtex
@article{https://doi.org/10.5281/zenodo.18881670,
  doi = {10.5281/zenodo.18881670},
  
  url = {https://doi.org/10.5281/zenodo.18881670},
  
  author = {Khasia, Vladimer},
  
  title = {BASIS: Balanced Activation Sketching with Invariant Scalars for "Ghost Backpropagation"},
  
  publisher = {Zenodo},
  
  year = {2026},
  
  copyright = {All Rights Reserved}
}
```










