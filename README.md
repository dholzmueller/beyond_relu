## Beyond ReLU

This codebase can be used to reproduce the plots of our paper 
[Beyond ReLU: How Activations Affect Neural Kernels and 
Random Wide Networks](https://arxiv.org/abs/2506.22429).
In particular, it contains code for 
computing the infinite-width NTK and NNGP kernels 
of shallow and deep fully-connected neural networks for 
different activation functions and depths. 
Our quadrature is designed to work with activations 
that can be non-smooth at zero. 
We also use code from 
[this repository](https://github.com/Pehlevan-Group/NTK_Learning_Curves) (MIT license)
to compute the eigenvalues of the integral operators 
associated to these kernels.

To generate the plots, run `plots.py`. 
Generation takes a few seconds on a CPU. 
Dependencies: `numpy, scipy, torch, matplotlib`.

If you are using this code for research purposes, please cite our 
[paper](https://arxiv.org/abs/2506.22429):
```
@inproceedings{
holzmuller2026beyond,
title={Beyond Re{LU}: How Activations Affect  Neural Kernels and Random Wide Networks},
author={David Holzm{\"u}ller and Max Sch{\"o}lpple},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=aRxjE5hQLA}
}
```