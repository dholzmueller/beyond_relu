import scipy
import numpy as np
import torch

import gegenbauer
from ntk import NeuralSphereKernel


class SpectrumComputation:
    def __init__(self, d, kmax, num_quad_points=5000):
        alpha = d / 2.0 - 1
        z, w = scipy.special.roots_gegenbauer(num_quad_points, alpha)
        self.z = torch.as_tensor(z)
        self.w = torch.as_tensor(w)
        self.d = d
        self.degens = torch.as_tensor(np.array([gegenbauer.degeneracy(d, k) for k in range(kmax)]))

        Q = torch.as_tensor(gegenbauer.get_gegenbauer_fast2(z, kmax, d))
        self.scaled_Q = Q * self.degens[:, None]

    def compute(self, kernel: NeuralSphereKernel):
        kernel_values = kernel(self.z)
        scale = gegenbauer.surface_area( self.d - 1) / gegenbauer.surface_area(self.d)
        return scale * (self.scaled_Q @ (kernel_values * self.w)) / self.degens
