from typing import Callable

import numpy as np
import torch


class QuadrantCorrGaussQuad:
    def __init__(self, deg: int = 50, limit: int = 12.0):
        self.limit = limit
        self.deg = deg
        self.points_y, self.weights_y = (torch.as_tensor(t) for t in np.polynomial.legendre.leggauss(deg))
        self.points_x = 0.5*(self.points_y+1.0)  # move to interval [0, 1]
        self.weights_x = 0.5*self.weights_y

    def quad(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], rhos: torch.Tensor):
        # approximately integrate f(x, y) * p(x, y) on [0, \infty]^2, where p is the pdf of the Gaussian with
        # mean zero and covariance [[1 rho], [rho 1]]
        # we use a reparametrization of the integral to make it work better for |rho| close to 1
        # rhos must be in [-1, 1] and rhos.shape = (n_rhos,)
        assert len(rhos.shape) == 1
        rhos = rhos.clamp(-1.0, 1.0)[None, None, :]
        sx = np.sqrt(0.5*(1. + rhos))  # sigma_x
        sy = np.sqrt(0.5*(1. - rhos))  # sigma_y
        sxy = sx * sy
        x = self.points_x[:, None, None]
        y = self.points_y[None, :, None]
        weights = self.weights_x[:, None, None] * self.weights_y[None, :, None]
        c = self.limit
        A = c**2 * sxy[0,0] / (2*np.pi) * torch.sum(x * f(c*sxy*x*(1+y), c*sxy*x*(1-y))
                                            * torch.exp(-0.5*((c*sy*x)**2 + (c*sx*x*y)**2)) * weights, dim=(0,1))
        B = c**2 * sx[0,0] / (2*np.pi) * torch.sum(f(c*sx*(x + sy*(1+y)), c*sx*(x + sy*(1-y)))
                                           * torch.exp(-0.5*((c*(sy + x))**2 + (c*sx*y)**2)) * weights, dim=(0,1))
        return A + B


class DualAct:
    def __call__(self, rhos: torch.Tensor) -> torch.Tensor:
        # rhos.shape = (n_rhos,), values in [-1, 1]
        raise NotImplementedError()


class QuadGaussDualAct(DualAct):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        self.f = f
        self.quad = QuadrantCorrGaussQuad()

    def __call__(self, rhos: torch.Tensor) -> torch.Tensor:
        # rhos should be 1D, returns the same shape
        # compute integrals of quadrants 1-4
        quad1 = self.quad.quad(lambda x, y: self.f(x) * self.f(y), rhos)
        quad2 = self.quad.quad(lambda x, y: self.f(x) * self.f(-y), -rhos)  # equal to quad3 by symmetry
        quad4 = self.quad.quad(lambda x, y: self.f(-x) * self.f(-y), rhos)
        return quad1 + 2*quad2 + quad4


class MCDualAct:  # for testing
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], n_mc: int = 1000000):
        self.f = f
        self.n_mc = n_mc

    def __call__(self, rhos: torch.Tensor) -> torch.Tensor:
        results = []
        for rho in rhos:
            points = torch.as_tensor(np.random.multivariate_normal(mean=np.zeros(2), cov=np.array([[1., rho], [rho, 1.]]),
                                                   size=self.n_mc))
            results.append(torch.dot(self.f(points[:, 0]), self.f(points[:, 1])) / self.n_mc)
        return torch.as_tensor(results)


def derivative_act(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    # assumes f is an activation function (elementwise)
    def df(x):
        x = x.detach().clone().requires_grad_(True)
        y = f(x)
        return torch.autograd.grad(y, x, torch.ones_like(y))[0]
    return df


class NeuralSphereKernel:
    def __init__(self, act: Callable[[torch.Tensor], torch.Tensor], n_layers: int,
                 sigma_b: float, sigma_i: float, ntk: bool, sigma_w: float = 1.0, dual_act_cls: type = QuadGaussDualAct):
        assert n_layers >= 1
        self.act = act
        self.n_layers = n_layers
        self.dual_act_cls = dual_act_cls
        self.ntk = ntk
        self.sbsq = sigma_b**2
        self.sisq = sigma_i**2
        self.swsq = sigma_w**2
        self.alphas = [self.sbsq * self.sisq + self.swsq]
        for l in range(1, n_layers):
            dual_act = dual_act_cls(lambda x: act(x * np.sqrt(self.alphas[l-1])))
            self.alphas.append(self.sbsq * self.sisq + self.swsq * dual_act(torch.as_tensor([1.])).item())


    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        nngp_values = [self.sbsq * self.sisq + self.swsq * t]
        for l in range(1, self.n_layers):
            dual_act = self.dual_act_cls(lambda x: self.act(x * np.sqrt(self.alphas[l - 1])))
            nngp_values.append(self.sbsq * self.sisq + self.swsq * dual_act(nngp_values[l-1] / self.alphas[l-1]))

        if not self.ntk:
            return nngp_values[-1]

        ntk_values = [self.sbsq * (1. - self.sisq) + nngp_values[0]]
        for l in range(1, self.n_layers):
            dual_act = self.dual_act_cls(lambda x: derivative_act(self.act)(x * np.sqrt(self.alphas[l - 1])))
            ntk_values.append(self.sbsq * (1. - self.sisq) + nngp_values[l]
                              + self.swsq * ntk_values[l-1] * dual_act(nngp_values[l-1] / self.alphas[l-1]))

        return ntk_values[-1]

