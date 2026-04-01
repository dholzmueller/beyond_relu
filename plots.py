from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
# matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 10.95,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'legend.framealpha': 0.5,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
})
import matplotlib.pyplot as plt

from ntk import NeuralSphereKernel
from spectrum import SpectrumComputation
import matplotlib.colors as mcolors

def darker(color, amount=0.2):
    r, g, b, a = mcolors.to_rgba(color)
    amount = max(0.0, min(1.0, amount))
    factor = 1.0 - amount
    return (r * factor, g * factor, b * factor, a)


def plot_spectra_general(config, filename: str, powers_config: List[Tuple[float, Dict[str, Any]]]):
    kmax = 100
    x = torch.arange(kmax) + 1
    comp = SpectrumComputation(d=2, kmax=kmax, num_quad_points=1000)
    plt.figure(figsize=(5, 4))

    pow_handles = []
    kernel_handles = []

    for pow, plot_kwargs in powers_config:
        h, = plt.loglog(x, x.float() ** -pow, '--', label=rf'$y=x^{{-{pow:g}}}$', **plot_kwargs)
        pow_handles.append(h)

    for name, kernel, plot_kwargs in config:
        default_style = dict(linestyle='None', marker='.')
        y = comp.compute(kernel)
        mask = y >= 1e-14  # avoid hidden points that are shown in some PDF readers (Mi Browser on the phone)
        h, = plt.loglog(x[mask], y[mask], **(default_style | plot_kwargs), label=name)
        kernel_handles.append(h)

    plt.grid(True)
    plt.xlabel('Eigenvalue index $l+1$')
    plt.ylabel('Eigenvalue $\\mu_{l+1}$')
    plt.ylim(bottom=1e-14)

    ax = plt.gca()
    legend1 = ax.legend(handles=kernel_handles, loc='lower left')
    ax.legend(handles=pow_handles, loc='upper right')
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(Path('plots') / filename)
    plt.close()


def plot_multilayer():
    alpha = 1.0
    config = [
        ('3-layer SELU NTK', NeuralSphereKernel(torch.selu, n_layers=3, ntk=True, sigma_b=1.0,
                                               sigma_i=1.0), dict(marker='*', color=darker('tab:red'), alpha=alpha)),
        ('4-layer SELU NTK', NeuralSphereKernel(torch.selu, n_layers=4, ntk=True, sigma_b=1.0,
                                                sigma_i=1.0), dict(marker='.', color='tab:red')),
        ('3-layer SELU NNGP', NeuralSphereKernel(torch.selu, n_layers=3, ntk=False, sigma_b=1.0,
                                                sigma_i=1.0), dict(marker='*', color=darker('tab:orange'), alpha=alpha)),
        ('4-layer SELU NNGP', NeuralSphereKernel(torch.selu, n_layers=4, ntk=False, sigma_b=1.0,
                                                sigma_i=1.0), dict(marker='.', color='tab:orange')),
        ('3-layer ELU NTK', NeuralSphereKernel(torch.nn.functional.elu, n_layers=3, ntk=True, sigma_b=1.0,
                                               sigma_i=1.0), dict(marker='*', color=darker('tab:cyan'), alpha=alpha)),
        ('4-layer ELU NTK', NeuralSphereKernel(torch.nn.functional.elu, n_layers=4, ntk=True, sigma_b=1.0,
                                               sigma_i=1.0), dict(marker='.', color='tab:cyan')),
        ('3-layer ELU NNGP', NeuralSphereKernel(torch.nn.functional.elu, n_layers=3, ntk=False, sigma_b=1.0,
                                               sigma_i=1.0), dict(marker='*', color=darker('tab:blue'), alpha=alpha)),
        ('4-layer ELU NNGP', NeuralSphereKernel(torch.nn.functional.elu, n_layers=4, ntk=False, sigma_b=1.0,
                                               sigma_i=1.0), dict(marker='.', color='tab:blue')),
    ]
    plot_spectra_general(config, filename='deep_spectra.pdf',
                         powers_config=[(2, dict(color='tab:red')),
                                        (4, dict(color='tab:cyan')),
                                        (6, dict(color='tab:blue'))])


def plot_twolayer():
    config = [
        ('ReLU NTK', NeuralSphereKernel(torch.relu, n_layers=2, ntk=True, sigma_b=1.0, sigma_i=1.0),
         dict(marker='*', color=darker('tab:green'))),
        ('ReLU NTK (no bias)', NeuralSphereKernel(torch.relu, n_layers=2, ntk=True, sigma_b=0.0,
                                                  sigma_i=0.0), dict(marker='.', color='tab:green')),
        ('SELU NTK', NeuralSphereKernel(torch.selu, n_layers=2, ntk=True, sigma_b=1.0, sigma_i=1.0),
         dict(marker='*', color=darker('tab:blue'))),
        ('SELU NTK (no bias)', NeuralSphereKernel(torch.selu, n_layers=2, ntk=True, sigma_b=0.0,
                                                sigma_i=0.0), dict(marker='.', color='tab:blue')),
        ('ELU NTK', NeuralSphereKernel(F.elu, n_layers=2, ntk=True, sigma_b=1.0, sigma_i=1.0),
         dict(marker='*', color=darker('tab:orange'))),
        ('ELU NTK (no bias)', NeuralSphereKernel(F.elu, n_layers=2, ntk=True, sigma_b=0.0,
                                                          sigma_i=0.0), dict(marker='.', color='tab:orange')),
        ('GELU NTK', NeuralSphereKernel(F.gelu, n_layers=2, ntk=True, sigma_b=1.0, sigma_i=1.0),
         dict(marker='*', color=darker('tab:purple'))),
        ('GELU NTK (no bias)', NeuralSphereKernel(F.gelu, n_layers=2, ntk=True, sigma_b=0.0,
                                                         sigma_i=0.0), dict(marker='.', color='tab:purple')),
    ]
    plot_spectra_general(config, filename='shallow_spectra.pdf',
                         powers_config=[(2, dict(color='tab:red')),
                                        (4, dict(color='tab:cyan')),
                                        (6, dict(color='tab:blue'))])


if __name__ == '__main__':
    plot_multilayer()
    plot_twolayer()
