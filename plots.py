from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any

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

from ntk import MCDualAct, QuadGaussDualAct, NeuralSphereKernel
from spectrum import SpectrumComputation

import matplotlib.colors as mcolors

def brighter(color, amount=0.3):
    """
    Return a slightly brighter version of a Matplotlib color.

    Parameters
    ----------
    color : any Matplotlib color
        Examples: 'tab:red', '#336699', (0.2, 0.4, 0.6), etc.
    amount : float, default=0.15
        How much to brighten, in [0, 1].
        0   -> unchanged
        1   -> white

    Returns
    -------
    tuple
        Brightened RGB tuple.
    """
    r, g, b = mcolors.to_rgb(color)
    amount = max(0.0, min(1.0, amount))
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )


import matplotlib.colors as mcolors

def darker(color, amount=0.2):
    r, g, b, a = mcolors.to_rgba(color)
    amount = max(0.0, min(1.0, amount))
    factor = 1.0 - amount
    return (r * factor, g * factor, b * factor, a)


import colorsys

def desaturate(color, amount=0.3):
    r, g, b, a = mcolors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    amount = max(0.0, min(1.0, amount))
    s *= (1 - amount)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, a)


import colorsys
import matplotlib.colors as mcolors

def shift_hue(color, amount=-0.08):
    r, g, b, a = mcolors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + amount) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, a)


import math
import matplotlib.colors as mcolors

def shift_hue_perceptual(color, amount=-0.1):
    """
    Shift hue in Oklab space for a more perceptually uniform result.

    Parameters
    ----------
    color : any Matplotlib color
        Examples: 'tab:red', '#336699', (0.2, 0.4, 0.6), etc.
    amount : float
        Hue shift as a fraction of a full turn.
        Examples:
            0.0   -> unchanged
            0.1   -> +36 degrees
            0.5   -> opposite hue
            -0.1  -> -36 degrees

    Returns
    -------
    tuple
        RGB tuple in [0, 1].

    Notes
    -----
    - This is more perceptually reasonable than shifting hue in HSV/HLS.
    - For very saturated colors, the rotated color can fall outside sRGB gamut.
      In that case this function clips to [0, 1], which can slightly alter the
      final appearance.
    """

    def srgb_to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    def linear_to_srgb(c):
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1 / 2.4)) - 0.055

    def rgb_to_oklab(r, g, b):
        # sRGB -> linear
        r = srgb_to_linear(r)
        g = srgb_to_linear(g)
        b = srgb_to_linear(b)

        # linear RGB -> LMS
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l ** (1/3)
        m_ = m ** (1/3)
        s_ = s ** (1/3)

        # LMS -> Oklab
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return L, a, b

    def oklab_to_rgb(L, a, b):
        # Oklab -> LMS cube roots
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

        # Undo cube root
        l = l_ ** 3
        m = m_ ** 3
        s = s_ ** 3

        # LMS -> linear RGB
        r = (+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)
        g = (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)
        b = (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)

        # linear -> sRGB
        r = linear_to_srgb(r)
        g = linear_to_srgb(g)
        b = linear_to_srgb(b)

        return r, g, b

    rgba = mcolors.to_rgba(color)
    r, g, b, alpha = rgba

    L, a, b_ = rgb_to_oklab(r, g, b)

    # Convert a,b to polar coordinates = chroma,hue
    C = math.hypot(a, b_)
    h = math.atan2(b_, a)

    # Rotate hue
    h += amount * 2 * math.pi

    # Back to Cartesian
    a2 = C * math.cos(h)
    b2 = C * math.sin(h)

    r2, g2, b2 = oklab_to_rgb(L, a2, b2)

    # Clip back into sRGB gamut
    r2 = min(1.0, max(0.0, r2))
    g2 = min(1.0, max(0.0, g2))
    b2 = min(1.0, max(0.0, b2))

    return (r2, g2, b2, alpha)


def plot_dual_acts(act: Callable[[torch.Tensor], torch.Tensor]):
    x = torch.linspace(-1, 1, 20)
    plt.plot(x, MCDualAct(act)(x), label='MC')
    plt.plot(x, QuadGaussDualAct(act)(x), label='QuadGauss')
    plt.legend()
    plt.show()


def plot_dual_acts_2():
    x = torch.linspace(-1, 1, 100)
    plt.plot(x, MCDualAct(torch.relu)(x), label='relu')
    plt.plot(x, MCDualAct(torch.selu)(x), label='selu')
    plt.plot(x, MCDualAct(torch.nn.functional.elu)(x), label='elu')
    plt.plot(x, MCDualAct(torch.nn.functional.silu)(x), label='silu')
    # plt.plot(x, QuadGaussDualAct(act)(x), label='QuadGauss')
    plt.legend()
    plt.show()


def plot_nngp(act: Callable[[torch.Tensor], torch.Tensor]):
    x = torch.linspace(-1, 1, 1000)
    kernel = NeuralSphereKernel(act, n_layers=6, ntk=True, sigma_b=1.0, sigma_i=1.0)
    plt.plot(x, kernel(x), label='kernel')
    plt.legend()
    plt.show()


def plot_spectra():
    kernels = {
        '2-layer ELU NNGP': NeuralSphereKernel(torch.nn.functional.elu, n_layers=2, ntk=False, sigma_b=1.0, sigma_i=1.0),
        '2-layer ELU NNGP, no bias': NeuralSphereKernel(torch.nn.functional.elu, n_layers=2, ntk=False, sigma_b=0.0, sigma_i=0.0),
        '2-layer SELU NNGP': NeuralSphereKernel(torch.selu, n_layers=2, ntk=False, sigma_b=1.0,
                                               sigma_i=1.0),
        '2-layer SELU NNGP, no bias': NeuralSphereKernel(torch.selu, n_layers=2, ntk=False, sigma_b=0.0,
                                                        sigma_i=0.0),
        '2-layer ReLU NNGP': NeuralSphereKernel(torch.relu, n_layers=2, ntk=False, sigma_b=1.0, sigma_i=1.0),
        '2-layer ReLU NNGP, no bias': NeuralSphereKernel(torch.relu, n_layers=2, ntk=False, sigma_b=0.0, sigma_i=0.0),
        '2-layer GELU NNGP': NeuralSphereKernel(torch.nn.functional.gelu, n_layers=2, ntk=False, sigma_b=1.0, sigma_i=1.0),
        '2-layer GELU NNGP, no bias': NeuralSphereKernel(torch.nn.functional.gelu, n_layers=2, ntk=False, sigma_b=0.0, sigma_i=0.0),
    }
    kernels = {
        '2-layer ELU NNGP': NeuralSphereKernel(torch.nn.functional.elu, n_layers=2, ntk=False, sigma_b=1.0,
                                               sigma_i=1.0),
        '2-layer ELU NTK': NeuralSphereKernel(torch.nn.functional.elu, n_layers=2, ntk=True, sigma_b=1.0,
                                               sigma_i=1.0),
        '2-layer SELU NNGP': NeuralSphereKernel(torch.selu, n_layers=2, ntk=False, sigma_b=1.0,
                                                sigma_i=1.0),
        '2-layer SELU NTK': NeuralSphereKernel(torch.selu, n_layers=2, ntk=True, sigma_b=1.0,
                                                         sigma_i=1.0),
        '3-layer ELU NNGP': NeuralSphereKernel(torch.nn.functional.elu, n_layers=3, ntk=False, sigma_b=1.0,
                                               sigma_i=1.0),
        '3-layer ELU NTK': NeuralSphereKernel(torch.nn.functional.elu, n_layers=3, ntk=True, sigma_b=1.0,
                                              sigma_i=1.0),
        '3-layer SELU NNGP': NeuralSphereKernel(torch.selu, n_layers=3, ntk=False, sigma_b=1.0,
                                                sigma_i=1.0),
        '3-layer SELU NTK': NeuralSphereKernel(torch.selu, n_layers=3, ntk=True, sigma_b=1.0,
                                               sigma_i=1.0),
    }
    kmax = 100
    x = torch.arange(kmax)+1
    comp = SpectrumComputation(d=2, kmax=kmax, num_quad_points=500)
    for pow in (2, 4, 6):
        plt.loglog(x, x.float() ** -pow, '--', label=f'x**(-{pow})')
    for name, kernel in kernels.items():
        plt.loglog(x, comp.compute(kernel), '.', color=None, label=name)
    plt.grid(True)
    plt.xlabel('Eigenvalue index $l$')
    plt.ylabel('Eigenvalue $\\mu_l$')
    plt.ylim(bottom=1e-12)
    plt.legend()
    plt.show()


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
        # default_style = dict(linestyle='None', marker='.', alpha=0.5, markeredgecolor='none')
        h, = plt.loglog(x, comp.compute(kernel), **(default_style | plot_kwargs), label=name)
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
                         powers_config=[(2, dict(color='tab:cyan')),
                                        (4, dict(color='tab:brown')),
                                        (6, dict(color='tab:red'))])


if __name__ == '__main__':
    # what to show?
    # - effect of depth, nngp vs ntk, different activations, two-layer special cases
    # - depth can probably be compared in the same plot as activations. Maybe also NTK?

    # plot_dual_acts(torch.nn.functional.gelu)
    # plot_dual_acts_2()
    # plot_nngp(torch.selu)
    # plot_spectra()
    plot_multilayer()
    plot_twolayer()
