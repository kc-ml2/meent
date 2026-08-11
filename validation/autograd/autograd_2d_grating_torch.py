"""Autograd validation for the torch backend (meent/on_torch) on a 2D grating.

For every combination of

    permittivity case         : iso, aniso   (+ degenerate, opt-in)
    modeling / Fourier scheme : discrete, continuous, vector
    parameter                 : width, height, thickness, permittivity

this script compares d(loss)/d(parameter) obtained by torch autograd against a
central finite difference of the *same* forward pass, and reports PASS/FAIL per
parameter component.  Exit code is 0 only if every check passes.

    loss = de_ti[cy, cx]   (0th-order transmitted diffraction efficiency)

Geometry (the same physical structure in all three modes):

    layer 0 : rectangular pillar (width x height) of permittivity `eps_obj`,
              centered in an n_bg background
    layer 1 : uniform film of n_film

Notes on the parameters
-----------------------
* `width` / `height` are native parameters of vector modeling (the lx / ly of a
  'rectangle' instruction).  *Both* raster schemes -- discrete and continuous
  alike -- have no such parameter: they take the identical (Layers, H, W) ucell
  tensor and differ only in fourier_type, i.e. in how the convolution matrices
  are built from that fixed grid.  The grid coordinates are constants there, so
  the raster cases encode width into the cell *values* through a smooth sigmoid
  occupancy mask of half-width width/2.  The mask is plain torch, so any
  autograd/numerical mismatch it produces comes from meent, not the mask.  A hard
  (0/1) mask would make the analytic derivative identically zero and the
  comparison meaningless.  So for raster this checks that the chain
  scalar -> ucell -> conv matrix -> solver -> loss stays differentiable end to
  end; only vector modeling differentiates a genuine geometric edge position.
* `permittivity` is the pillar's eps.  meent takes refractive index, so the
  forward pass feeds it n = sqrt(eps); the reported gradient is d(loss)/d(eps).
* `thickness` has one entry per layer; both are checked.

Anisotropy
----------
The `permittivity` case selects how the pillar's eps is given:

    iso         (12,)            -- scalar; ucell is (Layers, H, W)
    aniso       (10, 12, 14)     -- diagonal; ucell is (Layers, H, W, 3)
    degenerate  (12, 12, 12)     -- diagonal but all components equal

`aniso` is intentionally non-degenerate: if any two components were equal, a bug
that swaps or collapses them would still produce matching numbers.  The
background and the film stay isotropic, which also exercises meent's
scalar -> 3-vector promotion (_to_index_vector) in vector modeling.

`degenerate` is a regression guard, not a curiosity.  A gradient-based
optimization started from an isotropic material sits exactly on that point, and
to_conv_mat_raster_continuous used to dispatch its "compress once and share"
shortcut on *value* equality (`torch.equal(lx, ly) and torch.equal(ly, lz)`).
Only layer[..., 0] was then read, so eps_x collected the summed isotropic
gradient while eps_y and eps_z came back exactly 0 -- correct forward result,
silently wrong backward one, and a first optimizer step that later iterations
never undo.  The shortcut now triggers on tensor identity (`lx is ly is lz`)
instead, and this case pins that down.

Usage
-----
    python validation/autograd/autograd_2d_grating_torch.py
    python validation/autograd/autograd_2d_grating_torch.py --cases aniso degenerate
    python validation/autograd/autograd_2d_grating_torch.py --eps 9 11 13
    python validation/autograd/autograd_2d_grating_torch.py --modes vector --fto 3 3
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Allow running the file directly from a checkout, without installing meent.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import meent


TYPE_FLOAT = torch.float64

MODES = ('discrete', 'continuous', 'vector')
PARAMS = ('width', 'height', 'thickness', 'permittivity')

# Pillar permittivity per case. See the "Anisotropy" section of the module docstring.
CASES = {
    'iso': (12.0,),
    'aniso': (10.0, 12.0, 14.0),
    'degenerate': (12.0, 12.0, 12.0),
}
DEFAULT_CASES = tuple(CASES)

# Central-difference step per parameter. Chosen near eps_machine**(1/3) * (curvature
# length of the loss): ~1E-3 nm for the nm-scale geometry, ~1E-5 for eps ~ 12.
DELTA = {'width': 1E-3, 'height': 1E-3, 'thickness': 1E-3, 'permittivity': 1E-5}


@dataclass
class Config:
    period: tuple = (300., 300.)
    wavelength: float = 900.
    fto: tuple = (2, 2)
    theta_deg: float = 20.
    phi_deg: float = 30.
    pol: float = 0.5
    n_top: float = 1.
    n_bot: float = 1.
    n_bg: float = 1.45          # layer-0 background index
    n_film: float = 2.0         # layer-1 uniform film index
    grid: tuple = (40, 40)      # raster sampling (ny, nx)
    tau: float = 6.0            # sigmoid edge width [nm] of the raster occupancy mask
    # Pillar permittivity: 1 value (isotropic) or 3 values (eps_x, eps_y, eps_z).
    # See CASES and the "Anisotropy" section of the module docstring.
    eps_obj: tuple = (12.0,)

    @property
    def aniso(self):
        return len(self.eps_obj) == 3


def initial_params(cfg, requires_grad=False):
    def mk(value):
        return torch.tensor(value, dtype=TYPE_FLOAT, requires_grad=requires_grad)

    return {
        'width': mk([120.]),
        'height': mk([90.]),
        'thickness': mk([220., 150.]),
        'permittivity': mk(list(cfg.eps_obj)),  # n = sqrt(eps)
    }


def soft_pillar_layer(width, height, n_obj, cfg):
    """Raster layer 0: sigmoid-smoothed rectangular pillar, differentiable in width/height."""
    ny, nx = cfg.grid
    px, py = cfg.period

    x = (torch.arange(nx, dtype=TYPE_FLOAT) + 0.5) * (px / nx)
    y = (torch.arange(ny, dtype=TYPE_FLOAT) + 0.5) * (py / ny)

    mask_x = torch.sigmoid((width / 2 - (x - px / 2).abs()) / cfg.tau)
    mask_y = torch.sigmoid((height / 2 - (y - py / 2).abs()) / cfg.tau)
    occupancy = mask_y[:, None] * mask_x[None, :]

    if cfg.aniso:
        # (ny, nx, 1) * (3,) -> (ny, nx, 3): one interpolation per diagonal component,
        # the shape meent's ucell setter reads as anisotropic.
        return cfg.n_bg + (n_obj - cfg.n_bg) * occupancy[..., None]

    return cfg.n_bg + (n_obj - cfg.n_bg) * occupancy


def make_forward(mode, cfg):
    """Return forward(params) -> scalar loss, for one modeling / Fourier scheme."""
    # 'vector' ignores fourier_type; ucell being a list is what selects vector modeling.
    fourier_type = 1 if mode == 'continuous' else 0

    mee = meent.call_mee(
        backend=2,
        fto=list(cfg.fto),
        wavelength=cfg.wavelength,
        period=list(cfg.period),
        n_top=cfg.n_top,
        n_bot=cfg.n_bot,
        theta=math.radians(cfg.theta_deg),
        phi=math.radians(cfg.phi_deg),
        pol=cfg.pol,
        thickness=[1., 1.],  # overwritten every forward pass
        device=0,
        type_complex=0,
        fourier_type=fourier_type,
    )

    cx, cy = cfg.period[0] / 2, cfg.period[1] / 2

    def forward(p):
        n_obj = torch.sqrt(p['permittivity'])

        if mode == 'vector':
            # n_obj is a 1-element tensor when isotropic and a 3-element one when
            # anisotropic; meent's modeler dispatches on numel() == 3, so the same
            # instruction covers both. The scalar n_bg background is promoted to
            # (n, n, n) by meent itself.
            mee.ucell = [
                [cfg.n_bg, [['rectangle', cx, cy, p['width'], p['height'], n_obj, 0, 0, 0]]],
                [cfg.n_film, []],
            ]
        else:
            layer_0 = soft_pillar_layer(p['width'], p['height'], n_obj, cfg)
            # torch.stack needs matching shapes, so the isotropic film is materialized as
            # (ny, nx, 3) too when the pillar is anisotropic. Vector modeling has no such
            # constraint -- there the film layer stays a genuine scalar-index layer.
            film_shape = (*cfg.grid, 3) if cfg.aniso else cfg.grid
            layer_1 = torch.full(film_shape, cfg.n_film, dtype=TYPE_FLOAT)
            mee.ucell = torch.stack([layer_0, layer_1])

        mee.thickness = p['thickness']

        de_ti = mee.conv_solve().res.de_ti

        return de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]

    return forward


def grad_autograd(forward, cfg):
    params = initial_params(cfg, requires_grad=True)
    loss = forward(params)
    loss.backward()

    grads = {}
    for name, tensor in params.items():
        # A None grad means the parameter was detached from the graph somewhere inside
        # the solver; keep it distinguishable from a genuine zero gradient.
        grads[name] = None if tensor.grad is None else tensor.grad.detach().clone()

    return grads, loss.item()


def grad_numerical(forward, cfg, name, delta):
    params = {k: v.detach().clone() for k, v in initial_params(cfg).items()}
    flat = params[name].reshape(-1)

    out = []
    for i in range(flat.numel()):
        center = flat[i].item()

        flat[i] = center + delta
        loss_p = forward(params).item()

        flat[i] = center - delta
        loss_m = forward(params).item()

        flat[i] = center
        out.append((loss_p - loss_m) / (2 * delta))

    return torch.tensor(out, dtype=TYPE_FLOAT).reshape(params[name].shape)


def check(case, mode, cfg, params_to_check, rtol, atol):
    print(f'\n=== {case} / {mode} ===')
    forward = make_forward(mode, cfg)

    t0 = time.time()
    grads_ad, loss = grad_autograd(forward, cfg)
    t_ad = time.time() - t0
    print(f'loss (de_ti 0th order) = {loss:.12f}    [autograd pass: {t_ad:.2f} s]')

    header = f'{"parameter":<14}{"i":>3}  {"autograd":>22}{"numerical":>22}{"abs.diff":>12}{"rel.diff":>12}  result'
    print(header)
    print('-' * len(header))

    rows_failed = []
    for name in params_to_check:
        delta = DELTA[name]
        g_num = grad_numerical(forward, cfg, name, delta)
        g_ad = grads_ad[name]

        if g_ad is None:
            print(f'{name:<14}{"-":>3}  {"None (detached from graph)":>56}{"":>12}{"":>12}  FAIL')
            rows_failed.append((case, mode, name, -1))
            continue

        g_ad = g_ad.reshape(-1)
        g_num = g_num.reshape(-1)

        for i in range(g_num.numel()):
            a, n = g_ad[i].item(), g_num[i].item()
            abs_diff = abs(a - n)
            rel_diff = abs_diff / max(abs(n), 1E-300)
            ok = abs_diff <= atol + rtol * abs(n)
            if not ok:
                rows_failed.append((case, mode, name, i))
            print(f'{name:<14}{i:>3}  {a:>22.12e}{n:>22.12e}{abs_diff:>12.2e}{rel_diff:>12.2e}  '
                  f'{"PASS" if ok else "FAIL"}')

    return rows_failed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    parser.add_argument('--params', nargs='+', default=list(PARAMS), choices=PARAMS)
    parser.add_argument('--fto', nargs=2, type=int, default=[2, 2])
    parser.add_argument('--grid', nargs=2, type=int, default=[40, 40],
                        help='raster sampling (ny, nx); ignored by vector modeling')
    parser.add_argument('--rtol', type=float, default=1E-4)
    parser.add_argument('--atol', type=float, default=1E-8)
    parser.add_argument('--cases', nargs='+', default=list(DEFAULT_CASES), choices=list(CASES),
                        help='permittivity case(s) to run; see the Anisotropy note above')
    parser.add_argument('--eps', nargs='+', type=float, default=None,
                        help='override --cases with one custom permittivity: 1 value '
                             '(isotropic) or 3 values (eps_x, eps_y, eps_z)')
    args = parser.parse_args()

    if args.eps is not None:
        if len(args.eps) not in (1, 3):
            parser.error('--eps takes 1 value (isotropic) or 3 values (eps_x, eps_y, eps_z)')
        cases = {'custom': tuple(args.eps)}
    else:
        cases = {name: CASES[name] for name in args.cases}

    base = Config(fto=tuple(args.fto), grid=tuple(args.grid))

    print('meent torch autograd validation -- 2D grating')
    print(f'  period {base.period}, wavelength {base.wavelength}, fto {list(base.fto)}, '
          f'theta {base.theta_deg} deg, phi {base.phi_deg} deg, pol {base.pol}')
    print(f'  raster grid {base.grid}, mask edge width tau = {base.tau} nm')
    for name, eps in cases.items():
        kind = 'anisotropic -- i = 0,1,2 is eps_x, eps_y, eps_z' if len(eps) == 3 else 'isotropic'
        print(f'  case {name:<11} pillar permittivity {list(eps)}  ({kind})')
    print(f'  tolerance: |ad - num| <= {args.atol:g} + {args.rtol:g} * |num|')

    failed = []
    for case, eps in cases.items():
        cfg = Config(fto=tuple(args.fto), grid=tuple(args.grid), eps_obj=eps)
        for mode in args.modes:
            failed += check(case, mode, cfg, args.params, args.rtol, args.atol)

    n_checks = len(cases) * len(args.modes) * len(args.params)
    print('\n' + '=' * 60)
    if failed:
        print(f'FAIL: {len(failed)} mismatch(es)')
        for case, mode, name, i in failed:
            print(f'  - {case} / {mode} / {name}[{i}]')
    else:
        print(f'PASS: autograd matches numerical gradients for all {len(cases)} case(s) x '
              f'{len(args.modes)} mode(s) x {len(args.params)} parameter(s) = {n_checks} checks')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
