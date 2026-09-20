"""Conditional flow matching model for the GMC boundary sampler."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    c_dim: int = 5
    y_dim: int = 4
    hidden_dim: int = 160
    depth: int = 4
    expansion: int = 2
    time_embed_dim: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(**d)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        device = t.device
        dtype = t.dtype
        freqs = torch.exp(
            torch.linspace(0.0, math.log(1000.0), half, device=device, dtype=dtype)
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, expansion: int = 2):
        super().__init__()
        inner = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_dim),
        )
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )

    def forward(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        h = self.ff(self.norm(z))
        scale, shift = self.film(q).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        return z + h


class BoundaryVelocityNet(nn.Module):
    """v_theta(c, y, t) for conditional flow matching.

    The conditioning variables c are not transported; this network returns only the
    velocity in target space.  The training loop enforces the straight-line CFM loss.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(config.c_dim, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.input_proj = nn.Linear(config.c_dim + config.y_dim, h)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(h, 2 * h, config.expansion) for _ in range(config.depth)]
        )
        self.out_norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, config.y_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, y], dim=-1)
        z = self.input_proj(x)
        et = self.time_mlp(self.time_embedding(t))
        ec = self.cond_mlp(c)
        q = torch.cat([et, ec], dim=-1)
        for block in self.blocks:
            z = block(z, q)
        return self.out(self.out_norm(z))


def cfm_loss(model: BoundaryVelocityNet, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(y)
    t = torch.rand((y.shape[0], 1), device=y.device, dtype=y.dtype)
    yt = (1.0 - t) * eps + t * y
    target_v = y - eps
    pred_v = model(c, yt, t)
    return F.mse_loss(pred_v, target_v)


def cfm_loss_fixed(
    model: BoundaryVelocityNet,
    c: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    """Deterministic validation form of the straight-line CFM objective."""
    yt=(1.0-t)*eps+t*y
    target_v=y-eps
    return F.mse_loss(model(c,yt,t),target_v)


@torch.no_grad()
def sample_rk4(
    model: BoundaryVelocityNet,
    c: torch.Tensor,
    n_steps: int = 12,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw encoded target samples by integrating dy/dt=v_theta(c,y,t)."""

    if seed is not None:
        g = torch.Generator(device=c.device)
        g.manual_seed(seed)
        y = torch.randn((c.shape[0], model.config.y_dim), generator=g, device=c.device, dtype=c.dtype)
    else:
        y = torch.randn((c.shape[0], model.config.y_dim), device=c.device, dtype=c.dtype)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = torch.full((c.shape[0], 1), i * dt, device=c.device, dtype=c.dtype)
        dt_t = torch.tensor(dt, device=c.device, dtype=c.dtype)
        k1 = model(c, y, t0)
        k2 = model(c, y + 0.5 * dt_t * k1, t0 + 0.5 * dt)
        k3 = model(c, y + 0.5 * dt_t * k2, t0 + 0.5 * dt)
        k4 = model(c, y + dt_t * k3, t0 + dt)
        y = y + (dt_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y

@torch.no_grad()
def integrate_rk4_from_latent(
    model: BoundaryVelocityNet,
    c: torch.Tensor,
    y0: torch.Tensor,
    n_steps: int = 12,
) -> torch.Tensor:
    """Integrate the learned CFM ODE from user-supplied standard-normal latents.

    This is the response-matrix construction path used by Stage 8 Ultra.  Supplying
    the latent vectors explicitly allows scrambled Sobol / common-random-number
    sampling of the latent Gaussian, which substantially reduces the sampling noise
    in estimated response-matrix columns without changing the trained flow.
    """
    if y0.shape != (c.shape[0], model.config.y_dim):
        raise ValueError(f"latent shape {tuple(y0.shape)} incompatible with c={tuple(c.shape)}")
    y = y0.to(device=c.device, dtype=c.dtype).clone()
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t0 = torch.full((c.shape[0], 1), i * dt, device=c.device, dtype=c.dtype)
        k1 = model(c, y, t0)
        k2 = model(c, y + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = model(c, y + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = model(c, y + dt * k3, t0 + dt)
        y = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return y


def sobol_standard_normal(
    n: int,
    dim: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scrambled Sobol points transformed to N(0,1) by the inverse error function."""
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive")
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
    # Draw on CPU because SobolEngine is CPU based, then transfer once.
    u = engine.draw(n).to(dtype=dtype)
    eps = torch.finfo(dtype).eps
    u = torch.clamp(u, eps, 1.0 - eps)
    z = torch.sqrt(torch.tensor(2.0, dtype=dtype)) * torch.erfinv(2.0*u - 1.0)
    return z.to(device=device)

# -----------------------------------------------------------------------------
# Exact cached-condition RK4 inference
# -----------------------------------------------------------------------------
@dataclass
class _ConditionCache:
    input_c: torch.Tensor
    ec_silu: torch.Tensor
    film_c: list[torch.Tensor]


def _prepare_condition_cache(model: BoundaryVelocityNet, c: torch.Tensor) -> _ConditionCache:
    """Precompute every network term that is invariant along the CFM ODE.

    This is an algebraic refactor of ``BoundaryVelocityNet.forward``.  It does not
    approximate the vector field: Linear([c,y]) and Linear(SiLU([et,ec])) are split
    into their c/y and time/condition contributions and recombined exactly.
    """
    cfg=model.config; h=cfg.hidden_dim
    ip=model.input_proj
    input_c=F.linear(c,ip.weight[:,:cfg.c_dim],ip.bias)
    ec=model.cond_mlp(c); ecs=F.silu(ec)
    film_c=[]
    for block in model.blocks:
        lin=block.film[1]
        # q=[et,ec], so the second h columns multiply SiLU(ec).  Put the bias in
        # the invariant contribution; the time contribution is bias-free.
        film_c.append(F.linear(ecs,lin.weight[:,h:],lin.bias))
    return _ConditionCache(input_c,ecs,film_c)


def _time_film_terms(model: BoundaryVelocityNet, t_value: float, *, device, dtype) -> list[torch.Tensor]:
    """Per-block FiLM contribution for one scalar ODE time, shape [1,2h]."""
    h=model.config.hidden_dim
    t=torch.full((1,1),float(t_value),device=device,dtype=dtype)
    et=model.time_mlp(model.time_embedding(t)); ets=F.silu(et)
    out=[]
    for block in model.blocks:
        lin=block.film[1]
        out.append(F.linear(ets,lin.weight[:,:h],None))
    return out


def _velocity_from_cache(
    model: BoundaryVelocityNet,
    cache: _ConditionCache,
    y: torch.Tensor,
    time_terms: list[torch.Tensor],
) -> torch.Tensor:
    cfg=model.config
    ip=model.input_proj
    z=cache.input_c + F.linear(y,ip.weight[:,cfg.c_dim:],None)
    for k,block in enumerate(model.blocks):
        hff=block.ff(block.norm(z))
        film=cache.film_c[k] + time_terms[k]  # [B,2h] + [1,2h]
        scale,shift=film.chunk(2,dim=-1)
        hff=hff*(1.0+scale)+shift
        z=z+hff
    return model.out(model.out_norm(z))


@torch.inference_mode()
def integrate_rk4_from_latent_cached(
    model: BoundaryVelocityNet,
    c: torch.Tensor,
    y0: torch.Tensor,
    n_steps: int = 12,
) -> torch.Tensor:
    """RK4 with exact caching of condition- and scalar-time-only network work."""
    if y0.shape != (c.shape[0], model.config.y_dim):
        raise ValueError(f"latent shape {tuple(y0.shape)} incompatible with c={tuple(c.shape)}")
    y=y0.to(device=c.device,dtype=c.dtype).clone(); cache=_prepare_condition_cache(model,c)
    # RK4 only needs the half-step grid 0, 1/(2N), ..., 1.
    tterms=[_time_film_terms(model,j/(2.0*n_steps),device=c.device,dtype=c.dtype) for j in range(2*n_steps+1)]
    dt=1.0/float(n_steps)
    for i in range(n_steps):
        k1=_velocity_from_cache(model,cache,y,tterms[2*i])
        k2=_velocity_from_cache(model,cache,y+0.5*dt*k1,tterms[2*i+1])
        k3=_velocity_from_cache(model,cache,y+0.5*dt*k2,tterms[2*i+1])
        k4=_velocity_from_cache(model,cache,y+dt*k3,tterms[2*i+2])
        y=y+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
    return y

@torch.inference_mode()
def sample_rk4_cached(
    model: BoundaryVelocityNet,
    c: torch.Tensor,
    n_steps: int = 12,
    seed: int | None = None,
) -> torch.Tensor:
    """Random-latent counterpart of :func:`integrate_rk4_from_latent_cached`."""
    if seed is not None:
        g=torch.Generator(device=c.device); g.manual_seed(int(seed))
        y0=torch.randn((c.shape[0],model.config.y_dim),generator=g,device=c.device,dtype=c.dtype)
    else:
        y0=torch.randn((c.shape[0],model.config.y_dim),device=c.device,dtype=c.dtype)
    return integrate_rk4_from_latent_cached(model,c,y0,n_steps=n_steps)

def gather_condition_cache(cache: _ConditionCache, ids: torch.Tensor) -> _ConditionCache:
    """Gather/broadcast precomputed unique-condition cache rows for latent samples."""
    return _ConditionCache(cache.input_c[ids], cache.ec_silu[ids], [x[ids] for x in cache.film_c])


@torch.inference_mode()
def integrate_rk4_from_latent_prepared(
    model: BoundaryVelocityNet,
    cache: _ConditionCache,
    y0: torch.Tensor,
    n_steps: int = 12,
) -> torch.Tensor:
    """RK4 from a precomputed condition cache (no condition network evaluation)."""
    y=y0.clone()
    tterms=[_time_film_terms(model,j/(2.0*n_steps),device=y.device,dtype=y.dtype) for j in range(2*n_steps+1)]
    dt=1.0/float(n_steps)
    for i in range(n_steps):
        k1=_velocity_from_cache(model,cache,y,tterms[2*i])
        k2=_velocity_from_cache(model,cache,y+0.5*dt*k1,tterms[2*i+1])
        k3=_velocity_from_cache(model,cache,y+0.5*dt*k2,tterms[2*i+1])
        k4=_velocity_from_cache(model,cache,y+dt*k3,tterms[2*i+2])
        y=y+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
    return y


def prepare_condition_cache(model: BoundaryVelocityNet, c: torch.Tensor) -> _ConditionCache:
    """Public wrapper used by high-throughput response construction."""
    return _prepare_condition_cache(model,c)
