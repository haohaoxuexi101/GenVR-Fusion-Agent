"""Coordinate transforms for 2D rectangular GMC cell models.

Stage-8 HPC v2 adds an optional ``excess_chord`` path parameterization.  Instead
of asking the CFM to learn an unconstrained total optical path length ``L``, the
network learns the non-negative excess over the straight entry-to-exit chord,

    Delta L = L - L_chord >= 0.

At decode time ``L = L_chord + Delta L`` by construction.  This removes the
large post-hoc chord-repair rate observed with older checkpoints while keeping
backward compatibility with checkpoints whose transform has ``path_mode=absolute``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch


_EPS = 1.0e-8
_CLIP = 1.0e-5


def _logit_np(x: np.ndarray, clip: float = _CLIP) -> np.ndarray:
    x = np.clip(x, clip, 1.0 - clip)
    return np.log(x / (1.0 - x))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _angle_radius_np(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle = np.arctan2(v, u)
    frac = (angle + np.pi) / (2.0 * np.pi)
    radius = np.sqrt(np.maximum(0.0, u * u + v * v))
    return frac, np.clip(radius, 0.0, 1.0)


def _from_angle_radius_np(frac: np.ndarray, radius: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle = frac * (2.0 * np.pi) - np.pi
    return radius * np.cos(angle), radius * np.sin(angle)


def _positive_bounds(values: np.ndarray, qlo: float, qhi: float) -> Tuple[float, float]:
    lv = np.log10(np.maximum(values, _EPS))
    lo, hi = np.quantile(lv, [qlo, qhi])
    if hi - lo < 1.0e-6:
        lo -= 1.0
        hi += 1.0
    margin = 0.10 * (hi - lo)
    return float(lo - margin), float(hi + margin)


def _perimeter_point_np(p: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode canonical perimeter fraction to a point on a W x H rectangle."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0 - 1.0e-12)
    W = np.asarray(W, dtype=np.float64); H = np.asarray(H, dtype=np.float64)
    s = p * (2.0 * (W + H))
    x = np.empty_like(s); y = np.empty_like(s)
    m0 = s < W
    m1 = (~m0) & (s < W + H)
    m2 = (~m0) & (~m1) & (s < 2.0 * W + H)
    m3 = ~(m0 | m1 | m2)
    x[m0] = s[m0]; y[m0] = 0.0
    x[m1] = W[m1]; y[m1] = s[m1] - W[m1]
    x[m2] = 2.0 * W[m2] + H[m2] - s[m2]; y[m2] = H[m2]
    x[m3] = 0.0; y[m3] = 2.0 * (W[m3] + H[m3]) - s[m3]
    return x, y


class RectTargetTransformMixin:
    """Shared target encoding for exit p, projected direction, and path length."""

    def _encode_positive_np(self, x: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        lx = np.log10(np.maximum(x, self.eps))
        lo, hi = bounds
        z = (lx - lo) / (hi - lo)
        return _logit_np(z, self.clip)

    def _decode_positive_np(self, zenc: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        z = _sigmoid_np(zenc)
        lo, hi = bounds
        lx = lo + z * (hi - lo)
        return np.maximum(10.0 ** lx, 0.0)

    def _path_baseline_np(self, conditions: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _path_variable_np(self, conditions: np.ndarray | None, p: np.ndarray, L: np.ndarray) -> np.ndarray:
        mode = getattr(self, "path_mode", "absolute")
        if mode == "absolute":
            return np.asarray(L, dtype=np.float64)
        if mode != "excess_chord":
            raise ValueError(f"unknown path_mode={mode}")
        if conditions is None:
            raise ValueError("conditions are required for path_mode='excess_chord'")
        chord = self._path_baseline_np(conditions, p)
        # Direct random-flight MC can differ from the analytic chord by tiny roundoff. Clamp only
        # that roundoff; the transform itself then guarantees L_decoded >= chord.
        return np.maximum(np.asarray(L, dtype=np.float64) - chord, 0.0)

    def encode_targets(self, targets: np.ndarray, conditions: np.ndarray | None = None) -> np.ndarray:
        p = targets[:, 0]
        u = targets[:, 1]
        v = targets[:, 2]
        L = targets[:, 3]
        angle_frac, radius = _angle_radius_np(u, v)
        path_var = self._path_variable_np(conditions, p, L)
        encoded = np.stack(
            [
                _logit_np(p, self.clip),
                _logit_np(angle_frac, self.clip),
                _logit_np(radius, self.clip),
                self._encode_positive_np(path_var, self.logL_bounds),
            ],
            axis=1,
        )
        return encoded.astype(np.float32)

    def decode_targets_np(self, encoded: np.ndarray, conditions: np.ndarray | None = None) -> np.ndarray:
        p = _sigmoid_np(encoded[:, 0])
        angle_frac = _sigmoid_np(encoded[:, 1])
        radius = _sigmoid_np(encoded[:, 2])
        u, v = _from_angle_radius_np(angle_frac, radius)
        path_var = self._decode_positive_np(encoded[:, 3], self.logL_bounds)
        mode = getattr(self, "path_mode", "absolute")
        if mode == "absolute":
            L = path_var
        elif mode == "excess_chord":
            if conditions is None:
                raise ValueError("conditions are required to decode excess_chord path targets")
            chord = self._path_baseline_np(conditions, p)
            # Add only a float32-scale safety ulp so a subsequent float64 geometry
            # check cannot see L infinitesimally below the chord after casting.
            L = chord * (1.0 + 2.5e-7) + path_var + self.eps
        else:
            raise ValueError(f"unknown path_mode={mode}")
        return np.stack([p, u, v, L], axis=1).astype(np.float32)

    def decode_targets_torch(self, encoded: torch.Tensor, conditions_raw: torch.Tensor | None = None) -> torch.Tensor:
        # Torch decode is retained for compatibility.  The constrained mode is used
        # primarily by batched response construction, whose raw geometry currently
        # lives in NumPy; requiring raw conditions here avoids silently using encoded c.
        p = torch.sigmoid(encoded[:, 0])
        angle_frac = torch.sigmoid(encoded[:, 1])
        radius = torch.sigmoid(encoded[:, 2])
        angle = angle_frac * (2.0 * torch.pi) - torch.pi
        u = radius * torch.cos(angle)
        v = radius * torch.sin(angle)
        z = torch.sigmoid(encoded[:, 3])
        lo, hi = self.logL_bounds
        logL = lo + z * (hi - lo)
        path_var = torch.pow(torch.tensor(10.0, device=encoded.device, dtype=encoded.dtype), logL)
        if getattr(self, "path_mode", "absolute") == "absolute":
            L = path_var
        else:
            if conditions_raw is None:
                raise ValueError("raw conditions required for excess_chord torch decode")
            # This branch is intentionally simple and exact; it is not on the hot
            # response-builder path.
            c = conditions_raw.detach().cpu().numpy()
            chord = torch.as_tensor(self._path_baseline_np(c, p.detach().cpu().numpy()), device=encoded.device, dtype=encoded.dtype)
            L = chord + path_var
        return torch.stack([p, u, v, L], dim=1)


@dataclass
class BoundaryTransform(RectTargetTransformMixin):
    """Boundary-model transform.

    ``path_mode='absolute'`` reproduces legacy checkpoints.
    ``path_mode='excess_chord'`` is recommended for new training because it makes
    the geometric inequality L >= chord exact by construction.
    """

    logW_bounds: Tuple[float, float]
    logH_bounds: Tuple[float, float]
    logL_bounds: Tuple[float, float]
    path_mode: str = "absolute"
    clip: float = _CLIP
    eps: float = _EPS

    @staticmethod
    def _baseline_static(conditions: np.ndarray, p: np.ndarray) -> np.ndarray:
        W = conditions[:, 0].astype(np.float64); H = conditions[:, 1].astype(np.float64)
        yin = conditions[:, 2].astype(np.float64)
        xo, yo = _perimeter_point_np(p, W, H)
        return np.hypot(xo, yo - yin)

    def _path_baseline_np(self, conditions: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._baseline_static(np.asarray(conditions), np.asarray(p))

    @staticmethod
    def fit(conditions: np.ndarray, targets: np.ndarray, qlo: float = 1.0e-5, qhi: float = 1.0 - 1.0e-5,
            path_mode: str = "absolute") -> "BoundaryTransform":
        W = conditions[:, 0]; H = conditions[:, 1]; L = targets[:, 3]
        if path_mode == "absolute":
            path = L
        elif path_mode == "excess_chord":
            path = np.maximum(L - BoundaryTransform._baseline_static(conditions, targets[:, 0]), 0.0)
        else:
            raise ValueError(f"unknown path_mode={path_mode}")
        return BoundaryTransform(_positive_bounds(W, qlo, qhi), _positive_bounds(H, qlo, qhi),
                                 _positive_bounds(path, qlo, qhi), path_mode=path_mode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BoundaryTransform":
        return BoundaryTransform(
            logW_bounds=tuple(d["logW_bounds"]),
            logH_bounds=tuple(d["logH_bounds"]),
            logL_bounds=tuple(d["logL_bounds"]),
            path_mode=str(d.get("path_mode", "absolute")),
            clip=float(d.get("clip", _CLIP)),
            eps=float(d.get("eps", _EPS)),
        )

    def encode_conditions(self, conditions: np.ndarray) -> np.ndarray:
        W = conditions[:, 0]; H = conditions[:, 1]; y = conditions[:, 2]
        u = conditions[:, 3]; v = conditions[:, 4]
        yin_frac = np.clip(y / np.maximum(H, self.eps), 0.0, 1.0)
        angle_frac, radius = _angle_radius_np(u, v)
        encoded = np.stack([
            self._encode_positive_np(W, self.logW_bounds),
            self._encode_positive_np(H, self.logH_bounds),
            _logit_np(yin_frac, self.clip),
            _logit_np(angle_frac, self.clip),
            _logit_np(radius, self.clip),
        ], axis=1)
        return encoded.astype(np.float32)


@dataclass
class InternalTransform(RectTargetTransformMixin):
    """Internal-source transform.

    conditions columns: W,H,x_in,y_in,u_in,v_in, all in optical coordinates except
    direction components.  ``excess_chord`` is recommended for new checkpoints.
    """

    logW_bounds: Tuple[float, float]
    logH_bounds: Tuple[float, float]
    logL_bounds: Tuple[float, float]
    path_mode: str = "absolute"
    clip: float = _CLIP
    eps: float = _EPS

    @staticmethod
    def _baseline_static(conditions: np.ndarray, p: np.ndarray) -> np.ndarray:
        W = conditions[:, 0].astype(np.float64); H = conditions[:, 1].astype(np.float64)
        xin = conditions[:, 2].astype(np.float64); yin = conditions[:, 3].astype(np.float64)
        xo, yo = _perimeter_point_np(p, W, H)
        return np.hypot(xo - xin, yo - yin)

    def _path_baseline_np(self, conditions: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self._baseline_static(np.asarray(conditions), np.asarray(p))

    @staticmethod
    def fit(conditions: np.ndarray, targets: np.ndarray, qlo: float = 1.0e-5, qhi: float = 1.0 - 1.0e-5,
            path_mode: str = "absolute") -> "InternalTransform":
        W = conditions[:, 0]; H = conditions[:, 1]; L = targets[:, 3]
        if path_mode == "absolute":
            path = L
        elif path_mode == "excess_chord":
            path = np.maximum(L - InternalTransform._baseline_static(conditions, targets[:, 0]), 0.0)
        else:
            raise ValueError(f"unknown path_mode={path_mode}")
        return InternalTransform(_positive_bounds(W, qlo, qhi), _positive_bounds(H, qlo, qhi),
                                 _positive_bounds(path, qlo, qhi), path_mode=path_mode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InternalTransform":
        return InternalTransform(
            logW_bounds=tuple(d["logW_bounds"]),
            logH_bounds=tuple(d["logH_bounds"]),
            logL_bounds=tuple(d["logL_bounds"]),
            path_mode=str(d.get("path_mode", "absolute")),
            clip=float(d.get("clip", _CLIP)),
            eps=float(d.get("eps", _EPS)),
        )

    def encode_conditions(self, conditions: np.ndarray) -> np.ndarray:
        W = conditions[:, 0]; H = conditions[:, 1]; x = conditions[:, 2]; y = conditions[:, 3]
        u = conditions[:, 4]; v = conditions[:, 5]
        xfrac = np.clip(x / np.maximum(W, self.eps), 0.0, 1.0)
        yfrac = np.clip(y / np.maximum(H, self.eps), 0.0, 1.0)
        angle_frac, radius = _angle_radius_np(u, v)
        encoded = np.stack([
            self._encode_positive_np(W, self.logW_bounds),
            self._encode_positive_np(H, self.logH_bounds),
            _logit_np(xfrac, self.clip),
            _logit_np(yfrac, self.clip),
            _logit_np(angle_frac, self.clip),
            _logit_np(radius, self.clip),
        ], axis=1)
        return encoded.astype(np.float32)
