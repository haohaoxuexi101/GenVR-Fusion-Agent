"""Structured 2D benchmark definitions for stage-2 GMC reproduction.

The geometries follow the numerical descriptions in Farmer's dissertation closely
where dimensions and cross sections are stated.  Some fine layout details in the
published figures are not code-level specified, so these constructors keep the
benchmark parameters explicit and easy to modify.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


SourceKind = Literal["left_boundary", "volume_box"]


@dataclass
class Structured2DProblem:
    name: str
    nx: int
    ny: int
    width: float
    height: float
    sigma_a: np.ndarray  # shape (ny, nx)
    sigma_s: np.ndarray  # shape (ny, nx)
    source_kind: SourceKind
    source_box: Optional[Tuple[float, float, float, float]] = None  # xmin,xmax,ymin,ymax
    boundary_source_y_range: Optional[Tuple[float, float]] = None

    @property
    def dx(self) -> float:
        return self.width / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def volume(self) -> float:
        # Unit out-of-plane thickness.
        return self.dx * self.dy

    def cell_indices(self, x: float, y: float) -> Tuple[int, int] | None:
        if x < 0.0 or x >= self.width or y < 0.0 or y >= self.height:
            return None
        ix = min(self.nx - 1, max(0, int(x / self.dx)))
        iy = min(self.ny - 1, max(0, int(y / self.dy)))
        return ix, iy


def _cell_centers(problem: Structured2DProblem) -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(problem.nx) + 0.5) * problem.dx
    ys = (np.arange(problem.ny) + 0.5) * problem.dy
    return np.meshgrid(xs, ys)


def make_lattice_problem(nx: int = 112, ny: int = 112) -> Structured2DProblem:
    """7 x 7 cm heterogeneous lattice benchmark.

    Stated material data:
      background/source: sigma_a=0, sigma_s=1 cm^-1
      absorbers:         sigma_a=9.5, sigma_s=0.5 cm^-1
      central source:    1 x 1 cm subdivision

    The absorber pattern is an editable 1-cm checker-like layout chosen to match
    the qualitative Figure 3.1 lattice: strong absorbers around a central source.
    """
    width = height = 7.0
    sa = np.zeros((ny, nx), dtype=np.float64)
    ss = np.ones((ny, nx), dtype=np.float64)
    prob = Structured2DProblem(
        name="lattice_7cm",
        nx=nx,
        ny=ny,
        width=width,
        height=height,
        sigma_a=sa,
        sigma_s=ss,
        source_kind="volume_box",
        source_box=(3.0, 4.0, 3.0, 4.0),
    )
    X, Y = _cell_centers(prob)

    # A compact, explicit absorber layout.  Modify this list if an author code or
    # exact geometry file becomes available.
    # Integer-centimeter absorber pattern digitized from Fig. 3.1a.
    # Blue absorber squares surround the 1 cm central source in a staggered
    # reactor-lattice-like pattern.
    absorber_boxes = [
        (1.0, 2.0, 5.0, 6.0), (5.0, 6.0, 5.0, 6.0),
        (2.0, 3.0, 4.0, 5.0), (4.0, 5.0, 4.0, 5.0),
        (1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 3.0, 4.0),
        (2.0, 3.0, 2.0, 3.0), (4.0, 5.0, 2.0, 3.0),
        (1.0, 2.0, 1.0, 2.0), (3.0, 4.0, 1.0, 2.0), (5.0, 6.0, 1.0, 2.0),
    ]
    for xmin, xmax, ymin, ymax in absorber_boxes:
        mask = (X >= xmin) & (X < xmax) & (Y >= ymin) & (Y < ymax)
        sa[mask] = 9.5
        ss[mask] = 0.5
    return prob


def make_hohlraum_problem(nx: int = 112, ny: int = 112) -> Structured2DProblem:
    """1.3 x 1.3 cm linearized hohlraum benchmark.

    Stated material data from the dissertation:
      black walls:          sigma_a=50, sigma_s=50
      red source stripe:    sigma_a=5,  sigma_s=95
      green frame:          sigma_a=10, sigma_s=90
      central blue absorber:sigma_a=95, sigma_s=5
      white near-vacuum:    sigma_a=0,  sigma_s=5

    The coordinates are parameterized from the axis ticks and layout in Fig. 3.1d.
    """
    width = height = 1.3
    sa = np.zeros((ny, nx), dtype=np.float64)
    ss = np.full((ny, nx), 5.0, dtype=np.float64)  # white near-vacuum
    prob = Structured2DProblem(
        name="linearized_hohlraum_1p3cm",
        nx=nx,
        ny=ny,
        width=width,
        height=height,
        sigma_a=sa,
        sigma_s=ss,
        source_kind="left_boundary",
        boundary_source_y_range=(0.25, 1.05),
    )
    X, Y = _cell_centers(prob)

    # Black absorbing outer wall.
    wall = 0.05
    mask_wall = (X < wall) | (X >= width - wall) | (Y < wall) | (Y >= height - wall)
    sa[mask_wall] = 50.0
    ss[mask_wall] = 50.0

    # Red boundary-source stripe on the left wall, approximately y in [0.25, 1.05].
    mask_red = (X < wall) & (Y >= 0.25) & (Y <= 1.05)
    sa[mask_red] = 5.0
    ss[mask_red] = 95.0

    # Green frame around the central absorber, using the visible tick positions.
    frame_outer = (X >= 0.45) & (X <= 0.85) & (Y >= 0.25) & (Y <= 1.05)
    absorber = (X >= 0.53) & (X <= 0.77) & (Y >= 0.34) & (Y <= 0.96)
    frame = frame_outer & (~absorber)
    sa[frame] = 10.0
    ss[frame] = 90.0

    # Blue central absorber.
    sa[absorber] = 95.0
    ss[absorber] = 5.0
    return prob

def make_tokamak_square_discrete_problem(nx: int = 224, ny: int = 224):
    width = height = 14.0
    sa = np.zeros((ny, nx), dtype=np.float64)
    ss = np.ones((ny, nx), dtype=np.float64)
    
    # 1. 【整体右移】为了防止左边管道撞墙，将源的中心从 X=7.0 移动到 X=8.0
    prob = Structured2DProblem(
        name="tokamak_radial_square",
        nx=nx,
        ny=ny,
        width=width,
        height=height,
        sigma_a=sa,
        sigma_s=ss,
        source_kind="volume_box",
        source_box=(7.0, 9.0, 4.0, 6.0), # 新的源位置：X范围 7~9，中心为 X=8.0
    )
    X, Y = _cell_centers(prob)

    # 保持方块大小不变，确保不会因为网格产生畸变
    box_size = 0.75 
    half = box_size / 2.0
    absorber_centers = []

    # 新的几何中心：跟随源的中心移动到了 X=8.0
    cx, cy = 8.0, 5.0
    
    # =========================================================
    # 2. 【内层包层：减少数量，防止重叠】
    # =========================================================
    R_inner = 2.4
    # 方块数量从 13 降到 11，拉开明显间隙
    theta_inner = np.linspace(np.radians(150), np.radians(390), 11) 
    for t in theta_inner:
        x = cx + R_inner * np.cos(t)
        y = cy + R_inner * np.sin(t)
        absorber_centers.append((x, y))

    # =========================================================
    # 3. 【外层包层：减少数量，防止重叠】
    # =========================================================
    R_outer = 4.0  
    theta_outer_start = np.radians(160)
    theta_outer_end = np.radians(380)
    
    # 方块数量从 20 降到 15，拉开明显间隙
    theta_outer = np.linspace(theta_outer_start, theta_outer_end, 15) 
    for t in theta_outer:
        x = cx + R_outer * np.cos(t)
        y = cy + R_outer * np.sin(t)
        absorber_centers.append((x, y))

    # =========================================================
    # 4. 【顶部管道：降低高度、调整倾斜度，防止越界截断】
    # =========================================================
    left_top_x = cx + R_outer * np.cos(theta_outer_start)
    right_top_x = cx + R_outer * np.cos(theta_outer_end)
    
    # 留出 0.8 cm 的缝隙，不与 U 型圆弧粘连
    base_y = cy + R_outer * np.sin(theta_outer_start) + 0.8  
    top_y = 13.0       # 最高点从 13.5 降到 13.0，防止撞顶壁
    tilt_offset = -2.5 # 倾斜度从 -3.5 改为 -2.5，结合整体右移，彻底解决左侧越界
    
    # 管道的方块数也从 6 降到 5，让上下间隙变大，更加透气
    blocks_per_wall = 5

    wall_bases_x = np.linspace(left_top_x, right_top_x, 5) 

    for start_x in wall_bases_x:
        end_x = start_x + tilt_offset
        for i in range(blocks_per_wall):
            fraction = i / (blocks_per_wall - 1)
            px = start_x + fraction * (end_x - start_x)
            py = base_y + fraction * (top_y - base_y)
            absorber_centers.append((px, py))

    # =========================================================
    # 5. 渲染防截断校验
    # =========================================================
    for center_x, center_y in absorber_centers:
        xmin, xmax = center_x - half, center_x + half
        ymin, ymax = center_y - half, center_y + half
        
        # 加上 max 和 min 函数进行极端的边界保护，即使有误差也不会报错越界
        mask = (X >= max(0.0, xmin)) & (X < min(width, xmax)) & (Y >= max(0.0, ymin)) & (Y < min(height, ymax))
        sa[mask] = 9.5
        ss[mask] = 0.5
        
    return prob

def make_tokamak_coils_problem(nx: int = 150, ny: int = 150): 
    width = height = 1.3  # 与 lattice 一致的整体尺寸
    s = width / 1.3       # 从原始 1.3 cm 几何整体放大到 7 cm
    sa = np.zeros((ny, nx), dtype=np.float64)
    ss = np.full((ny, nx), 5.0, dtype=np.float64)  
    
    # 源从左侧边界入射（left_boundary），不再使用体积源方块
    
    prob = Structured2DProblem(
        name="tokamak_d_shape_coils_red_cs",
        nx=nx,
        ny=ny,
        width=width,
        height=height,
        sigma_a=sa,
        sigma_s=ss,
        source_kind="left_boundary",
        boundary_source_y_range=(0.25 * s, 1.05 * s),
    )
    X, Y = _cell_centers(prob)

    # =========================================================================
    # 1. 生成 D 字形掩码
    # =========================================================================
    def get_d_mask(R0, Z0, a, kappa, delta):
        Y_norm = (Y - Z0) / (a * kappa)
        valid_y = np.abs(Y_norm) <= 1.0
        Y_safe = np.clip(Y_norm, -1.0, 1.0)
        theta = np.arcsin(Y_safe)
        X_right = R0 + a * np.cos(theta + delta * Y_safe)
        X_left  = R0 - a * np.cos(theta - delta * Y_safe)
        return valid_y & (X >= X_left) & (X <= X_right)

    R0, Z0 = 0.65 * s, 0.65 * s
    mask_outer = get_d_mask(R0, Z0, a=0.35 * s, kappa=1.40, delta=0.5)
    mask_mid   = get_d_mask(R0, Z0, a=0.28 * s, kappa=1.25, delta=0.45)
    mask_inner = get_d_mask(R0, Z0, a=0.18 * s, kappa=1.10, delta=0.3)

    # =========================================================================
    # 2. 生成外围磁场线圈 (将左侧红色线圈与外侧黑色线圈分离)
    # =========================================================================
    coil_size = 0.06 * s
    half = coil_size / 2.0
    
    # 【修改点】分离出左侧的中心螺线管 (CS)
    cs_centers = [
        (0.18 * s, 0.25 * s), (0.18 * s, 0.45 * s), (0.18 * s, 0.65 * s), (0.18 * s, 0.85 * s), (0.18 * s, 1.05 * s)
    ]
    
    # 其余的外围极向场线圈 (PF)
    pf_centers = [
        (0.40 * s, 1.20 * s), (0.65 * s, 1.24 * s), (0.90 * s, 1.15 * s), # 顶部
        (0.40 * s, 0.10 * s), (0.65 * s, 0.06 * s), (0.90 * s, 0.15 * s), # 底部
        (1.12 * s, 0.90 * s), (1.18 * s, 0.65 * s), (1.12 * s, 0.40 * s)  # 右侧
    ]
    
    mask_cs = np.zeros_like(X, dtype=bool)
    for cx, cy in cs_centers:
        box = (X >= cx - half) & (X < cx + half) & (Y >= cy - half) & (Y < cy + half)
        mask_cs = mask_cs | box

    mask_pf = np.zeros_like(X, dtype=bool)
    for cx, cy in pf_centers:
        box = (X >= cx - half) & (X < cx + half) & (Y >= cy - half) & (Y < cy + half)
        mask_pf = mask_pf | box

    # =========================================================================
    # 3. 涂层与材料赋值
    # =========================================================================
    
    # 涂层 1: 最外层黑框 + 黑色的 PF 线圈
    wall = (mask_outer & ~mask_mid) | mask_pf
    sa[wall] = 50.0
    ss[wall] = 50.0

    # 【修改点】涂层 2: 左侧红色的 CS 线圈 (使用原定的红色材料属性)
    sa[mask_cs] = 5.0
    ss[mask_cs] = 95.0

    # 涂层 3: 绿框 (包层)
    frame = mask_mid & ~mask_inner
    sa[frame] = 10.0
    ss[frame] = 90.0

    # 涂层 4: 蓝色核心
    sa[mask_inner] = 95.0
    ss[mask_inner] = 5.0

    return prob

def material_rgb(problem: Structured2DProblem) -> np.ndarray:
    """Small RGB material map for visual checks."""
    img = np.ones((problem.ny, problem.nx, 3), dtype=np.float32)
    sa, ss = problem.sigma_a, problem.sigma_s
    # white void stays white
    black = (np.isclose(sa, 50.0) & np.isclose(ss, 50.0))
    red = (np.isclose(sa, 5.0) & np.isclose(ss, 95.0))
    green = (np.isclose(sa, 10.0) & np.isclose(ss, 90.0))
    blue = ((np.isclose(sa, 95.0) & np.isclose(ss, 5.0)) |
            (np.isclose(sa, 9.5) & np.isclose(ss, 0.5)))
    img[black] = (0.0, 0.0, 0.0)
    img[red] = (0.9, 0.05, 0.05)
    img[green] = (0.1, 0.6, 0.1)
    img[blue] = (0.05, 0.05, 0.8)
    src = np.isclose(sa, 0.0) & np.isclose(ss, 1.0)
    if problem.source_box is not None:
        X, Y = _cell_centers(problem)
        xmin, xmax, ymin, ymax = problem.source_box
        sbox = (X >= xmin) & (X < xmax) & (Y >= ymin) & (Y < ymax)
        img[sbox] = (0.8, 0.05, 0.8)
    return img
