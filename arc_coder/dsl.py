"""
Object-Centric Domain Specific Language (DSL) for ARC-AGI.
High-performance grid and object manipulation library for program synthesis.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, deque
import numpy as np


# ---------------------------------------------------------
# 1. Core Grid Primitives & Transformations
# ---------------------------------------------------------

def to_array(grid: Any) -> np.ndarray:
    """Converts input grid (list of lists or ndarray) to 2D numpy array of int32."""
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {arr.shape}")
    return arr


def to_list(grid: np.ndarray) -> List[List[int]]:
    """Converts numpy array to standard list of lists of python ints."""
    return grid.tolist()


def get_shape(grid: Any) -> Tuple[int, int]:
    """Returns (height, width) of the grid."""
    g = to_array(grid)
    return int(g.shape[0]), int(g.shape[1])


def get_background_color(grid: Any) -> int:
    """Returns the most frequent color in the grid (defaults to 0 on tie)."""
    g = to_array(grid)
    colors, counts = np.unique(g, return_counts=True)
    # If 0 is among top, prefer 0 as background
    max_count = np.max(counts)
    top_colors = colors[counts == max_count]
    if 0 in top_colors:
        return 0
    return int(top_colors[0])


def rotate_grid(grid: Any, k: int = 1) -> np.ndarray:
    """Rotates grid 90 degrees counter-clockwise k times (k=1: 90 deg, k=2: 180, k=3: 270)."""
    return np.rot90(to_array(grid), k=k)


def flip_h(grid: Any) -> np.ndarray:
    """Flips grid horizontally (left to right)."""
    return np.fliplr(to_array(grid))


def flip_v(grid: Any) -> np.ndarray:
    """Flips grid vertically (upside down)."""
    return np.flipud(to_array(grid))


def transpose_grid(grid: Any) -> np.ndarray:
    """Transposes grid (swaps rows and columns)."""
    return to_array(grid).T


def scale_grid(grid: Any, factor: int) -> np.ndarray:
    """Scales each cell into a factor x factor block."""
    g = to_array(grid)
    return np.repeat(np.repeat(g, factor, axis=0), factor, axis=1)


def tile_grid(grid: Any, n_rows: int, n_cols: int) -> np.ndarray:
    """Repeats grid n_rows times vertically and n_cols times horizontally."""
    g = to_array(grid)
    return np.tile(g, (n_rows, n_cols))


def color_map(grid: Any, mapping: Dict[int, int]) -> np.ndarray:
    """Maps colors in grid according to mapping dictionary."""
    g = to_array(grid).copy()
    for src, dst in mapping.items():
        g[g == src] = dst
    return g


def crop_bbox(grid: Any, min_r: int, min_c: int, max_r: int, max_c: int) -> np.ndarray:
    """Crops grid to the inclusive bounding box [min_r:max_r+1, min_c:max_c+1]."""
    g = to_array(grid)
    return g[min_r : max_r + 1, min_c : max_c + 1].copy()


def crop_to_content(grid: Any, bg_color: Optional[int] = None) -> np.ndarray:
    """Crops grid to non-background bounding box."""
    g = to_array(grid)
    if bg_color is None:
        bg_color = get_background_color(g)
    coords = np.argwhere(g != bg_color)
    if coords.size == 0:
        return g
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return crop_bbox(g, min_r, min_c, max_r, max_c)


# ---------------------------------------------------------
# 2. Object Extraction & Object Representation
# ---------------------------------------------------------

class ArcObject:
    """Represents a connected or single-colored component in an ARC grid."""

    def __init__(self, color: int, coords: Set[Tuple[int, int]]):
        self.color = int(color)
        self.coords = set((int(r), int(c)) for r, c in coords)
        self.size = len(self.coords)
        if self.coords:
            rows = [r for r, _ in self.coords]
            cols = [c for _, c in self.coords]
            self.min_r, self.max_r = min(rows), max(rows)
            self.min_c, self.max_c = min(cols), max(cols)
            self.height = self.max_r - self.min_r + 1
            self.width = self.max_c - self.min_c + 1
            self.center_r = float(np.mean(rows))
            self.center_c = float(np.mean(cols))
        else:
            self.min_r = self.max_r = self.min_c = self.max_c = 0
            self.height = self.width = 0
            self.center_r = self.center_c = 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.min_r, self.min_c, self.max_r, self.max_c)

    def extract_patch(self, grid: np.ndarray, bg: int = 0) -> np.ndarray:
        """Extracts bounding box mask with object's color."""
        patch = np.full((self.height, self.width), bg, dtype=np.int32)
        for r, c in self.coords:
            patch[r - self.min_r, c - self.min_c] = self.color
        return patch

    def __repr__(self):
        return f"ArcObject(color={self.color}, size={self.size}, bbox={self.bbox})"


def find_objects(
    grid: Any,
    connectivity: int = 4,
    bg_color: Optional[int] = None,
    monochrome: bool = True,
) -> List[ArcObject]:
    """
    Finds contiguous objects in the grid using BFS.
    - connectivity: 4 (orthogonal) or 8 (orthogonal + diagonal).
    - bg_color: color to ignore (defaults to get_background_color(grid)).
    - monochrome: if True, objects must be of the same color.
    """
    g = to_array(grid)
    h, w = g.shape
    if bg_color is None:
        bg_color = get_background_color(g)

    visited = set()
    objects = []

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            val = g[r, c]
            if val == bg_color or (r, c) in visited:
                continue

            obj_color = val
            comp_coords = set()
            queue = deque([(r, c)])
            visited.add((r, c))

            while queue:
                curr_r, curr_c = queue.popleft()
                comp_coords.add((curr_r, curr_c))

                for dr, dc in deltas:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                        n_val = g[nr, nc]
                        if n_val != bg_color and (not monochrome or n_val == obj_color):
                            visited.add((nr, nc))
                            queue.append((nr, nc))

            objects.append(ArcObject(color=obj_color, coords=comp_coords))

    return objects


def count_objects(grid: Any, color: Optional[int] = None) -> int:
    """Counts objects in the grid, optionally filtering by color."""
    objs = find_objects(grid)
    if color is not None:
        return sum(1 for o in objs if o.color == color)
    return len(objs)


def get_largest_object(grid: Any) -> Optional[ArcObject]:
    """Returns the object with the largest pixel count."""
    objs = find_objects(grid)
    return max(objs, key=lambda o: o.size) if objs else None


def get_smallest_object(grid: Any) -> Optional[ArcObject]:
    """Returns the object with the smallest pixel count."""
    objs = find_objects(grid)
    return min(objs, key=lambda o: o.size) if objs else None


# ---------------------------------------------------------
# 3. Object-Centric Operations
# ---------------------------------------------------------

def recolor_object(grid: Any, obj: ArcObject, new_color: int) -> np.ndarray:
    """Returns a new grid with the specified object recolored."""
    g = to_array(grid).copy()
    for r, c in obj.coords:
        g[r, c] = new_color
    return g


def move_object(grid: Any, obj: ArcObject, dr: int, dc: int, bg_color: int = 0) -> np.ndarray:
    """Returns a new grid with the object translated by (dr, dc)."""
    g = to_array(grid).copy()
    h, w = g.shape

    # Erase old position
    for r, c in obj.coords:
        g[r, c] = bg_color

    # Draw at new position (clamped to grid)
    for r, c in obj.coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            g[nr, nc] = obj.color

    return g


def fill_between(
    grid: Any,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: int,
) -> np.ndarray:
    """Draws a straight line (orthogonal or Bresenham) between p1 and p2."""
    g = to_array(grid).copy()
    r1, c1 = p1
    r2, c2 = p2

    # Orthogonal line优先
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            g[r1, c] = color
    elif c1 == c2:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            g[r, c1] = color
    else:
        # Diagonal / Bresenham step
        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1
        r, c = r1, c1
        while (r, c) != (r2, c2):
            g[r, c] = color
            if r != r2:
                r += dr
            if c != c2:
                c += dc
        g[r2, c2] = color

    return g


def apply_gravity(grid: Any, direction: str = "down", bg_color: int = 0) -> np.ndarray:
    """Simulates gravity causing non-background pixels to fall toward direction."""
    g = to_array(grid).copy()
    h, w = g.shape

    if direction == "down":
        for c in range(w):
            non_bg = [g[r, c] for r in range(h) if g[r, c] != bg_color]
            new_col = [bg_color] * (h - len(non_bg)) + non_bg
            for r in range(h):
                g[r, c] = new_col[r]
    elif direction == "up":
        for c in range(w):
            non_bg = [g[r, c] for r in range(h) if g[r, c] != bg_color]
            new_col = non_bg + [bg_color] * (h - len(non_bg))
            for r in range(h):
                g[r, c] = new_col[r]
    elif direction == "left":
        for r in range(h):
            non_bg = [g[r, c] for c in range(w) if g[r, c] != bg_color]
            new_row = non_bg + [bg_color] * (w - len(non_bg))
            for c in range(w):
                g[r, c] = new_row[c]
    elif direction == "right":
        for r in range(h):
            non_bg = [g[r, c] for c in range(w) if g[r, c] != bg_color]
            new_row = [bg_color] * (w - len(non_bg)) + non_bg
            for c in range(w):
                g[r, c] = new_row[c]

    return g


def complete_symmetry(grid: Any, axis: str = "vertical") -> np.ndarray:
    """Completes grid symmetry by reflecting non-background pixels across axis."""
    g = to_array(grid).copy()
    if axis == "vertical":
        flipped = flip_h(g)
    elif axis == "horizontal":
        flipped = flip_v(g)
    else:
        return g

    # Overlay non-zero pixels
    mask = (g == 0) & (flipped != 0)
    g[mask] = flipped[mask]
    return g
