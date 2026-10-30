"""
Autonomous Agentic Solver for ARC-AGI.
Executes Strategy A (Direct Grid), Strategy B (Object-Centric DSL),
and Strategy C (Heuristic Synthesis) with automated iterative verification.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arc_coder.dsl import *
from arc_coder.execute import test_solution, grids_equal


class ArcAgenticSolver:
    """Multi-strategy autonomous solver for ARC tasks."""

    def __init__(self, task: Dict[str, Any], task_id: str = "task_demo"):
        self.task = task
        self.task_id = task_id
        self.train_pairs = task.get("train", [])
        self.test_pairs = task.get("test", [])

    def analyze_task(self) -> Dict[str, Any]:
        """Analyzes shape deltas and color palettes across train pairs."""
        shapes_in = [tuple(np.array(p["input"]).shape) for p in self.train_pairs]
        shapes_out = [tuple(np.array(p["output"]).shape) for p in self.train_pairs]
        same_shape = shapes_in == shapes_out

        in_colors = set(c for p in self.train_pairs for row in p["input"] for c in row)
        out_colors = set(c for p in self.train_pairs for row in p["output"] for c in row)
        new_colors = out_colors - in_colors

        info = {
            "shapes_in": shapes_in,
            "shapes_out": shapes_out,
            "same_shape": same_shape,
            "in_colors": in_colors,
            "out_colors": out_colors,
            "new_colors": new_colors,
        }
        return info

    def try_strategy_a_geometric(self) -> Optional[Tuple[str, List]]:
        """Tests standard geometric transformations: identity, rotations, flips, symmetries."""
        candidates = [
            ("Identity", "def transform(g):\n    return np.array(g).tolist()"),
            ("Rotate 90", "def transform(g):\n    return rotate_grid(g, 1).tolist()"),
            ("Rotate 180", "def transform(g):\n    return rotate_grid(g, 2).tolist()"),
            ("Rotate 270", "def transform(g):\n    return rotate_grid(g, 3).tolist()"),
            ("Flip Horizontal", "def transform(g):\n    return flip_h(g).tolist()"),
            ("Flip Vertical", "def transform(g):\n    return flip_v(g).tolist()"),
            ("Crop to Content", "def transform(g):\n    return crop_to_content(g).tolist()"),
            ("Symmetry Vertical", "def transform(g):\n    return complete_symmetry(g, 'vertical').tolist()"),
            ("Symmetry Horizontal", "def transform(g):\n    return complete_symmetry(g, 'horizontal').tolist()"),
            ("Gravity Down", "def transform(g):\n    return apply_gravity(g, 'down').tolist()"),
        ]

        for name, code in candidates:
            passed, msg, test_preds = test_solution(code, self.task)
            if passed:
                return code, test_preds
        return None

    def try_strategy_b_color_mapping(self) -> Optional[Tuple[str, List]]:
        """Deduces consistent 1-to-1 global color remapping if shapes match."""
        info = self.analyze_task()
        if not info["same_shape"]:
            return None

        # Build color mapping from first pair
        p0_in = np.array(self.train_pairs[0]["input"])
        p0_out = np.array(self.train_pairs[0]["output"])
        mapping = {}

        for c_in in range(10):
            mask = p0_in == c_in
            if not mask.any():
                continue
            out_vals = p0_out[mask]
            unique_out = np.unique(out_vals)
            if len(unique_out) == 1:
                mapping[int(c_in)] = int(unique_out[0])
            else:
                return None  # Inconsistent 1-to-1 mapping

        code = (
            f"def transform(g):\n"
            f"    mapping = {mapping}\n"
            f"    return color_map(g, mapping).tolist()\n"
        )
        passed, msg, test_preds = test_solution(code, self.task)
        if passed:
            return code, test_preds
        return None

    def try_strategy_c_tiling_scaling(self) -> Optional[Tuple[str, List]]:
        """Tests scaling and tiling transformations."""
        info = self.analyze_task()
        s_in = info["shapes_in"][0]
        s_out = info["shapes_out"][0]

        # Check integer scaling
        if s_out[0] % s_in[0] == 0 and s_out[1] % s_in[1] == 0:
            scale_r = s_out[0] // s_in[0]
            scale_c = s_out[1] // s_in[1]

            if scale_r == scale_c and scale_r > 1:
                code = f"def transform(g):\n    return scale_grid(g, {scale_r}).tolist()\n"
                passed, msg, test_preds = test_solution(code, self.task)
                if passed:
                    return code, test_preds

            code_tile = f"def transform(g):\n    return tile_grid(g, {scale_r}, {scale_c}).tolist()\n"
            passed, msg, test_preds = test_solution(code_tile, self.task)
            if passed:
                return code_tile, test_preds

        return None

    def solve(self, solutions_dir: str = "solutions") -> Tuple[bool, Optional[str], Optional[List]]:
        """Executes full strategy pipeline until solved."""
        os.makedirs(solutions_dir, exist_ok=True)
        print(f"Solving Task {self.task_id}...")

        # 1. Strategy A: Direct Geometry
        res = self.try_strategy_a_geometric()
        if res:
            code, test_preds = res
            self._save_solution(code, solutions_dir)
            print(f"[SOLVED] Task {self.task_id} via Strategy A (Geometric)!")
            return True, code, test_preds

        # 2. Strategy B: Color Mapping
        res = self.try_strategy_b_color_mapping()
        if res:
            code, test_preds = res
            self._save_solution(code, solutions_dir)
            print(f"[SOLVED] Task {self.task_id} via Strategy B (Color Mapping)!")
            return True, code, test_preds

        # 3. Strategy C: Tiling & Scaling
        res = self.try_strategy_c_tiling_scaling()
        if res:
            code, test_preds = res
            self._save_solution(code, solutions_dir)
            print(f"[SOLVED] Task {self.task_id} via Strategy C (Tiling/Scaling)!")
            return True, code, test_preds

        print(f"[UNSOLVED] Task {self.task_id} could not be solved by basic primitives.")
        return False, None, None

    def _save_solution(self, code: str, directory: str):
        path = os.path.join(directory, f"{self.task_id}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Auto-synthesized solution for {self.task_id}\n")
            f.write("import numpy as np\n")
            f.write("from arc_coder.dsl import *\n\n")
            f.write(code + "\n")
        print(f"Saved verified solution to {path}")
