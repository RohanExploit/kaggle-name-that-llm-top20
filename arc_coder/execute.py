"""
Execution and Verification Harness for ARC-AGI Agentic Coders.
Safely executes candidate transform() functions against train pairs,
computes pixel-level diffs on failure, and predicts test outputs.
"""

import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Import full DSL into evaluation namespace
import arc_coder.dsl as dsl
from arc_coder.dsl import *


def create_execution_namespace() -> Dict[str, Any]:
    """Builds the execution environment with NumPy and all DSL functions available."""
    ns = {
        "np": np,
        "numpy": np,
    }
    # Export all public dsl symbols into namespace
    for name in dir(dsl):
        if not name.startswith("_"):
            ns[name] = getattr(dsl, name)
    return ns


def grids_equal(g1: Any, g2: Any) -> bool:
    """Checks whether two grids are strictly identical in shape and values."""
    try:
        a1 = np.array(g1, dtype=np.int32)
        a2 = np.array(g2, dtype=np.int32)
        return bool(np.array_equal(a1, a2))
    except Exception:
        return False


def compute_grid_diff(pred: Any, expected: Any) -> Dict[str, Any]:
    """Computes detailed diff metrics between predicted and expected grids."""
    try:
        p_arr = np.array(pred, dtype=np.int32)
        e_arr = np.array(expected, dtype=np.int32)

        if p_arr.shape != e_arr.shape:
            return {
                "shape_match": False,
                "expected_shape": e_arr.shape,
                "got_shape": p_arr.shape,
                "pixel_mismatches": int(e_arr.size),
                "accuracy": 0.0,
            }

        mismatches = int(np.sum(p_arr != e_arr))
        total_pixels = int(e_arr.size)
        accuracy = 1.0 - (mismatches / max(total_pixels, 1))

        return {
            "shape_match": True,
            "expected_shape": e_arr.shape,
            "got_shape": p_arr.shape,
            "pixel_mismatches": mismatches,
            "total_pixels": total_pixels,
            "accuracy": accuracy,
        }
    except Exception as e:
        return {"error": str(e), "shape_match": False}


def test_solution(
    code_str: str,
    task: Dict[str, Any],
    timeout_seconds: float = 3.0,
) -> Tuple[bool, str, Optional[List[List[List[int]]]]]:
    """
    Tests candidate code containing transform(input_grid) against all train pairs in task.
    Returns:
    - passed (bool): True if transform() perfectly reproduces all train outputs.
    - message (str): 'PASS' or detailed failure report with diffs.
    - test_predictions: List of 2D grid outputs for each test input (if passed), else None.
    """
    ns = create_execution_namespace()

    # 1. Compile and execute candidate code
    try:
        exec(code_str, ns)
    except Exception:
        err = traceback.format_exc(limit=2)
        return False, f"Syntax/Compilation Error:\n{err}", None

    if "transform" not in ns or not callable(ns["transform"]):
        return False, "Error: Code must define a callable function 'transform(input_grid)'.", None

    transform_fn = ns["transform"]

    # 2. Verify across all training pairs
    for idx, pair in enumerate(task.get("train", [])):
        inp = pair["input"]
        expected = pair["output"]

        try:
            t0 = time.time()
            got = transform_fn(inp)
            elapsed = time.time() - t0

            if elapsed > timeout_seconds:
                return False, f"Timeout Error: Execution took {elapsed:.2f}s on train pair {idx}", None

        except Exception:
            err = traceback.format_exc(limit=3)
            return False, f"Runtime Exception on train pair {idx}:\n{err}", None

        # Verify equivalence
        if not grids_equal(got, expected):
            diff = compute_grid_diff(got, expected)
            if not diff.get("shape_match", True):
                report = (
                    f"FAILED on train pair {idx}: Shape mismatch! "
                    f"Expected shape {diff['expected_shape']}, got {diff['got_shape']}"
                )
            else:
                report = (
                    f"FAILED on train pair {idx}: Pixel mismatch! "
                    f"{diff['pixel_mismatches']}/{diff['total_pixels']} pixels incorrect "
                    f"({diff['accuracy']:.1%} accuracy)"
                )
            return False, report, None

    # 3. Passed all train pairs! Run on test pairs
    test_preds = []
    for t_idx, test_item in enumerate(task.get("test", [])):
        test_inp = test_item["input"]
        try:
            pred = transform_fn(test_inp)
            pred_arr = np.array(pred, dtype=np.int32)
            test_preds.append(pred_arr.tolist())
        except Exception:
            err = traceback.format_exc(limit=2)
            return False, f"Passed train pairs, but crashed on test input {t_idx}:\n{err}", None

    return True, "PASS", test_preds
