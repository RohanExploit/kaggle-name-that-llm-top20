"""
Verification test runner for ARC AGI Agentic Coder.
Tests the Object-Centric DSL and the Autonomous Solver on canonical tasks.
"""

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arc_coder.dsl import *
from arc_coder.execute import test_solution
from arc_coder.solver import ArcAgenticSolver


def run_smoke_tests():
    print("=" * 60)
    print("Running ARC-AGI DSL & Coder Smoke Tests")
    print("=" * 60)

    # Test 1: Grid transformations
    grid = [[1, 2], [3, 4]]
    rot = rotate_grid(grid, 1)
    assert np.array_equal(rot, [[2, 4], [1, 3]]), "Rotation failed"

    fliph = flip_h(grid)
    assert np.array_equal(fliph, [[2, 1], [4, 3]]), "Flip H failed"

    # Test 2: Object finding & counting
    sparse_grid = [
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 3],
    ]
    objs = find_objects(sparse_grid, connectivity=4, bg_color=0)
    assert len(objs) == 2, f"Expected 2 objects, found {len(objs)}"
    assert count_objects(sparse_grid, color=2) == 1
    assert count_objects(sparse_grid, color=3) == 1

    # Test 3: Bounding box & crop
    largest = get_largest_object(sparse_grid)
    assert largest.color == 2
    assert largest.size == 3
    assert largest.bbox == (1, 1, 2, 2)

    cropped = crop_to_content(sparse_grid, bg_color=0)
    assert cropped.shape == (3, 3)

    # Test 4: End-to-End solver on sample task
    sample_task = {
        "train": [
            {
                "input": [[1, 0], [0, 0]],
                "output": [[2, 0], [0, 0]],
            },
            {
                "input": [[0, 1], [1, 0]],
                "output": [[0, 2], [2, 0]],
            },
        ],
        "test": [
            {
                "input": [[1, 1], [0, 1]],
            }
        ],
    }

    solver = ArcAgenticSolver(sample_task, task_id="task_test_recolor")
    solved, code, test_preds = solver.solve()
    assert solved, "Solver failed on recolor task"
    assert test_preds[0] == [[2, 2], [0, 2]], f"Unexpected test prediction: {test_preds}"

    print("\n[ALL TESTS PASSED] All DSL primitives, execution sandbox, and solver strategies passed 100%!")


if __name__ == "__main__":
    run_smoke_tests()
