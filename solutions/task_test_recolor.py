# Auto-synthesized solution for task_test_recolor
import numpy as np
from arc_coder.dsl import *

def transform(g):
    mapping = {0: 0, 1: 2}
    return color_map(g, mapping).tolist()

