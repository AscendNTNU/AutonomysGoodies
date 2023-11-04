import random
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from rrt import PathNode, RRT

class PathNode(PathNode):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
        self.cost = 1 << 32

class RRTStar(RRT):
    def __init__(self, start, goal, obstacles, sample_area, segment_length=1, goal_sample_rate=5, max_iterations=5000, robot_clearance=1):
        super().__init__(start, goal, obstacles, sample_area, segment_length, goal_sample_rate, max_iterations, robot_clearance)