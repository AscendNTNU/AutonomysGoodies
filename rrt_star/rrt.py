import random
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

class PathNode:
    """
    RRT Path Node
    """
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.parent = None
        self.children = []

    def dist_to_node(self, node):
        return np.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)

    # @staticmethod
    # def dist_to_node(node1, node2):
    #     return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

class AreaRect:
    def __init__(self, area: Tuple[float, float, float, float]) -> None:
        self.xmin = float(area[0])
        self.xmax = float(area[1])
        self.ymin = float(area[2])
        self.ymax = float(area[3])

class RRT:
    """
    Main class for RRT path planning
    """

    def __init__(self, start, goal, 
                       obstacles, sample_area, 
                       segment_length = 1,
                       goal_sample_rate = 5, 
                       max_iterations = 5000, 
                       robot_clearance = 1):
        """
        start: [x, y],
        goal:  [x, y],
        obstacles:   [[x, y, radius], ...]
        sample_area: [xmin, xmax, ymin, ymax] 
        """

        self.start = PathNode(start[0], start[1])
        self.goal  = PathNode(goal[0], goal[1])
        self.sample_area = sample_area
        self.segment_length = segment_length
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.obstacles = obstacles
        self.node_list = []
        self.robot_clearance = robot_clearance


    def planning(self):
        """
        Main RRT path planning algorighm
        """

        self.node_list = [self.start]
        for i in range(self.max_iterations):
            if i % 100 == 0:
                print(f"Iteration: {i}")

            random_node = self.get_random_node()
            nearest_node_index = self.get_nearest_node_index(self.node_list, random_node)
            nearest_node = self.node_list[nearest_node_index]

            new_node = self.steer(nearest_node, random_node, self.segment_length)
            if not self.check_collision(nearest_node, new_node, self.obstacles, self.robot_clearance):
                self.node_list.append(new_node)
            else:
                continue

            if i % 5 == 0:
                self.draw_graph(random_node)

            # Check if we are close to the goal
            if self.goal.dist_to_node(self.node_list[-1]) <= self.segment_length:
                final_node = self.steer(self.node_list[-1], self.goal, self.segment_length)
                if not self.check_collision(self.node_list[-1], final_node, self.obstacles, self.robot_clearance):
                    self.node_list.append(final_node)
                    return self.generate_path(self.node_list, len(self.node_list) - 1)
        
        return None

    def steer(self, from_node, to_node, segment_length=1):
        if from_node.x == to_node.x and from_node.y == to_node.y:
            return PathNode(from_node.x, from_node.y)
        heading = np.array([to_node.x - from_node.x, to_node.y - from_node.y])
        length = np.linalg.norm(heading)

        if length > segment_length:
            length = segment_length

        [new_x, new_y] = heading / np.linalg.norm(heading) * length + np.array([from_node.x, from_node.y])
        
        new_node = PathNode(new_x, new_y)
        new_node.parent = from_node
        from_node.children.append([new_node, length])
        return new_node
    
    def generate_path(self, node_list, goal_index):
        # Iterate over the linked list
        nodes = [node_list[goal_index]]
        node = node_list[goal_index]
        while node.parent:
            node = node.parent
            nodes.append(node)
        return nodes

    def get_random_node(self):
        # To make the algorithm more informed, the random sampling can be pushed 
        # in the direction of the goal, i.e. setting the random point to the goal
        if random.randint(0, 100) > self.goal_sample_rate:
            node = PathNode(
                random.uniform(self.sample_area[0], self.sample_area[1]),
                random.uniform(self.sample_area[2], self.sample_area[3])
            )
        else:
            node = PathNode(self.goal.x, self.goal.y)
        return node 
    
    @staticmethod
    def get_nearest_node_index(node_list, random_node):
        # No need to calculate the acctual distance, just distance²
        index = -1
        min_val = 1 << 32
        for i, node in enumerate(node_list):
            dist = (node.x - random_node.x)**2 + (node.y - random_node.y)**2
            if dist < min_val:
                min_val = dist
                index = i
        return index
    
    @staticmethod
    def get_dist_to_nearest_node(node_list, node):
        min_val = 1 << 32
        for _node in node_list:
            dist = (node.x - _node.x)**2 + (node.z - _node.y)**2
            if dist < min_val:
                min_val = dist
        return min_val
    
    @staticmethod
    def check_collision(from_node, to_node, obstacles, clearance):
        """
        Checks for collision between spheares and and line, given some clearance:
            1. Checks each of the start and end points with each obstacle
            2. Finds the closest point on the line from start to end to each obstacle center,
               and checks if this point is to close or not
        """

        from_node = np.array([from_node.x, from_node.y])
        to_node = np.array([to_node.x, to_node.y])
        heading = to_node - from_node

        length = np.linalg.norm(heading)
        if length:
            heading /= length

        for [x, y, radius] in obstacles:
            obstacle_xy = np.array([x, y])
            if np.linalg.norm(obstacle_xy - from_node) <= radius + clearance:
                return True
            elif np.linalg.norm(obstacle_xy - to_node) <= radius + clearance:
                return True

            # Project x,y onto the line (start, end)
            lhs = from_node - obstacle_xy
            dot_product = np.dot(lhs, heading)
            dot_product = max(0.0, dot_product)
            dot_product = min(dot_product, length)
            projected_point = from_node + heading * dot_product

            if np.linalg.norm(projected_point - obstacle_xy) <= radius + clearance:
                return True
        return False
    
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * np.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * np.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)
    
    def draw_graph(self, node=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        if node is not None:
            plt.plot(node.x, node.y, "^k")
            if self.robot_clearance > 0.0:
                self.plot_circle(node.x, node.y, self.robot_clearance, '-r')
        
        for node in self.node_list:
            if node.parent:
                [path_x, path_y] = [[node.parent.x, node.x], [node.parent.y, node.y]]
                plt.plot(path_x, path_y, "-g")

        for (ox, oy, size) in self.obstacles:
            self.plot_circle(ox, oy, size, '-r')

        plt.plot([self.sample_area[0], self.sample_area[1], self.sample_area[1], self.sample_area[0], self.sample_area[0]],
                 [self.sample_area[2], self.sample_area[2], self.sample_area[3], self.sample_area[3], self.sample_area[2]], "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis("equal")
        plt.axis(self.sample_area)
        plt.grid(True)
        plt.pause(0.01)


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        sample_area=[-2, 12, 0, 16],
        obstacles=obstacleList,
        robot_clearance=0.8
    )
    
    path = rrt.planning()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        rrt.draw_graph()
        plt.plot([node.x for node in path], [node.y for node in path], '-r')
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()


if __name__ == '__main__':
    main()