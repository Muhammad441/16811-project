import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage.morphology import distance_transform_edt
import pdb

class CostMap:
    def __init__(self, x_len = 500, y_len = 500):
        # Assuming resolution = 1
        self.x_len = x_len
        self.y_len = y_len

        self.map = np.ones((x_len, y_len))
        self.threshold = 50
    
    def addObstacle(self, cx, cy, lx, ly):
        min_x = max(int(cx - lx/2), 0)
        max_x = min(int(cx + lx/2), self.x_len - 1)

        min_y = max(int(cy - ly/2), 0)
        max_y = min(int(cy + ly/2), self.y_len - 1)

        self.map[min_x:max_x, min_y:max_y] = 0

    def computeCost(self):
        distance_field = distance_transform_edt(self.map)
        high_value_flags = distance_field > self.threshold 
        self.cost_map = self.threshold - distance_field
        self.cost_map[high_value_flags] = 0

    def visualize(self, map):
        plt.imshow(map)
        plt.show()

class Map1(CostMap):
    def __init__(self):
        CostMap.__init__(self)
        self.addObstacle(cx = 75, cy = 300, lx = 150, ly = 15)
        self.addObstacle(cx = 245, cy = 450, lx = 15, ly = 200)
        self.computeCost()



def main():
    cost_map = Map1()
    cost_map.visualize(cost_map.map)
    # cost_map = CostMap()
    # cost_map.addObstacle(cx = 20, cy = 30, lx = 100, ly = 30)
    # cost_map.distanceField()
    # cost_map.visualize(cost_map.map)

if __name__ == "__main__":
    main()