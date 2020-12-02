import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from manipulator import Manipulator
import costmap
import copy 
import pdb

class Planner:
    def trajectoryCost(self, traj):
        obstacle_cost = 0
        smoothness_cost = 0
        for state in traj:
            links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)
            for i in range(len(links_pts_x)):
                for j in range(len(links_pts_x[i])):
                    obstacle_cost = obstacle_cost + self.map.cost_map[int(links_pts_y[i][j])][int(links_pts_x[i][j])]

        for i in range(1, len(traj)):
            smoothness_cost += np.sum(np.abs(traj[i] - traj[i-1]))
        
        return obstacle_cost, smoothness_cost