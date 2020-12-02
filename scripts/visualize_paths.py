import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from manipulator import Manipulator
from costmap import Map1, Map2
import costmap
import copy 
import pdb
import random 
from numpy import linalg as LA
from scipy import optimize
from sympy import Polygon
from scipy.ndimage.morphology import binary_fill_holes as imfill
from PIL import Image, ImageDraw
from planner import Planner

def interpolatePath(start, goal, num_waypoints, num_links):
    traj = np.zeros((num_waypoints, num_links))
    traj[0] = start
    traj[-1] = goal
    for i in range(1, num_waypoints):
        traj[i] = traj[0] + (goal - start)*i/num_waypoints
    return traj

def Map1():
    manipulator = Manipulator(base_position = np.array((100, 200)))
    map = costmap.Map1()
    vis = Visualizer(map.map)
    traj = np.load('../data/ga/ga_map1_0.npy')
    path = []
    for point in traj:
        path.append(manipulator.ForwardKinematics(point))
    
    vis.traj_vis(path)

def Map2():
    manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                              base_position = np.array((100, 200)))
    map = costmap.Map3()
    vis = Visualizer(map.map)
    traj = np.load('../data/sim_anneal/sim_anneal_map2_0.npy')
    path = []
    for point in traj:
        path.append(manipulator.ForwardKinematics(point))
    
    vis.traj_vis(path)

def Map3():
    manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                              base_position = np.array((250, 200)))
    map = costmap.Map4()
    vis = Visualizer(map.map)
    traj = np.load('../data/ga/final_results/ga_map3_2.npy')
    path = []
    for i in range(traj.shape[0] - 1):
        states = interpolatePath(traj[i], traj[i+1], num_links = 5, num_waypoints = 10)
        for state in states:
            path.append(manipulator.ForwardKinematics(state))
    # for point in traj:
    #     path.append(manipulator.ForwardKinematics(point))
    
    vis.traj_vis(path)

def main():
    Map3()
if __name__ == "__main__":
    main()