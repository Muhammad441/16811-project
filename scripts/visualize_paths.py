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
from ga_planner import GAPlanner

def interpolatePath(start, goal, num_waypoints, num_links):
    traj = np.zeros((num_waypoints, num_links))
    traj[0] = start
    traj[-1] = goal
    for i in range(1, num_waypoints):
        traj[i] = traj[0] + (goal - start)*i/num_waypoints
    return traj

def Map1():
    # manipulator = Manipulator(base_position = np.array((100, 200)))
    # map = costmap.Map1()
    # vis = Visualizer(map.map)
    # traj = np.load('../data/ga/ga_map1_0.npy')
    # path = []
    # for point in traj:
    #     path.append(manipulator.ForwardKinematics(point))
    
    # vis.traj_vis(path)
    start = np.array((0.1,0,0,0))
    goal = np.array((0.1,1.7,1.7,1.5))

    for i in range(3, 18):
        manipulator = Manipulator(base_position = np.array((100, 200)))
        map = costmap.Map1()
        vis = Visualizer(map.map)
        traj = np.load('../data/sim_anneal/sim_anneal_map1_' + str(i)+ '.npy')

        planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                map = map, manipulator = manipulator)
        # print(planner.trajectoryCost(traj))
        map3_obstacle_cost, map3_smoothness_cost = planner.trajectoryCost(traj)
        print(map3_obstacle_cost, map3_smoothness_cost)

        path = []
        for i in range(len(traj)):
            # if(i%2 != 0):
            #     continue
            state = traj[i]
            path.append(manipulator.ForwardKinematics(state))
        vis.traj_vis(path)
        # vis.stationary_traj_vis(path)
        # with open('../data/ga/ga_map1.csv', mode='a') as file_:
        #     file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
        #     file_.write("\n")

def Map2():
    # manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
    #                           base_position = np.array((100, 200)))
    # map = costmap.Map3()
    # vis = Visualizer(map.map)
    # traj = np.load('../data/sim_anneal/sim_anneal_map2_0.npy')
    # path = []
    # for point in traj:
    #     path.append(manipulator.ForwardKinematics(point))
    
    # vis.traj_vis(path)

    for i in range(1, 18):
        manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                                base_position = np.array((100, 200)))
        map = costmap.Map3()
        vis = Visualizer(map.map)
        start = np.array((2.0,2.0,1.56,1.56, 1.56))
        goal = np.array((0,1.56,1.2,0,0))
        traj = np.load('../data/ga/final_results/ga_map2_' + str(i)+ '.npy')
        planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                map = map, manipulator = manipulator)
        # print(planner.trajectoryCost(traj))
        map3_obstacle_cost, map3_smoothness_cost = planner.trajectoryCost(traj)
        print(map3_obstacle_cost, map3_smoothness_cost)

        path = []
        for i in range(traj.shape[0] - 1):
            states = interpolatePath(traj[i], traj[i+1], num_links = 5, num_waypoints = 10)
            for state in states:
                path.append(manipulator.ForwardKinematics(state))
        vis.traj_vis(path)        
        # with open('../data/ga/ga_map2.csv', mode='a') as file_:
        #     file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
        #     file_.write("\n")



def Map3():

    for i in range(11, 19):
        manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                                base_position = np.array((250, 200)))
        map = costmap.Map4()
        vis = Visualizer(map.map)
        traj = np.load('../data/sim_anneal/sim_anneal_map3_' + str(i)+ '.npy')
        start = np.array((2.0,2.0,1.56,1.56, 1.56))
        goal = np.array((0,1.56,1.2,0,0))
        planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                map = map, manipulator = manipulator)
        # print(planner.trajectoryCost(traj))
        map3_obstacle_cost, map3_smoothness_cost = planner.trajectoryCost(traj)
        print(map3_obstacle_cost, map3_smoothness_cost)

        path = []
        for i in range(traj.shape[0] - 1):
            states = interpolatePath(traj[i], traj[i+1], num_links = 5, num_waypoints = 10)
            for state in states:
                path.append(manipulator.ForwardKinematics(state))
        # for i in range(traj.shape[0]):
        #     # states = interpolatePath(traj[i], traj[i+1], num_links = 5, num_waypoints = 10)
        #     path.append(manipulator.ForwardKinematics(traj[i]))
        vis.traj_vis(path)        

        # with open('../data/ga/ga_map3.csv', mode='a') as file_:
        #     file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
        #     file_.write("\n")

        # path = []
        # for i in range(traj.shape[0] - 1):
        #     states = interpolatePath(traj[i], traj[i+1], num_links = 5, num_waypoints = 10)
        #     for state in states:
        #         path.append(manipulator.ForwardKinematics(state))
    # for point in traj:
    #     path.append(manipulator.ForwardKinematics(point))
    
    vis.traj_vis(path)

def main():
    Map3()
if __name__ == "__main__":
    main()