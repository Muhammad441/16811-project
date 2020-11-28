import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from costmap import Map1
import copy 
import pdb

class Manipulator:
    def __init__(self, num_links = 4, link_lengths = np.ones(4)*75, base_position = np.array((200,400))):
        self.num_links = num_links
        self.link_lengths = link_lengths
        self.base_position = base_position
        self.collision_resolution = 2

    def ForwardKinematics(self, state):
        assert(state.shape[0] == self.num_links)
        
        x_c_p = self.base_position[0]
        y_c_p = self.base_position[1]
        x_c = [x_c_p]
        y_c = [y_c_p]
        for i in range(state.shape[0]):
            x_p = self.link_lengths[i]*np.cos(state[i])
            y_p = self.link_lengths[i]*np.sin(state[i])
            x_i = x_c_p + x_p 
            y_i = y_c_p + y_p
            x_c.append(x_i)
            y_c.append(y_i)
            x_c_p = x_i
            y_c_p = y_i

        return [x_c, y_c]
    
    def linkPts(self, state):
        links_pts_x = []
        links_pts_y = []
        lengths_pts = []

        x_c_p = self.base_position[0]
        y_c_p = self.base_position[1]

        for arm_idx in range(self.num_links):
            length = 0
            link_pts_x = []
            link_pts_y = []
            lengths = []
            while length < self.link_lengths[arm_idx]:
                x_p = length*np.cos(state[arm_idx])
                y_p = length*np.sin(state[arm_idx])
                link_pts_x.append(x_c_p + x_p) 
                link_pts_y.append(y_c_p + y_p) 
                lengths.append(length) 
                length += self.collision_resolution
            x_c_p = link_pts_x[-1]
            y_c_p = link_pts_y[-1]
            links_pts_x.append(link_pts_x)
            links_pts_y.append(link_pts_y)
            lengths_pts.append(lengths)
        
        return links_pts_x, links_pts_y, lengths_pts
    
class ChompPlanner:
    def __init__(self, start, goal, num_waypoints, map, manipulator):
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints
        self.map = map
        self.manipulator = manipulator

        self.traj = self.seedPath()

    def seedPath(self):
        traj = np.zeros((self.num_waypoints, self.manipulator.num_links))
        traj[0] = self.start
        traj[-1] = self.goal
        for i in range(1, self.num_waypoints):
            traj[i] = traj[0] + (self.goal - self.start)*i/self.num_waypoints
        return traj

    def obstacleGradient(self, state):
        links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)
        assert(len(links_pts_x) == len(links_pts_y))
        assert(len(links_pts_x[0]) == len(links_pts_y[0]))
        
        gradient = np.zeros(state.shape)

        for i in range(self.manipulator.num_links):
            for j in range(i, self.manipulator.num_links):
                for pt in range(len(links_pts_x[j])):
                    grad_x = self.gx[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]
                    grad_y = self.gy[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]

                    # print(links_pts_x[j][pt], links_pts_y[j][pt], grad_x, grad_y)

                    if(i==j):
                        gradient[i] += lengths_pts[i][pt]*(-np.sin(state[i]))*grad_x \
                                  + lengths_pts[i][pt]*(np.cos(state[i]))*grad_y
                    else:
                        gradient[i] += lengths_pts[i][-1]*(-np.sin(state[i]))*grad_x + \
                                    lengths_pts[i][-1]*(np.cos(state[i]))*grad_y
        return gradient/(len(links_pts_x)*len(links_pts_x[0]))

    def gradientOpt(self, num_opt_steps = 30, stepsize = 0.01):
        self.gx, self.gy = np.gradient(self.map.cost_map)
        for opt_step in range(num_opt_steps):
            gradients = np.zeros((self.num_waypoints, self.manipulator.num_links))
            for waypoint in range(1, self.num_waypoints - 1):
                obstacle_gradient = self.obstacleGradient(self.traj[waypoint])

                smoothnessGradient = np.zeros(self.traj[waypoint].shape)
                if(waypoint > 0):
                    smoothnessGradient += self.traj[waypoint] - self.traj[waypoint - 1] 
                
                if(waypoint < self.num_waypoints - 1):
                    smoothnessGradient += self.traj[waypoint+1] - self.traj[waypoint] 
                gradients[waypoint] = obstacle_gradient 

            self.traj = self.traj - stepsize*gradients

def main():
    manipulator = Manipulator(base_position = np.array((150,250)))
    map = Map1()
    vis = Visualizer(map.map)
    start = np.array((0.1,0.2,0.1,0.1))
    goal = np.array((0.1,0.2,-1.5,0.1))

    planner = ChompPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)
    

    planner.gradientOpt()

    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.traj[i]))

    vis.traj_vis(path)

if __name__ == "__main__":
    main()