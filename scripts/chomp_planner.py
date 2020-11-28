import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from manipulator import Manipulator
from costmap import Map1
import copy 
import pdb
    
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
                    # if(int(links_pts_y[j][pt]) >= self.map.y_len):
                    #     grad_y = -1
                    # elif(int(links_pts_y[j][pt]) < 0):
                    #     grad_y = 1
                    # else:
                    # pdb.set_trace() 
                    grad_y = self.gy[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]
                    
                    grad_x = self.gx[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]

                    # print(links_pts_x[j][pt], links_pts_y[j][pt], grad_x, grad_y)

                    if(i==j):
                        gradient[i] += lengths_pts[i][pt]*(-np.sin(state[i]))*grad_x \
                                  + lengths_pts[i][pt]*(np.cos(state[i]))*grad_y
                    else:
                        gradient[i] += lengths_pts[i][-1]*(-np.sin(state[i]))*grad_x + \
                                    lengths_pts[i][-1]*(np.cos(state[i]))*grad_y
        return gradient/(len(links_pts_x)*len(links_pts_x[0]))

    def gradientOpt(self, num_opt_steps = 1000, stepsize = 0.002, threshold = 0.01):
        self.gy, self.gx = np.gradient(self.map.cost_map)
        for opt_step in range(num_opt_steps):
            print("Iteration : {}".format(opt_step))

            flag = False
            gradients = np.zeros((self.num_waypoints, self.manipulator.num_links))
            for waypoint in range(1, self.num_waypoints - 1):
                obstacle_gradient = self.obstacleGradient(self.traj[waypoint])

                smoothnessGradient = np.zeros(self.traj[waypoint].shape)
                if(waypoint > 0):
                    smoothnessGradient += self.traj[waypoint] - self.traj[waypoint - 1] 
                
                if(waypoint < self.num_waypoints - 1):
                    smoothnessGradient += -self.traj[waypoint+1] + self.traj[waypoint] 
                
                gradients[waypoint] = 100*smoothnessGradient + obstacle_gradient
            self.traj = self.traj - stepsize*gradients
            if(np.max(gradients) > threshold):
                flag = True
            if(flag == False):
                break

def main():
    manipulator = Manipulator(base_position = np.array((100, 200)))
    map = Map1()
    vis = Visualizer(map.map)
    start = np.array((0.1,0,0,0))
    goal = np.array((0.1,1.7,1.7,1.5))

    planner = ChompPlanner(start = start, goal = goal, num_waypoints = 30, 
                 map = map, manipulator = manipulator)
    

    planner.gradientOpt(num_opt_steps=600, threshold = 1)

    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.traj[i]))

    vis.traj_vis(path)
    plt.show()

if __name__ == "__main__":
    main()