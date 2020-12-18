import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from manipulator import Manipulator
from planner import Planner
import costmap
import copy 
import pdb
    
class ChompPlanner(Planner):
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

    def interpolatePath(self, start, goal, num_waypoints):
        traj = np.zeros((num_waypoints, self.manipulator.num_links))
        traj[0] = start
        traj[-1] = goal
        for i in range(1, num_waypoints):
            traj[i] = traj[0] + (goal - start)*i/num_waypoints
        return traj        

    def obstacleGradient(self, state):
        links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)
        assert(len(links_pts_x) == len(links_pts_y))
        assert(len(links_pts_x[0]) == len(links_pts_y[0]))
        
        gradient = np.zeros(state.shape)

        for i in range(self.manipulator.num_links):
            for j in range(i, self.manipulator.num_links):
                for pt in range(len(links_pts_x[j])):
                    grad_y = self.gy[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]
                    
                    grad_x = self.gx[int(links_pts_y[j][pt])][int(links_pts_x[j][pt])]

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
            # print("Iteration : {}".format(opt_step))

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

def Map1(reverse_flag = False):
    manipulator = Manipulator(base_position = np.array((100, 200)))
    map = costmap.Map1()
    vis = Visualizer(map.map)
    if(not reverse_flag):
        goal = np.array((0.1,0,0,0))
        start = np.array((0.1,1.7,1.7,1.5))
    else:
        start = np.array((0.1,0,0,0))
        goal = np.array((0.1,1.7,1.7,1.5))


    planner = ChompPlanner(start = start, goal = goal, num_waypoints = 30, 
                 map = map, manipulator = manipulator)
    

    planner.gradientOpt(num_opt_steps=600, threshold = 1) #600

    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.traj[i]))
    vis.traj_vis(path)
    plt.show()

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)
    return planner.traj, obstacle_cost, smoothness_cost

def Map2(reverse_flag = False):
    manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                              base_position = np.array((100, 200)))
    map = costmap.Map3()
    vis = Visualizer(map.map)
    if(not reverse_flag):
        start = np.array((0.1,1.7,1.7,1.5,3.14))
        goal = np.array((0.1,0.1,1.4,1.4, 1.4))
    else:
        start = np.array((0.1,0.1,1.4,1.4, 1.4))
        goal = np.array((0.1,1.7,1.7,1.5,3.14))

    planner = ChompPlanner(start = start, goal = goal, num_waypoints = 30, 
                 map = map, manipulator = manipulator)
    

    planner.gradientOpt(num_opt_steps=1000, threshold = 1) #1000

    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.traj[i]))
    vis.traj_vis(path)
    plt.show()
    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)

    return planner.traj, obstacle_cost, smoothness_cost

def Map3(reverse_flag = False):
    manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
                              base_position = np.array((250, 200)))
    map = costmap.Map4()
    vis = Visualizer(map.map)
    if(not reverse_flag):
        start = np.array((0,1.56,1.2,0,0))
        goal = np.array((2.0,2.0,1.56,1.56, 1.56))
    else:
        start = np.array((2.0,2.0,1.56,1.56, 1.56))
        goal = np.array((0,1.56,1.2,0,0))

    planner = ChompPlanner(start = start, goal = goal, num_waypoints = 30, 
                 map = map, manipulator = manipulator)
    

    planner.gradientOpt(num_opt_steps=9000, threshold = 1) #5000

    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.traj[i]))
    vis.traj_vis(path)
    plt.show()

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)
    return planner.traj, obstacle_cost, smoothness_cost
    
def main():
    map1_cost = []
    map2_cost = []
    map3_cost = []
    for i in range(20):
        print("i ", i)
        map1_traj, map1_obstacle_cost, map1_smoothness_cost = Map1(i>=10)
        map1_cost.append([map1_obstacle_cost, map1_smoothness_cost])
        np.save('../data/chomp/chomp_map1_' + str(i) + '.npy', map1_traj)

        with open('../data/chomp/chomp_map1.csv', mode='a') as file_:
            file_.write("{},{}".format(map1_obstacle_cost, map1_smoothness_cost))
            file_.write("\n")

        map2_traj, map2_obstacle_cost, map2_smoothness_cost = Map2(i>=10)
        map2_cost.append([map2_obstacle_cost, map2_smoothness_cost])
        np.save('../data/chomp/chomp_map2_' + str(i) + '.npy', map2_traj)

        with open('../data/chomp/chomp_map2.csv', mode='a') as file_:
            file_.write("{},{}".format(map2_obstacle_cost, map2_smoothness_cost))
            file_.write("\n")

        map3_traj, map3_obstacle_cost, map3_smoothness_cost = Map3(i>=10)
        map3_cost.append([map3_obstacle_cost, map3_smoothness_cost])
        np.save('../data/chomp/chomp_map3_' + str(i) + '.npy', map3_traj)

        with open('../data/chomp/chomp_map3.csv', mode='a') as file_:
            file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
            file_.write("\n")

    np.savetxt("../data/chomp/chomp_map1.csv", np.array(map1_cost), delimiter=",")
    np.savetxt("../data/chomp/chomp_map2.csv", np.array(map2_cost), delimiter=",")
    np.savetxt("../data/chomp/chomp_map3.csv", np.array(map3_cost), delimiter=",")

if __name__ == "__main__":
    main()