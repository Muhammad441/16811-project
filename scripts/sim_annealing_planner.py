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

def autoTrajCost(z):
    traj = z.reshape(30, -1)
    cost = 0
    for i in range(len(traj)):
        # if(i > 0):
        #     cost += np.sum(np.abs(traj[i] - traj[i-1]))
        if(i < len(traj) - 1):
            cost += np.sum(np.abs(traj[i+1] - traj[i]))
    return cost

class SimAnnealingPlanner(Planner):
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

    def trajObstacleCost(self, traj):
        cost = 0
        for state in traj:
            links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)
            for i in range(len(links_pts_x)):
                for j in range(len(links_pts_x[i])):
                    # pdb.set_trace()
                    cost = cost + self.map.cost_map[int(links_pts_y[i][j])][int(links_pts_x[i][j])]
        return cost 

    def trajSmoothCost(self, traj):
        cost = 0
        for i in range(len(traj)):
            if(i > 0):
                cost += np.sum(np.abs(traj[i] - traj[i-1]))
            if(i < len(traj) - 1):
                cost += np.sum(np.abs(traj[i+1] - traj[i]))
        # print(cost)
        return cost
    
    def trajCost(self, traj):
        # return self.trajObstacleCost(traj) + 300*self.trajSmoothCost(traj)
        # return self.trajSmoothCost(traj) + 200*self.trajObstacleCost(traj)
        return self.trajUnfitScore(traj)
    
    def norm(self, state_1, state_2):
        return LA.norm(abs(state_1 - state_2)%(2*np.pi))

    def getNeighbor(self, traj):
        max_change = 0.2
        threshold = 0.1
        neighbor = copy.deepcopy(traj)

        for count in range(5):
            idx = random.randint(1, traj.shape[0] - 2)
            flag = True
            while flag:
                new_state = (np.random.rand(neighbor[idx].shape[0])*2 - 1)*max_change + neighbor[idx]
                if(self.norm(new_state, neighbor[idx]) < threshold):
                    neighbor[idx] = copy.deepcopy(new_state)
                    flag = False

        return neighbor


    def optimize(self, num_opt_steps = 10, initial_temperature = 10000):
        traj_curr = copy.deepcopy(self.traj)
        traj_best = copy.deepcopy(self.traj)
        curr_cost = self.trajCost(traj_curr)
        best_cost = self.trajCost(traj_best)

        temp_curr = initial_temperature

        for opt_step in range(num_opt_steps):
            print("Opt step : {}, best_score : {}".format(opt_step, best_cost))
            traj_neighbor = self.getNeighbor(traj_curr)
            neighbor_cost = self.trajCost(traj_neighbor)
            temp_curr *= 0.99
            if(neighbor_cost < curr_cost):
                traj_curr = copy.deepcopy(traj_neighbor)
                curr_cost = neighbor_cost
                if(neighbor_cost < best_cost):
                    traj_best = copy.deepcopy(traj_neighbor)
                    best_cost = neighbor_cost
            else:
                if (np.exp((curr_cost - neighbor_cost)/temp_curr) > random.random()):
                        traj_curr = copy.deepcopy(traj_neighbor)
                        curr_cost = neighbor_cost
            if(best_cost <= 30):
                break
                    
        self.traj = traj_best

    def trajObstacleCost(self, traj):
        cost = 0
        for state in traj:
            links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)
            for i in range(len(links_pts_x)):
                for j in range(len(links_pts_x[i])):
                    cost = cost + self.map.cost_map[int(links_pts_y[i][j])][int(links_pts_x[i][j])]
        return cost 

    def autoTrajCost(self, z):
        traj_temp = copy.deepcopy(self.traj)
        traj_temp[1:-1] = z.reshape(13, -1)
        cost = self.trajSmoothCost(traj_temp) + 20*self.trajObstacleCost(traj_temp)
        return cost
        

    def autoOptimize(self):
        x0 = self.traj[1:-1].reshape(-1)
        print(x0.shape)
        minimizer_kwargs = {"method": "BFGS"}
        res = optimize.basinhopping(self.autoTrajCost, x0, minimizer_kwargs = {"method": "BFGS"},
        niter = 15)
        self.traj[1:-1] = res.x.reshape(13, -1)
        pdb.set_trace()


    def trajUnfitScore(self,traj):
        trajCost = 0.0

        ## Compute cost for polygon area
        img = Image.new('L', (self.map.cost_map.shape[1], self.map.cost_map.shape[0]), 0)
        mask=None
        smoothingCost = 0
        for i in range(self.num_waypoints-1):
            state1 = traj[i,:]
            state2 = traj[i+1,:]
            fine_traj = self.interpolatePath(state1,state2,10)
            for k in range(fine_traj.shape[0]-1):
                fine_state1 = fine_traj[k,:]
                fine_state2 = fine_traj[k+1,:]
                links_pts_x1, links_pts_y1, lengths_pts1 = self.manipulator.linkPts(fine_state1)
                links_pts_x2, links_pts_y2, lengths_pts2 = self.manipulator.linkPts(fine_state2)
                poly_c = []
                for j in range(len(links_pts_x1)):
                    for pt in range(len(links_pts_x1[j])):
                        poly_c.append((links_pts_x1[j][pt],links_pts_y1[j][pt]))
                for j in range(len(links_pts_x2)):
                    for pt in range(len(links_pts_x2[j])):
                        poly_c.append((links_pts_x2[-(1+j)][pt],links_pts_y2[-(1+j)][pt]))
                ImageDraw.Draw(img).polygon(poly_c, outline=1, fill=1)
                img_ = Image.new('L', (self.map.cost_map.shape[1], self.map.cost_map.shape[0]), 0)
                ImageDraw.Draw(img_).polygon(poly_c, outline=1, fill=1)
                smoothingCost += np.sum(np.array(img_))
        mask = np.array(img)
        filled = imfill(mask)
        area = self.map.cost_map[filled].sum()
        self.beta = 1
        trajCost += self.beta*area + 0.1*smoothingCost

        return trajCost

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

    planner = SimAnnealingPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)    

    planner.optimize(num_opt_steps=1000)

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)

    return planner.traj, obstacle_cost, smoothness_cost

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


    planner = SimAnnealingPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)
    

    planner.optimize(num_opt_steps=1500)

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

    planner = SimAnnealingPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)
    planner.optimize(num_opt_steps=5000)

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)
    return planner.traj, obstacle_cost, smoothness_cost

def main():
    map1_cost = []
    map2_cost = []
    map3_cost = []
    for i in range(1, 20):
        map1_traj, map1_obstacle_cost, map1_smoothness_cost = Map1(i>=10)
        map1_cost.append([map1_obstacle_cost, map1_smoothness_cost])
        np.save('../data/sim_anneal/sim_anneal_map1_' + str(i) + '.npy', map1_traj)

        with open('../data/sim_anneal/sim_anneal_map1.csv', mode='a') as file_:
            file_.write("{},{}".format(map1_obstacle_cost, map1_smoothness_cost))
            file_.write("\n")


        map2_traj, map2_obstacle_cost, map2_smoothness_cost = Map2(i>=10)
        map2_cost.append([map2_obstacle_cost, map2_smoothness_cost])
        np.save('../data/sim_anneal/sim_anneal_map2_' + str(i) + '.npy', map2_traj)
        with open('../data/sim_anneal/sim_anneal_map2.csv', mode='a') as file_:
            file_.write("{},{}".format(map2_obstacle_cost, map2_smoothness_cost))
            file_.write("\n")

        map3_traj, map3_obstacle_cost, map3_smoothness_cost = Map3(i>=10)
        map3_cost.append([map3_obstacle_cost, map3_smoothness_cost])
        np.save('../data/sim_anneal/sim_anneal_map3_' + str(i) + '.npy', map3_traj)
        with open('../data/sim_anneal/sim_anneal_map3.csv', mode='a') as file_:
            file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
            file_.write("\n")

    np.savetxt("../data/sim_anneal/sim_anneal_map1.csv", np.array(map1_cost), delimiter=",")
    np.savetxt("../data/sim_anneal/sim_anneal_map2.csv", np.array(map2_cost), delimiter=",")
    np.savetxt("../data/sim_anneal/sim_anneal_map3.csv", np.array(map3_cost), delimiter=",")

if __name__ == "__main__":
    main()