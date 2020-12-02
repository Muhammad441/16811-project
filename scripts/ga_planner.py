import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from costmap import Map1, Map2
import costmap
from manipulator import Manipulator
import copy 
import pdb
import heapq
from itertools import count
import pickle
from sympy import Polygon
from scipy.ndimage.morphology import binary_fill_holes as imfill
from PIL import Image, ImageDraw
from planner import Planner

class GAPlanner(Planner):
    def __init__(self, start, goal, num_waypoints, map, manipulator):
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints
        self.map = map
        self.manipulator = manipulator
        self.traj = self.seedPath(self.start,self.goal,self.num_waypoints)
        self.tiebreaker = count()
    
    def seedPath(self,start,goal,num_waypoints):
        traj = np.zeros((num_waypoints, self.manipulator.num_links))
        traj[0] = start
        traj[-1] = goal
        for i in range(1, num_waypoints):
            traj[i] = traj[0] + (goal - start)*i/num_waypoints
        return traj
    
    def optimize(self,population_size=50, alpha=1.0, beta=1.0, max_iters=50):
        self.alpha=alpha
        self.beta=beta
        self.population_size=population_size
        # self.trajUnfitScore(self.traj)
        self.population = self.generateInitialPopulation()
        for i in range(max_iters):
            self.population = self.next_generation(self.population)
            print ("Opt cost", self.population[0][0], "Iteration:", i)
            if self.population[0][0]==0:
                break
        self.traj = self.population[0][2]
    
    def push_heap(self,heap,value_data):
        cv = next(self.tiebreaker)
        if cv>12939220230:
            self.tiebreaker = count()
        if value_data[0] >= 0:
            heapq.heappush(heap,(value_data[0],cv,value_data[1]))

    def generateInitialPopulation(self):
        population = []
        fs = self.trajUnfitScore(self.traj)
        self.push_heap(population, (fs, self.traj))
        random_population = self.generateRandomPopulation(self.population_size-1)
        return population+random_population
    
    def generateRandomPopulation(self,num_indiv):
        population = []
        while len(population)!=num_indiv:
            traj = np.random.uniform(low=-3.14159, high=3.14159, size=(self.num_waypoints-2, self.manipulator.num_links))
            traj = np.concatenate([self.traj[0:1,:],traj,self.traj[self.num_waypoints-1:self.num_waypoints,:]],axis=0)
            fs = self.trajUnfitScore(traj)
            if fs >= 0:
                self.push_heap(population, (fs, traj))
        return population
    
    def selection(self,population):
        return heapq.nsmallest(int(self.population_size/2),population)
    
    def pairing(self,population):
        ## Assumes that the population is sorted
        parents = [[population[x][2],population[x+1][2]] 
                   for x in range(len(population)//2)]
        return parents
    
    def mating(self,parents):
        parent_0 = parents[0].reshape((1,-1),order='C')
        parent_1 = parents[1].reshape((1,-1),order='C')
        mating_point_1 = np.random.randint(self.manipulator.num_links+1, parent_0.shape[1]-self.manipulator.num_links)
        mating_point_2 = np.random.randint(self.manipulator.num_links+1, parent_0.shape[1]-self.manipulator.num_links)
        mp_min = min(mating_point_1,mating_point_2)
        mp_max = max(mating_point_1,mating_point_2)
        offspring_0 = np.concatenate([ parent_0[0:1,0:mp_min],parent_1[0:1,mp_min:mp_max],parent_0[0:1,mp_max:] ],axis=1)
        offspring_1 = np.concatenate([ parent_1[0:1,0:mp_min],parent_0[0:1,mp_min:mp_max],parent_1[0:1,mp_max:] ],axis=1)
        offspring_0 = offspring_0.reshape((self.num_waypoints,self.manipulator.num_links),order='C')
        offspring_1 = offspring_1.reshape((self.num_waypoints,self.manipulator.num_links),order='C')
        return [offspring_0, offspring_1]
    
    def mutation(self,population,num=2,eliteism=0):
        affected = []
        for i in range(num):
            sel_indiv = np.random.randint(eliteism,len(population))
            sel_wp = np.random.randint(1,self.num_waypoints-1)
            sel_lk = np.random.randint(0,self.manipulator.num_links)
            #population[sel_indiv][2][sel_wp,sel_lk] = np.random.uniform(low=-3.14159, high=3.14159)
            population[sel_indiv][2][sel_wp,sel_lk] = np.random.normal(population[sel_indiv][2][sel_wp,sel_lk], scale=0.3)
            population[sel_indiv][2][sel_wp,sel_lk] = np.clip(population[sel_indiv][2][sel_wp,sel_lk], -3.14159, 3.14159)
            affected.append(sel_indiv)
        new_pop = []
        for i in range(len(population)):
            if i in affected:
                fs = self.trajUnfitScore(population[i][2])
                self.push_heap(new_pop, (fs, population[i][2]))
            else:
                self.push_heap(new_pop, (population[i][0], population[i][2]))
        return new_pop
        

    def next_generation(self,population):
        selected_population = self.selection(population)
        sel_pol_num = len(selected_population)
        selected_population.sort()
        parents = self.pairing(selected_population)
        offsprings = [self.mating(parents[x]) for x in range(len(parents))]
        unmutated = copy.deepcopy(selected_population)
        for i in range(len(offsprings)):
            for j in range(2):
                traj = offsprings[i][j]
                fs = self.trajUnfitScore(traj)
                if fs >= 0:
                    self.push_heap(unmutated, (fs, traj))
        mutated = self.mutation(unmutated,num=100,eliteism=1)
        random_population = self.generateRandomPopulation(self.population_size-len(mutated))
        mutated = mutated+random_population
        mutated.sort()
        return mutated

    def PolyArea(self,x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) 
    
    def trajUnfitScore(self,traj):
        """Fitness score of a trajectory is comprised of:
        1) Trajectory has to be smooth. Distance of each intermediate state with prev and next
        2) Obstacle should be avoided by each state.
        Obstacle cost is the sum of obs cost of all intermediate points of all links
        """
        trajCost = 0.0

        ## Compute cost for polygon area
        img = Image.new('L', (self.map.cost_map.shape[1], self.map.cost_map.shape[0]), 0)
        mask=None
        smoothingCost = 0
        for i in range(self.num_waypoints-1):
            state1 = traj[i,:]
            state2 = traj[i+1,:]
            fine_traj = self.seedPath(state1,state2,10)
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
                # img_ = Image.new('L', (self.map.cost_map.shape[1], self.map.cost_map.shape[0]), 0)
                # ImageDraw.Draw(img_).polygon(poly_c, outline=1, fill=1)
                # smoothingCost += np.sum(np.array(img_))

        mask = np.array(img)
        # plt.imshow(mask)
        # plt.show()
        filled = imfill(mask)
        # plt.imshow(filled)
        # plt.show()
        area = self.map.cost_map[filled].sum()
        # print ("obstacle", np.sum(self.map.cost_map[filled]==50.0))
        trajCost += self.beta*area + 0.1*smoothingCost

        return trajCost

# def Map3():
#     manipulator = Manipulator(num_links = 5, link_lengths = np.ones(5)*75, 
#                               base_position = np.array((250, 200)))
#     map = costmap.Map4()
#     vis = Visualizer(map.map)
#     start = np.array((0,1.56,1.2,0,0))
#     goal = np.array((2.0,2.0,1.56,1.56, 1.56))
#     vis.state_vis(manipulator.ForwardKinematics(start))
#     vis.state_vis(manipulator.ForwardKinematics(goal))

#     planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
#                  map = map, manipulator = manipulator)

#     planner.optimize(population_size=50,alpha=900.0, beta=1., max_iters=300)
#     path = []
#     for i in range(planner.traj.shape[0]-1):
#         traj = planner.seedPath(planner.population[0][2][i,:],planner.population[0][2][i+1,:],10)
#         for j in range(traj.shape[0]):
#             path.append(manipulator.ForwardKinematics(traj[j]))
                
#     pdb.set_trace()
#     vis.traj_vis(path)
#     plt.show()

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

    planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)

    planner.optimize(population_size=50,alpha=900.0, beta=1., max_iters=2)

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)

    return planner.population[0][2], obstacle_cost, smoothness_cost

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

    planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)

    planner.optimize(population_size=50,alpha=900.0, beta=1., max_iters=2)

    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)
    return planner.population[0][2], obstacle_cost, smoothness_cost

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

    planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)

    planner.optimize(population_size=50,alpha=900.0, beta=1., max_iters=2)
    
    obstacle_cost, smoothness_cost = planner.trajectoryCost(planner.traj)
    print(obstacle_cost, smoothness_cost)
    return planner.population[0][2], obstacle_cost, smoothness_cost

def main():
    map1_cost = []
    map2_cost = []
    map3_cost = []
    for i in range(1, 20):
        np.random.seed(i)
        map1_traj, map1_obstacle_cost, map1_smoothness_cost = Map1(i>=10)
        map1_cost.append([map1_obstacle_cost, map1_smoothness_cost])
        np.save('../data/ga/ga_map1_' + str(i) + '.npy', map1_traj)
        with open('../data/ga/ga_map1.csv', mode='a') as file_:
            file_.write("{},{}".format(map1_obstacle_cost, map1_smoothness_cost))
            file_.write("\n")


        # map2_traj, map2_obstacle_cost, map2_smoothness_cost = Map2(i>=10)
        # map2_cost.append([map2_obstacle_cost, map2_smoothness_cost])
        # np.save('../data/ga/ga_map2_' + str(i) + '.npy', map2_traj)
        # with open('../data/ga/ga_map2.csv', mode='a') as file_:
        #     file_.write("{},{}".format(map2_obstacle_cost, map2_smoothness_cost))
        #     file_.write("\n")

        # map3_traj, map3_obstacle_cost, map3_smoothness_cost = Map3(i>=10)
        # map3_cost.append([map3_obstacle_cost, map3_smoothness_cost])
        # np.save('../data/ga/ga_map3_' + str(i) + '.npy', map3_traj)
        # with open('../data/ga/ga_map3.csv', mode='a') as file_:
        #     file_.write("{},{}".format(map3_obstacle_cost, map3_smoothness_cost))
        #     file_.write("\n")

    np.savetxt("../data/ga/ga_map1.csv", np.array(map1_cost), delimiter=",")
    np.savetxt("../data/ga/ga_map2.csv", np.array(map2_cost), delimiter=",")
    np.savetxt("../data/ga/ga_map3.csv", np.array(map3_cost), delimiter=",")

if __name__ == "__main__":
    main()