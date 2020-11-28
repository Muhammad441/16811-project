import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from costmap import Map1, Map2
from manipulator import Manipulator
import copy 
import pdb
import heapq
from itertools import count
import pickle
from sympy import Polygon
from scipy.ndimage.morphology import binary_fill_holes as imfill
from PIL import Image, ImageDraw

class GAPlanner:
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
        mask = np.array(img)
        # plt.imshow(mask)
        # plt.show()
        filled = imfill(mask)
        # plt.imshow(filled)
        # plt.show()
        area = self.map.cost_map[filled].sum()
        # print ("obstacle", np.sum(self.map.cost_map[filled]==50.0))
        trajCost += self.beta*area

        return trajCost


def main():
    manipulator = Manipulator(base_position = np.array((150,250)))
    map = Map2()
    vis = Visualizer(map.map)
    start = np.array((0.1,0.1,1.2,0.0))
    goal = np.array((0.1,1.9,1.9,1.6))

    planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)

    #print(planner.trajUnfitScore(planner.traj))
    planner.optimize(population_size=50,alpha=900.0, beta=1., max_iters=50)
    
    # path = []
    # for i in range(planner.traj.shape[0]):
    #     path.append(manipulator.ForwardKinematics(planner.traj[i]))
    
    path = []
    for i in range(planner.traj.shape[0]-1):
        traj = planner.seedPath(planner.population[0][2][i,:],planner.population[0][2][i+1,:],10)
        for j in range(traj.shape[0]):
            path.append(manipulator.ForwardKinematics(traj[j]))

    # path = []
    # for i in range(planner.population[0][2].shape[0]):
    #     path.append(manipulator.ForwardKinematics(planner.population[0][2][i]))
    
    # path = []
    # with open('filename.pickle', 'rb') as handle:
    #     path = pickle.load(handle)

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(path, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vis.traj_vis(path)
    plt.show()

if __name__ == "__main__":
    main()