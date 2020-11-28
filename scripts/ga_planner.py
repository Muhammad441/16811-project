import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from costmap import Map1
from manipulator import Manipulator
import copy 
import pdb
import heapq
from itertools import count

class GAPlanner:
    def __init__(self, start, goal, num_waypoints, map, manipulator):
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints
        self.map = map
        self.manipulator = manipulator
        self.traj = self.seedPath()
        self.tiebreaker = count()
    
    def seedPath(self):
        traj = np.zeros((self.num_waypoints, self.manipulator.num_links))
        traj[0] = self.start
        traj[-1] = self.goal
        for i in range(1, self.num_waypoints):
            traj[i] = traj[0] + (self.goal - self.start)*i/self.num_waypoints
        return traj
    
    def optimize(self,population_size=50, alpha=1.0, max_iters=50):
        self.alpha=alpha
        self.population_size=population_size
        self.population = self.generateInitialPopulation()
        for i in range(max_iters):
            self.population = self.next_generation(self.population)
            print"Opt cost", self.population[0][0]
    
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
        return heapq.nsmallest(self.population_size/2,population)
    
    def pairing(self,population):
        ## Assumes that the population is sorted
        parents = [[population[x][2],population[x+1][2]] 
                   for x in range(len(population)//2)]
        return parents
    
    def mating(self,parents):
        parent_0 = parents[0].reshape((1,-1),order='C')
        parent_1 = parents[1].reshape((1,-1),order='C')
        mating_point = np.random.randint(self.manipulator.num_links+1, parent_0.shape[1]-self.manipulator.num_links)
        offspring_0 = np.concatenate([ parent_0[0:1,0:mating_point],parent_1[0:1,mating_point:] ],axis=1)
        offspring_1 = np.concatenate([ parent_1[0:1,0:mating_point],parent_0[0:1,mating_point:] ],axis=1)
        offspring_0 = offspring_0.reshape((self.num_waypoints,self.manipulator.num_links),order='C')
        offspring_1 = offspring_1.reshape((self.num_waypoints,self.manipulator.num_links),order='C')
        return [offspring_0, offspring_1]
    
    def mutation(self,population,num=2):
        affected = []
        for i in range(num):
            sel_indiv = np.random.randint(0,len(population))
            sel_wp = np.random.randint(1,self.num_waypoints-1)
            sel_lk = np.random.randint(0,self.manipulator.num_links)
            population[sel_indiv][2][sel_wp,sel_lk] = np.random.uniform(low=-3.14159, high=3.14159)
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
        mutated = self.mutation(unmutated,num=2)
        random_population = self.generateRandomPopulation(self.population_size-len(mutated))
        mutated = mutated+random_population
        mutated.sort()
        return mutated
        

    
    def trajUnfitScore(self,traj):
        """Fitness score of a trajectory is comprised of:
        1) Trajectory has to be smooth. Distance of each intermediate state with prev and next
        2) Obstacle should be avoided by each state.
        Obstacle cost is the sum of obs cost of all intermediate points of all links
        """
        trajCost = 0.0

        ## Compute cost due to smoothness
        for i in range(1,self.num_waypoints-1):
            dist_p = np.linalg.norm(traj[i,:]-traj[i-1,:])
            dist_n = np.linalg.norm(traj[i+1,:]-traj[i,:])
            trajCost += self.alpha*np.exp(dist_p+dist_n)

        ## Compute cost due to obstacles
        for i in range(self.num_waypoints):
            state = traj[i,:]
            links_pts_x, links_pts_y, lengths_pts = self.manipulator.linkPts(state)

            this_wp_cost = 0.0
            for j in range(self.manipulator.num_links):
                for pt in range(len(links_pts_x[j])):
                    pt_y = int(links_pts_y[j][pt])
                    pt_x = int(links_pts_x[j][pt])
                    if pt_x >= self.map.x_len or pt_x < 0 or pt_y >= self.map.y_len or pt_y < 0:
                        return -1
                    this_wp_cost += self.map.cost_map[pt_y, pt_x]
            trajCost+=this_wp_cost
        return trajCost


def main():
    manipulator = Manipulator(base_position = np.array((150,250)))
    map = Map1()
    vis = Visualizer(map.map)
    start = np.array((0.1,0.2,0.1,0.1))
    goal = np.array((0.1,0.2,-1.5,0.1))

    planner = GAPlanner(start = start, goal = goal, num_waypoints = 10, 
                 map = map, manipulator = manipulator)

    planner.optimize(population_size=50,alpha=900.0, max_iters=1000)
    #print(planner.trajUnfitScore(planner.traj))
    
    path = []
    for i in range(planner.traj.shape[0]):
        path.append(manipulator.ForwardKinematics(planner.population[0][2][i,:]))

    vis.traj_vis(path)
    plt.show()

if __name__ == "__main__":
    main()