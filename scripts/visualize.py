import numpy as np
import matplotlib.pyplot as plt 

class Visualizer:
    def __init__(self, map):
        self.map = map

    def state_vis(self, state):
        fig, ax = plt.subplots()
        x_c, y_c = state[0], state[1]
        ax.imshow(self.map)
        ax.plot(x_c,y_c,'-*',label="robot")
        ax.plot(x_c[0],y_c[0],'^',label="start")
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    def traj_vis(self, traj):
        fig, ax = plt.subplots()
        ax.imshow(self.map)
        for i in range(len(traj)):
            state = traj[i]
            x_c, y_c = state[0], state[1]
            ax.plot(x_c,y_c,'-*',label="robot")
            ax.plot(x_c[0],y_c[0],'^',label="start")
            ax.set_aspect('equal', adjustable='box')
        plt.show()