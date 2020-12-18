import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import pdb
import time 
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

        state = traj[0]
        x_c, y_c = state[0], state[1]
        Ln, = ax.plot(x_c,y_c,'-*',label="robot")
        plt.ion()
        plt.show()
        for i in range(len(traj)):
            state = traj[i]
            x_c, y_c = state[0], state[1]
            Ln.set_xdata(x_c)
            Ln.set_ydata(y_c)
            plt.pause(0.2)
            # plt.waitforbuttonpress()
        plt.waitforbuttonpress()

    def stationary_traj_vis(self, traj):
        fig, ax = plt.subplots()
        for state in traj:
            x_c, y_c = state[0], state[1]
            ax.imshow(self.map)
            ax.plot(x_c,y_c,'-*')
            ax.plot(x_c[0],y_c[0],'^')
            ax.set_aspect('equal', adjustable='box')
        # plt.legend()
        plt.show()
