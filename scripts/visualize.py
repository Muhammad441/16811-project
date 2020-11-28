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
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(traj)):
            # fig, ax = plt.plot()
            ax.imshow(self.map)
            state = traj[i]
            x_c, y_c = state[0], state[1]
            ax.plot(x_c,y_c,'-*',label="robot")
            # ax.plot(x_c[0],y_c[0],'^',label="start")
            # # plt.set_aspect('equal', adjustable='box')
            # plt.show()
            # plt.show()
            fig.canvas.draw()
            time.sleep(0.1)
            fig.canvas.flush_events()

            # fig.clf()
            # plt.clf()
            # ax.clear()
        plt.ioff()
        
