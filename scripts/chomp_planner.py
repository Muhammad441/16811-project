import numpy as np
import matplotlib.pyplot as plt
from visualize import Visualizer
from costmap import Map1

class Manipulator:
    def __init__(self, num_links = 4, link_lengths = np.ones(4)*75, base_position = np.array((200,400))):
        self.num_links = num_links
        self.link_lengths = link_lengths
        self.base_position = base_position

    def ForwardKinematics(self, state):
        assert(state.shape[0] == self.num_links)
        
        x_c_p = self.base_position[0]
        y_c_p = self.base_position[1]
        x_c = [x_c_p]
        y_c = [y_c_p]
        for i in range(state.shape[0]):
            x_p = self.link_lengths[i]*np.cos(np.sum(state[0:i+1]))
            y_p = self.link_lengths[i]*np.sin(np.sum(state[0:i+1]))
            x_i = x_c_p + x_p 
            y_i = y_c_p + y_p
            x_c.append(x_i)
            y_c.append(y_i)
            x_c_p = x_i
            y_c_p = y_i

        return [x_c, y_c]

def main():
    manipulator = Manipulator()
    map = Map1()
    vis = Visualizer(map.map)
    state = manipulator.ForwardKinematics(np.array((0.1,0.2,0.1,0.1)))
    vis.state_vis(state)

if __name__ == "__main__":
    main()