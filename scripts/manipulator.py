import numpy as np 

class Manipulator:
    def __init__(self, num_links = 4, link_lengths = np.ones(4)*75, base_position = np.array((200,400))):
        self.num_links = num_links
        self.link_lengths = link_lengths
        self.base_position = base_position
        self.collision_resolution = 2

    def ForwardKinematics(self, state):
        assert(state.shape[0] == self.num_links)
        
        x_c_p = self.base_position[0]
        y_c_p = self.base_position[1]
        x_c = [x_c_p]
        y_c = [y_c_p]
        for i in range(state.shape[0]):
            x_p = self.link_lengths[i]*np.cos(state[i])
            y_p = self.link_lengths[i]*np.sin(state[i])
            x_i = x_c_p + x_p 
            y_i = y_c_p + y_p
            x_c.append(x_i)
            y_c.append(y_i)
            x_c_p = x_i
            y_c_p = y_i

        return [x_c, y_c]
    
    def linkPts(self, state):
        links_pts_x = []
        links_pts_y = []
        lengths_pts = []

        x_c_p = self.base_position[0]
        y_c_p = self.base_position[1]

        for arm_idx in range(self.num_links):
            length = 0
            link_pts_x = []
            link_pts_y = []
            lengths = []
            while length < self.link_lengths[arm_idx]:
                x_p = length*np.cos(state[arm_idx])
                y_p = length*np.sin(state[arm_idx])
                link_pts_x.append(x_c_p + x_p) 
                link_pts_y.append(y_c_p + y_p) 
                lengths.append(length) 
                length += self.collision_resolution
            x_c_p = link_pts_x[-1]
            y_c_p = link_pts_y[-1]
            links_pts_x.append(link_pts_x)
            links_pts_y.append(link_pts_y)
            lengths_pts.append(lengths)
        
        return links_pts_x, links_pts_y, lengths_pts