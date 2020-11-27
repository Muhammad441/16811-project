import numpy as np
import matplotlib.pyplot as plt 

m_K = 4
m_N = 10

def generate_random_state(K):
    state = np.random.uniform(low=-3.14159, high=3.14159, size=(K,))
    return state 


def generate_random_obstacles(N,xdim,ydim):
    obstacles_x = np.random.uniform(low=-xdim/2., high=xdim/2., size=(N,1))
    obstacles_y = np.random.uniform(low=-ydim/2., high=ydim/2., size=(N,1))
    obstacles = np.concatenate([obstacles_x,obstacles_y],axis=1)
    return obstacles

def state_to_coordinates(state,length):
    x_c = [0.0]
    y_c = [0.0]
    x_c_p = 0.0
    y_c_p = 0.0
    for i in range(state.shape[0]):
        x_p = length*np.cos(np.sum(state[0:i+1]))
        y_p = length*np.sin(np.sum(state[0:i+1]))
        x_i = x_c_p + x_p 
        y_i = y_c_p + y_p
        x_c.append(x_i)
        y_c.append(y_i)
        x_c_p = x_i
        y_c_p = y_i 
    return x_c, y_c


def visualize_state_obstacles(state,obstacles,length,xdim,ydim):
    fig, ax = plt.subplots()
    x_c, y_c = state_to_coordinates(state,length)
    ax.plot(x_c,y_c,'-*',label="robot")
    ax.plot(0,0,'^',label="start")
    circles = []
    for i in range(obstacles.shape[0]):
        circle = plt.Circle((obstacles[i,0], obstacles[i,1]), 0.2, color='r')
        circles.append(circle)
    for i in range(len(circles)):
        ax.add_artist(circles[i])
    plt.xlim([-xdim/2.,xdim/2.])
    plt.ylim([-ydim/2.,ydim/2.])
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()



def main():
    state = generate_random_state(m_K)
    obstacles = generate_random_obstacles(m_N,5.,5.)
    visualize_state_obstacles(state,obstacles,1,5.,5.)


main()