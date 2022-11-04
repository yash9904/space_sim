import streamlit as st
import numpy as np

from time import sleep


#%%
def r(p, q):
    '''
    Parameters
    ----------
    p : Location of point p
    q : Location of point q
    Returns
    -------
    float
        Euclidean distance between the points
    '''
    d = np.sqrt(np.dot(p - q, (p - q).T))
    return d if d > 50 else 50

def ru(p, q):
    
    
    '''
    Parameters
    ----------
    p : Location of point p
    q : Location of point q
    Returns
    -------
    ndarray of floats
        unit vector pointing from p to q
    '''
    x = q[0] - p[0]
    y = q[1] - p[1]
    
    d = r(p, q)
    
    return np.array([x / d, y / d])
#%%


mp = 4 * pow(10, 14)
mq = 2 * pow(10, 14)

G = 6.6743 * 10**(-11)




#%%
def mass_circle_p(x, y):
    global mp, p
    return ((x - p[0]) ** 2) + ((y - p[1]) ** 2) <= 100

def mass_circle_q(x, y):
    global mq, q
    return ((x - q[0]) ** 2) + ((y - q[1]) ** 2) <= 50

size = (1000, 1000)
space = np.zeros(size, np.float32)

def update_space():
    global mp, mq, space, p, q
    
    return np.fromfunction(mass_circle_p, space.shape) + np.fromfunction(mass_circle_q, space.shape)
    
del_t = 1
n_iter = 1000

#%%

st.title('Objects in Space')
st.sidebar.title('Configurations')
px = st.sidebar.slider('x coordinate of first object', 0, size[0] - 1, value = 200)
py = st.sidebar.slider('y coordinate of first object', 0, size[1] - 1, value = 200)
p = np.array([px, py], dtype = np.int32)

qx = st.sidebar.slider('x coordinate of second object', 0, size[0] - 1, value = 500)
qy = st.sidebar.slider('y coordinate of second object', 0, size[1] - 1, value = 600)
q = np.array([qx, qy], dtype = np.int32)

upu = st.sidebar.slider('x velocity of first object', -3, 3, value = 1)
upv = st.sidebar.slider('y velocity of first object', -3, 3, value = 0)
uqu = st.sidebar.slider('x velocity of second object', -3, 3, value = -1)
uqv = st.sidebar.slider('y velocity of second object', -3, 3, value = 0)

up = [upu, upv]
uq = [uqu, uqv]


#%%
data_dict = {'Property': ['Mass (e14)(kg)', 'x-Velocity(m/s)', 'y-Velocity(m/s)'], 
             'Object #1': [mp//10e13, up[0], up[1]],
             'Object #2': [mq//10e13, uq[0], uq[1]]}

st.sidebar.dataframe(data_dict)

#%%
widget = st.empty()
t = 1
while True:
    sleep(0.01)
    ap = G * mq * ru(p, q) / (r(p, q)**2)
    aq = G * mp * ru(q, p) / (r(p, q)**2)
    
    up += (ap * del_t)
    uq += (aq * del_t)
    
    p += np.int32((up * del_t))
    q += np.int32((uq * del_t))
    
    # Periodic boundary
    p %= size[0]
    q %= size[1]

    px, py = p.astype(np.uint16)
    qx, qy = q.astype(np.uint16)
    
    
    space[px - t: px + t, py - t: py + t] = 1
    space[qx - t: qx + t, qy - t: qy + t] = 1
    
    new_space = space + update_space()
    widget.image(new_space, clamp = True)
    