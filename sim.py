#%%
import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

mpl.rcParams['figure.max_open_warning'] = 0
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

up = [2, -3]
uq = [4, 2]

p = np.array([400.0, 400.0])
q = np.array([300.0, 800.0])
#%%
def mass_circle_p(x, y):
    global mp, p
    return ((x - p[0]) ** 2) + ((y - p[1]) ** 2) <= 400

def mass_circle_q(x, y):
    global mq, q
    return ((x - q[0]) ** 2) + ((y - q[1]) ** 2) <= 200

size = (1000, 1000)
space = np.zeros(size, np.float32)

def update_space():
    global mp, mq, space, p, q
    
    return np.fromfunction(mass_circle_p, space.shape) + np.fromfunction(mass_circle_q, space.shape)
    
del_t = 1
n_iter = 1000

#%%
now = datetime.now().strftime("%d%m%Y_%H%M%S")
frames_dir = 'frames_dir_space_sim' + now

if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)

t = 2
vid_dir = f'space_sim_t{del_t * n_iter}_{now}.mp4'

#%%
for i in tqdm(range(n_iter), desc = 'Simulating...'):    

    ap = G * mq * ru(p, q) / (r(p, q)**2)
    aq = G * mp * ru(q, p) / (r(p, q)**2)
    
    up += (ap * del_t)
    uq += (aq * del_t)
    
    p += (up * del_t)
    q += (uq * del_t)
    
    # Periodic boundary
    p %= 1000
    q %= 1000

    px, py = p.astype(np.uint16)
    qx, qy = q.astype(np.uint16)
    
    
    space[px - t: px + t, py - t: py + t] = 1
    space[qx - t: qx + t, qy - t: qy + t] = 1
    
    new_space = space + update_space()
    
    fig = plt.figure(figsize = (20, 15))
    
    plt.imshow(new_space, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(frames_dir + "/img.{0:05d}.png".format(i))
    plt.close(fig)

imshape = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0])).shape
imshape = (imshape[1], imshape[0])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

result = cv2.VideoWriter(vid_dir, 
                         fourcc,
                         20, imshape)

for file in tqdm(sorted(os.listdir(frames_dir)), desc = "Saving ..."):
    image = cv2.imread(os.path.join(frames_dir, file))
    result.write(image)
result.release()

print(f'\nSimulation saved as: {vid_dir}')
