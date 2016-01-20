import numpy as np
from scipy.ndimage.filters import gaussian_filter

class PotentialField:
    def __init__(self, start, goal, cspace):
        self.start = start
        self.goal = goal
        self.cspace = cspace

    def get_attractive(self, ksi, d):
        uattr = np.zeros((181, 361))
        for t1 in range(0, 181, 1):
            for t2 in range(0, 361, 1):
                ro = np.linalg.norm(np.array([t1, t2] - self.goal))
                uattr[t1, t2] = 0.5 * ksi * ro**2 if ro <= d else d * ksi * ro
        return uattr
     
    def get_repulsive(self, s):
        repulsive = self.cspace.copy() * 200
        repulsive = gaussian_filter(repulsive, sigma=s)
        return repulsive
