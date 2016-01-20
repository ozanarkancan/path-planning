import numpy as np
from numpy import cos, sin
from scipy.ndimage.filters import gaussian_filter

class Robot:
    def __init__(self, base, l1, l2):
        self.base = base
        self.l1 = l1
        self.l2 = l2

    def forward(self, t1, t2):
        x = self.l1 * cos(t1) + self.l2 * cos(t1 - t2)
        y = self.l1 * sin(t1) + self.l2 * sin(t1 - t2)
        return np.array([x + self.base[0], y + self.base[1]])

    def inverse(self, x, y):
        x = x - self.base[0]
        y = y - self.base[1]
        c2 = (x**2 + y**2 - self.l1**2 - self.l2**2)/(2 * self.l1*self.l2)
        s2 = np.sqrt(1 - c2**2)#+/-
        t2 = np.arctan2(s2, c2)
        
        cb = (self.l1**2 + x**2 + y**2 - self.l2**2)/(2*self.l1*np.sqrt(x**2 + y**2))
        sb = np.sqrt(1 - cb**2)
        t1 = np.arctan2(y, x) + np.arctan2(sb, cb)

        return np.array([t1, t2])

    def get_links(self, t1, t2):
        tmp = (cos(t1) * self.l1, sin(t1) * self.l1)
        l1 = [self.base, np.array([tmp[0] + self.base[0], tmp[1] + self.base[1]])]
        l2 = [l1[1], self.forward(t1, t2)]
        return (l1, l2)
