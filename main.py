import numpy as np
from numpy import cos, sin
from robot import Robot
from potential_field import PotentialField
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
import argparse
from astar import *

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="main")
    parser.add_argument("--base", default=[20., 0.], nargs='+', type=float,
        help="base position")
    parser.add_argument("--l1", default=52., type=float,
        help="length of the link 1")
    parser.add_argument("--l2", default=20., type=float,
        help="length of the link 2")
    parser.add_argument("--circle", default=[40, 60, 10], nargs='+', type=float,
        help="central position and radius of the circle")
    parser.add_argument("--start", default=[20., 70.], nargs='+', type=float,
        help="start position")
    parser.add_argument("--goal", default=[60., 40.], nargs='+', type=float,
        help="goal position")
    parser.add_argument("--lr", default=1, type=float,
        help="learning rate")
    parser.add_argument("--ksi", default=0.5, type=float,
        help="ksi parameter of attractive field function")
    parser.add_argument("--d", default=6, type=float,
        help="d parameter of attractive field function")
    parser.add_argument("--sigma", default=4, type=float,
        help="sigma for gaussian filter")
    parser.add_argument("--movie", default=False, action='store_true',
        help="create movie")
    return parser

def point_circle_intersect(point, circle):
    """
    point: (x, y)
    circle: (x, y, r)
    """

    d = np.sqrt((circle[0] - point[0])**2 + (circle[1] - point[1])**2)
    return d <= r

def line_circle_intersect(line, circle):
    """
    line: [(x1, y1), (x2, y2)]
    circle: [(x, y,) r]
    """

    E, L = line
    C, r = circle

    d = L - E
    f = E - C

    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - r**2;

    disc = b**2 - 4 * a * c
    if disc < 0:
        return False
    else:
        disc = np.sqrt(disc)
        t1 = (-b - disc)/(2*a)
        t2 = (-b + disc)/(2*a)
        if t1 >= 0 and t1 <= 1:
            return True

        if t2 >= 0 and t2 <= 1:
            return False

        return False

def intersect_boundries(l1, l2):
    left = filter(lambda x: x[0] <= 0, l1 + l2)
    right = filter(lambda x: x[0] >= 100, l1 + l2)
    down = filter(lambda x: x[1] <= 0, [l1[1]]+l2)
    up = filter(lambda x: x[1] >= 100, l1+l2)
    ls = filter(lambda x: len(x) > 0, [left, right, up, down])
    return len(ls) != 0

def configuration_space(robot, circle):
    cspace = np.zeros((181, 361))
    for t1 in xrange(0, 181, 1):
        for t2 in xrange(0, 361, 1):
            l1, l2 = robot.get_links(np.deg2rad(t1), np.deg2rad(t2))
            test1 = line_circle_intersect(l1, circle)
            test2 = line_circle_intersect(l2, circle)
            test3 = intersect_boundries(l1, l2)

            if test1 or test2 or test3:
                cspace[t1, t2] = 1
    
    res = np.where(cspace == 1)
    return cspace

def gradient_descent(pf, start, goal, lr=1):
    curr = start
    path = [curr]
    
    for i in range(200):
        #print "Epoch: ", (i + 1)
        x = round(curr[0])
        y = round(curr[1])
        pmax = 1000000
        #find the direction of max gradient
        dx = 0
        dy = 0

        #shuffling the order helps to choose neighbour with same value randomly
        nbs = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
        np.random.shuffle(nbs)

        for nx, ny in nbs:
            if nx == 0 and ny == 0:
                continue
            cmax = pf[x + nx, y + ny] - pf[x, y]
            if cmax < pmax:
                dx = nx
                dy = ny
                pmax = cmax
                
        curr = (curr[0] + lr*dx, curr[1] + lr*dy)
        curr = (curr[0] % 181, curr[1] % 361)
        path.append(curr)
        if np.abs(curr[0] - goal[0]) < 1 and np.abs(curr[1] - goal[1]) < 1:
            curr = goal
            break
    path.append(curr)
    return path

def plot_cspace(ax, cspace, start, goal, title):
    t1 = np.arange(181)
    t2 = np.arange(361)
    res = np.where(cspace == 1)
    ax.plot(res[0], res[1], '.')
    x, y = start
    x2, y2 = goal
    ax.plot(x,y, 'rd')
    ax.plot(x2,y2, 'gd')
    l1 = ax.annotate('start', xy=(x, y))
    l2 = ax.annotate('goal', xy=(x2, y2))
    
    ax.set_title(title)
    ax.set_xlim(0, 181)
    ax.set_ylim(0, 361)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')

def plot_surface(ax, m, start, goal, title):
    t1 = np.arange(181)
    t2 = np.arange(361)
    X, Y = np.meshgrid(t1, t2, indexing='ij')
    surf = ax.plot_surface(X, Y, m, cmap=cm.jet)
    ax.set_title(title)
    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2')
    x, y, _ = proj3d.proj_transform(start[0],start[1],m[start[0], start[1]], ax.get_proj())
    x2, y2, _ = proj3d.proj_transform(goal[0],goal[1],m[goal[0], goal[1]], ax.get_proj())
    l1 = ax.annotate('start', xy=(x, y))
    l2 = ax.annotate('goal', xy=(x2, y2))

def plot_contour(ax, m, start, goal, title):
    t1 = np.arange(181)
    t2 = np.arange(361)
    X, Y = np.meshgrid(t1, t2, indexing='ij')
    surf = ax.contourf(X, Y, m)
    ax.set_title(title)
    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2')
    x, y = start
    x2, y2 = goal
    l1 = ax.annotate('start', xy=(x, y))
    l2 = ax.annotate('goal', xy=(x2, y2))
    plt.colorbar(surf)

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, robot, circle, cspace, start, goal, path, title):
        fig = plt.figure()
        fig.canvas.set_window_title(title)
        self.ax1 = fig.add_subplot(1, 2, 1)
        self.ax2 = fig.add_subplot(1, 2, 2)
        
        self.ax1.add_patch(mpatches.Rectangle((0, 0), 100, 100, alpha=1, facecolor='none'))
        self.ax1.plot(start[0], start[1], 'rd')#start
        self.ax1.plot(goal[0], goal[1], 'gd')#goal
        r = circle[1]
        xfirst = circle[0][0] + (r * cos(0))
        yfirst = circle[0][1] + (r * sin(0))
        
        for i in xrange(0, 361, 10):
            theta = (i*np.pi) / 180
            xc = circle[0][0] + (r*cos(theta))
            yc = circle[0][1] + (r*sin(theta))
            self.ax1.add_line(mlines.Line2D([xfirst, xc],[yfirst, yc]))
            xfirst = xc
            yfirst = yc

        t1, t2 = robot.inverse(start[0], start[1])
        l1, l2 = robot.get_links(t1, t2)
        self.l1= mlines.Line2D([l1[0][0], l1[1][0]],[l1[0][1], l1[1][1]], color='k')
        self.ax1.add_line(self.l1)
        self.l2 = mlines.Line2D([l2[0][0], l2[1][0]],[l2[0][1], l2[1][1]], color='k')
        self.ax1.add_line(self.l2)
        self.ax1.set_title('World')
        t3, t4 = robot.inverse(goal[0], goal[1])
        start_deg = (np.rad2deg(t1), np.rad2deg(t2))
        goal_deg = (np.rad2deg(t3), np.rad2deg(t4))

        plot_cspace(self.ax2, cspace, start_deg, goal_deg, 'Configuration Space')
        
        self.path = path

        self.world_lines = []
        self.cspace_lines = []

        for j in range(len(self.path)):
            line1 = mlines.Line2D([], [], color='g', marker='.')
            line2 = mlines.Line2D([], [], color='g', marker='.')
            self.ax1.add_line(line1)
            self.ax2.add_line(line2)
            self.world_lines.append(line1)
            self.cspace_lines.append(line2)
        
        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True)

    def new_frame_seq(self):
        return iter(range(len(self.path)))

    def _draw_frame(self, framedata):
        i = framedata
        if i == 0:
            self._init_draw()
        t1, t2 = self.path[i]
        
        l1, l2 = robot.get_links(np.deg2rad(t1), np.deg2rad(t2))

        self.l1.set_data([l1[0][0], l1[1][0]],[l1[0][1], l1[1][1]])
        self.l2.set_data([l2[0][0], l2[1][0]],[l2[0][1], l2[1][1]])
        
        self.world_lines[i].set_data([l2[1][0]],[l2[1][1]])
        self.cspace_lines[i].set_data([t1], [t2])

    def _init_draw(self):
        self.l1.set_data([], [])
        self.l2.set_data([], [])
        lines = self.world_lines + self.cspace_lines
        for line in lines:
            line.set_data([], [])


if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    
    robot = Robot(args['base'], args['l1'], args['l2'])
    c = args['circle']
    circle = [np.array([c[0], c[1]]), c[2]]
    cspace = configuration_space(robot, circle)
    t1, t2 = robot.inverse(args['start'][0], args['start'][1])

    t1 = np.rad2deg(t1)
    t2 = np.rad2deg(t2)
    start = np.array([t1, t2])
    
    t1, t2 = robot.inverse(args['goal'][0], args['goal'][1])
    t1 = np.rad2deg(t1)
    t2 = np.rad2deg(t2)
    goal = np.array([t1, t2])
    
    pf = PotentialField(start, goal, cspace)
    att = pf.get_attractive(ksi=args['ksi'], d=args['d'])
    rep = pf.get_repulsive(s=args['sigma'])
    total = att + rep

    fig = plt.figure()
    ax = plt.subplot2grid((2,1), (0, 0), projection='3d')
    plot_surface(ax, att, start, goal, 'Attractive Field')
    ax = plt.subplot2grid((2,1), (1, 0))
    plot_contour(ax, att, start, goal, 'Attractive Field')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.figure()
    ax2 = plt.subplot2grid((2, 1), (0, 0), projection='3d')
    plot_surface(ax2, rep, start, goal, 'Repulsive Field')
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    plot_contour(ax2, rep, start, goal, 'Repulsive Field')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.figure()
    ax3 = plt.subplot2grid((2,1), (0,0), projection='3d')
    plot_surface(ax3, total, start, goal, 'Potential Field')
    ax3 = plt.subplot2grid((2,1), (1,0))
    plot_contour(ax3, total, start, goal, 'Potential Field')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    
    path = gradient_descent(total, start, goal, lr=args['lr'])
    ani = SubplotAnimation(robot, circle, cspace, args['start'], args['goal'], path, 'Potential Field Approach')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if args['movie']:
        ani.save('sol_potential_field.mp4', writer = 'avconv', fps=10)
    plt.show()
    
    path_finder = Astar(start, goal, cspace)
    path = path_finder.search()
    ani = SubplotAnimation(robot, circle, cspace, args['start'], args['goal'], path, 'A Star Path Finding')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if args['movie']:
        ani.save('sol_astar.mp4', writer = 'avconv', fps=10)
    plt.show()
