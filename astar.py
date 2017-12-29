import numpy as np
from sets import Set

class Node:
    def __init__(self, loc):
        self.parent = None
        self.loc = loc
        self.f = 0.
        self.g = 0.
        self.h = 0.

    def __eq__(self, other):    
        return np.array_equal(self.loc, other.loc)

    def __hash__(self):
        return hash((self.loc[0], self.loc[1]))

    def neighbours(self, cspace, goal):
        nbs = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        
        available = []
        for nx, ny in nbs:
            if nx == 0 and ny == 0:
                continue
            cx = self.loc[0] + nx
            cy = self.loc[1] + ny
            #check whether node is inside the frame
            # and it is not obstacle
            if cx < 0 or cx > 180 or cy < 0 or cy > 360 or cspace[cx, cy] == 1:
                continue
            else:
                n = Node(np.array([cx, cy]))
                n.parent = self
                n.g = self.g + np.sqrt(nx**2 + ny**2)
                n.h = np.linalg.norm(n.loc - goal)
                n.f = n.g + n.h
                available.append(n)
        return available

class Astar:
    def __init__(self, start, goal, cspace):
        """
        start and goal are in degree
        """
        self.real_start = start
        self.real_goal = goal
        self.start = np.array([int(round(start[0])), int(round(start[1]))])
        self.goal = np.array([int(round(goal[0])), int(round(goal[1]))])
        self.cspace = cspace

    def search(self):
        openset = Set()
        closedset = Set()

        start = Node(self.start)
        start.h = np.linalg.norm(start.loc - self.goal)
        openset.add(start)

        final_node = None
        reached = False
        while openset and not reached:
            q = min(openset, key=lambda o:o.f)
            openset.remove(q)
            final_node = q
            successors = q.neighbours(self.cspace, self.goal)
            for successor in successors:
                if np.array_equal(successor.loc, self.goal):
                    final_node = successor
                    reached = True
                    break
                if successor in closedset:
                    continue
                if successor in openset:
                    for s in openset:
                        if np.array_equal(s.loc, successor.loc) and successor.f < s.f:
                            s.parent = successor.parent
                            s.g = successor.g
                            s.h = successor.parent
                            s.f = successor.parent
                            break
                else:
                    openset.add(successor)
            closedset.add(q)
        
        path = [self.real_start, self.real_goal]
        while final_node.parent != None:
            path.insert(1, final_node.loc)
            final_node = final_node.parent
        return path
