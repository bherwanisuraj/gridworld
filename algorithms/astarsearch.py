import heapq

class Node:

    def __int__(self, position, name, parent = None):
        self.name = name
        self.position = position
        self.parent = parent
        self.neighbours = []
        self.g = 0
        self.h = 0
        self.f = 0

    def addN(self, v):
        self.neighbours.append(v)

    def __lt__(self, other_node):
        return self.f < other_node.f

    def __repr__(self):
        self.name


class Edge:
    def __int__(self, weight, target):
        self.weight = weight
        self.target = target

class SearchAlgo:
    pass