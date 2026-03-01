import networkx as nx
import math

class CarSim:
    def __init__(self, graph_file, speed=0.3, init_x=0, init_y=0, init_yaw=0):
        self.graph = nx.read_graphml(graph_file)
        self.path = []

        self.speed = speed # m/s
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.dets = []

    def find_path(self, visit_points):
        self.path = []
        for src, dst in list(zip(visit_points, visit_points[1:])):
            path = nx.dijkstra_path(self.graph, src, dst)
            self.path += list(zip(path, path[1:]))

    def update(self):
        self.x = self.graph.nodes[self.path[0][0]]['x']
        self.y = self.graph.nodes[self.path[0][0]]['y']
        self.yaw = math.atan2(
            self.graph.nodes[self.path[0][1]]['y'] - self.graph.nodes[self.path[0][0]]['y'],
            self.graph.nodes[self.path[0][1]]['x'] - self.graph.nodes[self.path[0][0]]['x']
        )
        
        # Move along the edge
        edge_length = math.sqrt(
            (self.graph.nodes[self.path[0][1]]['x'] - self.graph.nodes[self.path[0][0]]['x'])**2
            +
            (self.graph.nodes[self.path[0][1]]['y'] - self.graph.nodes[self.path[0][0]]['y'])**2
        )
        if edge_length > 0:
            self.x += self.speed * math.cos(self.yaw) / edge_length
            self.y += self.speed * math.sin(self.yaw) / edge_length
        
        # Check if we reached the end of the edge
        if math.sqrt((self.x - self.graph.nodes[self.path[0][1]]['x'])**2 + (self.y - self.graph.nodes[self.path[0][1]]['y'])**2) < self.speed:
            self.path.pop(0)