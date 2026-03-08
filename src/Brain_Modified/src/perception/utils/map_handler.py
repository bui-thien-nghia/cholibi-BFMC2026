import networkx as nx
import numpy as np
import os

class MapHandler:
    """
    Handles loading the GraphML track map and performing pathfinding/localization lookups.
    """
    def __init__(self, map_path='Misc/Competition_track_graph.graphml'):
        if not os.path.exists(map_path):
            print(f"[ERROR] MapHandler: Map file not found at {map_path}")
            self.G = None
            return
            
        # Load the graph
        self.G = nx.read_graphml(map_path)
        
        # Build a fast lookup for node coordinates
        self.nodes_data = {}
        for node, data in self.G.nodes(data=True):
            # Convert string coordinates from graphml to floats
            # Use 'x'/'y' (attr.name) or 'd0'/'d1' (id) as fallbacks
            x = float(data.get('x', data.get('d0', 0)))
            y = float(data.get('y', data.get('d1', 0)))
            self.nodes_data[node] = np.array([x, y])
            
        print(f"[INFO] MapHandler: Loaded {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")

    def find_nearest_node(self, x, y):
        """
        Finds the ID of the node closest to the given global coordinates.
        """
        if self.G is None: return None
        
        pos = np.array([x, y])
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, node_pos in self.nodes_data.items():
            dist = np.linalg.norm(pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
                
        return nearest_node

    def get_path(self, start_node, end_node):
        """
        Calculates the shortest path waypoints between two nodes.
        Returns: List of [x, y] coordinates.
        """
        if self.G is None: return []
        
        try:
            # Note: We use string IDs because read_graphml loads IDs as strings
            path_ids = nx.shortest_path(self.G, source=str(start_node), target=str(end_node))
            waypoints = [self.nodes_data[node_id] for node_id in path_ids]
            return waypoints
        except nx.NetworkXNoPath:
            print(f"[WARNING] MapHandler: No path found between {start_node} and {end_node}")
            return []
        except Exception as e:
            print(f"[ERROR] MapHandler pathfinding error: {e}")
            return []

    def get_node_pose(self, node_id):
        """Returns the [x, y] of a specific node."""
        return self.nodes_data.get(str(node_id))
