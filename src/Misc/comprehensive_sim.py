"""
TODO:
- Add sprite for traffic signs
"""

import sys
import os
import math
import json
from collections import deque
from enum import Enum
import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon, Wedge
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available")

sys.path.insert(0, os.path.abspath('../Computer/src/servers/trafficCommunicationServer/Useful'))
sys.path.insert(0, os.path.abspath('../Brain/src/statemachine'))
sys.path.insert(0, os.path.abspath('../Perception/final'))

try:
    import networkx as nx
except ImportError as e:
    print(f"[ERROR] networkx required: {e}")
    sys.exit(1)

try:
    from modeChanger import StateChanger
    from systemMode import SystemModeRebuilt
    HAS_MODE_CHANGER = True
    print('Using custom StateChanger and SystemModeRebuilt')
except ImportError:
    HAS_MODE_CHANGER = False
    print("[WARNING] modeChanger or systemMode not available, using fallback StateChanger with basic logic")
    class StateChanger:
        LANE_KEEPING_NORMAL = {'mode': 'lane_keeping_normal'}
        LANE_KEEPING_SLOW = {'mode': 'lane_keeping_slow'}
        LANE_KEEPING_FAST = {'mode': 'lane_keeping_fast'}
        STOP = {'mode': 'stop'}
        TURN = {'mode': 'turn'}
        OVERTAKING = {'mode': 'overtaking'}
        TAILING = {'mode': 'tailing'}
        PARKING = {'mode': 'parking'}

        """Fallback StateChanger - provides basic mode transitions based on detections"""
        def __init__(self):
            self.cur_state = None
            self.last_indices = []
            self.last_boxes = []
        
        def record_detection(self, indices, boxes):
            """Store detection indices and boxes"""
            self.last_indices = indices if indices else []
            self.last_boxes = boxes if boxes else []
        
        def change_state(self):
            """Determine new state based on stored detections - HIGH to LOW priority"""
            # Priority: red_light/stop_sign > pedestrian/cyclist > vehicles > intersection > yellow_light > green_light
            if not self.last_indices:
                self.cur_state = None
                return
            
            # HIGHEST PRIORITY: Must stop immediately
            if any(idx == 5 for idx in self.last_indices):  # red_light
                self.cur_state = 'STOP'
                return
            if any(idx == 16 for idx in self.last_indices):  # stop_sign
                self.cur_state = 'STOP'
                return
            
            # HIGH PRIORITY: Pedestrians and cyclists
            if any(idx == 0 for idx in self.last_indices):  # pedestrian
                self.cur_state = 'STOP'
                return
            if any(idx == 1 for idx in self.last_indices):  # cyclist
                self.cur_state = 'STOP'
                return
            
            # MEDIUM PRIORITY: Other vehicles
            if any(idx in [2, 3, 4] for idx in self.last_indices):  # car, bus, truck
                self.cur_state = 'TAILING'
                return
            
            # MEDIUM-LOW: Intersections
            if any(idx == 9 for idx in self.last_indices):  # intersection
                self.cur_state = 'TURN'
                return
            
            # LOW PRIORITY: Traffic lights (yellow before green)
            if any(idx == 6 for idx in self.last_indices):  # yellow_light
                self.cur_state = 'LANE_KEEPING_SLOW'
                return
            if any(idx == 7 for idx in self.last_indices):  # green_light
                self.cur_state = 'LANE_KEEPING_NORMAL'
                return
            
            # No matching detections
            self.cur_state = None
        
        def _get_state(self):
            """Return current state as SystemModeRebuilt enum"""
            if self.cur_state is None:
                return None
            mode_map = {
                'STOP': SystemModeRebuilt.STOP,
                'LANE_KEEPING_SLOW': SystemModeRebuilt.LANE_KEEPING_SLOW,
                'LANE_KEEPING_FAST': SystemModeRebuilt.LANE_KEEPING_FAST,
                'TAILING': SystemModeRebuilt.TAILING,
                'TURN': SystemModeRebuilt.TURN,
            }
            return mode_map.get(self.cur_state, None)
    print('Using fallback StateChanger with basic logic')



# ============================================================================
# [FEATURE 1] SCENE OBJECT SYSTEM - Traffic signs, lights, and vehicles
# ============================================================================
class TrafficSign:
    """Traffic sign on the map with image support"""
    # Mapping of sign types to image filenames
    IMAGE_MAP = {
        'stop_sign': 'stop_sign.png',
        'pedestrian_sign': 'pedestrian_sign.png',
        'parking_sign': 'parking_sign.png',
        'enter_highway_sign': 'enter_highway_sign.png',
        'leave_highway_sign': 'leave_highway_sign.png',
        'oneway_sign': 'oneway_sign.png',
        'priority_sign': 'priority_sign.png',
        'roundabout_sign': 'roundabout_sign.png',
        'notallowed_sign': 'notallowed_sign.png',
    }
    
    def __init__(self, sign_type, x, y, radius=0.3):
        self.sign_type = sign_type
        self.x = x
        self.y = y
        self.radius = radius
        self.class_id = {
            'pedestrian': 0, 'cyclist': 1, 'car': 2, 'bus': 3, 'truck': 4,
            'red_light': 5, 'yellow_light': 6, 'green_light': 7,
            'stop_sign': 16, 'pedestrian_sign': 8, 'parking_sign': 12, 
            'enter_highway_sign': 9, 'leave_highway_sign': 10,
            'oneway_sign': 11, 'priority_sign': 13,
            'roundabout_sign': 15, 'notallowed_sign': 14
        }.get(sign_type, 0)
        self.image = None
        self._load_image()
    
    def _load_image(self):
        """Load image file if available"""
        if not HAS_MATPLOTLIB:
            return
        img_filename = self.IMAGE_MAP.get(self.sign_type)
        if img_filename:
            img_path = os.path.join(os.path.dirname(__file__), 'traffic_signs', img_filename)
            if os.path.exists(img_path):
                try:
                    from PIL import Image
                    self.image = Image.open(img_path)
                except Exception as e:
                    print(f"[WARNING] Could not load image {img_path}: {e}")
    
    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    # def draw(self, ax):
    #     colors = {
    #         'stop_sign': 'red', 'pedestrian_sign': 'yellow', 'parking_sign': 'green',
    #         'enter_highway_sign': 'blue', 'roundabout_sign': 'orange',
    #         'priority_sign': 'purple', 'oneway_sign': 'brown', 'notallowed_sign': 'black',
    #         'leave_highway_sign': 'gray'
    #     }
    #     color = colors.get(self.sign_type, 'yellow')
        
    #     if self.image:
    #         try:
    #             # Zoom adjusted to match traffic light size (radius 0.2)
    #             imagebox = OffsetImage(self.image, zoom=0.05, alpha=0.8)
    #             ab = AnnotationBbox(imagebox, (self.x, self.y), frameon=False)
    #             ax.add_artist(ab)
    #         except Exception:
    #             # Fallback to circle if image fails
    #             circle = plt.Circle((self.x, self.y), self.radius, color=color, alpha=0.6)
    #             ax.add_patch(circle)
    #     else:
    #         circle = plt.Circle((self.x, self.y), self.radius, color=color, alpha=0.6)
    #         ax.add_patch(circle)
    
    def draw_and_return(self, ax):
        colors = {
            'stop_sign': 'red', 'pedestrian_sign': 'yellow', 'parking_sign': 'green',
            'enter_highway_sign': 'blue', 'roundabout_sign': 'orange',
            'priority_sign': 'purple', 'oneway_sign': 'brown', 'notallowed_sign': 'black',
            'leave_highway_sign': 'gray'
        }
        color = colors.get(self.sign_type, 'yellow')
        
        if self.image:
            try:
                imagebox = OffsetImage(self.image, zoom=0.1, alpha=0.8)
                ab = AnnotationBbox(imagebox, (self.x, self.y), frameon=False)
                ax.add_artist(ab)
                return ab
            except Exception:
                circle = plt.Circle((self.x, self.y), self.radius, color=color, alpha=0.6)
                ax.add_patch(circle)
                return circle
        else:
            circle = plt.Circle((self.x, self.y), self.radius, color=color, alpha=0.6)
            ax.add_patch(circle)
            return circle


class TrafficLight:
    """Traffic light on the map"""
    def __init__(self, x, y, node_id=None, state='green'):
        self.x = x
        self.y = y
        self.node_id = node_id
        self.state = state
        self.timer = 0
        self.durations = {'green': 10, 'yellow': 3, 'red': 8}
        self.radius = 0.2
    
    def update(self, dt):
        self.timer += dt
        if self.timer >= self.durations[self.state]:
            transitions = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}
            self.state = transitions[self.state]
            self.timer = 0
    
    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def draw(self, ax):
        colors = {'red': 'red', 'yellow': 'gold', 'green': 'lime'}
        circle = plt.Circle((self.x, self.y), self.radius, color=colors[self.state], ec='black', linewidth=2)
        ax.add_patch(circle)
    
    def draw_and_return(self, ax):
        colors = {'red': 'red', 'yellow': 'gold', 'green': 'lime'}
        circle = plt.Circle((self.x, self.y), self.radius, color=colors[self.state], ec='black', linewidth=2)
        ax.add_patch(circle)
        return circle


class Intersection:
    """Traffic intersection for scenario testing"""
    def __init__(self, inter_id, x, y):
        self.id = inter_id
        self.x = x
        self.y = y
        self.radius = 0.25
    
    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def draw(self, ax):
        line, = ax.plot(self.x, self.y, marker='*', color='purple', markersize=15, zorder=5)
        return line


class Vehicle:
    """Other vehicles on the map"""
    def __init__(self, vehicle_id, x, y):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.radius = 0.15
        self.vtype = 'car'
    
    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def draw(self, ax):
        rect = mpatches.Rectangle((self.x - self.radius, self.y - self.radius),
                                  self.radius*2, self.radius*2, color='red', alpha=0.7)
        ax.add_patch(rect)
    
    def draw_and_return(self, ax):
        rect = mpatches.Rectangle((self.x - self.radius, self.y - self.radius),
                                  self.radius*2, self.radius*2, color='red', alpha=0.7)
        ax.add_patch(rect)
        return rect


class SceneManager:
    """Manages all scene objects"""
    def __init__(self):
        self.signs = {}
        self.lights = {}
        self.vehicles = {}
        self.intersections = {}
        self.sign_counter = 0
        self.light_counter = 0
        self.vehicle_counter = 0
        self.intersection_counter = 0
    
    def add_sign(self, sign_type, x, y):
        sign_id = f"sign_{self.sign_counter}"
        self.signs[sign_id] = TrafficSign(sign_type, x, y)
        self.sign_counter += 1
        return sign_id
    
    def add_light(self, x, y, state='green'):
        light_id = f"light_{self.light_counter}"
        self.lights[light_id] = TrafficLight(x, y, state=state)
        self.light_counter += 1
        return light_id
    
    def add_vehicle(self, x, y):
        vehicle_id = f"car_{self.vehicle_counter}"
        self.vehicles[vehicle_id] = Vehicle(vehicle_id, x, y)
        self.vehicle_counter += 1
        return vehicle_id
    
    def add_intersection(self, x, y):
        inter_id = f"intersection_{self.intersection_counter}"
        self.intersections[inter_id] = Intersection(inter_id, x, y)
        self.intersection_counter += 1
        return inter_id
    
    def remove_sign(self, sign_id):
        if sign_id in self.signs:
            del self.signs[sign_id]
    
    def remove_light(self, light_id):
        if light_id in self.lights:
            del self.lights[light_id]
    
    def remove_vehicle(self, vehicle_id):
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
    
    def remove_intersection(self, inter_id):
        if inter_id in self.intersections:
            del self.intersections[inter_id]
    
    def update(self, dt):
        for light in self.lights.values():
            light.update(dt)
    
    def get_nearby_signs(self, x, y, radius=2.0):
        nearby = []
        for sign in self.signs.values():
            dist = sign.distance_to(x, y)
            if dist < radius:
                nearby.append((dist, sign))
        return sorted(nearby, key=lambda p: p[0])
    
    def get_nearby_lights(self, x, y, radius=2.0):
        nearby = []
        for light in self.lights.values():
            dist = light.distance_to(x, y)
            if dist < radius:
                nearby.append((dist, light))
        return sorted(nearby, key=lambda p: p[0])
    
    def get_nearby_vehicles(self, x, y, radius=2.0):
        nearby = []
        for vehicle in self.vehicles.values():
            dist = vehicle.distance_to(x, y)
            if dist < radius:
                nearby.append((dist, vehicle))
        return sorted(nearby, key=lambda p: p[0])
    
    def get_nearby_intersections(self, x, y, radius=2.0):
        nearby = []
        for intersection in self.intersections.values():
            dist = intersection.distance_to(x, y)
            if dist < radius:
                nearby.append((dist, intersection))
        return sorted(nearby, key=lambda p: p[0])


# ============================================================================
# [FEATURE 2] FIELD OF VIEW (FOV) - Trapezoid detection area
# ============================================================================
class FieldOfView:
    """Trapezoid FOV rotating with car heading"""
    def __init__(self, width_near=1.0, width_far=1.5, length=2.0):
        self.width_near = width_near
        self.width_far = width_far
        self.length = length
    
    def get_polygon(self, car_x, car_y, car_yaw):
        """Get FOV polygon rotated by car yaw (rotated 90° clockwise)"""
        car_yaw_adjusted = car_yaw - math.pi / 2
        cos_y = math.cos(car_yaw_adjusted)
        sin_y = math.sin(car_yaw_adjusted)
        
        pts_local = [
            (-self.width_near/2, 0),
            (self.width_near/2, 0),
            (self.width_far/2, self.length),
            (-self.width_far/2, self.length)
        ]
        
        pts_global = []
        for x, y in pts_local:
            gx = car_x + x * cos_y - y * sin_y
            gy = car_y + x * sin_y + y * cos_y
            pts_global.append([gx, gy])
        
        return np.array(pts_global)
    
    def contains_point(self, px, py, car_x, car_y, car_yaw):
        """Check if point is inside FOV trapezoid"""
        poly = self.get_polygon(car_x, car_y, car_yaw)
        path = mpatches.Path(poly)
        return path.contains_point([px, py])


# ============================================================================
# [FEATURE 3] CAR SIMULATOR - Enhanced with FOV and scene interaction
# ============================================================================
class ComprehensiveCarSimulator:
    """Main car simulator with scene interaction and FOV"""
    
    def __init__(self, graph_file, start_node, end_node=None, speed=0.5, dt=0.05, scene_manager=None, waypoint_path=None):
        print("[INFO] Initializing Car Simulator...")
        
        try:
            self.graph = nx.read_graphml(graph_file)
            print(f"[SUCCESS] Loaded graph with {self.graph.number_of_nodes()} nodes")
        except Exception as e:
            print(f"[ERROR] Failed to load graph: {e}")
            raise
        
        self.speed = speed
        self.dt = dt
        self.current_time = 0
        self.total_distance = 0
        
        # Support both single path (start_node, end_node) and multi-path (waypoint_path)
        if waypoint_path is not None:
            # Multiple paths: waypoint_path is a list of nodes [n1, n2, n3, n4, ...]
            # Paths are n1->n2, n2->n3, n3->n4, etc.
            self.waypoint_path = [str(n) for n in waypoint_path]
            self.start_node = self.waypoint_path[0]
            self.end_node = self.waypoint_path[-1]
            self.segment_idx = 0  # Current segment (0 = path to waypoint_path[1], 1 = path to waypoint_path[2], etc.)
            print(f"[INFO] Multi-path mode: {len(self.waypoint_path)-1} segments")
        else:
            # Single path mode
            self.waypoint_path = None
            self.start_node = str(start_node)
            self.end_node = str(end_node)
            self.segment_idx = 0
        
        self.path_nodes = []
        self.path_edges = []
        self.current_edge_idx = 0
        self.position_on_edge = 0
        
        # New: Waypoint tracking for path management
        self.waypoint_nodes = []
        self.current_waypoint_idx = 0
        self.visited_waypoints = set()
        self.x = float(self.graph.nodes[self.start_node]['x'])
        self.y = float(self.graph.nodes[self.start_node]['y'])
        self.yaw = 0.0
        self.position_history = deque(maxlen=1000)
        self.position_history.append((self.x, self.y))
        
        self.mode_changer = StateChanger()
        self.current_mode = SystemModeRebuilt.LANE_KEEPING_NORMAL
        self.current_detections = []
        self.lookup_nodes = []  # For turn recognition visualization
        
        # Bezier curve following for TURN mode
        self.is_following_curve = False
        self.curve_points = []  # Bezier curve waypoints
        self.curve_progress = 0  # Current position along curve (0-1)
        self.frozen_lookup_nodes = []  # Frozen nodes when entering TURN mode
        self.turn_curve_start_edge = None  # Edge index where curve following should start (at node 1)
        
        self.scene_manager = scene_manager or SceneManager()
        self.fov = FieldOfView(width_near=0.5, width_far=0.8, length=1.5)
        
        self.stopped = False
        self.stop_reason = ""
        
        self._plan_path()
        print(f"[SUCCESS] Path planned: {len(self.path_nodes)} nodes")
    
    def _plan_path(self):
        """Plan merged path through all segments (if multi-path) or single path"""
        try:
            if self.waypoint_path is not None:
                # Multi-path mode: merge all segments into one path
                print(f"[INFO] Merging {len(self.waypoint_path)-1} segments into one path...")
                merged_path_nodes = []
                for i in range(len(self.waypoint_path) - 1):
                    segment_start = self.waypoint_path[i]
                    segment_end = self.waypoint_path[i + 1]
                    segment_path = nx.dijkstra_path(self.graph, segment_start, segment_end)
                    if i == 0:
                        merged_path_nodes.extend(segment_path)
                    else:
                        merged_path_nodes.extend(segment_path[1:])  # Skip start node to avoid duplicate
                    print(f"  Segment {i+1}: {segment_start} -> {segment_end} ({len(segment_path)} nodes)")
                self.path_nodes = merged_path_nodes
            else:
                # Single path mode
                self.path_nodes = nx.dijkstra_path(self.graph, self.start_node, self.end_node)
                print(f"[INFO] Planning path: {self.start_node} -> {self.end_node}")
            
            self.path_edges = list(zip(self.path_nodes[:-1], self.path_nodes[1:]))
            self.current_edge_idx = 0
            
            # Create waypoints at regular intervals
            waypoint_interval = max(1, len(self.path_nodes) // 10)
            self.waypoint_nodes = [(self.path_nodes[i], i) for i in range(0, len(self.path_nodes), waypoint_interval)]
            if self.path_nodes[-1] not in [w[0] for w in self.waypoint_nodes]:
                self.waypoint_nodes.append((self.path_nodes[-1], len(self.path_nodes)-1))
            self.visited_waypoints = set()
            print(f"[SUCCESS] Path planned: {len(self.path_nodes)} nodes, {len(self.waypoint_nodes)} waypoints")
        except nx.NetworkXNoPath as e:
            print(f"[ERROR] Path planning failed: {e}")
            raise
    
    def _get_current_edge_endpoints(self):
        if self.current_edge_idx >= len(self.path_edges):
            return None, None
        src_node, dst_node = self.path_edges[self.current_edge_idx]
        src_x = float(self.graph.nodes[src_node]['x'])
        src_y = float(self.graph.nodes[src_node]['y'])
        dst_x = float(self.graph.nodes[dst_node]['x'])
        dst_y = float(self.graph.nodes[dst_node]['y'])
        return (src_x, src_y), (dst_x, dst_y)
    
    def _detect_objects(self): # HIGH-LV CTRL
        """[ENHANCEMENT] Detect objects within FOV trapezoid + waypoint tracking"""
        """
        enter highway signs must retain lane keeping fast
        yellow light got lane keeping normal, expected lane keeping slow
        intersection got lane keeping fast, expoected lane keeping normal
        intersection icon does not disappear after deleting
        """
        self.current_detections = []
        indices = []
        boxes = []
        
        # Detect based ONLY on FOV - use larger radius to ensure FOV can catch objects
        nearby_signs = self.scene_manager.get_nearby_signs(self.x, self.y, radius=5.0)
        for dist, sign in nearby_signs:
            if self.fov.contains_point(sign.x, sign.y, self.x, self.y, self.yaw):
                self.current_detections.append(sign.sign_type)
                indices.append(sign.class_id)
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        nearby_lights = self.scene_manager.get_nearby_lights(self.x, self.y, radius=5.0)
        for dist, light in nearby_lights:
            if self.fov.contains_point(light.x, light.y, self.x, self.y, self.yaw):
                det = f"{light.state}_light"
                self.current_detections.append(det)
                light_class_ids = {'red': 5, 'yellow': 6, 'green': 7}
                indices.append(light_class_ids[light.state])
                boxes.append([0.5, 0.5, 0.3, 0.9])
        
        nearby_vehicles = self.scene_manager.get_nearby_vehicles(self.x, self.y, radius=5.0)
        for dist, vehicle in nearby_vehicles:
            if self.fov.contains_point(vehicle.x, vehicle.y, self.x, self.y, self.yaw):
                self.current_detections.append(vehicle.vtype)
                vtype_ids = {'car': 2, 'bus': 3, 'truck': 4, 'cyclist': 1, 'pedestrian': 0}
                indices.append(vtype_ids.get(vehicle.vtype, 2))
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        nearby_intersections = self.scene_manager.get_nearby_intersections(self.x, self.y, radius=5.0)
        for dist, intersection in nearby_intersections:
            if self.fov.contains_point(intersection.x, intersection.y, self.x, self.y, self.yaw):
                self.current_detections.append('intersection')
                indices.append(13)  # intersection as priority_sign class (index 13) for normal lane keeping
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        # Record detections (state will be updated later)
        if indices:
            self.mode_changer.record_detection(indices, boxes)
            # Always print detections when objects are found in FOV
            if self.current_detections:
                det_str = ', '.join(self.current_detections[:3])
                print(f"[DETECTION] {det_str:30s}")
        
        # NEW: Track waypoint visits
        if self.waypoint_nodes and self.current_waypoint_idx < len(self.waypoint_nodes):
            next_wp = self.waypoint_nodes[self.current_waypoint_idx][0]
            next_wp_idx = self.waypoint_nodes[self.current_waypoint_idx][1]
            if next_wp not in self.visited_waypoints and self.current_edge_idx >= next_wp_idx - 2:
                self.visited_waypoints.add(next_wp)
                self.current_waypoint_idx += 1
                print(f"[WAYPOINT] {len(self.visited_waypoints)}/{len(self.waypoint_nodes)} reached")
    
    def _calculate_turn_lookup(self):
        """Calculate and send node degrees and yaw differences for turn recognition"""
        try:
            # Get current node and next 2 nodes (reduced from 4)
            lookup_indices = []
            lookup_nodes = []
            
            # Current node (destination of current edge - where car is heading)
            if self.current_edge_idx < len(self.path_edges):
                current_node = self.path_edges[self.current_edge_idx][1]
                lookup_nodes.append(current_node)
                lookup_indices.append(self.current_edge_idx)
            
            # Next 2 nodes (up to 2 forward)
            for i in range(1, 3):
                if self.current_edge_idx + i < len(self.path_edges):
                    next_node = self.path_edges[self.current_edge_idx + i][1]
                    lookup_nodes.append(next_node)
                    lookup_indices.append(self.current_edge_idx + i)
            
            # Pad with current node if less than 3 nodes
            while len(lookup_nodes) < 3:
                lookup_nodes.append(lookup_nodes[-1])
                lookup_indices.append(lookup_indices[-1])
            
            # Calculate node degrees (number of connections)
            node_degrees = []
            for node in lookup_nodes:
                degree = len(list(self.graph.neighbors(node)))
                node_degrees.append(degree)
            
            # Calculate absolute yaw differences (direction change between edges)
            yaw_diffs = []
            for idx in lookup_indices[:2]:  # Only 2 yaw differences (for 2 lookahead nodes)
                if idx < len(self.path_edges) - 1:
                    # Current edge
                    curr_src, curr_dst = self.path_edges[idx]
                    curr_x1 = float(self.graph.nodes[curr_src]['x'])
                    curr_y1 = float(self.graph.nodes[curr_src]['y'])
                    curr_x2 = float(self.graph.nodes[curr_dst]['x'])
                    curr_y2 = float(self.graph.nodes[curr_dst]['y'])
                    curr_yaw = math.atan2(curr_y2 - curr_y1, curr_x2 - curr_x1)
                    
                    # Next edge
                    next_src, next_dst = self.path_edges[idx + 1]
                    next_x1 = float(self.graph.nodes[next_src]['x'])
                    next_y1 = float(self.graph.nodes[next_src]['y'])
                    next_x2 = float(self.graph.nodes[next_dst]['x'])
                    next_y2 = float(self.graph.nodes[next_dst]['y'])
                    next_yaw = math.atan2(next_y2 - next_y1, next_x2 - next_x1)
                    
                    # Absolute yaw difference in degrees
                    yaw_diff = abs(math.degrees(next_yaw - curr_yaw))
                    # Normalize to 0-180
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    yaw_diffs.append(yaw_diff)
                else:
                    yaw_diffs.append(0)
            
            # Send to mode_changer
            self.mode_changer.record_lookup(node_degrees, yaw_diffs)
            
            # Store for visualization (current + 2 lookahead nodes)
            self.lookup_nodes = lookup_nodes[:3]
            
        except Exception as e:
            # Silently fail if lookup calculation fails
            pass
    
    def _generate_bezier_curve(self, p0, p1, p2, num_points=50):
        """Generate Bezier curve points from 3 control points (quadratic Bezier)"""
        try:
            curve_pts = []
            for i in range(num_points + 1):
                t = i / num_points
                # Quadratic Bezier formula: B(t) = (1-t)²*P0 + 2*(1-t)*t*P1 + t²*P2
                mt = 1 - t
                x = mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0]
                y = mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1]
                curve_pts.append((x, y))
            return curve_pts
        except Exception as e:
            print(f"[ERROR] Bezier curve generation failed: {e}")
            return []
    
    def _get_point_on_curve(self, progress):
        """Get position on curve based on progress (0-1)"""
        if not self.curve_points or progress < 0 or progress > 1:
            return None
        idx = min(int(progress * len(self.curve_points)), len(self.curve_points) - 1)
        return self.curve_points[idx]
    
    def _calculate_curve_length(self):
        """Calculate total length of Bezier curve"""
        if len(self.curve_points) < 2:
            return 1.0
        total_length = 0.0
        for i in range(len(self.curve_points) - 1):
            p1 = self.curve_points[i]
            p2 = self.curve_points[i + 1]
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += dist
        return max(total_length, 0.1)  # Avoid division by zero
    
    def _update_state(self):
        """Update state based on recorded detections and lookups"""
        # Always call change_state to evaluate current detections
        self.mode_changer.change_state()
        new_mode = self.mode_changer._get_state()
        if new_mode is not None:
            self.current_mode = new_mode
        else:
            self.current_mode = SystemModeRebuilt.LANE_KEEPING_NORMAL
        
        # Handle TURN mode: freeze lookup and generate Bezier curve
        mode_str = ''
        if self.current_mode is not None and hasattr(self.current_mode, 'value'):
            mode_str = self.current_mode.value.get('mode', '').lower()
        
        # Enter TURN mode: generate Bezier curve with current node + 2 lookahead nodes, start immediately
        if mode_str == 'turn' and not self.is_following_curve:
            self.frozen_lookup_nodes = self.lookup_nodes.copy()
            if len(self.frozen_lookup_nodes) >= 3:  # Need current + 2 lookahead nodes
                # Bezier control points: p0=current node, p1=lookahead node 1, p2=lookahead node 2
                current_node = self.frozen_lookup_nodes[0]   # Current node
                p1_node = self.frozen_lookup_nodes[1]        # Lookahead node 1
                p2_node = self.frozen_lookup_nodes[2]        # Lookahead node 2
                
                try:
                    # Get current car position as p0
                    p0 = (self.x, self.y)
                    p1 = (float(self.graph.nodes[p1_node]['x']), float(self.graph.nodes[p1_node]['y']))
                    p2 = (float(self.graph.nodes[p2_node]['x']), float(self.graph.nodes[p2_node]['y']))
                    
                    self.curve_points = self._generate_bezier_curve(p0, p1, p2, num_points=100)
                    # Start curve following immediately
                    self.is_following_curve = True
                    self.curve_progress = 0
                    print(f"[TURN] Bezier curve generated with {len(self.curve_points)} points, starting immediately")
                except Exception as e:
                    print(f"[ERROR] Failed to generate Bezier curve: {e}")
        
        # Exit TURN mode
        elif mode_str != 'turn' and self.is_following_curve:
            self.is_following_curve = False
            self.curve_points = []
            self.curve_progress = 0
            self.frozen_lookup_nodes = []
            self.lookup_nodes = []
            print(f"[TURN] Exited curve following mode")
        
        # Print mode change when detections exist
        if self.current_detections:
            mode_str_upper = self.get_state()['mode'].upper()
            print(f"  → MODE: {mode_str_upper}")
    
    def update(self): # LOW-LV CTRL
        """Update simulation by one timestep - timer always advances"""
        # Timer always advances, even when stopped
        self.current_time += self.dt
        
        if self.current_edge_idx >= len(self.path_edges):
            self.stopped = True
            self.stop_reason = "Reached destination"
            return
        
        self._detect_objects()
        
        # Only update lookup if not following a curve
        if not self.is_following_curve:
            self._calculate_turn_lookup()  # Calculate turn recognition data
        
        self._update_state()  # Update state based on detections and lookups
        
        mode_str = ''
        if self.current_mode is not None and hasattr(self.current_mode, 'value'):
            mode_str = self.current_mode.value.get('mode', '').lower()
        
        # Handle movement - either follow curve or normal path
        if self.is_following_curve:
            # === CURVE FOLLOWING LOGIC - Adjust here as needed ===
            # Following Bezier curve from current node through 2 lookahead nodes
            target_pos = self._get_point_on_curve(self.curve_progress)
            if target_pos:
                self.x, self.y = target_pos
                # Progress increment: controls curve following speed
                self.curve_progress += (self.speed * self.dt) / self._calculate_curve_length()
                
                # Check if reached end of curve
                if self.curve_progress >= 1.0:
                    self.is_following_curve = False
                    self.curve_points = []
                    self.frozen_lookup_nodes = []
                    self.lookup_nodes = []
                    self.curve_progress = 0
                    # Skip 1 edge to position at next node after curve end
                    self.current_edge_idx += 3
                    self.position_on_edge = 0
                    print(f"[TURN] Reached end of curve, resuming normal navigation at edge {self.current_edge_idx}")
                
                # Update yaw to face direction of curve
                if self.curve_progress < 1.0:
                    next_pos = self._get_point_on_curve(min(self.curve_progress + 0.01, 1.0))
                    if next_pos:
                        self.yaw = math.atan2(next_pos[1] - self.y, next_pos[0] - self.x)
            
            self.total_distance += self.speed * self.dt
            self.position_history.append((self.x, self.y))
            self.scene_manager.update(self.dt)
            return
        
        # Normal path following
        
        # if 'stop' in mode_str or 'stop_sign' in self.current_detections or 'red_light' in self.current_detections:
        #     self.stop_reason = f"Detection: {', '.join(self.current_detections[:2]) if self.current_detections else 'stop_mode'}"
        #     self.scene_manager.update(self.dt)
        #     self.position_history.append((self.x, self.y))
        #     return
        
        start_pt, end_pt = self._get_current_edge_endpoints()
        if not (start_pt and end_pt):
            self.stopped = True
            self.stop_reason = "Path error"
            return
        
        edge_length = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
        
        effective_speed = self.speed
        if 'slow' in mode_str:
            effective_speed = self.speed * 0.5
        elif 'fast' in mode_str:
            effective_speed = self.speed * 1.5
        elif 'stop' in mode_str:
            effective_speed = 0
        
        movement = (effective_speed * self.dt) / edge_length if edge_length > 0 else 0
        self.position_on_edge += movement

        self.x = start_pt[0] + self.position_on_edge * (end_pt[0] - start_pt[0])
        self.y = start_pt[1] + self.position_on_edge * (end_pt[1] - start_pt[1])
        
        if edge_length > 0:
            self.yaw = math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
        
        self.total_distance += effective_speed * self.dt
        
        if self.position_on_edge >= 1.0:
            self.current_edge_idx += 1
            self.position_on_edge = 0
        
        self.position_history.append((self.x, self.y))
        self.scene_manager.update(self.dt)
    
    def get_state(self):
        mode_str = 'lane_keeping_normal'
        if self.current_mode is not None:
            if hasattr(self.current_mode, 'mode'):
                mode_str = self.current_mode.mode.lower()
            elif hasattr(self.current_mode, 'value') and isinstance(self.current_mode.value, dict):
                mode_str = self.current_mode.value.get('mode', 'lane_keeping_normal').lower()
        
        return {
            'x': self.x,
            'y': self.y,
            'yaw': math.degrees(self.yaw),
            'mode': mode_str,
            'position': (self.x, self.y),
            'time': self.current_time,
            'distance': self.total_distance,
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'detections': self.current_detections,
            'fov_polygon': self.fov.get_polygon(self.x, self.y, self.yaw)
        }


# ============================================================================
# [FEATURE 4] INTERACTIVE VISUALIZATION - Click to add/remove objects
# ============================================================================
class InteractiveVisualizer:
    """Interactive visualization with mouse controls + playback features"""
    
    def __init__(self, sim, graph):
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib required")
        
        self.sim = sim
        self.graph = graph
        self.paused = False
        self.speed_multiplier = 1.0
        self.mode = 'normal'  # normal, add_red_light, add_yellow_light, add_green_light, add_sign, add_vehicle, add_intersection, remove, add_path_node
        
        # Two-panel layout: map on left, info on right
        self.fig, (self.ax_main, self.ax_info) = plt.subplots(
            1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [3, 1]}
        )
        
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('BFMC Simulation - Interactive Path Following\nSPACE:Play/Pause BKSP:Restart | R/Y/G:Lights B:Sign C:Car I:Intersection U:Path DEL:Remove', fontsize=10, weight='bold')
        self.ax_main.grid(True, alpha=0.2)
        
        # Cache graph plotting (only draw once)
        self._plot_graph()
        self.graph_plotted = True
        
        self.car_marker, = self.ax_main.plot([], [], 'bs', markersize=12, label='Car', zorder=10)
        self.fov_patch = None
        self.trajectory_line, = self.ax_main.plot([], [], 'b-', alpha=0.3, linewidth=1, zorder=2)
        
        # Store patches and lines for proper cleanup
        self.drawn_patches = []
        self.drawn_lines = []
        self.drawn_texts = []
        
        # Info panel on right (turned off axis)
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=8, verticalalignment='top', family='monospace')
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        print("[INFO] Controls: R=RedLight Y=YellowLight G=GreenLight B=SignMenu C=Car I=Intersection U=Path DEL=Remove SPACE=Pause BKSP=Restart H=Help")
    
    def _plot_graph(self):
        for node_id in self.graph.nodes():
            x = float(self.graph.nodes[node_id]['x'])
            y = float(self.graph.nodes[node_id]['y'])
            self.ax_main.plot(x, y, 'k.', markersize=6, alpha=0.4, zorder=1)
        
        for edge in self.graph.edges():
            src_x = float(self.graph.nodes[edge[0]]['x'])
            src_y = float(self.graph.nodes[edge[0]]['y'])
            dst_x = float(self.graph.nodes[edge[1]]['x'])
            dst_y = float(self.graph.nodes[edge[1]]['y'])
            self.ax_main.plot([src_x, dst_x], [src_y, dst_y], 'k-', alpha=0.25, linewidth=1.5, zorder=0)
    
    def _on_key(self, event):
        if event.key == 'r':
            self.mode = 'add_red_light'
            print("[MODE] Add Red Light - Click to place")
        elif event.key == 'y':
            self.mode = 'add_yellow_light'
            print("[MODE] Add Yellow Light - Click to place")
        elif event.key == 'g':
            self.mode = 'add_green_light'
            print("[MODE] Add Green Light - Click to place")
        elif event.key == 'b':
            self.mode = 'select_sign'
            print(f"[MODE] Select Traffic Sign Type:")
            print(f"  1=Stop, 2=Pedestrian, 3=Parking, 4=Enter Highway, 5=Leave Highway")
            print(f"  6=OneWay, 7=Priority, 8=Roundabout, 9=Not Allowed")
        elif event.key == '1':
            if self.mode == 'select_sign':
                self.mode = 'add_stop_sign'
                print(f"[MODE] Add Stop Sign - Click to place")
        elif event.key == '2':
            if self.mode == 'select_sign':
                self.mode = 'add_pedestrian_sign'
                print(f"[MODE] Add Pedestrian Sign - Click to place")
        elif event.key == '3':
            if self.mode == 'select_sign':
                self.mode = 'add_parking_sign'
                print(f"[MODE] Add Parking Sign - Click to place")
        elif event.key == '4':
            if self.mode == 'select_sign':
                self.mode = 'add_enter_highway_sign'
                print(f"[MODE] Add Enter Highway Sign - Click to place")
        elif event.key == '5':
            if self.mode == 'select_sign':
                self.mode = 'add_leave_highway_sign'
                print(f"[MODE] Add Leave Highway Sign - Click to place")
        elif event.key == '6':
            if self.mode == 'select_sign':
                self.mode = 'add_oneway_sign'
                print(f"[MODE] Add One Way Sign - Click to place")
        elif event.key == '7':
            if self.mode == 'select_sign':
                self.mode = 'add_priority_sign'
                print(f"[MODE] Add Priority Sign - Click to place")
        elif event.key == '8':
            if self.mode == 'select_sign':
                self.mode = 'add_roundabout_sign'
                print(f"[MODE] Add Roundabout Sign - Click to place")
        elif event.key == '9':
            if self.mode == 'select_sign':
                self.mode = 'add_notallowed_sign'
                print(f"[MODE] Add Not Allowed Sign - Click to place")
        elif event.key == 'c':
            self.mode = 'add_vehicle'
            print("[MODE] Add Vehicle - Click to place")
        elif event.key == 'i':
            self.mode = 'add_intersection'
            print("[MODE] Add Intersection - Click to place")
        elif event.key == 'u':
            self.mode = 'add_path_node'
            print("[MODE] Add/Remove Path Nodes - Click to add, click again to remove")
        elif event.key == 'delete':
            self.mode = 'remove' if self.mode != 'remove' else 'normal'
            print(f"[MODE] {'Remove mode ACTIVE' if self.mode == 'remove' else 'Normal mode'}")
        elif event.key == ' ':
            self.paused = not self.paused
            print(f"[PLAYBACK] {'PAUSED' if self.paused else 'PLAYING'}")
        elif event.key == 'backspace':
            self.sim.stopped = False
            self.sim.current_time = 0
            self.sim.total_distance = 0
            self.sim.position_history.clear()
            self.sim.current_edge_idx = 0
            self.sim.position_on_edge = 0
            self.sim.current_waypoint_idx = 0
            self.sim.visited_waypoints.clear()
            start_x = float(self.graph.nodes[self.sim.path_nodes[0]]['x'])
            start_y = float(self.graph.nodes[self.sim.path_nodes[0]]['y'])
            self.sim.x = start_x
            self.sim.y = start_y
            self.paused = False
            print("[RESTART] Simulation reset")
        elif event.key == '+' or event.key == '=':
            self.speed_multiplier = min(5.0, self.speed_multiplier + 0.5)
            old_speed = self.sim.speed
            self.sim.speed = old_speed * 1.5
            print(f"[SPEED] {self.sim.speed:.2f} units/sec")
        elif event.key == '-' or event.key == '_':
            if self.speed_multiplier > 0.5:
                old_speed = self.sim.speed
                self.sim.speed = max(0.1, old_speed * 0.67)
                self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)
                print(f"[SPEED] {self.sim.speed:.2f} units/sec")
        elif event.key == 'h':
            print("[CONTROLS] R/Y/G:Lights B:SignMenu(then 1-9) C:Car I:Intersection U:Path +-:Speed SPACE:Pause BKSP:Restart DEL:Remove Q:Quit")
        elif event.key == 'q':
            plt.close(self.fig)
    
    def _on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        if self.mode == 'remove':
            self._remove_nearby_object(event.xdata, event.ydata)
            return
        
        elif self.mode == 'add_path_node':
            # Check if clicking on existing path node to remove
            for i, node_id in enumerate(self.sim.path_nodes):
                try:
                    nx = float(self.graph.nodes[node_id]['x'])
                    ny = float(self.graph.nodes[node_id]['y'])
                    dist = math.sqrt((nx - event.xdata)**2 + (ny - event.ydata)**2)
                    if dist < 0.5:
                        self.sim.path_nodes.pop(i)
                        self._recalculate_path()
                        print(f"✓ Path node removed. Path now {len(self.sim.path_nodes)} nodes")
                        return
                except:
                    pass
            print(f"[INFO] Click on waypoint to remove, or click elsewhere to add")
            return
        
        elif self.mode == 'add_red_light':
            self.sim.scene_manager.add_light(event.xdata, event.ydata, 'red')
            print(f"✓ Red light at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
        
        elif self.mode == 'add_yellow_light':
            self.sim.scene_manager.add_light(event.xdata, event.ydata, 'yellow')
            print(f"✓ Yellow light at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
        
        elif self.mode == 'add_green_light':
            self.sim.scene_manager.add_light(event.xdata, event.ydata, 'green')
            print(f"✓ Green light at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
        
        elif self.mode.startswith('add_') and self.mode.endswith('_sign'):
            sign_type = self.mode.replace('add_', '')
            self.sim.scene_manager.add_sign(sign_type, event.xdata, event.ydata)
            sign_display = sign_type.replace('_', ' ').title()
            print(f"✓ {sign_display} at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
        
        elif self.mode == 'add_vehicle':
            self.sim.scene_manager.add_vehicle(event.xdata, event.ydata)
            print(f"✓ Vehicle at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
        
        elif self.mode == 'add_intersection':
            self.sim.scene_manager.add_intersection(event.xdata, event.ydata)
            print(f"✓ Intersection at ({event.xdata:.2f}, {event.ydata:.2f})")
            self.mode = 'normal'
    
    def _recalculate_path(self):
        """Recalculate path and waypoints after modification"""
        if len(self.sim.path_nodes) >= 2:
            try:
                self.sim.path_edges = [(self.sim.path_nodes[i], self.sim.path_nodes[i+1]) 
                                       for i in range(len(self.sim.path_nodes)-1)]
                waypoint_interval = max(1, len(self.sim.path_nodes) // 10)
                self.sim.waypoint_nodes = [(self.sim.path_nodes[i], i) 
                                           for i in range(0, len(self.sim.path_nodes), waypoint_interval)]
                if self.sim.path_nodes[-1] not in [w[0] for w in self.sim.waypoint_nodes]:
                    self.sim.waypoint_nodes.append((self.sim.path_nodes[-1], len(self.sim.path_nodes)-1))
                self.sim.current_edge_idx = 0
                self.sim.position_on_edge = 0
            except Exception as e:
                print(f"[ERROR] Path recalculation failed: {e}")
    
    def _remove_nearby_object(self, x, y, radius=0.5):
        for sign_id, sign in list(self.sim.scene_manager.signs.items()):
            if sign.distance_to(x, y) < radius:
                self.sim.scene_manager.remove_sign(sign_id)
                print(f"✓ Sign removed")
                return
        
        for light_id, light in list(self.sim.scene_manager.lights.items()):
            if light.distance_to(x, y) < radius:
                self.sim.scene_manager.remove_light(light_id)
                print(f"✓ Light removed")
                return
        
        for car_id, car in list(self.sim.scene_manager.vehicles.items()):
            if car.distance_to(x, y) < radius:
                self.sim.scene_manager.remove_vehicle(car_id)
                print(f"✓ Vehicle removed")
                return
        
        for inter_id, inter in list(self.sim.scene_manager.intersections.items()):
            if inter.distance_to(x, y) < radius:
                self.sim.scene_manager.remove_intersection(inter_id)
                print(f"✓ Intersection removed")
                return
        
        print(f"[INFO] No objects found at ({x:.2f}, {y:.2f})")
    
    def update_frame(self, frame):
        # Only update if not paused
        if not self.paused:
            self.sim.update()
        
        state = self.sim.get_state()
        
        # Update car marker only (minimal updates)
        self.car_marker.set_data([state['x']], [state['y']])
        
        # Update FOV patch
        if self.fov_patch:
            self.fov_patch.remove()
        fov_poly = state['fov_polygon']
        self.fov_patch = Polygon(fov_poly, alpha=0.05, color='green', zorder=9)
        self.ax_main.add_patch(self.fov_patch)
        
        # Update trajectory line
        if len(self.sim.position_history) > 1:
            hist = list(self.sim.position_history)
            xs = [pt[0] for pt in hist]
            ys = [pt[1] for pt in hist]
            self.trajectory_line.set_data(xs, ys)
        
        # Remove previously drawn objects (patches, lines, texts)
        for patch in self.drawn_patches:
            patch.remove()
        self.drawn_patches.clear()
        
        for line in self.drawn_lines:
            line.remove()
        self.drawn_lines.clear()
        
        for text in self.drawn_texts:
            text.remove()
        self.drawn_texts.clear()
        
        # Draw scene objects
        for sign in self.sim.scene_manager.signs.values():
            patch = sign.draw_and_return(self.ax_main)
            if patch:
                self.drawn_patches.append(patch)
        for light in self.sim.scene_manager.lights.values():
            patch = light.draw_and_return(self.ax_main)
            if patch:
                self.drawn_patches.append(patch)
        for vehicle in self.sim.scene_manager.vehicles.values():
            patch = vehicle.draw_and_return(self.ax_main)
            if patch:
                self.drawn_patches.append(patch)
        for intersection in self.sim.scene_manager.intersections.values():
            marker = intersection.draw(self.ax_main)
            if marker:
                self.drawn_patches.append(marker)
        
        # Draw waypoints with bright colors (cyan for unvisited, lime for visited)
        for wp_node, wp_idx in self.sim.waypoint_nodes:
            try:
                wp_x = float(self.graph.nodes[wp_node]['x'])
                wp_y = float(self.graph.nodes[wp_node]['y'])
                
                if wp_node in self.sim.visited_waypoints:
                    line, = self.ax_main.plot(wp_x, wp_y, marker='o', color='lime', markersize=7, zorder=11, markeredgecolor='darkgreen', markeredgewidth=0.5)
                    self.drawn_lines.append(line)
                else:
                    line, = self.ax_main.plot(wp_x, wp_y, marker='o', color='cyan', markersize=7, zorder=11, markeredgecolor='blue', markeredgewidth=0.5)
                    self.drawn_lines.append(line)
                    text = self.ax_main.text(wp_x + 0.15, wp_y + 0.15, str(wp_idx), fontsize=7, color='cyan', weight='bold', zorder=12)
                    self.drawn_texts.append(text)
            except:
                pass
        
        # Draw lookup nodes for turn recognition (circles matching waypoint style)
        for i, lookup_node in enumerate(self.sim.lookup_nodes):
            try:
                node_x = float(self.graph.nodes[lookup_node]['x'])
                node_y = float(self.graph.nodes[lookup_node]['y'])
                
                if i == 0:
                    # Current node - darker red
                    color = 'darkred'
                    edge_color = 'darkred'
                else:
                    # Lookahead nodes - bright red
                    color = 'red'
                    edge_color = 'darkred'
                
                line, = self.ax_main.plot(node_x, node_y, marker='o', color=color, markersize=7, 
                                         zorder=10, markeredgecolor=edge_color, markeredgewidth=0.5)
                self.drawn_lines.append(line)
            except:
                pass
        
        # Draw Bezier curve if following
        if self.sim.is_following_curve and self.sim.curve_points:
            try:
                curve_xs = [pt[0] for pt in self.sim.curve_points]
                curve_ys = [pt[1] for pt in self.sim.curve_points]
                line, = self.ax_main.plot(curve_xs, curve_ys, color='purple', linewidth=1.5, 
                                         alpha=0.6, zorder=5, linestyle='--')
                self.drawn_lines.append(line)
            except:
                pass
        
        mode_colors = {
            'lane_keeping_normal': 'blue', 'lane_keeping_slow': 'orange', 'lane_keeping_fast': 'red',
            'stop': 'darkred', 'turn': 'purple', 'overtaking': 'cyan', 'tailing': 'magenta', 'parking': 'green'
        }
        mode_color = mode_colors.get(state['mode'], 'black')
        
        status_str = 'PAUSED' if self.paused else ('STOPPED' if state['stopped'] else 'RUNNING')
        waypoint_progress = f"{len(self.sim.visited_waypoints)}/{len(self.sim.waypoint_nodes)}"
        
        info_text = f"""MODE: {state['mode'].upper()}
Time: {state['time']:.1f}s | Dist: {state['distance']:.2f}m
Speed: {self.sim.speed:.2f} u/s

Pos: ({state['x']:.2f}, {state['y']:.2f})
Yaw: {state['yaw']:.1f}°

Detect: {', '.join(state['detections'][:2]) if state['detections'] else 'None'}

Objects:
  Signs: {len(self.sim.scene_manager.signs)}
  Lights: {len(self.sim.scene_manager.lights)}
  Vehicles: {len(self.sim.scene_manager.vehicles)}
  Intersections: {len(self.sim.scene_manager.intersections)}

Waypoints: {waypoint_progress}
Status: {status_str}

KEYS: R/Y/G B C I O DEL
+-:Speed SPACE:Pause BKSP:Restart
Q:Quit H:Help"""
        self.info_text.set_text(info_text)
        self.info_text.set_color(mode_color)
        
        return self.car_marker, self.trajectory_line, self.info_text
    
    def run(self, max_frames=10000):
        # Optimized: 50ms interval = ~20fps (was causing lag at 30ms)
        anim = FuncAnimation(self.fig, self.update_frame, frames=max_frames,
                            interval=50, repeat=False, blit=False)
        plt.tight_layout()
        plt.show()


# ============================================================================
# [FEATURE 5] MAIN EXECUTION - CLI argument support
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='BFMC Car Simulation')
    parser.add_argument('--start', type=int, default=342, help='Start node ID (single path mode)')
    parser.add_argument('--end', type=int, default=98, help='End node ID (single path mode)')
    parser.add_argument('--waypoints', type=str, default=None, help='Multi-path mode: comma-separated node IDs (e.g., "342,100,200,98")')
    parser.add_argument('--speed', type=float, default=2.0, help='Car speed (units/sec)')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step (seconds)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BFMC 2026 COMPREHENSIVE CAR SIMULATION")
    print("="*70)
    
    # Determine mode and print info
    if args.waypoints:
        waypoint_list = [int(n.strip()) for n in args.waypoints.split(',')]
        print(f"Multi-path mode: {len(waypoint_list)-1} segments")
        print(f"Waypoints: {' -> '.join(map(str, waypoint_list))}")
        print(f"Speed: {args.speed} | DT: {args.dt}")
    else:
        print(f"Single-path mode: {args.start} -> {args.end}")
        print(f"Speed: {args.speed} | DT: {args.dt}")
        waypoint_list = None
    
    graph_file = os.path.abspath('Competition_track_graph.graphml')
    if not os.path.exists(graph_file):
        print(f"[ERROR] Graph file not found: {graph_file}")
        return
    
    try:
        scene_manager = SceneManager()
        
        if args.waypoints:
            # Multi-path mode
            waypoint_list = [int(n.strip()) for n in args.waypoints.split(',')]
            sim = ComprehensiveCarSimulator(
                graph_file=graph_file,
                start_node=waypoint_list[0],
                end_node=None,
                speed=args.speed,
                dt=args.dt,
                scene_manager=scene_manager,
                waypoint_path=waypoint_list
            )
        else:
            # Single-path mode
            sim = ComprehensiveCarSimulator(
                graph_file=graph_file,
                start_node=args.start,
                end_node=args.end,
                speed=args.speed,
                dt=args.dt,
                scene_manager=scene_manager
            )
        
        sim._plan_path()
        
        if args.headless:
            frame_count = 0
            while not sim.stopped and sim.current_time < 1000:
                sim.update()
                frame_count += 1
                if frame_count % 20 == 0:  # Print every ~1 second
                    state = sim.get_state()
                    det_str = ', '.join(state['detections'][:2]) if state['detections'] else 'None'
                    print(f"t={state['time']:.1f}s | {state['mode']:20s} | "
                          f"Pos=({state['x']:.2f}, {state['y']:.2f}) | "
                          f"Det: {det_str:30s}")
            
            state = sim.get_state()
            print(f"\n[COMPLETE] {state['stop_reason']} | Time: {state['time']:.1f}s | Dist: {state['distance']:.2f}m")
        else:
            graph = nx.read_graphml(graph_file)
            visualizer = InteractiveVisualizer(sim, graph)
            visualizer.run()
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()