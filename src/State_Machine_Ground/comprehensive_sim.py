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
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
HAS_MATPLOTLIB = True

sys.path.insert(0, os.path.abspath('../Brain/src/statemachine'))

import networkx as nx

from carMode import CarModeChanger
HAS_MODE_CHANGER = True
print('Using custom StateChanger and CarMode')

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
                imagebox = OffsetImage(self.image, zoom=0.05, alpha=0.8)
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
    def __init__(self, width_near=1.0, width_far=2, length=1.0):
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
    
    def __init__(self, graph_file, start_node, end_node=None, speed=0.5, dt=0.05, scene_manager=None, waypoint_path=None, use_mpc=False):
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
        
        self.mode_changer = CarModeChanger()
        self.current_mode = self.mode_changer._get_mode()
        self.current_detections = []
        self.lookup_nodes = []  # For turn recognition visualization
        
        # MPC controller setup
        self.use_mpc = use_mpc
        self.mpc_controller = None
        self.mpc_velocity = speed
        self.mpc_target_velocity = speed
        if self.use_mpc:
            self.mpc_controller = SimpleMPCController(max_steering_angle=25.0, wheelbase=0.3, dt=dt)
            print("[INFO] MPC controller enabled (±25° steering constraint)")
        
        # Bezier curve following for TURN/OVERTAKING mode
        self.is_following_curve = False
        self.curve_points = []  # Bezier curve waypoints
        self.curve_progress = 0  # Current position along curve (0-1)
        self.frozen_lookup_nodes = []  # Frozen nodes when entering TURN mode
        self.turn_curve_start_edge = None  # Edge index where curve following should start (at node 1)
        self.steer_angle = 0.0  # Current steering angle in degrees
        
        self.scene_manager = scene_manager or SceneManager()
        self.fov = FieldOfView(width_near=0.5, width_far=1, length=0.75)
        
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
    
    def _detect_objects(self):
        """Detect objects within FOV and update state machine."""
        self.current_detections = []
        indices = []
        boxes = []
        
        # Detect objects within FOV trapezoid
        for dist, sign in self.scene_manager.get_nearby_signs(self.x, self.y, radius=5.0):
            if self.fov.contains_point(sign.x, sign.y, self.x, self.y, self.yaw):
                self.current_detections.append(sign.sign_type)
                indices.append(sign.class_id)
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        for dist, light in self.scene_manager.get_nearby_lights(self.x, self.y, radius=5.0):
            if self.fov.contains_point(light.x, light.y, self.x, self.y, self.yaw):
                self.current_detections.append(f"{light.state}_light")
                indices.append({'red': 5, 'yellow': 6, 'green': 7}[light.state])
                boxes.append([0.5, 0.5, 0.3, 0.9])
        
        for dist, vehicle in self.scene_manager.get_nearby_vehicles(self.x, self.y, radius=5.0):
            if self.fov.contains_point(vehicle.x, vehicle.y, self.x, self.y, self.yaw):
                self.current_detections.append(vehicle.vtype)
                indices.append({'car': 2, 'bus': 3, 'truck': 4, 'cyclist': 1, 'pedestrian': 0}.get(vehicle.vtype, 2))
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        for dist, inter in self.scene_manager.get_nearby_intersections(self.x, self.y, radius=5.0):
            if self.fov.contains_point(inter.x, inter.y, self.x, self.y, self.yaw):
                self.current_detections.append('intersection')
                indices.append(13)
                boxes.append([0.5, 0.5, 0.3, 0.3])
        
        self.mode_changer.record_detection(indices, boxes)
        if self.current_detections:
            print(f"[DETECTION] {', '.join(self.current_detections[:3]):30s}")
        
        # Track waypoint visits
        if self.waypoint_nodes and self.current_waypoint_idx < len(self.waypoint_nodes):
            next_wp, next_wp_idx = self.waypoint_nodes[self.current_waypoint_idx]
            if next_wp not in self.visited_waypoints and self.current_edge_idx >= next_wp_idx - 2:
                self.visited_waypoints.add(next_wp)
                self.current_waypoint_idx += 1
                print(f"[WAYPOINT] {len(self.visited_waypoints)}/{len(self.waypoint_nodes)} reached")
    
    def _calculate_turn_lookup(self):
        """Calculate yaw differences for turn recognition."""
        try:
            yaw_diffs = []
            for i in range(2):
                if self.current_edge_idx + i < len(self.path_edges) - 1:
                    curr_src, curr_dst = self.path_edges[self.current_edge_idx + i]
                    next_src, next_dst = self.path_edges[self.current_edge_idx + i + 1]
                    
                    curr_yaw = math.atan2(
                        float(self.graph.nodes[curr_dst]['y']) - float(self.graph.nodes[curr_src]['y']),
                        float(self.graph.nodes[curr_dst]['x']) - float(self.graph.nodes[curr_src]['x'])
                    )
                    next_yaw = math.atan2(
                        float(self.graph.nodes[next_dst]['y']) - float(self.graph.nodes[next_src]['y']),
                        float(self.graph.nodes[next_dst]['x']) - float(self.graph.nodes[next_src]['x'])
                    )
                    yaw_diff = abs(math.degrees(next_yaw - curr_yaw))
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    yaw_diffs.append(yaw_diff)
            
            self.mode_changer.record_lookup(yaw_diffs)
            self.lookup_nodes = [self.path_edges[min(self.current_edge_idx + i, len(self.path_edges) - 1)][1] for i in range(3)]
        except Exception as e:
            pass
    
    def _generate_bezier_curve(self, list_p, num_points=100):
        """Generate Bezier curve points from n control points (nth degree polynomial Bezier)"""
        try:
            def bernstein_poly(n, i, t):
                from math import comb
                return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            
            n = len(list_p) - 1
            curve_points = []
            for j in range(num_points):
                t = j / (num_points - 1)
                x = sum(bernstein_poly(n, i, t) * list_p[i][0] for i in range(n + 1))
                y = sum(bernstein_poly(n, i, t) * list_p[i][1] for i in range(n + 1))
                curve_points.append((x, y))
            return curve_points
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
        """Update state based on recorded detections and manage Bezier curve following."""
        self.mode_changer.change_state()
        self.current_mode = self.mode_changer._get_mode()

        # Enter TURN mode: generate Bezier curve
        if self.current_mode and self.current_mode.value.get('mode') == 'turn' and not self.is_following_curve:
            if len(self.lookup_nodes) >= 3:
                try:
                    p0 = (self.x, self.y)
                    p1 = (float(self.graph.nodes[self.lookup_nodes[1]]['x']), float(self.graph.nodes[self.lookup_nodes[1]]['y']))
                    p2 = (float(self.graph.nodes[self.lookup_nodes[2]]['x']), float(self.graph.nodes[self.lookup_nodes[2]]['y']))
                    self.curve_points = self._generate_bezier_curve([p0, p1, p2])
                    self.is_following_curve = True
                    self.curve_progress = 0
                    print(f"[TURN] Bezier curve started")
                except Exception as e:
                    pass

        # !!! WORK IN PROGRESS AREA !!!
        # Enter OVERTAKING mode: generate Bezier curve
        elif self.current_mode and self.current_mode.value.get('mode') == 'overtaking' and not self.is_following_curve:
            # Step 1: Look for anchor node meeting all conditions
            nearest_jump_node = None
            nearest_return_node = None
            min_jump_dist = float('inf')
            min_return_dist = float('inf')
            overtake_entry_path = []
            tailing_fallback = False
            tailing_reason = ''

            self.mode_changer._update_switch('overtaking_possible', True)
            overtake_entry_path.append(self.path_edges[self.current_edge_idx][0])

            # Calculate car's original direction (current edge)
            current_edge_yaw = self.yaw
            if self.current_edge_idx < len(self.path_edges):
                src, dst = self.path_edges[self.current_edge_idx]
                current_edge_yaw = math.atan2(
                    float(self.graph.nodes[dst]['y']) - float(self.graph.nodes[src]['y']),
                    float(self.graph.nodes[dst]['x']) - float(self.graph.nodes[src]['x'])
                )

            for node in self.graph.nodes:
                if node in self.path_nodes:
                    continue

                node_x = float(self.graph.nodes[node]['x'])
                node_y = float(self.graph.nodes[node]['y'])
                dist = math.sqrt((node_x - self.x)**2 + (node_y - self.y)**2)

                jump_yaw = math.atan2(
                    node_y - self.y,
                    node_x - self.x
                )
                yaw_diff = abs(math.degrees(jump_yaw - current_edge_yaw))
                yaw_diff = min(yaw_diff, 360 - yaw_diff)

                if yaw_diff <= 30 and 1.0 < dist < 3.0:
                    if dist < min_jump_dist:
                        min_jump_dist = dist
                        nearest_jump_node = node

            # Step 2: If anchor found, retain OVERTAKING; else fall back to TAILING
            if nearest_jump_node:
                overtake_entry_path.append(nearest_jump_node)

                anchor_path_yaw = math.atan2(
                    float(self.graph.nodes[nearest_jump_node]['y']) - float(self.graph.nodes[list(self.graph.successors(nearest_jump_node))[0]]['y']),
                    float(self.graph.nodes[nearest_jump_node]['x']) - float(self.graph.nodes[list(self.graph.successors(nearest_jump_node))[0]]['y'])
                )
                opposite_heading = abs(math.degrees(anchor_path_yaw - current_edge_yaw)) > 90

                if opposite_heading:
                    traversal_node = list(self.graph.predecessors(nearest_jump_node))[0]
                    while len(overtake_entry_path) < 8 and self.graph.degree(traversal_node) <= 2:
                        if self.graph.degree(traversal_node) > 2:
                            tailing_fallback = True
                            tailing_reason = 'Cannot overtake near nodes with 2 child paths'
                            break
                        overtake_entry_path.append(traversal_node)
                        traversal_node = list(self.graph.predecessors(nearest_jump_node))[0]
                else:
                    traversal_node = list(self.graph.successors(nearest_jump_node))[0]
                    while len(overtake_entry_path) < 8:
                        if self.graph.degree(traversal_node) > 2:
                            tailing_fallback = True
                            tailing_reason = 'Cannot overtake near nodes with 2 child paths'
                            break
                        overtake_entry_path.append(traversal_node)
                        traversal_node = list(self.graph.successors(nearest_jump_node))[0]

                for node in self.path_nodes:
                    node_x = float(self.graph.nodes[node]['x'])
                    node_y = float(self.graph.nodes[node]['y'])
                    dist = math.sqrt((node_x - self.graph.nodes[overtake_entry_path[-1]]['x'])**2 + (node_y - self.graph.nodes[overtake_entry_path[-1]]['y'])**2)

                    jump_yaw = math.atan2(
                        node_y - self.graph.nodes[overtake_entry_path[-1]]['y'],
                        node_x - self.graph.nodes[overtake_entry_path[-1]]['x']
                    )
                    yaw_diff = abs(math.degrees(jump_yaw - current_edge_yaw))
                    yaw_diff = min(yaw_diff, 360 - yaw_diff)

                    # Condition 3: Within neighbor's small and big radius area (3 meters = 3.0)
                    if yaw_diff <= 30 and 1.0 < dist < 3.0:
                        if dist < min_return_dist:
                            min_return_dist = dist
                            nearest_return_node = node
                    
                if nearest_return_node:
                    overtake_entry_path.append(nearest_return_node)
                else:
                    tailing_fallback = True
                    tailing_reason = 'No possible remerging after overtaking'
            else:
                tailing_fallback = True
                tailing_reason = 'No possible overtaking anchor node'
                    
            # TODO: Draw a follow line for OVERTAKING, and give OVERTAKING condition to StateChanger
            if tailing_fallback:
                print(f"[TAILING] Fallback to tailing for this reason: {tailing_reason}")
                self.mode_changer._update_switch('overtaking_possible', False)
            else:
                print(f"[OVERTAKING] Possible path found")
                self.mode_changer._update_switch('overtaking_possible', True)
                overtake_entry_path = [[float(self.graph.nodes[node]['x']), float(self.graph.nodes[node]['y'])] for node in overtake_entry_path]
                # self.curve_points = [
                #     *self._generate_bezier_curve(overtake_entry_path[:4]),
                #     *self._generate_bezier_curve(overtake_entry_path[4:6]),
                #     *self._generate_bezier_curve(overtake_entry_path[6:10])
                # ]
                self.curve_points = self._generate_bezier_curve(overtake_entry_path)
                self.is_following_curve = True
                self.curve_progress = 0

        # Exit TURN, OVERTAKING AND PARKING mode
        elif self.current_mode and self.current_mode.value.get('mode') not in ['turn', 'overtaking', 'parking'] and self.is_following_curve:
            self.is_following_curve = False
            self.mode_changer.following_curve = False
            self.curve_points = []
            self.curve_progress = 0
            self.mode_changer._update_switch('in_special_mode', False)
        # !!! END OF WORK IN PROGRESS AREA!!!

        if self.current_detections:
            print(f"  → MODE: {self.current_mode.value.get('mode', 'unknown').upper()}")

    def update(self):
        """Update simulation by one timestep."""
        self.current_time += self.dt
        self.mode_changer.update_timer(self.dt)
        
        if self.current_edge_idx >= len(self.path_edges):
            self.stopped = True
            self.stop_reason = "Reached destination"
            return
        
        self._detect_objects()
        self._calculate_turn_lookup()
        self._update_state()
        
        current_speed = self.mode_changer._get_speed()
        effective_speed = current_speed.value / 100.0  # Convert cm/s to m/s
        
        # MPC-based movement (alternative to edge-based)
        if self.use_mpc and self.mpc_controller:
            # Smooth velocity ramping
            max_accel = 0.5
            velocity_diff = self.mpc_target_velocity - self.mpc_velocity
            self.mpc_velocity += max(min(velocity_diff, max_accel * self.dt), -max_accel * self.dt)
            self.mpc_velocity = max(0, self.mpc_velocity)
            
            # Get next 5 path points for MPC
            remaining_nodes = self.path_nodes[self.current_edge_idx:]
            target_points = [(float(self.graph.nodes[node_id]['x']), float(self.graph.nodes[node_id]['y'])) 
                           for node_id in remaining_nodes[:5]]
            
            if not target_points:
                self.stopped = True
                self.stop_reason = "Reached destination"
                return
            
            # Compute MPC steering
            steer_angle = self.mpc_controller.compute_steering(self.x, self.y, self.yaw, target_points)
            self.steer_angle = steer_angle
            
            # Update yaw using kinematic model
            self.yaw = self.mpc_controller.update_heading(self.yaw, self.mpc_velocity, steer_angle)
            
            # Update position
            self.x += self.mpc_velocity * math.cos(self.yaw) * self.dt
            self.y += self.mpc_velocity * math.sin(self.yaw) * self.dt
            
            self.total_distance += self.mpc_velocity * self.dt
            self.position_history.append((self.x, self.y))
            
            # Check if reached next waypoint (within 0.3m)
            if self.current_edge_idx < len(self.path_nodes):
                next_node = self.path_nodes[self.current_edge_idx]
                next_x = float(self.graph.nodes[next_node]['x'])
                next_y = float(self.graph.nodes[next_node]['y'])
                dist_to_next = math.sqrt((next_x - self.x)**2 + (next_y - self.y)**2)
                if dist_to_next < 0.3:
                    self.current_edge_idx += 1
                    if self.current_edge_idx >= len(self.path_nodes):
                        self.stopped = True
                        self.stop_reason = "Reached destination"
            
            self.scene_manager.update(self.dt)
            return
        
        # Handle Bezier curve following
        if self.is_following_curve:
            target_pos = self._get_point_on_curve(self.curve_progress)
            if target_pos:
                self.x, self.y = target_pos
                
                # Calculate yaw based on tangent to the curve
                lookahead_progress = min(self.curve_progress + 0.02, 1.0)
                next_pos = self._get_point_on_curve(lookahead_progress)
                if next_pos and next_pos != target_pos:
                    self.yaw = math.atan2(next_pos[1] - self.y, next_pos[0] - self.x)
                
                # Advance progress
                self.curve_progress += (effective_speed * self.dt) / self._calculate_curve_length()

                # Check if curve following is complete
                if self.curve_progress >= 1.0:
                    self.is_following_curve = False
                    self.curve_points = []
                    self.curve_progress = 0
                    # Advance edge index past the curve
                    self.current_edge_idx += 3
                    self.position_on_edge = 0

            self.total_distance += effective_speed * self.dt
            self.position_history.append((self.x, self.y))
            self.scene_manager.update(self.dt)
            return
        
        # Normal edge-based movement
        start_pt, end_pt = self._get_current_edge_endpoints()
        if not (start_pt and end_pt):
            self.stopped = True
            self.stop_reason = "Path error"
            return
        
        edge_length = math.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
        movement = (effective_speed * self.dt) / edge_length if edge_length > 0 else 0
        self.position_on_edge += movement
        
        self.x = start_pt[0] + self.position_on_edge * (end_pt[0] - start_pt[0])
        self.y = start_pt[1] + self.position_on_edge * (end_pt[1] - start_pt[1])
        self.yaw = math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
        
        self.total_distance += effective_speed * self.dt
        
        # Calculate steer angle based on next waypoint direction
        if self.current_edge_idx + 1 < len(self.path_edges):
            next_edge = self.path_edges[self.current_edge_idx + 1]
            next_dst_node = next_edge[1]
            next_x = float(self.graph.nodes[next_dst_node]['x'])
            next_y = float(self.graph.nodes[next_dst_node]['y'])
            target_yaw = math.atan2(next_y - self.y, next_x - self.x)
            self.steer_angle = math.degrees(target_yaw - self.yaw)
            # Normalize to [-180, 180]
            while self.steer_angle > 180:
                self.steer_angle -= 360
            while self.steer_angle < -180:
                self.steer_angle += 360
        
        if self.position_on_edge >= 1.0:
            self.current_edge_idx += 1
            self.position_on_edge = 0
        
        self.position_history.append((self.x, self.y))
        self.scene_manager.update(self.dt)
    def get_state(self):
        mode_str = self.current_mode.value.get('mode', 'unknown').lower()
        return {
            'x': self.x,
            'y': self.y,
            'yaw': math.degrees(self.yaw),
            'steer_angle': self.steer_angle,
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
# [FEATURE 4] MPC CONTROLLER - Alternative to edge-based movement
# ============================================================================
class SimpleMPCController:
    """MPC-based steering control with kinematic bicycle model"""
    def __init__(self, max_steering_angle=25.0, wheelbase=0.3, dt=0.05):
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        self.dt = dt
        self.last_steer_angle = 0.0
        self.steer_rate_limit = 5.0  # degrees per second
        self.k_p = 25.0  # Proportional gain for heading control (tuned down for smoothness)
    
    def compute_steering(self, current_x, current_y, current_yaw, target_path_points):
        """Compute steering angle using cross-track error minimization."""
        if not target_path_points or len(target_path_points) < 2:
            return 0.0
        
        # Find lookahead point
        lookahead_distance = 0.5
        target_x, target_y = target_path_points[0]
        
        # Search for point at lookahead distance
        distance = 0.0
        for i in range(len(target_path_points) - 1):
            p1 = target_path_points[i]
            p2 = target_path_points[i + 1]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if distance + segment_length >= lookahead_distance:
                # Interpolate on this segment
                t = (lookahead_distance - distance) / segment_length if segment_length > 0 else 0
                target_x = p1[0] + t * (p2[0] - p1[0])
                target_y = p1[1] + t * (p2[1] - p1[1])
                break
            distance += segment_length
        
        # Calculate heading to target
        target_yaw = math.atan2(target_y - current_y, target_x - current_x)
        yaw_error = target_yaw - current_yaw
        
        # Normalize error to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # Proportional control with gain
        steer_command = self.k_p * yaw_error
        steer_command = math.degrees(steer_command)
        
        # Clamp to max steering angle
        steer_command = max(-self.max_steering_angle, min(self.max_steering_angle, steer_command))
        
        # Apply rate limiting for smooth steering
        max_change = self.steer_rate_limit * self.dt
        steer_angle = max(self.last_steer_angle - max_change,
                         min(self.last_steer_angle + max_change, steer_command))
        self.last_steer_angle = steer_angle
        
        return steer_angle
    
    def update_heading(self, current_yaw, velocity, steer_angle_degrees):
        """Update heading using kinematic bicycle model."""
        steer_rad = math.radians(steer_angle_degrees)
        yaw_rate = (velocity / self.wheelbase) * math.tan(steer_rad)
        new_yaw = current_yaw + yaw_rate * self.dt
        return new_yaw


# ============================================================================
# [FEATURE 5] INTERACTIVE VISUALIZATION - Click to add/remove objects
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
        self.mode = 'normal'

        # Two-panel layout: map on left, info on right
        self.fig, (self.ax_main, self.ax_info) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]}
        )

        self.ax_main.set_aspect('equal')
        self.ax_main.set_facecolor('#1a1a1a')
        self.fig.patch.set_facecolor('#0d0d0d')
        self.ax_main.set_title('BFMC Simulation', fontsize=9, weight='bold', color='#00ff00')
        self.ax_main.grid(True, alpha=0.15, color='#333333', linestyle=':')

        # Pre-load background image for faster rendering
        self.bg_image_artist = None
        self._load_background_image()
        self._plot_graph()

        # Persistent artists - only update what changes
        self.car_marker, = self.ax_main.plot([], [], 'r*', markersize=15, zorder=15)  # High zorder for visibility
        self.trajectory_line, = self.ax_main.plot([], [], '#00ff00', alpha=0.5, linewidth=2, zorder=14)

        # Scene object artists cache (mapped by object ID)
        self.sign_artists = {}  # sign_id -> patch
        self.light_artists = {}  # light_id -> patch
        self.vehicle_artists = {}  # car_id -> patch
        self.intersection_artists = {}  # inter_id -> marker

        # Waypoint artists cache
        self.wp_unvisited = {}  # node_id -> artist
        self.wp_visited = {}  # node_id -> artist

        # Info panel
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#1a1a1a')
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=7.5, verticalalignment='top', family='monospace', color='#00ff00')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Note: Don't hide graph lines - they provide helpful context
        # Instead, we'll keep graph elements low zorder (zorder=1) so objects appear on top

        print("[INFO] Controls: R/Y/G B C I DEL SPACE BKSP +/- H Q")

    def _load_background_image(self):
        """Load and display background track image."""
        try:
            img_path = os.path.abspath('track/comp_track.png')
            if os.path.exists(img_path):
                img = Image.open(img_path)
                w, h = img.size
                img = img.resize((w//8, h//8), Image.Resampling.LANCZOS)  # Resize for performance
                img_array = np.array(img)
                
                # Calculate extent from graph bounds properly [left, right, bottom, top]
                x_coords = [float(self.graph.nodes[n]['x']) for n in self.graph.nodes()]
                y_coords = [float(self.graph.nodes[n]['y']) for n in self.graph.nodes()]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add small padding for better alignment
                x_padding = (x_max - x_min) * 0.0175
                y_padding = (y_max - y_min) * 0.025
                extent = [x_min - x_padding, x_max + x_padding, y_min - y_padding, y_max + y_padding]
                
                # Display image as background (zorder=0 keeps it behind everything)
                self.bg_image_artist = self.ax_main.imshow(img_array, extent=extent, aspect='auto', zorder=0, alpha=0.85, origin='upper')
                print("[SUCCESS] Background image loaded and cached")
            else:
                print(f"[INFO] Background image not found at {img_path}")
        except Exception as e:
            print(f"[WARNING] Could not load background image: {e}")
    
    def _plot_graph(self):
        for node_id in self.graph.nodes():
            x = float(self.graph.nodes[node_id]['x'])
            y = float(self.graph.nodes[node_id]['y'])
            self.ax_main.plot(x, y, 'c.', markersize=5, alpha=0.6, zorder=1)  # Cyan nodes
        
        for edge in self.graph.edges():
            src_x = float(self.graph.nodes[edge[0]]['x'])
            src_y = float(self.graph.nodes[edge[0]]['y'])
            dst_x = float(self.graph.nodes[edge[1]]['x'])
            dst_y = float(self.graph.nodes[edge[1]]['y'])
            self.ax_main.plot([src_x, dst_x], [src_y, dst_y], 'c-', alpha=0.35, linewidth=1.5, zorder=1)  # Cyan edges
    
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
        """Optimized frame update - only draw what changes"""
        # Update simulation
        if not self.paused:
            self.sim.update()

        state = self.sim.get_state()

        # 1. Update car marker (changes every frame)
        self.car_marker.set_data([state['x']], [state['y']])

        # 2. Update trajectory line (extends every frame)
        if len(self.sim.position_history) > 1:
            hist = list(self.sim.position_history)
            self.trajectory_line.set_data(
                [pt[0] for pt in hist], [pt[1] for pt in hist]
            )

        # 3. Update scene objects - only add new ones or update existing
        self._update_scene_objects()

        # 4. Update waypoints - only add new ones
        self._update_waypoints()

        # 5. Update Bezier curve
        self._update_bezier_curve()

        # 6. Update info text
        self._update_info_text(state)

        return self.car_marker, self.trajectory_line, self.info_text

    def _update_scene_objects(self):
        """Update scene objects - add new ones, keep existing"""
        # Signs
        for sign_id, sign in self.sim.scene_manager.signs.items():
            if sign_id not in self.sign_artists:
                patch = sign.draw_and_return(self.ax_main)
                if patch:
                    self.sign_artists[sign_id] = patch

        # Lights
        for light_id, light in self.sim.scene_manager.lights.items():
            if light_id not in self.light_artists:
                patch = light.draw_and_return(self.ax_main)
                if patch:
                    self.light_artists[light_id] = patch

        # Vehicles
        for car_id, car in self.sim.scene_manager.vehicles.items():
            if car_id not in self.vehicle_artists:
                patch = car.draw_and_return(self.ax_main)
                if patch:
                    self.vehicle_artists[car_id] = patch

        # Intersections
        for inter_id, inter in self.sim.scene_manager.intersections.items():
            if inter_id not in self.intersection_artists:
                marker = inter.draw(self.ax_main)
                if marker:
                    self.intersection_artists[inter_id] = marker

        # Remove deleted objects
        existing_signs = set(self.sim.scene_manager.signs.keys())
        for sign_id in list(self.sign_artists.keys()):
            if sign_id not in existing_signs:
                try:
                    self.sign_artists[sign_id].remove()
                except:
                    pass
                del self.sign_artists[sign_id]

        existing_lights = set(self.sim.scene_manager.lights.keys())
        for light_id in list(self.light_artists.keys()):
            if light_id not in existing_lights:
                try:
                    self.light_artists[light_id].remove()
                except:
                    pass
                del self.light_artists[light_id]

        existing_vehicles = set(self.sim.scene_manager.vehicles.keys())
        for car_id in list(self.vehicle_artists.keys()):
            if car_id not in existing_vehicles:
                try:
                    self.vehicle_artists[car_id].remove()
                except:
                    pass
                del self.vehicle_artists[car_id]

        existing_intersections = set(self.sim.scene_manager.intersections.keys())
        for inter_id in list(self.intersection_artists.keys()):
            if inter_id not in existing_intersections:
                try:
                    self.intersection_artists[inter_id].remove()
                except:
                    pass
                del self.intersection_artists[inter_id]

    def _update_waypoints(self):
        """Update waypoint markers - only add new ones"""
        for wp_node, wp_idx in self.sim.waypoint_nodes:
            if wp_node in self.sim.visited_waypoints and wp_node not in self.wp_visited:
                try:
                    wp_x = float(self.graph.nodes[wp_node]['x'])
                    wp_y = float(self.graph.nodes[wp_node]['y'])
                    line, = self.ax_main.plot(wp_x, wp_y, marker='o', color='lime', markersize=5,
                                             zorder=11, markeredgecolor='darkgreen', markeredgewidth=0.3)
                    self.wp_visited[wp_node] = line
                except:
                    pass
            elif wp_node not in self.sim.visited_waypoints and wp_node not in self.wp_unvisited:
                try:
                    wp_x = float(self.graph.nodes[wp_node]['x'])
                    wp_y = float(self.graph.nodes[wp_node]['y'])
                    line, = self.ax_main.plot(wp_x, wp_y, marker='o', color='cyan', markersize=5,
                                             zorder=11, markeredgecolor='blue', markeredgewidth=0.3)
                    self.wp_unvisited[wp_node] = line
                except:
                    pass

    def _update_bezier_curve(self):
        """Update Bezier curve display"""
        if self.sim.is_following_curve and self.sim.curve_points:
            try:
                curve_xs = [pt[0] for pt in self.sim.curve_points]
                curve_ys = [pt[1] for pt in self.sim.curve_points]
                line, = self.ax_main.plot(curve_xs, curve_ys, color='magenta', linewidth=1,
                                         alpha=0.5, zorder=5, linestyle='--')
                self.ax_main.add_line(line)
            except:
                pass

    def _update_info_text(self, state):
        """Update info panel text"""
        mode_colors = {
            'straight': 'cyan',
            'turn': 'magenta',
            'overtaking': 'red',
            'tailing': 'orange',
            'parking': 'lime'
        }
        mode_color = mode_colors.get(state['mode'], 'white')

        status_str = 'PAUSED' if self.paused else ('STOPPED' if state['stopped'] else 'RUNNING')
        waypoint_progress = f"{len(self.sim.visited_waypoints)}/{len(self.sim.waypoint_nodes)}"

        current_speed = self.sim.mode_changer._get_speed()
        speed_value = current_speed.value / 100.0 if hasattr(current_speed, 'value') else 0

        info_text = f"""Mode: {state['mode'].upper()}
Speed: {speed_value:.2f} m/s

Time: {state['time']:.1f}s
Dist: {state['distance']:.1f}m

Pos: ({state['x']:.2f}, {state['y']:.2f})
Yaw: {state['yaw']:.0f}°
Steer: {state['steer_angle']:.0f}°

Detect: {', '.join(state['detections'][:2]) if state['detections'] else 'None'}

Objects:
  Signs: {len(self.sim.scene_manager.signs)}
  Lights: {len(self.sim.scene_manager.lights)}
  Cars: {len(self.sim.scene_manager.vehicles)}
  Inter: {len(self.sim.scene_manager.intersections)}

Waypoints: {waypoint_progress}
Status: {status_str}"""
        self.info_text.set_text(info_text)
        self.info_text.set_color(mode_color)
    
    def run(self, max_frames=10000):
        # Blitting disabled to support dynamic object placement (signs, lights, vehicles, intersections)
        # FuncAnimation with blit=True only redraws returned artists, missing dynamically added objects
        anim = FuncAnimation(self.fig, self.update_frame, frames=max_frames,
                            interval=30, repeat=False, blit=False)
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
    parser.add_argument('--mpc', action='store_true', help='Use MPC-based movement instead of edge-based')
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
    
    if args.mpc:
        print(f"Movement: MPC-based (±25° steering constraint)")
    else:
        print(f"Movement: Edge-based")
    
    graph_file = os.path.abspath('track/Competition_track_graph.graphml')
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
                waypoint_path=waypoint_list,
                use_mpc=args.mpc
            )
        else:
            # Single-path mode
            sim = ComprehensiveCarSimulator(
                graph_file=graph_file,
                start_node=args.start,
                end_node=args.end,
                speed=args.speed,
                dt=args.dt,
                scene_manager=scene_manager,
                use_mpc=args.mpc
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