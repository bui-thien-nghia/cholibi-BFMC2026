"""
BFMC 2026 Comprehensive Car Simulation — Tkinter Edition

Tkinter-based visualization with state machine validation.
Zero external GUI dependencies — uses Python's built-in tkinter.
Features:
  - Graph-based path planning (Dijkstra)
  - Field of View (FOV) trapezoid detection
  - Traffic signs, lights, vehicles, parking spots, intersections
  - Overtaking (stationary vehicle), tailing (moving vehicle), parking maneuvers
  - Bezier curve following for turn/overtake/park
  - MPC controller (optional)
  - Interactive object placement via keyboard + mouse
  - Real-time info panel

Controls:
  h             Show help
  space         Pause / Resume
  BackSpace     Restart simulation
  +/-           Speed up / slow down
  r/y/g         Place red / yellow / green traffic light
  b then 1-9    Place traffic sign (1=Stop, 2=Ped, 3=Park, ...)
  c             Place stationary vehicle
  m             Place moving vehicle
  p             Place parking spot
  i             Place intersection
  Delete        Toggle remove mode (click to delete nearest object)
  q / Escape    Quit
"""

import sys
import os
import math
import argparse
import time as _time
import tkinter as tk
from collections import deque
from enum import Enum

import numpy as np
import networkx as nx
from PIL import Image, ImageTk

# Ensure local carMode.py takes priority over the Brain copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.abspath('../Brain/src/statemachine'))
from carMode import CarModeChanger, CarSpeed
print('[OK] Using custom CarModeChanger')

# ============================================================================
# CONSTANTS  (hex colors for tkinter Canvas)
# ============================================================================
C_BLACK      = '#0D0D0D'
C_DARK_BG    = '#1A1A1A'
C_GREEN      = '#00FF00'
C_DARK_GREEN = '#00B400'
C_CYAN       = '#00C8C8'
C_CYAN_DIM   = '#005050'
C_RED        = '#FF3232'
C_YELLOW     = '#FFD700'
C_LIME       = '#00FF00'
C_WHITE      = '#FFFFFF'
C_MAGENTA    = '#FF00FF'
C_ORANGE     = '#FFA500'
C_BLUE       = '#0064FF'
C_PURPLE     = '#A020F0'
C_GRAY       = '#808080'
C_DARK_GRAY  = '#3C3C3C'
C_TRAIL      = '#00C800'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test (no matplotlib dependency)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _bezier_curve(control_points, num_points=80):
    """Nth-degree Bezier curve from N+1 control points."""
    from math import comb
    n = len(control_points) - 1
    if n < 1:
        return list(control_points)
    pts = []
    for j in range(num_points):
        t = j / (num_points - 1)
        x = sum(comb(n, i) * (t ** i) * ((1 - t) ** (n - i)) * control_points[i][0] for i in range(n + 1))
        y = sum(comb(n, i) * (t ** i) * ((1 - t) ** (n - i)) * control_points[i][1] for i in range(n + 1))
        pts.append((x, y))
    return pts


def _curve_length(points):
    """Total arc length of a polyline."""
    if len(points) < 2:
        return 0.1
    total = sum(math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
                for i in range(len(points) - 1))
    return max(total, 0.1)


def _catmull_rom_spline(points, num_points=80, alpha=0.5,
                        heading_start=None, heading_end=None):
    """Centripetal Catmull-Rom spline through control points.

    Adds phantom endpoints using headings (if given) or segment reflection.
    alpha: 0=uniform, 0.5=centripetal, 1.0=chordal.
    """
    if len(points) < 2:
        return list(points)
    pts = list(points)
    # Phantom before first point
    seg_d = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    if heading_start is not None:
        pts.insert(0, (pts[0][0] - seg_d * math.cos(heading_start),
                        pts[0][1] - seg_d * math.sin(heading_start)))
    else:
        pts.insert(0, (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1]))
    # Phantom after last point
    seg_d = math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
    if heading_end is not None:
        pts.append((pts[-1][0] + seg_d * math.cos(heading_end),
                    pts[-1][1] + seg_d * math.sin(heading_end)))
    else:
        pts.append((2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1]))

    result = []
    n_segs = len(pts) - 3
    per_seg = max(2, num_points // max(n_segs, 1))

    for s in range(n_segs):
        P = [np.array(pts[s + k], dtype=float) for k in range(4)]
        t0 = 0.0
        t1 = t0 + max(float(np.linalg.norm(P[1] - P[0])) ** (2 * alpha), 1e-10)
        t2 = t1 + max(float(np.linalg.norm(P[2] - P[1])) ** (2 * alpha), 1e-10)
        t3 = t2 + max(float(np.linalg.norm(P[3] - P[2])) ** (2 * alpha), 1e-10)
        for j in range(per_seg):
            t = t1 + (t2 - t1) * j / max(per_seg - 1, 1)
            f01 = (t - t0) / (t1 - t0) if abs(t1 - t0) > 1e-10 else 0.0
            f12 = (t - t1) / (t2 - t1) if abs(t2 - t1) > 1e-10 else 0.0
            f23 = (t - t2) / (t3 - t2) if abs(t3 - t2) > 1e-10 else 0.0
            A1 = P[0] + f01 * (P[1] - P[0])
            A2 = P[1] + f12 * (P[2] - P[1])
            A3 = P[2] + f23 * (P[3] - P[2])
            f02 = (t - t0) / (t2 - t0) if abs(t2 - t0) > 1e-10 else 0.0
            f13 = (t - t1) / (t3 - t1) if abs(t3 - t1) > 1e-10 else 0.0
            B1 = A1 + f02 * (A2 - A1)
            B2 = A2 + f13 * (A3 - A2)
            C = B1 + f12 * (B2 - B1)
            if s == 0 or j > 0:
                result.append((float(C[0]), float(C[1])))
    return result


def _cubic_bezier_tangent(p0, heading0, p2, heading2, num_points=80):
    """Cubic Bézier with tangent-directed control handles.

    Handles placed at 1/3 chord distance along heading directions, ensuring
    smooth entry/exit aligned with car heading.
    """
    chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
    d = max(chord / 3.0, 0.05)
    cp1 = (p0[0] + d * math.cos(heading0), p0[1] + d * math.sin(heading0))
    cp2 = (p2[0] - d * math.cos(heading2), p2[1] - d * math.sin(heading2))
    return _bezier_curve([p0, cp1, cp2, p2], num_points)


def _hermite_spline(points, tangents, num_points=80):
    """Cubic Hermite spline through points with explicit tangent vectors."""
    if len(points) < 2:
        return list(points)
    result = []
    n_segs = len(points) - 1
    per_seg = max(2, num_points // n_segs)
    for s in range(n_segs):
        p0 = np.array(points[s], dtype=float)
        p1 = np.array(points[s + 1], dtype=float)
        m0 = np.array(tangents[min(s, len(tangents) - 1)], dtype=float)
        m1 = np.array(tangents[min(s + 1, len(tangents) - 1)], dtype=float)
        for j in range(per_seg):
            t = j / max(per_seg - 1, 1)
            t2, t3 = t * t, t * t * t
            pt = ((2*t3 - 3*t2 + 1) * p0 + (t3 - 2*t2 + t) * m0 +
                  (-2*t3 + 3*t2) * p1 + (t3 - t2) * m1)
            if s == 0 or j > 0:
                result.append((float(pt[0]), float(pt[1])))
    return result


def _bspline_curve(control_points, num_points=80, degree=3):
    """Clamped uniform B-spline (approximating curve)."""
    n = len(control_points)
    if n < 2:
        return list(control_points)
    k = min(degree, n - 1)
    # Clamped knot vector
    n_internal = max(0, n - k - 1)
    knots = [0.0] * (k + 1)
    for i in range(1, n_internal + 1):
        knots.append(i / (n_internal + 1))
    knots.extend([1.0] * (k + 1))
    pts = np.array(control_points, dtype=float)
    result = []
    for j in range(num_points):
        t = j / max(num_points - 1, 1)
        t = min(t, 1.0 - 1e-10)
        # Find knot span
        span = k
        for i in range(k, n):
            if knots[i] <= t < knots[i + 1]:
                span = i
                break
        # De Boor's algorithm
        d = [pts[min(max(span - k + r, 0), n - 1)].copy() for r in range(k + 1)]
        for r in range(1, k + 1):
            for s_ in range(k, r - 1, -1):
                idx = span - k + s_
                lk = knots[idx] if idx < len(knots) else 0.0
                rk = knots[idx + k - r + 1] if idx + k - r + 1 < len(knots) else 1.0
                dn = rk - lk
                a = (t - lk) / dn if abs(dn) > 1e-10 else 0.0
                d[s_] = (1.0 - a) * d[s_ - 1] + a * d[s_]
        result.append((float(d[k][0]), float(d[k][1])))
    return result


def _circular_arc(p0, p1, p2, num_points=80):
    """Circular arc through three points (linear fallback if collinear)."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        # Collinear — two linear segments
        half = num_points // 2
        pts = []
        for j in range(half):
            t = j / max(half - 1, 1)
            pts.append((ax + t * (bx - ax), ay + t * (by - ay)))
        for j in range(num_points - half):
            t = j / max(num_points - half - 1, 1)
            pts.append((bx + t * (cx - bx), by + t * (cy - by)))
        return pts
    ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / D
    uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / D
    a0 = math.atan2(ay - uy, ax - ux)
    a1 = math.atan2(by - uy, bx - ux)
    a2 = math.atan2(cy - uy, cx - ux)

    def _adiff(a, b):
        d = b - a
        while d > math.pi:  d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    d01 = _adiff(a0, a1)
    d02 = _adiff(a0, a2)
    # Ensure p1 lies on the arc between p0 and p2
    if (d01 > 0 and d02 > 0 and d01 < d02) or \
       (d01 < 0 and d02 < 0 and d01 > d02):
        sweep = d02
    else:
        sweep = d02 - (2 * math.pi if d02 > 0 else -2 * math.pi)
    R = math.hypot(ax - ux, ay - uy)
    result = []
    for j in range(num_points):
        t = j / max(num_points - 1, 1)
        angle = a0 + sweep * t
        result.append((ux + R * math.cos(angle), uy + R * math.sin(angle)))
    return result


# ============================================================================
# SCENE OBJECTS
# ============================================================================
class TrafficSign:
    """Traffic sign (data only — rendering handled by TkinterRenderer)."""
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
    CLASS_IDS = {
        'pedestrian': 0, 'cyclist': 1, 'car': 2, 'bus': 3, 'truck': 4,
        'red_light': 5, 'yellow_light': 6, 'green_light': 7,
        'stop_sign': 16, 'pedestrian_sign': 8, 'parking_sign': 12,
        'enter_highway_sign': 9, 'leave_highway_sign': 10,
        'oneway_sign': 11, 'priority_sign': 13,
        'roundabout_sign': 15, 'notallowed_sign': 14,
    }
    SIGN_COLORS = {
        'stop_sign': C_RED, 'pedestrian_sign': C_YELLOW, 'parking_sign': C_DARK_GREEN,
        'enter_highway_sign': C_BLUE, 'leave_highway_sign': C_GRAY,
        'oneway_sign': '#8B4513', 'priority_sign': C_PURPLE,
        'roundabout_sign': C_ORANGE, 'notallowed_sign': C_DARK_GRAY,
    }

    def __init__(self, sign_type, x, y, radius=0.3):
        self.sign_type = sign_type
        self.x, self.y = x, y
        self.radius = radius
        self.class_id = self.CLASS_IDS.get(sign_type, 0)

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)


class TrafficLight:
    """Cycling traffic light."""
    COLORS = {'red': C_RED, 'yellow': C_YELLOW, 'green': C_LIME}
    STATE_IDS = {'red': 5, 'yellow': 6, 'green': 7}

    def __init__(self, x, y, node_id=None, state='green'):
        self.x, self.y = x, y
        self.node_id = node_id
        self.state = state
        self.timer = 0.0
        self.durations = {'green': 10, 'yellow': 3, 'red': 8}
        self.radius = 0.2

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.durations[self.state]:
            self.state = {'green': 'yellow', 'yellow': 'red', 'red': 'green'}[self.state]
            self.timer = 0.0

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)


class Vehicle:
    """NPC vehicle with stationary / moving state for tailing / overtaking."""
    VTYPES = {'car': 2, 'bus': 3, 'truck': 4, 'cyclist': 1, 'pedestrian': 0}

    def __init__(self, vehicle_id, x, y, moving=False, speed=0.0, heading=0.0):
        self.id = vehicle_id
        self.x, self.y = x, y
        self.radius = 0.15
        self.vtype = 'car'
        self.moving = moving
        self.speed = speed          # world units / sec
        self.heading = heading      # radians
        self.path = []              # optional [(x,y), …]
        self.path_idx = 0

    def update(self, dt):
        if not self.moving or self.speed <= 0:
            return
        if self.path and self.path_idx < len(self.path) - 1:
            tx, ty = self.path[self.path_idx + 1]
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            if d < self.speed * dt:
                self.path_idx += 1
                self.x, self.y = tx, ty
            else:
                self.heading = math.atan2(dy, dx)
                self.x += self.speed * dt * math.cos(self.heading)
                self.y += self.speed * dt * math.sin(self.heading)
        else:
            # No path — drift forward slowly
            self.x += self.speed * dt * math.cos(self.heading)
            self.y += self.speed * dt * math.sin(self.heading)

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)


class ParkingSpot:
    """Rectangular parking bay with availability tracking."""

    def __init__(self, spot_id, x, y, angle=0.0, width=0.4, length=0.6, occupied=False):
        self.id = spot_id
        self.x, self.y = x, y
        self.angle = angle          # orientation (radians)
        self.width = width
        self.length = length
        self.occupied = occupied
        self.radius = max(width, length)

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def get_corners(self):
        """Four corners of the rectangle in world coordinates."""
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        hw, hl = self.width / 2, self.length / 2
        return [(self.x + cx * ca - cy * sa,
                 self.y + cx * sa + cy * ca)
                for cx, cy in [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]]

    def get_entry_point(self):
        """Point alongside the spot for the approach path."""
        return (self.x - self.length * math.sin(self.angle),
                self.y + self.length * math.cos(self.angle))


class Intersection:
    """Intersection marker (for testing scenarios)."""

    def __init__(self, inter_id, x, y):
        self.id = inter_id
        self.x, self.y = x, y
        self.radius = 0.25

    def distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)


# ============================================================================
# SCENE MANAGER
# ============================================================================
class SceneManager:
    """Central registry for all scene objects with spatial queries."""

    def __init__(self):
        self.signs = {}
        self.lights = {}
        self.vehicles = {}
        self.parking_spots = {}
        self.intersections = {}
        self._ctr = {'sign': 0, 'light': 0, 'vehicle': 0, 'parking': 0, 'inter': 0}

    # ---- add ----
    def add_sign(self, sign_type, x, y):
        sid = f"sign_{self._ctr['sign']}"; self._ctr['sign'] += 1
        self.signs[sid] = TrafficSign(sign_type, x, y); return sid

    def add_light(self, x, y, state='green'):
        lid = f"light_{self._ctr['light']}"; self._ctr['light'] += 1
        self.lights[lid] = TrafficLight(x, y, state=state); return lid

    def add_vehicle(self, x, y, moving=False, speed=0.0, heading=0.0):
        vid = f"car_{self._ctr['vehicle']}"; self._ctr['vehicle'] += 1
        self.vehicles[vid] = Vehicle(vid, x, y, moving=moving, speed=speed, heading=heading)
        return vid

    def add_parking_spot(self, x, y, angle=0.0, occupied=False):
        pid = f"park_{self._ctr['parking']}"; self._ctr['parking'] += 1
        self.parking_spots[pid] = ParkingSpot(pid, x, y, angle=angle, occupied=occupied)
        return pid

    def add_intersection(self, x, y):
        iid = f"inter_{self._ctr['inter']}"; self._ctr['inter'] += 1
        self.intersections[iid] = Intersection(iid, x, y); return iid

    # ---- remove ----
    def remove_sign(self, sid):      self.signs.pop(sid, None)
    def remove_light(self, lid):     self.lights.pop(lid, None)
    def remove_vehicle(self, vid):   self.vehicles.pop(vid, None)
    def remove_parking_spot(self, pid): self.parking_spots.pop(pid, None)
    def remove_intersection(self, iid): self.intersections.pop(iid, None)

    # ---- update ----
    def update(self, dt):
        for l in self.lights.values():   l.update(dt)
        for v in self.vehicles.values(): v.update(dt)

    # ---- spatial queries ----
    def _nearby(self, collection, x, y, radius):
        out = [(o.distance_to(x, y), o) for o in collection.values()]
        return sorted([(d, o) for d, o in out if d < radius], key=lambda p: p[0])

    def get_nearby_signs(self, x, y, r=2.0):         return self._nearby(self.signs, x, y, r)
    def get_nearby_lights(self, x, y, r=2.0):        return self._nearby(self.lights, x, y, r)
    def get_nearby_vehicles(self, x, y, r=2.0):      return self._nearby(self.vehicles, x, y, r)
    def get_nearby_parking_spots(self, x, y, r=2.0): return self._nearby(self.parking_spots, x, y, r)
    def get_nearby_intersections(self, x, y, r=2.0): return self._nearby(self.intersections, x, y, r)

    def remove_nearest(self, x, y, radius=0.5):
        """Remove the closest object within *radius*. Returns True on success."""
        best_d, best_id, best_col = radius, None, None
        for col in (self.signs, self.lights, self.vehicles, self.parking_spots, self.intersections):
            for oid, obj in col.items():
                d = obj.distance_to(x, y)
                if d < best_d:
                    best_d, best_id, best_col = d, oid, col
        if best_id is not None:
            del best_col[best_id]
            return True
        return False


# ============================================================================
# FIELD OF VIEW
# ============================================================================
class FieldOfView:
    """Trapezoid FOV that rotates with car heading."""

    def __init__(self, width_near=0.5, width_far=1.0, length=0.75):
        self.width_near = width_near
        self.width_far  = width_far
        self.length     = length

    def get_polygon(self, car_x, car_y, car_yaw):
        a = car_yaw - math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        wn, wf, L = self.width_near, self.width_far, self.length
        local = [(-wn / 2, 0), (wn / 2, 0), (wf / 2, L), (-wf / 2, L)]
        return [(car_x + lx * ca - ly * sa,
                 car_y + lx * sa + ly * ca) for lx, ly in local]

    def contains_point(self, px, py, car_x, car_y, car_yaw):
        return _point_in_polygon(px, py, self.get_polygon(car_x, car_y, car_yaw))


# ============================================================================
# MPC CONTROLLER
# ============================================================================
class SimpleMPCController:
    """MPC-like steering controller with kinematic bicycle model.

    Steering limit: [-25, 25] degrees.
    Uses pure-pursuit adaptive lookahead for path tracking.
    """

    def __init__(self, max_steering_angle=30.0, wheelbase=0.3, dt=0.05):
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        self.dt = dt
        self.last_steer = 0.0
        self.rate_limit = 120.0    # deg / s — responsive servo
        self.lookahead_base = 0.20 # base lookahead distance (metres)
        self.path_curvature = 0.0  # cached from last compute_steering()

    def compute_steering(self, cx, cy, cyaw, path_pts, velocity=0.3):
        """Compute steering angle toward *path_pts* using pure-pursuit.

        Uses the standard pure-pursuit formula:
            steer = atan(2 * L * sin(alpha) / ld)
        where alpha is heading error and ld is lookahead distance.

        Args:
            cx, cy: current position
            cyaw: current heading (radians)
            path_pts: list of (x, y) waypoints ahead
            velocity: current velocity (m/s) — used for adaptive lookahead
        Returns:
            steering angle in degrees, clamped to [-25, 25]
        """
        if not path_pts or len(path_pts) < 2:
            return 0.0
        # Adaptive lookahead: further ahead at higher speeds
        lookahead = self.lookahead_base + velocity
        tx, ty = path_pts[0]
        dist = 0.0
        for i in range(len(path_pts) - 1):
            p1, p2 = path_pts[i], path_pts[i + 1]
            sl = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist + sl >= lookahead:
                t = (lookahead - dist) / sl if sl > 0 else 0
                tx = p1[0] + t * (p2[0] - p1[0])
                ty = p1[1] + t * (p2[1] - p1[1])
                break
            dist += sl
        # Pure-pursuit steering formula
        alpha = math.atan2(ty - cy, tx - cx) - cyaw
        while alpha > math.pi:  alpha -= 2 * math.pi
        while alpha < -math.pi: alpha += 2 * math.pi
        ld = max(math.hypot(tx - cx, ty - cy), 0.01)
        steer_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        cmd = max(-self.max_steering_angle,
                  min(self.max_steering_angle, math.degrees(steer_rad)))
        mc = self.rate_limit * self.dt
        steer = max(self.last_steer - mc, min(self.last_steer + mc, cmd))
        self.last_steer = steer
        # Cache path curvature for MPC speed control
        self.path_curvature = self._compute_path_curvature(path_pts)
        return steer

    def update_heading(self, yaw, vel, steer_deg):
        """Update heading using kinematic bicycle model."""
        return yaw + (vel / self.wheelbase) * math.tan(math.radians(steer_deg)) * self.dt

    def _compute_path_curvature(self, path_pts):
        """Estimate max Menger curvature from the first few path segments."""
        max_k = 0.0
        n = min(len(path_pts), 10)
        for i in range(n - 2):
            p0, p1, p2 = path_pts[i], path_pts[i + 1], path_pts[i + 2]
            ax, ay = p1[0] - p0[0], p1[1] - p0[1]
            bx, by = p2[0] - p1[0], p2[1] - p1[1]
            cross = abs(ax * by - ay * bx)
            a_len = math.hypot(ax, ay)
            b_len = math.hypot(bx, by)
            c_len = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            denom = a_len * b_len * c_len
            if denom > 1e-10:
                max_k = max(max_k, 2.0 * cross / denom)
        return max_k

    def compute_speed(self, speed_min, speed_max, steer_angle):
        """Compute optimal speed within [speed_min, speed_max] cm/s.

        Uses cached path curvature (from last compute_steering) combined
        with steering-induced curvature.  Higher curvature → speed_min;
        straight path → speed_max.
        """
        if speed_max <= 0:
            return 0.0
        steer_k = abs(math.tan(math.radians(steer_angle)) / self.wheelbase)
        effective_k = max(self.path_curvature, steer_k * 0.5)
        k_thresh = 5     # 1/m — tuned for track geometry
        t = min(1.0, effective_k / k_thresh)
        speed = speed_max - t * (speed_max - speed_min)
        return max(speed_min, min(speed_max, speed))


# ============================================================================
# CAR SIMULATOR  (pure logic — no rendering dependency)
# ============================================================================
class ComprehensiveCarSimulator:
    """Core simulation: path planning, physics, detection, maneuvers."""

    def __init__(self, graph_file, start_node, end_node=None, speed=0.5, dt=0.05,
                 scene_manager=None, waypoint_path=None, curve_type='bezier3'):
        print("[INFO] Initializing Car Simulator...")
        self.graph = nx.read_graphml(graph_file)
        print(f"[OK] Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        self.speed = speed
        self.dt = dt
        self.current_time = 0.0
        self.total_distance = 0.0

        # Path mode
        if waypoint_path is not None:
            self.waypoint_path = [str(n) for n in waypoint_path]
            self.start_node = self.waypoint_path[0]
            self.end_node = self.waypoint_path[-1]
        else:
            self.waypoint_path = None
            self.start_node = str(start_node)
            self.end_node = str(end_node)

        self.path_nodes = []
        self.path_edges = []
        self.current_edge_idx = 0
        self.position_on_edge = 0.0

        # Waypoint tracking
        self.waypoint_nodes = []
        self.current_waypoint_idx = 0
        self.visited_waypoints = set()

        # Car state
        self.x = float(self.graph.nodes[self.start_node]['x'])
        self.y = float(self.graph.nodes[self.start_node]['y'])
        self.yaw = 0.0
        self.steer_angle = 0.0
        self.position_history = deque(maxlen=2000)
        self.position_history.append((self.x, self.y))

        # State machine
        self.mode_changer = CarModeChanger()
        self.current_mode = self.mode_changer._get_mode()
        self.current_detections = []
        self.lookup_nodes = []

        # MPC controller (always active)
        self.mpc_controller = SimpleMPCController(dt=dt)
        self.effective_speed_cms = 0.0   # last computed effective speed (cm/s)
        self.curve_type = curve_type     # 'bezier2'|'bezier3'|'catmull-rom'|'hermite'|'bspline'|'arc'

        # Curve following (shared by turn / overtake / park)
        self.is_following_curve = False
        self.curve_points = []
        self.curve_progress = 0.0

        # Overtaking
        self.overtake_merge_edge = None

        # Tailing
        self.tailing_target = None

        # Parking
        self.parking_phase = None          # 'entering' | 'parked' | 'exiting' | None
        self.parking_timer = 0.0
        self.parking_spot = None
        self.parking_park_duration = 3.0   # seconds to stay parked
        self.parking_return_curve = []
        self.parking_rejoin_edge = None

        # Scene
        self.scene_manager = scene_manager or SceneManager()
        self.fov = FieldOfView(width_near=0.5, width_far=1.0, length=0.75)

        self.stopped = False
        self.stop_reason = ""

        self._plan_path()
        # Initialize yaw from first edge direction
        if self.path_edges:
            sp, ep = self._get_edge_endpoints(0)
            if sp and ep:
                self.yaw = math.atan2(ep[1] - sp[1], ep[0] - sp[0])
        print(f"[OK] Path: {len(self.path_nodes)} nodes")

    # ------------------------------------------------------------------
    # PATH PLANNING
    # ------------------------------------------------------------------
    def _plan_path(self):
        try:
            if self.waypoint_path:
                merged = []
                for i in range(len(self.waypoint_path) - 1):
                    seg = nx.dijkstra_path(self.graph, self.waypoint_path[i], self.waypoint_path[i + 1])
                    merged.extend(seg if i == 0 else seg[1:])
                self.path_nodes = merged
            else:
                self.path_nodes = nx.dijkstra_path(self.graph, self.start_node, self.end_node)

            self.path_edges = list(zip(self.path_nodes[:-1], self.path_nodes[1:]))
            self.current_edge_idx = 0
            interval = max(1, len(self.path_nodes) // 10)
            self.waypoint_nodes = [(self.path_nodes[i], i) for i in range(0, len(self.path_nodes), interval)]
            if self.path_nodes[-1] not in [w[0] for w in self.waypoint_nodes]:
                self.waypoint_nodes.append((self.path_nodes[-1], len(self.path_nodes) - 1))
            self.visited_waypoints.clear()
        except nx.NetworkXNoPath as e:
            print(f"[ERROR] No path: {e}")
            raise

    def _get_edge_endpoints(self, idx=None):
        idx = idx if idx is not None else self.current_edge_idx
        if idx >= len(self.path_edges):
            return None, None
        s, d = self.path_edges[idx]
        return ((float(self.graph.nodes[s]['x']), float(self.graph.nodes[s]['y'])),
                (float(self.graph.nodes[d]['x']), float(self.graph.nodes[d]['y'])))

    def _get_point_on_curve(self, progress):
        if not self.curve_points or not (0 <= progress <= 1):
            return None
        idx = min(int(progress * (len(self.curve_points) - 1)), len(self.curve_points) - 1)
        return self.curve_points[idx]

    # ------------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------------
    def _detect_objects(self):
        self.current_detections = []
        indices, boxes = [], []
        has_moving, has_stationary = False, False

        for _, sign in self.scene_manager.get_nearby_signs(self.x, self.y, r=5.0):
            if self.fov.contains_point(sign.x, sign.y, self.x, self.y, self.yaw):
                self.current_detections.append(sign.sign_type)
                indices.append(sign.class_id)
                boxes.append([0.5, 0.5, 0.3, 0.3])

        for _, light in self.scene_manager.get_nearby_lights(self.x, self.y, r=5.0):
            if self.fov.contains_point(light.x, light.y, self.x, self.y, self.yaw):
                self.current_detections.append(f"{light.state}_light")
                indices.append(TrafficLight.STATE_IDS[light.state])
                boxes.append([0.5, 0.5, 0.3, 0.9])

        for _, veh in self.scene_manager.get_nearby_vehicles(self.x, self.y, r=5.0):
            if self.fov.contains_point(veh.x, veh.y, self.x, self.y, self.yaw):
                self.current_detections.append(veh.vtype)
                indices.append(Vehicle.VTYPES.get(veh.vtype, 2))
                boxes.append([0.5, 0.5, 0.3, 0.3])
                if veh.moving:
                    has_moving = True
                else:
                    has_stationary = True

        for _, inter in self.scene_manager.get_nearby_intersections(self.x, self.y, r=5.0):
            if self.fov.contains_point(inter.x, inter.y, self.x, self.y, self.yaw):
                self.current_detections.append('intersection')
                indices.append(13)
                boxes.append([0.5, 0.5, 0.3, 0.3])

        self.mode_changer.record_detection(indices, boxes)
        self.mode_changer.record_vehicle_state(has_moving, has_stationary)

        if self.current_detections:
            print(f"[DET] {', '.join(self.current_detections[:3]):30s}")

        # Waypoint visits
        if self.waypoint_nodes and self.current_waypoint_idx < len(self.waypoint_nodes):
            wp, wi = self.waypoint_nodes[self.current_waypoint_idx]
            if wp not in self.visited_waypoints and self.current_edge_idx >= wi - 2:
                self.visited_waypoints.add(wp)
                self.current_waypoint_idx += 1

    def _calculate_turn_lookup(self):
        try:
            diffs = []
            for i in range(1):  # look only 1 edge ahead
                if self.current_edge_idx + i < len(self.path_edges) - 1:
                    cs, cd = self.path_edges[self.current_edge_idx + i]
                    ns, nd = self.path_edges[self.current_edge_idx + i + 1]
                    cy = math.atan2(float(self.graph.nodes[cd]['y']) - float(self.graph.nodes[cs]['y']),
                                    float(self.graph.nodes[cd]['x']) - float(self.graph.nodes[cs]['x']))
                    ny = math.atan2(float(self.graph.nodes[nd]['y']) - float(self.graph.nodes[ns]['y']),
                                    float(self.graph.nodes[nd]['x']) - float(self.graph.nodes[ns]['x']))
                    d = abs(math.degrees(ny - cy))
                    diffs.append(min(d, 360 - d))
            self.mode_changer.record_lookup(diffs)
            self.lookup_nodes = [
                self.path_edges[min(self.current_edge_idx + i, len(self.path_edges) - 1)][1]
                for i in range(3)
            ]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MANEUVERS
    # ------------------------------------------------------------------
    def _start_turn(self):
        """Generate turn curve: current position + 2 nodes ahead.

        Curve type is determined by self.curve_type:
          bezier2     — Quadratic Bézier (3 control points)
          bezier3     — Cubic Bézier with heading-aligned tangent handles
          catmull-rom — Centripetal Catmull-Rom spline
          hermite     — Cubic Hermite with heading-derived tangent vectors
          bspline     — Clamped B-spline (5 control points with phantoms)
          arc         — Circular arc through 3 points
        """
        if len(self.lookup_nodes) < 3:
            return
        try:
            p0 = (self.x, self.y)
            p1 = (float(self.graph.nodes[self.lookup_nodes[1]]['x']),
                   float(self.graph.nodes[self.lookup_nodes[1]]['y']))
            p2 = (float(self.graph.nodes[self.lookup_nodes[2]]['x']),
                   float(self.graph.nodes[self.lookup_nodes[2]]['y']))

            heading0 = self.yaw
            heading2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])

            ct = self.curve_type
            if ct == 'bezier3':
                self.curve_points = _cubic_bezier_tangent(
                    p0, heading0, p2, heading2)
            elif ct == 'catmull-rom':
                self.curve_points = _catmull_rom_spline(
                    [p0, p1, p2], heading_start=heading0,
                    heading_end=heading2)
            elif ct == 'hermite':
                scale = max(chord / 3.0, 0.05)
                t0 = (scale * math.cos(heading0),
                      scale * math.sin(heading0))
                t2 = (scale * math.cos(heading2),
                      scale * math.sin(heading2))
                mid_dx, mid_dy = p2[0] - p0[0], p2[1] - p0[1]
                mid_len = math.hypot(mid_dx, mid_dy)
                if mid_len > 1e-6:
                    t1 = (scale * mid_dx / mid_len,
                          scale * mid_dy / mid_len)
                else:
                    t1 = t0
                self.curve_points = _hermite_spline(
                    [p0, p1, p2], [t0, t1, t2])
            elif ct == 'bspline':
                # Add phantom points for proper degree-3 curve
                d = max(chord / 4.0, 0.05)
                pre = (p0[0] - d * math.cos(heading0),
                       p0[1] - d * math.sin(heading0))
                post = (p2[0] + d * math.cos(heading2),
                        p2[1] + d * math.sin(heading2))
                self.curve_points = _bspline_curve(
                    [pre, p0, p1, p2, post])
            elif ct == 'arc':
                self.curve_points = _circular_arc(p0, p1, p2)
            else:  # bezier2 (default)
                self.curve_points = _bezier_curve([p0, p1, p2])

            self.is_following_curve = True
            self.curve_progress = 0.0
            print(f"[TURN] {ct} curve ({len(self.curve_points)} pts)")
        except Exception:
            pass

    def _start_overtaking(self):
        """Generate a 5-point Bezier overtaking path around a stationary vehicle."""
        target = None
        for _, v in self.scene_manager.get_nearby_vehicles(self.x, self.y, r=3.0):
            if self.fov.contains_point(v.x, v.y, self.x, self.y, self.yaw) and not v.moving:
                target = v
                break
        if target is None:
            print("[OVERTAKE] No stationary vehicle — falling back to tailing")
            return False

        offset = 0.5                          # lateral offset (metres)
        perp = self.yaw + math.pi / 2         # perpendicular to heading
        fwd = self.yaw
        d_approach = 0.4
        d_pass = max(0.3, target.distance_to(self.x, self.y))
        d_exit = d_pass + 0.5

        p0 = (self.x, self.y)
        p1 = (self.x + d_approach * math.cos(fwd) + offset * math.cos(perp),
              self.y + d_approach * math.sin(fwd) + offset * math.sin(perp))
        p2 = (self.x + d_pass * math.cos(fwd) + offset * math.cos(perp),
              self.y + d_pass * math.sin(fwd) + offset * math.sin(perp))
        p3 = (self.x + d_exit * math.cos(fwd) + offset * 0.3 * math.cos(perp),
              self.y + d_exit * math.sin(fwd) + offset * 0.3 * math.sin(perp))

        # Merge point: rejoin the original path a few edges ahead
        merge_idx = min(self.current_edge_idx + 6, len(self.path_edges) - 1)
        mn = self.path_edges[merge_idx][1]
        p4 = (float(self.graph.nodes[mn]['x']), float(self.graph.nodes[mn]['y']))

        self.curve_points = _bezier_curve([p0, p1, p2, p3, p4], num_points=120)
        self.is_following_curve = True
        self.curve_progress = 0.0
        self.overtake_merge_edge = merge_idx
        print(f"[OVERTAKE] Path generated ({len(self.curve_points)} pts)")
        return True

    def _start_tailing(self):
        """Begin tailing a moving vehicle (speed control only)."""
        for _, v in self.scene_manager.get_nearby_vehicles(self.x, self.y, r=3.0):
            if self.fov.contains_point(v.x, v.y, self.x, self.y, self.yaw):
                self.tailing_target = v
                print(f"[TAILING] Following vehicle at d={v.distance_to(self.x, self.y):.2f}")
                return True
        return False

    def _update_tailing(self):
        """Adjust speed to maintain safe following distance."""
        if self.tailing_target is None:
            return
        d = self.tailing_target.distance_to(self.x, self.y)
        in_fov = self.fov.contains_point(self.tailing_target.x, self.tailing_target.y,
                                          self.x, self.y, self.yaw)
        if not in_fov or d > 3.0:
            self.tailing_target = None
            return
        # Distance-based speed control
        if d < 0.3:
            self.mode_changer.cur_speed = CarSpeed.STOP
        elif d < 0.6:
            self.mode_changer.cur_speed = CarSpeed.SLOW
        else:
            self.mode_changer.cur_speed = CarSpeed.NORMAL

    def _start_parking(self):
        """Find an unoccupied parking spot and generate entry / exit curves."""
        for _, spot in self.scene_manager.get_nearby_parking_spots(self.x, self.y, r=3.0):
            if not spot.occupied:
                self.parking_spot = spot
                self.parking_phase = 'entering'
                self.parking_timer = 0.0

                entry_pt = spot.get_entry_point()
                p0 = (self.x, self.y)
                p1 = entry_pt
                p2 = (spot.x, spot.y)

                self.curve_points = _bezier_curve([p0, p1, p2], num_points=60)
                self.is_following_curve = True
                self.curve_progress = 0.0

                # Rejoin point on path after exit
                rejoin = min(self.current_edge_idx + 3, len(self.path_edges) - 1)
                rn = self.path_edges[rejoin][1]
                rpt = (float(self.graph.nodes[rn]['x']), float(self.graph.nodes[rn]['y']))
                self.parking_return_curve = _bezier_curve([p2, p1, rpt], num_points=60)
                self.parking_rejoin_edge = rejoin

                print(f"[PARKING] Entering spot at ({spot.x:.2f}, {spot.y:.2f})")
                return True

        print("[PARKING] No available spot found")
        return False

    def _update_parking(self, dt):
        if self.parking_phase is None:
            return

        if self.parking_phase == 'entering':
            if not self.is_following_curve:          # arrived at spot
                self.parking_phase = 'parked'
                self.parking_timer = 0.0
                self.mode_changer.cur_speed = CarSpeed.STOP
                if self.parking_spot:
                    self.parking_spot.occupied = True
                print("[PARKING] Parked — waiting...")

        elif self.parking_phase == 'parked':
            self.parking_timer += dt
            if self.parking_timer >= self.parking_park_duration:
                self.parking_phase = 'exiting'
                if self.parking_return_curve:
                    self.curve_points = self.parking_return_curve
                    self.is_following_curve = True
                    self.curve_progress = 0.0
                if self.parking_spot:
                    self.parking_spot.occupied = False
                print("[PARKING] Exiting spot...")

        elif self.parking_phase == 'exiting':
            if not self.is_following_curve:          # finished exiting
                if self.parking_rejoin_edge is not None:
                    self.current_edge_idx = self.parking_rejoin_edge
                    self.position_on_edge = 0.0
                self.parking_phase = None
                self.parking_spot = None
                self.parking_return_curve = []
                self.parking_rejoin_edge = None
                print("[PARKING] Exit complete — resuming path")

    # ------------------------------------------------------------------
    # STATE UPDATE
    # ------------------------------------------------------------------
    def _update_state(self):
        self.mode_changer.change_state()
        self.current_mode = self.mode_changer._get_mode()
        mode_str = self.current_mode.value.get('mode', '') if self.current_mode else ''

        # TURN
        if mode_str == 'turn' and not self.is_following_curve and not self.parking_phase:
            self._start_turn()

        # OVERTAKING
        elif mode_str == 'overtaking' and not self.is_following_curve and not self.parking_phase:
            if not self._start_overtaking():
                self._start_tailing()

        # TAILING
        elif mode_str == 'tailing':
            if self.tailing_target is None:
                self._start_tailing()
            self._update_tailing()

        # PARKING
        elif mode_str == 'parking' and not self.is_following_curve and not self.parking_phase:
            self._start_parking()

        # Exit curve when mode returns to normal
        elif mode_str not in ('turn', 'overtaking', 'parking') \
                and self.is_following_curve and not self.parking_phase:
            self.is_following_curve = False
            self.curve_points = []
            self.curve_progress = 0.0

        # Clear tailing reference when no longer relevant
        if mode_str != 'tailing' and self.tailing_target is not None:
            self.tailing_target = None

        if self.current_detections:
            print(f"  -> MODE: {mode_str.upper()}")

    # ------------------------------------------------------------------
    # SPEED FORMULA
    # ------------------------------------------------------------------
    def _compute_effective_speed_cms(self):
        """Compute effective speed in cm/s using MPC speed control.

        The MPC controller picks an optimal speed within each range based
        on upcoming path curvature and current steering demand:
          Stop:        0          cm/s
          Slow:       [15, 20]   cm/s
          Normal:     [30, 40]   cm/s
          Turn/steer: [20, 30]   cm/s
          Highway:    [40, 50]   cm/s
        """
        speed_enum = self.mode_changer._get_speed()
        mode_str = (self.current_mode.value.get('mode', '')
                    if self.current_mode else '')

        if speed_enum == CarSpeed.STOP:
            return 0.0

        if speed_enum == CarSpeed.SLOW:
            return self.mpc_controller.compute_speed(15.0, 20.0, self.steer_angle)

        if speed_enum == CarSpeed.FAST:          # highway
            return self.mpc_controller.compute_speed(40.0, 50.0, self.steer_angle)

        # NORMAL speed enum
        if mode_str == 'turn' or abs(self.steer_angle) > 5.0:
            return self.mpc_controller.compute_speed(20.0, 30.0, self.steer_angle)

        return self.mpc_controller.compute_speed(30.0, 40.0, self.steer_angle)

    # ------------------------------------------------------------------
    # MAIN UPDATE
    # ------------------------------------------------------------------
    def update(self):
        self.current_time += self.dt
        self.mode_changer.update_timer(self.dt)

        if self.current_edge_idx >= len(self.path_edges) and not self.parking_phase:
            self.stopped = True
            self.stop_reason = "Reached destination"
            return

        self._detect_objects()
        self._calculate_turn_lookup()
        self._update_state()
        self._update_parking(self.dt)

        # Compute MPC-based effective speed
        self.effective_speed_cms = self._compute_effective_speed_cms()
        eff_speed = self.effective_speed_cms / 100.0   # cm/s → m/s

        # Parked — freeze
        if self.parking_phase == 'parked':
            self.scene_manager.update(self.dt)
            return

        # Bezier curve following (turn / overtake / park)
        if self.is_following_curve and self.curve_points:
            self._curve_step(eff_speed)
            return

        # Normal edge-based movement with MPC steering
        self._edge_step(eff_speed)

    # ---- movement helpers (all use MPC + bicycle kinematics) ----
    def _curve_step(self, eff_speed):
        """Follow Bézier curve using MPC steering + bicycle model."""
        total_pts = len(self.curve_points)
        current_idx = min(int(self.curve_progress * (total_pts - 1)),
                          total_pts - 1)

        # Reference points ahead on the curve for MPC
        remaining_pts = self.curve_points[current_idx:]
        if len(remaining_pts) < 2:
            remaining_pts = self.curve_points[-2:]

        # MPC steering toward curve
        steer = self.mpc_controller.compute_steering(
            self.x, self.y, self.yaw, remaining_pts[:15], velocity=eff_speed)
        self.steer_angle = steer

        # Recompute speed now that steer_angle is up-to-date
        self.effective_speed_cms = self._compute_effective_speed_cms()
        eff_speed = self.effective_speed_cms / 100.0

        # Bicycle model kinematics
        self.yaw = self.mpc_controller.update_heading(self.yaw, eff_speed, steer)
        self.x += eff_speed * math.cos(self.yaw) * self.dt
        self.y += eff_speed * math.sin(self.yaw) * self.dt

        # Update curve progress by closest curve point
        best_idx, best_d = current_idx, float('inf')
        for i in range(max(0, current_idx - 2),
                       min(current_idx + 30, total_pts)):
            d = math.hypot(self.curve_points[i][0] - self.x,
                           self.curve_points[i][1] - self.y)
            if d < best_d:
                best_d = d
                best_idx = i
        self.curve_progress = best_idx / max(total_pts - 1, 1)

        # Check if curve is complete
        end_pt = self.curve_points[-1]
        dist_to_end = math.hypot(end_pt[0] - self.x, end_pt[1] - self.y)
        if self.curve_progress >= 0.92 or dist_to_end < 0.12:
            self.is_following_curve = False
            self.curve_progress = 0.0
            self.curve_points = []
            if self.overtake_merge_edge is not None:
                self.current_edge_idx = self.overtake_merge_edge
                self.overtake_merge_edge = None
            else:
                self._resync_edge_index()
            self.position_on_edge = 0.0

        self.total_distance += eff_speed * self.dt
        self.position_history.append((self.x, self.y))
        self.scene_manager.update(self.dt)

    def _resync_edge_index(self):
        """After a curve ends, find the closest path node and resume from there."""
        best_idx = self.current_edge_idx
        best_d = float('inf')
        search_end = min(len(self.path_nodes),
                         self.current_edge_idx + 30)
        for i in range(self.current_edge_idx, search_end):
            nx_ = float(self.graph.nodes[self.path_nodes[i]]['x'])
            ny_ = float(self.graph.nodes[self.path_nodes[i]]['y'])
            d = math.hypot(nx_ - self.x, ny_ - self.y)
            if d < best_d:
                best_d = d
                best_idx = i
        # Start from the nearest node (or one past if very close)
        self.current_edge_idx = min(best_idx, len(self.path_edges) - 1)

    def _edge_step(self, eff_speed):
        """Follow graph edges using MPC steering + bicycle model.

        Uses closest-point-on-path tracking: finds the nearest path node,
        advances edge_idx to it, then builds the reference path forward
        from there for pure-pursuit steering.
        """
        # --- Find closest remaining path node to car ---
        remaining_start = self.current_edge_idx
        search_end = min(len(self.path_nodes),
                         self.current_edge_idx + 40)
        best_i = self.current_edge_idx
        best_d = float('inf')
        for i in range(remaining_start, search_end):
            nx_ = float(self.graph.nodes[self.path_nodes[i]]['x'])
            ny_ = float(self.graph.nodes[self.path_nodes[i]]['y'])
            d = math.hypot(nx_ - self.x, ny_ - self.y)
            if d < best_d:
                best_d = d
                best_i = i

        # Advance edge_idx to the closest node (never go backward)
        if best_i > self.current_edge_idx:
            self.current_edge_idx = best_i

        if self.current_edge_idx >= len(self.path_edges):
            self.stopped = True
            self.stop_reason = "Reached destination"
            return

        # Build reference path from nearest node onward (plenty of lookahead)
        remaining = self.path_nodes[self.current_edge_idx:]
        ref_pts = [(float(self.graph.nodes[n]['x']),
                    float(self.graph.nodes[n]['y'])) for n in remaining[:20]]
        if len(ref_pts) < 2:
            self.stopped = True
            self.stop_reason = "Reached destination"
            return

        # MPC steering
        steer = self.mpc_controller.compute_steering(
            self.x, self.y, self.yaw, ref_pts, velocity=eff_speed)
        self.steer_angle = steer

        # Recompute speed now that steer_angle is up-to-date
        self.effective_speed_cms = self._compute_effective_speed_cms()
        eff_speed = self.effective_speed_cms / 100.0

        # Bicycle model kinematics
        self.yaw = self.mpc_controller.update_heading(self.yaw, eff_speed, steer)
        self.x += eff_speed * math.cos(self.yaw) * self.dt
        self.y += eff_speed * math.sin(self.yaw) * self.dt
        self.total_distance += eff_speed * self.dt

        self.position_history.append((self.x, self.y))
        self.scene_manager.update(self.dt)

    # ------------------------------------------------------------------
    def get_state(self):
        mode_str = self.current_mode.value.get('mode', 'unknown').lower() if self.current_mode else 'unknown'
        return {
            'x': self.x, 'y': self.y,
            'yaw': math.degrees(self.yaw),
            'steer_angle': self.steer_angle,
            'mode': mode_str,
            'time': self.current_time,
            'distance': self.total_distance,
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'detections': self.current_detections,
            'fov_polygon': self.fov.get_polygon(self.x, self.y, self.yaw),
            'parking_phase': self.parking_phase,
            'speed_cms': self.effective_speed_cms,
        }


# ============================================================================
# TKINTER RENDERER
# ============================================================================
class Camera:
    """World <-> screen coordinate transform."""

    def __init__(self, graph, map_w, map_h):
        xs = [float(graph.nodes[n]['x']) for n in graph.nodes()]
        ys = [float(graph.nodes[n]['y']) for n in graph.nodes()]
        self.wmin_x, self.wmax_x = min(xs) - 0.5, max(xs) + 0.5
        self.wmin_y, self.wmax_y = min(ys) - 0.5, max(ys) + 0.5
        self.ww = self.wmax_x - self.wmin_x
        self.wh = self.wmax_y - self.wmin_y
        self.map_w, self.map_h = map_w, map_h
        self.scale = min(map_w / self.ww, map_h / self.wh) * 0.95
        self.cx = map_w / 2
        self.cy = map_h / 2

    def w2s(self, wx, wy):
        """World -> screen."""
        sx = (wx - self.wmin_x - self.ww / 2) * self.scale + self.cx
        sy = (self.wmin_y + self.wh / 2 - wy) * self.scale + self.cy
        return int(sx), int(sy)

    def s2w(self, sx, sy):
        """Screen -> world."""
        wx = (sx - self.cx) / self.scale + self.wmin_x + self.ww / 2
        wy = self.wmin_y + self.wh / 2 - (sy - self.cy) / self.scale
        return wx, wy

    def dist_px(self, d):
        """World distance -> pixels."""
        return max(1, int(d * self.scale))


class TkinterRenderer:
    """Smooth interactive renderer using built-in tkinter Canvas.

    Static elements (track image, graph edges/nodes) are drawn once.
    Dynamic elements (car, FOV, trail, objects, info) are redrawn each frame
    via tag-based deletion, keeping the canvas item count low.
    """

    SCREEN_W, SCREEN_H = 1400, 780
    MAP_W   = 1060
    INFO_W  = 340
    TARGET_FPS = 60

    MODE_COLORS = {
        'straight': C_CYAN, 'turn': C_MAGENTA, 'overtaking': C_RED,
        'tailing': C_ORANGE, 'parking': C_LIME,
    }

    def __init__(self, sim, graph):
        self.sim   = sim
        self.graph = graph

        # ---- Tk root & canvas ----
        self.root = tk.Tk()
        self.root.title("BFMC 2026 — Comprehensive Simulation")
        self.root.geometry(f"{self.SCREEN_W}x{self.SCREEN_H}")
        self.root.resizable(False, False)
        self.root.configure(bg=C_DARK_BG)

        self.canvas = tk.Canvas(self.root, width=self.SCREEN_W, height=self.SCREEN_H,
                                bg=C_DARK_BG, highlightthickness=0)
        self.canvas.pack()

        self.camera = Camera(graph, self.MAP_W, self.SCREEN_H)

        self.paused     = False
        self.input_mode = 'normal'
        self.running    = True

        # Status bar
        self.status_msg   = "Press h for controls"
        self.status_timer = 3.0

        # Image references (prevent garbage-collection)
        self._bg_photo  = None          # track background PhotoImage
        self._sign_imgs = {}            # {sign_type: PhotoImage}

        # Draw persistent layers
        self._load_track_image()
        self._render_graph()
        self._load_sign_images()

        # Bindings
        self.root.bind('<KeyPress>', self._on_key_press)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Frame interval
        self._frame_ms = max(1, 1000 // self.TARGET_FPS)

        print("[OK] Tkinter renderer ready  |  Press h for controls")

    # ==============================================================
    # INIT HELPERS
    # ==============================================================
    def _load_track_image(self):
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'track', 'comp_track.png')
        if not os.path.exists(img_path):
            return
        try:
            pil_img = Image.open(img_path).convert('RGBA')
            cam = self.camera
            coeff = [0.98, 0.98]
            lt = cam.w2s(cam.wmin_x, cam.wmax_y)
            rb = cam.w2s(cam.wmax_x, cam.wmin_y)
            iw, ih = int((rb[0] - lt[0]) * coeff[0]), int((rb[1] - lt[1]) * coeff[1])
            if iw > 0 and ih > 0:
                offset = [int(rb[0] - lt[0] - iw) / 2, int((rb[1] - lt[1] - ih) / 2)]
                pil_img = pil_img.resize((iw, ih), Image.Resampling.LANCZOS)
                # Semi-transparent blend over dark background
                pil_img.putalpha(180)
                bg = Image.new('RGBA', pil_img.size, (26, 26, 26, 255))
                composited = Image.alpha_composite(bg, pil_img).convert('RGB')
                self._bg_photo = ImageTk.PhotoImage(composited)
                self.canvas.create_image(lt[0] + offset[0], lt[1] + offset[1], image=self._bg_photo,
                                         anchor='nw', tags='bg')
                print("[OK] Track background loaded")
        except Exception as e:
            print(f"[WARN] Background: {e}")

    def _render_graph(self):
        """Draw graph edges & nodes once (static — never deleted)."""
        cam = self.camera
        for e in self.graph.edges():
            s = cam.w2s(float(self.graph.nodes[e[0]]['x']), float(self.graph.nodes[e[0]]['y']))
            d = cam.w2s(float(self.graph.nodes[e[1]]['x']), float(self.graph.nodes[e[1]]['y']))
            self.canvas.create_line(s[0], s[1], d[0], d[1], fill=C_CYAN_DIM, tags='graph')
        for n in self.graph.nodes():
            s = cam.w2s(float(self.graph.nodes[n]['x']), float(self.graph.nodes[n]['y']))
            self.canvas.create_oval(s[0] - 1, s[1] - 1, s[0] + 1, s[1] + 1,
                                    fill=C_CYAN_DIM, outline='', tags='graph')

    def _load_sign_images(self):
        sign_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic_signs')
        for stype, fname in TrafficSign.IMAGE_MAP.items():
            p = os.path.join(sign_dir, fname)
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert('RGBA')
                    img = img.resize((24, 24), Image.Resampling.LANCZOS)
                    self._sign_imgs[stype] = ImageTk.PhotoImage(img)
                except Exception:
                    pass

    # ==============================================================
    # STATUS
    # ==============================================================
    def _status(self, msg, dur=2.0):
        self.status_msg   = msg
        self.status_timer = dur

    # ==============================================================
    # EVENT HANDLING
    # ==============================================================
    def _on_close(self):
        self.running = False
        self.root.destroy()

    def _on_key_press(self, event):
        key = event.keysym
        kl  = key.lower()

        if kl == 'space':
            self.paused = not self.paused
            self._status('PAUSED' if self.paused else 'PLAYING')
        elif kl in ('q', 'escape'):
            self._on_close()
        elif kl == 'backspace':
            self._restart()
        elif kl in ('equal', 'plus', 'kp_add'):
            self.sim.speed = min(10.0, self.sim.speed * 1.5)
            self._status(f'Speed: {self.sim.speed:.2f}')
        elif kl in ('minus', 'kp_subtract'):
            self.sim.speed = max(0.1, self.sim.speed * 0.67)
            self._status(f'Speed: {self.sim.speed:.2f}')
        elif kl == 'h':
            self._status(
                "r/y/g=Lights  b+1-9=Signs  c=StaticCar  m=MovCar  p=ParkSpot  i=Inter  Del=Remove",
                dur=5.0)
        elif kl == 'r':
            self.input_mode = 'add_red_light';    self._status('Click → Red Light')
        elif kl == 'y':
            self.input_mode = 'add_yellow_light'; self._status('Click → Yellow Light')
        elif kl == 'g':
            self.input_mode = 'add_green_light';  self._status('Click → Green Light')
        elif kl == 'b':
            self.input_mode = 'select_sign'
            self._status('1=Stop 2=Ped 3=Park 4=HwyIn 5=HwyOut 6=OneW 7=Prior 8=Rnd 9=NoEnt', dur=5.0)
        elif kl == 'c':
            self.input_mode = 'add_vehicle_static'; self._status('Click → Stationary Vehicle')
        elif kl == 'm':
            self.input_mode = 'add_vehicle_moving'; self._status('Click → Moving Vehicle')
        elif kl == 'p':
            self.input_mode = 'add_parking_spot';   self._status('Click → Parking Spot')
        elif kl == 'i':
            self.input_mode = 'add_intersection';   self._status('Click → Intersection')
        elif kl == 'delete':
            self.input_mode = 'remove' if self.input_mode != 'remove' else 'normal'
            self._status('REMOVE mode ON' if self.input_mode == 'remove' else 'Normal')
        elif kl in ('1','2','3','4','5','6','7','8','9') and self.input_mode == 'select_sign':
            sign_types = [
                'stop_sign', 'pedestrian_sign', 'parking_sign',
                'enter_highway_sign', 'leave_highway_sign',
                'oneway_sign', 'priority_sign', 'roundabout_sign', 'notallowed_sign',
            ]
            idx = int(kl) - 1
            if idx < len(sign_types):
                self.input_mode = f'add_{sign_types[idx]}'
                self._status(f"Click → {sign_types[idx].replace('_', ' ').title()}")

    def _on_canvas_click(self, event):
        if event.x >= self.MAP_W:
            return                          # click on info panel — ignore
        wx, wy = self.camera.s2w(event.x, event.y)
        self._on_click(wx, wy)

    def _on_click(self, wx, wy):
        m = self.input_mode
        if m == 'remove':
            self._status('Removed' if self.sim.scene_manager.remove_nearest(wx, wy) else 'Nothing nearby')

        elif m.startswith('add_') and m.endswith('_light'):
            colour = m.replace('add_', '').replace('_light', '')
            self.sim.scene_manager.add_light(wx, wy, colour)
            self._status(f'{colour.title()} light placed')
            self.input_mode = 'normal'

        elif m.startswith('add_') and m.endswith('_sign'):
            stype = m[4:]                   # strip 'add_'
            self.sim.scene_manager.add_sign(stype, wx, wy)
            self._status(f"{stype.replace('_', ' ').title()} placed")
            self.input_mode = 'normal'

        elif m == 'add_vehicle_static':
            self.sim.scene_manager.add_vehicle(wx, wy, moving=False, speed=0.0)
            self._status('Stationary vehicle placed')
            self.input_mode = 'normal'

        elif m == 'add_vehicle_moving':
            self.sim.scene_manager.add_vehicle(wx, wy, moving=True, speed=0.15,
                                               heading=self.sim.yaw)
            self._status('Moving vehicle placed')
            self.input_mode = 'normal'

        elif m == 'add_parking_spot':
            self.sim.scene_manager.add_parking_spot(wx, wy, angle=self.sim.yaw)
            self._status('Parking spot placed')
            self.input_mode = 'normal'

        elif m == 'add_intersection':
            self.sim.scene_manager.add_intersection(wx, wy)
            self._status('Intersection placed')
            self.input_mode = 'normal'

    def _restart(self):
        s = self.sim
        s.stopped = False
        s.current_time = 0.0
        s.total_distance = 0.0
        s.position_history.clear()
        s.current_edge_idx = 0
        s.position_on_edge = 0.0
        s.current_waypoint_idx = 0
        s.visited_waypoints.clear()
        s.is_following_curve = False
        s.curve_points = []
        s.curve_progress = 0.0
        s.parking_phase = None
        s.parking_spot = None
        s.tailing_target = None
        s.overtake_merge_edge = None
        s.x = float(self.graph.nodes[s.path_nodes[0]]['x'])
        s.y = float(self.graph.nodes[s.path_nodes[0]]['y'])
        s.yaw = 0.0
        s.position_history.append((s.x, s.y))
        self.paused = False
        self._status('Simulation reset')

    # ==============================================================
    # RENDERING
    # ==============================================================
    def _render(self):
        """Redraw all dynamic layers on top of the static background."""
        self.canvas.delete('dyn')           # wipe previous dynamic items
        cam   = self.camera
        state = self.sim.get_state()

        # 1. FOV polygon  (stipple simulates transparency)
        fov = state['fov_polygon']
        if len(fov) >= 3:
            pts = []
            for p in fov:
                pts.extend(cam.w2s(*p))
            self.canvas.create_polygon(pts, fill=C_GREEN, stipple='gray12',
                                       outline=C_GREEN, width=1, tags='dyn')

        # 2. Bezier curve
        if self.sim.is_following_curve and len(self.sim.curve_points) > 1:
            pts = []
            for p in self.sim.curve_points:
                pts.extend(cam.w2s(*p))
            self.canvas.create_line(pts, fill=C_MAGENTA, width=2, smooth=True, tags='dyn')

        # 3. Trajectory  (down-sample for performance when > 500 pts)
        history = list(self.sim.position_history)
        if len(history) > 500:
            step = len(history) // 500
            history = history[::step] + [history[-1]]
        if len(history) > 1:
            pts = []
            for p in history:
                pts.extend(cam.w2s(*p))
            if len(pts) >= 4:
                self.canvas.create_line(pts, fill=C_TRAIL, width=2, tags='dyn')

        # 4. Scene objects
        self._draw_scene()

        # 5. Waypoints
        for wn, _ in self.sim.waypoint_nodes:
            sx, sy = cam.w2s(float(self.graph.nodes[wn]['x']),
                             float(self.graph.nodes[wn]['y']))
            col  = C_LIME if wn in self.sim.visited_waypoints else C_CYAN
            ecol = C_DARK_GREEN if wn in self.sim.visited_waypoints else C_BLUE
            self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4,
                                    fill=col, outline=ecol, tags='dyn')

        # 6. Car  (direction triangle + centre dot)
        cx, cy = cam.w2s(state['x'], state['y'])
        yaw_s  = -state['yaw'] * math.pi / 180.0       # flip Y for screen
        sz = 8
        tip   = (cx + sz * math.cos(yaw_s),
                 cy + sz * math.sin(yaw_s))
        left  = (cx + sz * 0.6 * math.cos(yaw_s + 2.5),
                 cy + sz * 0.6 * math.sin(yaw_s + 2.5))
        right = (cx + sz * 0.6 * math.cos(yaw_s - 2.5),
                 cy + sz * 0.6 * math.sin(yaw_s - 2.5))
        self.canvas.create_polygon(tip[0], tip[1], left[0], left[1],
                                   right[0], right[1],
                                   fill=C_RED, outline='', tags='dyn')
        self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                fill='#FF7878', outline='', tags='dyn')

        # 7. Info panel
        self._draw_info(state)

        # 8. Status bar
        if self.status_timer > 0:
            self.canvas.create_text(10, self.SCREEN_H - 15,
                                    text=self.status_msg, fill=C_YELLOW,
                                    anchor='w', font=('Consolas', 10), tags='dyn')

    # ----------------------------------------------------------
    def _draw_scene(self):
        cam = self.camera

        # Signs
        for sign in self.sim.scene_manager.signs.values():
            sx, sy = cam.w2s(sign.x, sign.y)
            if sign.sign_type in self._sign_imgs:
                self.canvas.create_image(sx, sy, image=self._sign_imgs[sign.sign_type],
                                         tags='dyn')
            else:
                col = TrafficSign.SIGN_COLORS.get(sign.sign_type, C_YELLOW)
                r = cam.dist_px(sign.radius)
                self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                        fill=col, outline=C_WHITE, tags='dyn')

        # Lights
        for light in self.sim.scene_manager.lights.values():
            sx, sy = cam.w2s(light.x, light.y)
            col = TrafficLight.COLORS[light.state]
            r = cam.dist_px(light.radius)
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                    fill=col, outline='#282828', width=2, tags='dyn')

        # Vehicles
        for veh in self.sim.scene_manager.vehicles.values():
            sx, sy = cam.w2s(veh.x, veh.y)
            r = cam.dist_px(veh.radius)
            col = C_ORANGE if veh.moving else C_RED
            self.canvas.create_rectangle(sx - r, sy - r, sx + r, sy + r,
                                         fill=col, outline=C_WHITE, tags='dyn')
            if veh.moving:
                self.canvas.create_text(sx, sy, text='M', fill=C_WHITE,
                                        font=('Consolas', 8), tags='dyn')

        # Parking spots
        for spot in self.sim.scene_manager.parking_spots.values():
            corners = spot.get_corners()
            pts = []
            for c in corners:
                pts.extend(cam.w2s(*c))
            col = C_GRAY if spot.occupied else C_DARK_GREEN
            self.canvas.create_polygon(pts, fill='', outline=col, width=2, tags='dyn')
            if not spot.occupied:
                sc = cam.w2s(spot.x, spot.y)
                self.canvas.create_text(sc[0], sc[1], text='P', fill=C_LIME,
                                        font=('Consolas', 9, 'bold'), tags='dyn')

        # Intersections
        for inter in self.sim.scene_manager.intersections.values():
            sx, sy = cam.w2s(inter.x, inter.y)
            for a in range(0, 360, 45):
                r = math.radians(a)
                ex = int(sx + 8 * math.cos(r))
                ey = int(sy + 8 * math.sin(r))
                self.canvas.create_line(sx, sy, ex, ey,
                                        fill=C_PURPLE, width=2, tags='dyn')

    # ----------------------------------------------------------
    def _draw_info(self, state):
        """Render the right-side information panel."""
        tag = 'info'
        self.canvas.delete(tag)

        px = self.MAP_W

        # Panel background & separator
        self.canvas.create_rectangle(px, 0, self.SCREEN_W, self.SCREEN_H,
                                     fill=C_DARK_BG, outline='', tags=tag)
        self.canvas.create_line(px, 0, px, self.SCREEN_H,
                                fill=C_DARK_GRAY, width=2, tags=tag)

        mode_col = self.MODE_COLORS.get(state['mode'], C_WHITE)
        sp_cms = state.get('speed_cms', 0.0)
        sp_enum = self.sim.mode_changer._get_speed()
        sp_label = sp_enum.name if hasattr(sp_enum, 'name') else '?'
        status = 'PAUSED' if self.paused else ('STOPPED' if state['stopped'] else 'RUNNING')
        wpp = f"{len(self.sim.visited_waypoints)}/{len(self.sim.waypoint_nodes)}"

        rows = [
            ('Mode',    state['mode'].upper(),                             mode_col),
            ('Speed',   f'{sp_cms:.1f} cm/s  ({sp_label})',               C_WHITE),
            ('',        '',                                                C_WHITE),
            ('Time',    f"{state['time']:.1f}s",                           C_WHITE),
            ('Dist',    f"{state['distance']:.1f}m",                       C_WHITE),
            ('',        '',                                                C_WHITE),
            ('Pos',     f"({state['x']:.2f}, {state['y']:.2f})",          C_WHITE),
            ('Yaw',     f"{state['yaw']:.0f} deg",                         C_WHITE),
            ('Steer',   f"{state['steer_angle']:.0f} deg",                 C_WHITE),
            ('',        '',                                                C_WHITE),
            ('Detect',  ', '.join(state['detections'][:2]) or 'None',
                        C_YELLOW if state['detections'] else C_GRAY),
            ('',        '',                                                C_WHITE),
            ('Signs',   str(len(self.sim.scene_manager.signs)),            C_WHITE),
            ('Lights',  str(len(self.sim.scene_manager.lights)),           C_WHITE),
            ('Cars',    str(len(self.sim.scene_manager.vehicles)),         C_WHITE),
            ('Parking', str(len(self.sim.scene_manager.parking_spots)),    C_WHITE),
            ('Inter',   str(len(self.sim.scene_manager.intersections)),    C_WHITE),
            ('',        '',                                                C_WHITE),
            ('Waypts',  wpp,                                               C_WHITE),
            ('Status',  status,
                        C_GREEN if status == 'RUNNING' else C_YELLOW),
        ]
        if state.get('parking_phase'):
            rows.append(('Park', state['parking_phase'].upper(), C_LIME))

        # Title
        self.canvas.create_text(px + 10, 15, text='BFMC 2026 SIM', fill=C_GREEN,
                                anchor='w', font=('Consolas', 14, 'bold'), tags=tag)
        self.canvas.create_line(px + 5, 35, px + self.INFO_W - 5, 35,
                                fill=C_DARK_GRAY, tags=tag)

        y = 50
        for lbl, val, col in rows:
            if lbl == '' and val == '':
                y += 8
                continue
            self.canvas.create_text(px + 80, y, text=f'{lbl}:', fill=C_GRAY,
                                    anchor='e', font=('Consolas', 10), tags=tag)
            self.canvas.create_text(px + 85, y, text=val, fill=col,
                                    anchor='w', font=('Consolas', 10), tags=tag)
            y += 18

        # Help footer
        y = self.SCREEN_H - 80
        self.canvas.create_line(px + 5, y, px + self.INFO_W - 5, y,
                                fill=C_DARK_GRAY, tags=tag)
        for line in ["h=Help  space=Pause",
                      "Bksp=Reset  +/-=Speed",
                      "Click map to place objects"]:
            y += 16
            self.canvas.create_text(px + 10, y, text=line, fill='#505050',
                                    anchor='w', font=('Consolas', 9), tags=tag)

    # ==============================================================
    # MAIN LOOP
    # ==============================================================
    def _tick(self):
        if not self.running:
            return
        t0 = _time.perf_counter()

        if not self.paused and not self.sim.stopped:
            self.sim.update()

        self._render()

        if self.status_timer > 0:
            self.status_timer -= 1.0 / self.TARGET_FPS

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        delay = max(1, self._frame_ms - int(elapsed_ms))
        self.root.after(delay, self._tick)

    def run(self):
        self.root.after(10, self._tick)
        self.root.mainloop()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='BFMC 2026 Car Simulation (Tkinter)')
    parser.add_argument('--start',     type=int,   default=342, help='Start node ID')
    parser.add_argument('--end',       type=int,   default=98,  help='End node ID')
    parser.add_argument('--waypoints', type=str,   default=None,
                        help='Comma-separated waypoint nodes  e.g. "342,100,200,98"')
    parser.add_argument('--speed',     type=float, default=2.0, help='Car speed (units/s)')
    parser.add_argument('--headless',  action='store_true',     help='Run without GUI')
    parser.add_argument('--dt',        type=float, default=0.05, help='Time step (s)')
    parser.add_argument('--curve-type', type=str,   default='bezier3',
                        choices=['bezier2', 'bezier3', 'catmull-rom',
                                 'hermite', 'bspline', 'arc'],
                        help='Curve type for turns')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BFMC 2026 COMPREHENSIVE CAR SIMULATION")
    print("=" * 60)

    waypoint_list = None
    if args.waypoints:
        waypoint_list = [int(n.strip()) for n in args.waypoints.split(',')]
        print(f"Multi-path: {' -> '.join(map(str, waypoint_list))}")
    else:
        print(f"Path: {args.start} -> {args.end}")
    print(f"Speed: {args.speed}  DT: {args.dt}  MPC: always-on  Curve: {args.curve_type}")

    graph_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'track', 'Competition_track_graph.graphml')
    if not os.path.exists(graph_file):
        print(f"[ERROR] Graph not found: {graph_file}")
        return

    try:
        scene = SceneManager()

        if waypoint_list:
            sim = ComprehensiveCarSimulator(
                graph_file, waypoint_list[0], None,
                speed=args.speed, dt=args.dt,
                scene_manager=scene, waypoint_path=waypoint_list,
                curve_type=args.curve_type)
        else:
            sim = ComprehensiveCarSimulator(
                graph_file, args.start, args.end,
                speed=args.speed, dt=args.dt,
                scene_manager=scene, curve_type=args.curve_type)

        if args.headless:
            frame = 0
            while not sim.stopped and sim.current_time < 1000:
                sim.update()
                frame += 1
                if frame % 20 == 0:
                    s = sim.get_state()
                    det = ', '.join(s['detections'][:2]) or 'None'
                    print(f"t={s['time']:6.1f} | {s['mode']:12s} | "
                          f"({s['x']:.2f}, {s['y']:.2f}) | {det}")
            s = sim.get_state()
            print(f"\n[DONE] {s['stop_reason']}  |  {s['time']:.1f}s  |  {s['distance']:.2f}m  |  curve: {args.curve_type}")
        else:
            graph = nx.read_graphml(graph_file)
            renderer = TkinterRenderer(sim, graph)
            renderer.run()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
