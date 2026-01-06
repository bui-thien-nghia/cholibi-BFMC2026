import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def find_plateaus(img,
                  ascent_threshold=30,       
                  winning_percentage=0.4,
                  ascent_to_plateau=10,      
                  min_pixels_to_plateau=2,
                  lower_pixel_limit=3,       
                  upper_pixel_limit=150,     
                  confidence_interval=0.2):  
    """
    Scans image rows to find 'plateaus' of brightness (white lines) 
    that stand out from the background.
    """
    if isinstance(img, str):
        img = cv2.imread(img)

    # 1. Grayscale & Blur
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Strong blur to smooth out the shiny tarp noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, w = gray.shape
    boundary_points = []
    
    scan_rows = range(20, gray.shape[0], 4)

    for i in scan_rows:
        row = gray[i, :]
        
        grad = np.abs(np.diff(row))
        
        # Dynamic Thresholding: Adapts to lighting conditions of the row
        row_mean_grad = np.mean(grad)
        dynamic_thresh = row_mean_grad + (2.0 * np.std(grad))
        dynamic_thresh = max(ascent_threshold, min(dynamic_thresh, 80)) # Clamp

        peaks = np.where(grad > dynamic_thresh)[0]
        
        # Analyze potential lines (plateaus) between gradient peaks
        for k in range(len(peaks) - 1):
            p1 = peaks[k]
            p2 = peaks[k+1]
            width = p2 - p1
            
            # 1. Width Check
            if not (lower_pixel_limit < width < upper_pixel_limit):
                continue
            
            # 2. Brightness Check (Is the plateau actually bright?)
            center_idx = (p1 + p2) // 2
            plateau_val = row[center_idx]
            
            # Only accept if it's brighter than the "dark" surrounding
            # Simple check: Is it reasonably white? (>60/255)
            if plateau_val > 60:
                boundary_points.append((center_idx, i))

    return boundary_points

def deproject_points(points, h, theta, f, k, u0, v0, eps=1e-8):
    """
    Converts Image Pixels (u, v) -> World Coordinates (x, y) in cm.
    """
    world_points = []
    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad) 
    
    # Pre-calculate alpha
    alpha = f * k 
    
    for u, v in points:
    
        u_c = u - u0
        v_c = v - v0
        
        # Avoid division by zero at the horizon
        denom = (sin_theta * v_c) + (f * cos_theta)
        if abs(denom) < eps: continue
            
        beta = np.arctan2(v_c, f)
        total_angle = theta_rad + beta
        
        if total_angle <= 0: continue # Point is above horizon
        
        dist_y = h / np.tan(total_angle)
        dist_x = u_c * dist_y / np.sqrt(u_c**2 + f**2) # Approximate lateral
        # Better lateral approximation: x = u_c * dist_y / f (small angle approx)
        dist_x = (u_c * dist_y) / f

        # Filter points that are too far (noisy) or too close (hood)
        if 5 < dist_y < 300: 
            world_points.append((float(dist_x), float(dist_y)))
    
    return world_points

def dbscan_cluster(points, eps=15.0, min_samples=3):

    if len(points) < min_samples:
        return [-1] * len(points)
    
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    return clusters.labels_.tolist()

def find_lane_polynomials(points, labels):
    """
    Separates points into Left/Right groups and fits polynomials.
    """
    if not points or not labels: return None, None
    
    clusters = {}
    for pt, label in zip(points, labels):
        if label == -1: continue # Noise
        if label not in clusters: clusters[label] = []
        clusters[label].append(pt)
    
    if not clusters: return None, None

    # Sort clusters by size (biggest cluster is likely a lane)
    sorted_labels = sorted(clusters.keys(), key=lambda l: len(clusters[l]), reverse=True)
    
    left_cluster = []
    right_cluster = []
    
    # Logic: If we have at least 2 clusters, picking the biggest two
    # determining which is Left vs Right based on Mean X position.
    
    # 1. Extract best clusters
    c1 = np.array(clusters[sorted_labels[0]])
    mean_x1 = np.mean(c1[:, 0])
    
    if len(sorted_labels) >= 2:
        c2 = np.array(clusters[sorted_labels[1]])
        mean_x2 = np.mean(c2[:, 0])
        
        if mean_x1 < mean_x2:
            left_cluster = c1
            right_cluster = c2
        else:
            left_cluster = c2
            right_cluster = c1
    else:
        # Only one lane found
        if mean_x1 < 0: left_cluster = c1
        else: right_cluster = c1

    # 2. Fit Polynomials (x = ay^2 + by + c)
    def fit(pts):
        if len(pts) < 5: return None
        # Fit X as a function of Y (since lines are vertical-ish)
        # Poly: x = f(y)
        return np.poly1d(np.polyfit(pts[:, 1], pts[:, 0], 2))

    left_poly = fit(left_cluster) if len(left_cluster) > 0 else None
    right_poly = fit(right_cluster) if len(right_cluster) > 0 else None
    
    return left_poly, right_poly

def suggest_path(left_poly, right_poly):
    """
    Calculates the center path polynomial.
    """
    if left_poly and right_poly:
        # Average the coefficients
        avg_coeffs = (left_poly.coeffs + right_poly.coeffs) / 2
        return np.poly1d(avg_coeffs)
    elif left_poly:
        # Shift Left lane to right by assumed lane width (e.g. 37cm)
        # Note: This is rough, the Controller handles this better dynamically
        # but we return a visual guide here.
        c = left_poly.coeffs.copy()
        c[-1] += 37.0 
        return np.poly1d(c)
    elif right_poly:
        c = right_poly.coeffs.copy()
        c[-1] -= 37.0
        return np.poly1d(c)
    return None

def run_lane_detect(img, h, theta, f, k, eps=15.0, use_deprojected=True):
    """
    Main Pipeline Function
    """
    # 1. Find potential white line pixels (Image Space)
    points_uv = find_plateaus(img)
    if not points_uv: return None, None, None

    height, width = img.shape[:2] if not isinstance(img, str) else cv2.imread(img).shape[:2]

    # 2. Convert to World Space (cm)
    if use_deprojected:
        points_world = deproject_points(points_uv, h, theta, f, k, width/2, height/2)
        processing_points = points_world
    else:
        processing_points = points_uv

    if not processing_points: return None, None, None

    # 3. Cluster points to separate lanes
    labels = dbscan_cluster(processing_points, eps=eps)

    # 4. Fit lines
    left_poly, right_poly = find_lane_polynomials(processing_points, labels)
    
    # 5. Calculate Center Path
    path_poly = suggest_path(left_poly, right_poly)
    
    return path_poly, left_poly, right_poly
