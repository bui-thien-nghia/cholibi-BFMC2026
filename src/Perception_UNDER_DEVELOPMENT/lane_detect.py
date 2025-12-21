import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def find_plateaus(img,
                  ascent_threshold=30,
                  winning_percentage=0.5,
                  ascent_to_plateau=20,
                  min_pixels_to_plateau=5,
                  lower_pixel_limit=6,
                  upper_pixel_limit=200,
                  confidence_interval=0.08):
    '''
    # Status:
    0. Did not pass threshold
    1. Segment not long enough
    2. Segment too long
    3. Did not pass winning threshold
    4. Did not pass ascent to plateau threshold
    5. Did not pass noise test
    6. PassedDocstring for find_plateaus
    '''
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    boundary_points = []
    status = [0, 0, 0, 0, 0, 0, 0]

    for i, row in enumerate(gray):
        row = row.tolist()
        j = 0
        while j < w - 1:
            # If no abrupt change: skip
            if abs(row[j] - row[j + 1]) < ascent_threshold:
                status[0] += 1
                j += 1
                continue
            
            # If abrupt change: find plateau
            ascent_value_pos = j
            ascent_value = row[ascent_value_pos]
            buffer = [j + 1]
            j += 2
            while j < w - 1 and abs(row[j] - row[j - 1]) < ascent_threshold:
                buffer.append(j)
                j += 1
            
            if len(buffer) < lower_pixel_limit:
                status[1] += 1
                continue
            if len(buffer) > upper_pixel_limit:
                status[2] += 1
                continue
            
            # Range Boyer-Moore voting for plateau
            plateau_pos = buffer[0]
            plateau = row[plateau_pos]
            counter = 0
            for pos in buffer:
                if counter == 0:
                    plateau_pos = pos
                    plateau = row[plateau_pos]
                    counter += 1
                else:
                    if (1 - confidence_interval) * plateau <= row[pos] <= (1 + confidence_interval) * plateau:
                        counter += 1
                    else:
                        counter -= 1

        
            # Check if plateau is stable & does not come from white noise
            counter = sum(1 for pos in buffer if (1 - confidence_interval) * plateau <= row[pos] <= (1 + confidence_interval) * plateau)

            if not counter >= len(buffer) * winning_percentage:
                status[3] += 1
                continue
            if not abs(plateau - ascent_value) >= ascent_to_plateau:
                status[4] += 1
                continue
            if not abs(plateau_pos - ascent_value_pos) <= min_pixels_to_plateau:
                status[5] += 1
                continue
            
            status[6] += 1
            boundary_points.append((plateau_pos, i))

    return boundary_points, status

def deproject_points(points, h, theta, f, k, u0, v0, eps=1e-8):
    world_points = []
    theta_rad = theta * np.pi / 180
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad) 
    alpha = f / k
    for u, v in points:
        norm_u = (u - u0) / alpha
        norm_v = (v - v0) / alpha
        # scale = h / (cos_theta * norm_v - sin_theta + eps)
        # world_points.append((norm_u * scale, norm_v * scale))
        scale = h / (sin_theta * norm_v - cos_theta + eps)
        world_points.append((norm_u * scale, (cos_theta * norm_v + sin_theta) * scale))
    
    return world_points


def dbscan_cluster(points, eps=0.5, min_samples=2, standardise=True):
    """
    Cluster points using DBSCAN algorithm.
    Separates left and right lane boundaries based on spatial proximity.
    
    Args:
        points: List of (x, y) coordinates
        eps: Distance threshold for clustering
        min_samples: Minimum points per cluster
        standardise: Whether to standardize features
    
    Returns:
        List of cluster labels (noise points labeled as -1)
    """
    if len(points) < min_samples:
        return [-1] * len(points)
    
    rescaled_points = StandardScaler().fit_transform(np.array(points.copy())) if standardise else points
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(rescaled_points)

    return clusters.labels_.tolist()


def find_lane_points(points, labels):
    points_with_labels = {}
    for point, label in zip(points, labels):
        if label == -1:
            continue
        label = str(label)
        if label not in points_with_labels:
            points_with_labels[label] = [point]
        else:
            points_with_labels[label].append(point)

    max_len = 0
    second_max_len = 0
    first_lane_label = '0'
    second_lane_label = '0'
    for i in range(len(points_with_labels)):
        i = str(i)
        points_cnt = len(points_with_labels[i])
        if points_cnt >= max_len:
            if points_cnt != max_len:
                second_max_len = max_len
                second_lane_label = first_lane_label
            max_len = points_cnt
            first_lane_label = i
        elif len(points_with_labels[i]) > second_max_len:
            second_max_len = points_cnt
            second_lane_label = i

    return points_with_labels[first_lane_label], points_with_labels[second_lane_label]



def fit_polynomial(points, degree=2, threshold=10):
    if len(points) < threshold:
        return None
    
    try:
        points = np.array(points)
        sorted_points = points[points[:, 0].argsort()]
        coefficients = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], degree)
        return np.poly1d(coefficients)
    except Exception as e:
        print(f"Polynomial fitting error: {e}")
        return None


def suggest_path(left_lane, right_lane):
    left_lane = np.array(left_lane)
    right_lane = np.array(right_lane)
    left_lane = left_lane[left_lane[:, 1].argsort()]
    right_lane = right_lane[right_lane[:, 1].argsort()]

    min_y_left = np.min(left_lane[:, 1])
    max_y_left = np.max(left_lane[:, 1])
    min_y_right = np.min(right_lane[:, 1])
    max_y_right = np.max(right_lane[:, 1])

    suggested_points = []
    for y in range(min(min_y_left, min_y_right), max(max_y_left, max_y_right)):
        buffer_left = left_lane[left_lane[:, 1] == y]
        buffer_right = right_lane[right_lane[:, 1] == y]
        if not len(buffer_left) or not len(buffer_right):
            continue

        mean_x = (np.mean(buffer_left[:, 0]) + np.mean(buffer_right[:, 0])) / 2
        suggested_points.append((mean_x, y))

    return fit_polynomial(suggested_points)

def display_points(img, boundary_points):
    img = cv2.imread(img)
    for point in boundary_points:
        cv2.circle(img, point, 2, (0, 255, 0), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Boundary points found')


def display_clusters(points, labels, cmap='rainbow', dim=([],[])):
    """
    Display clustered points with different colors for each cluster.
    Visualizes the DBSCAN clustering results for left/right lane separation.
    
    Args:
        points: Array of (x, y) coordinates
        labels: Cluster labels from DBSCAN (-1 for noise points)
        cmap: Colormap name (default 'rainbow')
    """
    points = np.array(points.copy())
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if num_clusters == 0:
        print("No valid clusters to display")
        return
    
    colors = plt.colormaps[cmap](np.linspace(0, 1, num_clusters))
    
    for x, y, label in zip(points[:, 0], points[:, 1], labels):
        if label == -1:  # Skip noise points
            continue
        plt.scatter(x, -y, c=[colors[label]])
    
    plt.xlabel('x-camera')
    plt.ylabel('y-camera')
    if len(dim) == 2:
        if len(dim[0]) == 2:
            plt.xlim(dim[0][0], dim[0][1])
        elif len(dim[0]) == 1:
            plt.xlim(0, dim[0][0])

        if len(dim[1]) == 2:
            plt.ylim(dim[1][0], dim[1][1])
        elif len(dim[1]) == 1:
            plt.ylim(0, dim[1][0])
    plt.title(f'Clustered potential boundary points: {num_clusters} clusters detected')


def display_polynomials(list_polynomials, cmap='rainbow', dim=()):
    x = np.linspace(-100, 1000, 1000)
    colors = plt.colormaps[cmap](np.linspace(0, 1, len(list_polynomials)))
    for i, poly in enumerate(list_polynomials):
        y = poly(x)
        plt.plot(x, -y, label=f'Line {i}', color=tuple(colors[i].tolist()))
    
    plt.xlabel('x-camera')
    plt.ylabel('y-camera')
    plt.title(f'Lane detection result')
    if len(dim) == 2:
        if len(dim[0]) == 2:
            plt.xlim(dim[0][0], dim[0][1])
        elif len(dim[0]) == 1:
            plt.xlim(0, dim[0][0])

        if len(dim[1]) == 2:
            plt.ylim(dim[1][0], dim[1][1])
        elif len(dim[1]) == 1:
            plt.ylim(0, dim[1][0])
    plt.legend()