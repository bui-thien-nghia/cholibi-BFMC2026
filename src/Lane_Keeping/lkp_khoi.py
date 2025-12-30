import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# them cac import de matplotlib chay headless, khong can gui, tranh loi khi chay tren ubuntu

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use('Agg')


def find_plateaus(img,
                  ascent_threshold=50,
                  winning_percentage=0.4,
                  ascent_to_plateau=20,
                  min_pixels_to_plateau=3,
                  lower_pixel_limit=5,
                  upper_pixel_limit=100,
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
    if type(img) == str:
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
        world_points.append((float(norm_u * scale), float((cos_theta * norm_v + sin_theta) * scale)))
    
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
    """
    Returns respectively the left and right lane.
    """
    if labels == [-1] or len(labels) == 0:
        return {}
    
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
    left_lane_label = '0'
    right_lane_label = '0'
    for i in range(len(points_with_labels)):
        i = str(i)
        points_cnt = len(points_with_labels[i])
        if points_cnt >= max_len:
            if points_cnt != max_len:
                second_max_len = max_len
                right_lane_label = left_lane_label
            max_len = points_cnt
            left_lane_label = i
        elif len(points_with_labels[i]) > second_max_len:
            second_max_len = points_cnt
            right_lane_label = i

    left_mean_x = float(np.mean(np.array(points_with_labels[left_lane_label])[:, 0]))
    right_mean_x = float(np.mean(np.array(points_with_labels[right_lane_label])[:, 0]))

    if left_mean_x >= right_mean_x:
        left_lane_label, right_lane_label = right_lane_label, left_lane_label

    return points_with_labels[left_lane_label], points_with_labels[right_lane_label]



def fit_polynomial(points, degree=2, threshold=10):
    """
    Fits a polynomial x = f(y) to the given points.
    Changed to fit y = f(x) for better lane representation.
    """
    if len(points) < threshold:
        return None
    
    try:
        points = np.array(points)
        sorted_points = points[points[:, 1].argsort()]
        # Sort diem theo index 1 (y) thay cho index 0 (x)
        coefficients = np.polyfit(sorted_points[:, 1], sorted_points[:, 0], degree)
        return np.poly1d(coefficients)
    except Exception as e:
        print(f"Polynomial fitting error: {e}")
        return None


def suggest_path(left_lane_points, right_lane_points, diff_y=10): 
    left_lane_points = np.array(left_lane_points)
    right_lane_points = np.array(right_lane_points)
    

    
    min_y_left = np.min(left_lane_points[:, 1])
    max_y_left = np.max(left_lane_points[:, 1])
    min_y_right = np.min(right_lane_points[:, 1])
    max_y_right = np.max(right_lane_points[:, 1])

    left_lane_points = left_lane_points[left_lane_points[:, 1].argsort()]
    right_lane_points = right_lane_points[right_lane_points[:, 1].argsort()]

    suggested_points = []
    for y in np.arange(min(min_y_left, min_y_right), max(max_y_left, max_y_right)):
        buffer_left = left_lane_points[(left_lane_points[:, 1] >= y - diff_y) & (left_lane_points[:, 1] <= y + diff_y)]
        buffer_right = right_lane_points[(right_lane_points[:, 1] >= y - diff_y) & (right_lane_points[:, 1] <= y + diff_y)]
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
    # ve bang Y 
    y_max = 480
    if len(dim) == 2 and len(dim[1]) >= 1:
        y_max = dim[1][1] if len(dim[1]) == 2 else dim[1][0]

    y = np.linspace(0, y_max, 100)
    colors = plt.colormaps[cmap](np.linspace(0, 1, len(list_polynomials)))
    for i, poly in enumerate(list_polynomials):
        if poly == None: continue
        x = poly(y)
        plt.plot(x, -y, label=f'Line {i}', color=tuple(colors[i].tolist()))
    
    plt.xlabel('x-camera')
    plt.ylabel('y-camera')
    plt.title(f'Lane detection result')
    if len(dim) == 2:
        if len(dim[0]) == 2:
            plt.xlim(dim[0][0], dim[0][1])
        if len(dim[1]) == 2:
            plt.ylim(dim[1][0], dim[1][1])
    plt.legend()


def run_lane_detect(img, h, theta, f, k, eps=1e-8, use_deprojected=False, car_position=1024):
    if type(img) == str:
        img = cv2.imread(img)

    height, width, _ = img.shape
    points, _ = find_plateaus(img)
    
    deprojected = deproject_points(points, h, theta, f, k, width / 2, height / 2, eps=eps)
    labels = dbscan_cluster(deprojected, eps=0.3)
    if use_deprojected:
        points = deprojected
    first_lane, second_lane = find_lane_points(points, labels)
    suggested_path_poly = suggest_path(first_lane, second_lane)
    first_lane_poly = fit_polynomial(first_lane)
    second_lane_poly = fit_polynomial(second_lane)
    
    return suggested_path_poly, first_lane_poly, second_lane_poly


def add_lanes_to_image(img, list_polynomials, points_per_polynomial=100):
    if type(img) == str:
        img = cv2.imread(img)

    h, w = img.shape[:2]
    
    # Duyệt các điểm theo chiều dọc (y) từ trên xuống dưới
    for y in np.linspace(0, h-1, points_per_polynomial):
        
        # Duyệt qua từng đường line (poly)
        for poly in list_polynomials:
            if poly is None: continue
            
            try:
                # Tinh toạ độ x theo y, khong can giai root nx vi da fit x = f(y) o tren
                x_val = poly(y)
                
                # Kiểm tra nếu x nằm trong phạm vi chiều rộng ảnh thì mới vẽ
                if 0 <= x_val < w:
                    cv2.circle(img, (int(x_val), int(y)), 2, (0, 255, 0), 2)
            except Exception:
                pass

    return img

def find_x_at_y(poly, target_y, img_width=2048):
    if poly is None:
        return None
    try:
        x_val = poly(target_y)
        if 0 <= x_val < img_width:
            return float(x_val)
    except:
        pass
    return None

def calculate_centre_distance(path_poly, car_position, target_y=None, img_height=None):
    if path_poly is None:
        return None
    if target_y is None:
        if img_height is None:
            print("ERROR: img_height must be provided if target_y is not given.")
            return None
        target_y = img_height - 1
    path_x = find_x_at_y(path_poly, target_y)
    distance = car_position- path_x
    return distance

def main():
    # Get the directory where this script is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # img_path = os.path.join(script_dir, 'test.jpg')
    img_path='test.jpg'  # Hoặc thay đổi đường dẫn tới ảnh của bạn ở đây
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"ERROR: Không thể đọc file ảnh '{img_path}'.")
        print("Hãy đảm bảo file ảnh nằm cùng thư mục hoặc sửa lại đường dẫn trong code.")
        return
    
    # In file anh dang dung
    print(f"\n=== Using Image File ===")
    print(f"Image file: {img_path}")

    h = 0.225
    theta = 15
    f = 2.75e-3
    k = 1.3e-6
    
    # In cac tham so dang dung
    print("\n=== Lane Detection Parameters ===")
    print(f"h = {h}")
    print(f"theta = {theta}")
    print(f"f = {f}")
    print(f"k = {k}")

    

    path_poly, first_lane_poly, second_lane_poly = run_lane_detect(
        img,
        h=h,
        theta=theta,
        f=f,
        k=k,
        use_deprojected=False
    )
    print("\n=== Lane Detection Results ===")
    print(f"Suggested path: {path_poly}")
    print(f"First path: {first_lane_poly}")
    print(f"Second path: {second_lane_poly}")

    if path_poly is None:
        print("\nNo path detected, exiting.")
        return
    

    img_with_lanes = add_lanes_to_image(img.copy(), [path_poly, first_lane_poly, second_lane_poly])

    height, width= img.shape[:2]
    car_position = width / 2
    centre_dist = calculate_centre_distance(path_poly, car_position, target_y=height-1, img_height=height)

    if centre_dist is not None:
        print(f"\n=== Centre Distance Calculation ===")
        print(f"Center distance at bottom of image: {centre_dist:.2f} pixels")
        print(f"Path position at y = {height-1}: {car_position - centre_dist:.2f} px")
        print(f"Distance center: {centre_dist:.2f} px")

        if centre_dist == 0:
            print("Car is ON the path")
        elif centre_dist < 0:
            print("Car is LEFT of the path")
        elif centre_dist > 0:
            print("Car is RIGHT of the path")

        cv2.putText(img_with_lanes, f'Distance: {centre_dist:.2f} px', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.circle(img_with_lanes, (int(car_position), height-1), 10, (255, 0, 0), -1)
    
    else:
        print("\nWarning: Cannot calculate center distance")
    
    
    
    output_file = 'lane_detection_result.png'
    cv2.imwrite(output_file, img_with_lanes)
    print(f"\n=== Output ===")
    print(f"Result saved to {output_file}")

    print("\n=== Path X at Custom Y Positions ===")
    test_y = int(input("Enter y: "))
    x = find_x_at_y(path_poly, test_y, img_width=width)
    if x is not None:
        print(f"Path x at y={test_y:3d}: {x:.2f} px")
    else:
        print(f"Cannot find path x at y={test_y:3d}")

    
if __name__ == "__main__":
    main()
