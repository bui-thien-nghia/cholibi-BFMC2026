import numpy as np
import cv2

class Config:
    # --- 0. Image Path ---
    IMAGE = 'road7.jpeg'
    # --- 1. ROI (Region of Interest) ---
    # Cut the upper region and the under region from the roi
    ROI_START_Y = 0.45 # start taking the roi from ROI_START_Y% of the image | DEF 0.45
    ROI_END_Y = 0.85 # end of the roi | DEF 0.85
    
    # --- 2. Preprocessing Values --- 
    GAMMA_VALUE = 1.0 # the brightness (increase if running in heavily-shaded area) | DEF 1.0
    MORPH_KERNEL_RATIO = 0.01 # the ratio of the searching kernel comparing to the image width | DEF 0.01
    
    # --- 3. Adaptive HLS Values ---
    # The lowest threshold put to prevent encountering noise in the dark areas
    L_THRESH_MIN_FLOOR = 150 # the floor threshold for Light channel (white) | DEF 150
    S_THRESH_MIN_FLOOR = 100 # the floor threshold for Saturation channel (yellow) | DEF 100

    # Offset to calculate the adaptive threshold
    L_THRESH_OFFSET = 50 # the offset to subtract from the max Light channel value | DEF 50
    S_THRESH_OFFSET = 60 # the offset to subtract from the max Saturation channel value | DEF 60

    # --- 4. Curve Fitting Values ---
    SLOPE_THRESH = 0.5 # the minimum slope to consider a line | DEF 0.5
    ANCHOR_TOLERANCE = 50 # the tolerance from the center to consider left/right line | DEF 50
    INTERCEPT_MARGIN = 100 # the margin to consider the intercept of the curve | DEF 100

    # --- 5. Hough Transform Values ---
    HOUGH_THRESH = 100 # the minimum number of votes (intersections in Hough grid cell) | DEF 100
    HOUGH_MIN_LEN = 40 
    HOUGH_MAX_GAP = 5 # this let the hough transform connect the cut line
    
# functions to automatically canny using the calculated median lower and upper value
# choose between this and otsu
# is ever used, calib the sigma value again
def auto_canny(image, sigma=0.33):
    
    v = np.median(image)

    # default sigma is about 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    
    edged = cv2.Canny(image, lower, upper)
    return edged

# functions to canny using the otsu binarization (thresholding)
# choose between this and auto_canny above
def otsu_canny(img):
    high_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edged = cv2.Canny(img, int(low_thresh), int(high_thresh))
    # print(low_thresh)
    # print(high_thresh)
    return edged

# function to adjust the gamma value in case running in dark areas
# NOW UNUSED!!!
def adjust_gamma(image, gamma=1.0):
    
    # prevent to divide by zero in next step
    if gamma == 0: 
        gamma = 0.01
        
    invGamma = 1.0 / gamma # inversing the gamma value

    # construct the lookup table
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

# main process pipeline
def process_pipeline_adaptive(roi_img, morph=True): 
    
    binary_mask = adaptive_hls_binary(roi_img)

    # using morphological with both open and close including proper dilation and erosion
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) # kernel size is now (5, 5)

        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    closed_canny = cv2.Canny(binary_mask, 100, 200)
    
    
    return closed_canny

# function to get the kernel based on image shape
# NOW UNUSED!!!
# if used, please calib the ratio first
def get_adaptive_kernel(image, ratio=0.01):

    h, w = image.shape[:2]
    
    k_size = int(w * ratio)
    if k_size % 2 == 0: 
        k_size += 1
        
    k_size = max(3, k_size)
    
    print(k_size)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))


def fit_curves_from_points(lines, img_width):
    
    if lines is None: return None, None

    mid_width = img_width // 2 # Tìm tâm ảnh
    
    left_x, left_y = [], []
    right_x, right_y = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        if x2 - x1 == 0: continue
        slope = (y2 - y1) / (x2 - x1)
        
        if abs(slope) < 0.5: continue 

        
        if slope < 0 and x1 < mid_width and x2 < mid_width: 
            left_x.extend([x1, x2])
            left_y.extend([y1, y2])
            
        elif slope > 0 and x1 > mid_width and x2 > mid_width:
            right_x.extend([x1, x2])
            right_y.extend([y1, y2])

    


    left_fit = None
    right_fit = None

    # fit the second order poly with polyfit returning 3 coeffs
    if len(left_x) > 0:
        try:
            left_fit = np.polyfit(left_y, left_x, 2)
        except: pass 

    if len(right_x) > 0:
        try:
            right_fit = np.polyfit(right_y, right_x, 2)
        except: pass

    return left_fit, right_fit

# get the point to draw
def get_curve_points(image_shape, fit_coeffs, y_offset=0):
   
    if fit_coeffs is None: return None
    
    height, width = image_shape[:2]
    
    # in the region of interest, init the point from 0 to height
    plot_y = np.linspace(0, height - 1, num=height) 
    
    try:
        # find x by y
        plot_x = fit_coeffs[0] * plot_y**2 + fit_coeffs[1] * plot_y + fit_coeffs[2]
    except TypeError:
        return None


    # creating array x y
    # plus the y_offset to get in the right position in the original img
    pts = np.array([np.transpose(np.vstack([plot_x, plot_y + y_offset]))])
    return np.int32(pts)

def draw_equations(img, left_fit, right_fit):
    h, w = img.shape[:2]
    
    # scale based on img width
    font_scale = max(0.6, w / 1000.0)
    
    # thickness also based on font_scale (width)
    thickness = max(1, int(font_scale * 2))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255) 
    bg_color = (0, 0, 0)    

    # offset from left margin
    margin_x = int(20 * font_scale)
    start_y = int(50 * font_scale)
    line_spacing = int(40 * font_scale) # line spacing

    def draw_text_with_outline(text, x, y):
        # black border
        cv2.putText(img, text, (x, y), font, font_scale, bg_color, thickness + 3) 
        # white words
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)       

    
    if left_fit is not None:
        text_left = f"L: x = {left_fit[0]:.4f}y^2 + {left_fit[1]:.2f}y + {left_fit[2]:.0f}"
        draw_text_with_outline(text_left, margin_x, start_y)
    else:
        draw_text_with_outline("L: Not Detected", margin_x, start_y)

    if right_fit is not None:
        text_right = f"R: x = {right_fit[0]:.4f}y^2 + {right_fit[1]:.2f}y + {right_fit[2]:.0f}"
        draw_text_with_outline(text_right, margin_x, start_y + line_spacing)
    else:
        draw_text_with_outline("R: Not Detected", margin_x, start_y + line_spacing)


def adaptive_hls_binary(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # find the brightest in the current roi
    max_l = np.max(l_channel) 
    # prevent from going to low 
    thresh_l = max(150, max_l - 50) 
    l_mask = cv2.inRange(l_channel, int(thresh_l), 255)
    
    # calc the threshold for yellow line
    max_s = np.max(s_channel)
    thresh_s = max(100, max_s - 60) 
    s_mask = cv2.inRange(s_channel, int(thresh_s), 255)
    
    # combine the two masks for yellow and white
    combined = cv2.bitwise_or(l_mask, s_mask)
    return combined

def main():
    img = cv2.imread(Config.IMAGE)
    if img is None: return

    height, width = img.shape[:2]

    # cutting roi
    roi_start_y = int(height * 0.45)
    roi_img = img[roi_start_y:height, :]
    

    canny_roi = process_pipeline_adaptive(roi_img, morph=True)

    # use hough to decide whether the pts belong to the line or not
    lines = cv2.HoughLinesP(canny_roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # calculate the curve
    left_fit, right_fit = fit_curves_from_points(lines, width)

    # draw on original img
    line_image = np.zeros_like(img)
    
    # left point drawing
    if left_fit is not None:
        pts_left = get_curve_points(roi_img.shape, left_fit, roi_start_y)

        if pts_left is not None:
            # draw poly connecting pts
            cv2.polylines(line_image, [pts_left], isClosed=False, color=(0, 0, 255), thickness=40) 

    # right point drawing
    if right_fit is not None:
        pts_right = get_curve_points(roi_img.shape, right_fit, roi_start_y)

        if pts_right is not None:
            cv2.polylines(line_image, [pts_right], isClosed=False, color=(255, 0, 0), thickness=40) 

    # combine images
    combo_images = cv2.addWeighted(img, 0.8, line_image, 1, 1)
    


    draw_equations(combo_images, left_fit, right_fit)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny (Bilateral)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 800, 600)
    cv2.resizeWindow("Canny (Bilateral)", 800, 600)
    cv2.imshow("Result", combo_images)
    cv2.imshow("Canny (Bilateral)", canny_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
