import cv2
import numpy as np

class LaneDetector:
    def __init__(self, 
                 img_w=640, img_h=480,
                 xm_per_pix=0.135, # xcm per pixel in X
                 ym_per_pix=0.165 # xcm per pixel in Y
                 ):
        
        self.img_w = img_w
        self.img_h = img_h
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        
        # memory for tracking sliding windows
        self.left_fit_pixel = None
        self.right_fit_pixel = None
        self.detected = False

        src = np.float32([
            [img_w * 0.35, img_h * 0.40],  # Top Left (Narrower)
            [img_w * 0.65, img_h * 0.40],  # Top Right
            [img_w * 0.95, img_h * 0.90],  # Bot Right (Wider)
            [img_w * 0.05, img_h * 0.90]   # Bot Left
        ])
        
        dst = np.float32([
            [img_w * 0.2, 0],
            [img_w * 0.8, 0],
            [img_w * 0.8, img_h],
            [img_w * 0.2, img_h]
        ])

        # src = np.float32([
        #     [260, 220],  # Top Left  
        #     [380, 220],  # Top Right 
        #     [600, 460],  # Bot Right
        #     [40,  460]   # Bot Left
        # ])


        # dst = np.float32([
        #     [160, 0],   
        #     [480, 0],    
        #     [480, 480],
        #     [160, 480]
        # ])
        
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def preprocess(self, img):
        # Combines Color (HLS) and Gradient (Sobel) thresholds.
        # Perspective Transform (IPM)
        warped = cv2.warpPerspective(img, self.M, (self.img_w, self.img_h))
        
        # 2. Color Threshold 
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        # Filter for bright white
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > 100) & (l_channel > 150)] = 1 
        
        # 3. Sobel X (Vertical line detection)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0) # cv2.CV_64F
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel > 35) & (scaled_sobel < 100)] = 1
        
        # 4. Combine
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        return combined_binary, warped

    def find_lanes(self, binary_warped):
        """
        MANAGER FUNCTION:
        Decides whether to use the heavy 'Sliding Window' or the fast 'Search Around'.
        """
        if self.detected:
            # We have a previous detection, use the fast method!
            return self.search_around_poly(binary_warped)
        else:
            # We are lost, use the slow sliding window to find the lane first.
            return self.sliding_window(binary_warped)

    def sliding_window(self, binary_warped):
        """ The original 'Heavy' histogram search """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 12
        window_height = int(binary_warped.shape[0] // nwindows)
        margin = 60       
        minpix = 50       

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Process the pixels found
        return self._fit_polynomials(binary_warped, left_lane_inds, right_lane_inds)

    def search_around_poly(self, binary_warped):
        """ THE FAST METHOD: Search around the previous polynomial """
        margin = 60 # Width of the search band around the previous line

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # We need the PREVIOUS fits to define the search area
        left_fit = self.left_fit_pixel
        right_fit = self.right_fit_pixel

        # Imagine a tube around the old line. Check which pixels are inside.
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Safety Check: If we didn't find enough pixels, go back to sliding window next time
        if (np.sum(left_lane_inds) < 100) or (np.sum(right_lane_inds) < 100):
            self.detected = False
            return self.sliding_window(binary_warped)

        return self._fit_polynomials(binary_warped, left_lane_inds, right_lane_inds)

    def _fit_polynomials(self, binary_warped, left_lane_inds, right_lane_inds):
        """ Helper function to calculate math for both methods """
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        left_poly = None
        right_poly = None
        
        center_offset = self.img_w / 2.0
        
        # 1. Fit Left Lane
        if len(leftx) > 100:
            # Fit in PIXEL space (for tracking next frame)
            self.left_fit_pixel = np.polyfit(lefty, leftx, 2)
            
            # Fit in REAL WORLD space (for control)
            real_y = (self.img_h - lefty) * self.ym_per_pix
            real_x = (leftx - center_offset) * self.xm_per_pix
            left_fit_real = np.polyfit(real_y, real_x, 2)
            left_poly = np.poly1d(left_fit_real)
        else:
            self.detected = False # Lost lane

        # 2. Fit Right Lane
        if len(rightx) > 100:
            # Fit in PIXEL space
            self.right_fit_pixel = np.polyfit(righty, rightx, 2)
            
            # Fit in REAL WORLD space
            real_y = (self.img_h - righty) * self.ym_per_pix
            real_x = (rightx - center_offset) * self.xm_per_pix
            right_fit_real = np.polyfit(real_y, real_x, 2)
            right_poly = np.poly1d(right_fit_real)
        else:
            self.detected = False # Lost lane

        # If both lanes found, we keep 'detected = True' for the fast loop next time
        if left_poly is not None and right_poly is not None:
            self.detected = True
        
        return left_poly, right_poly
