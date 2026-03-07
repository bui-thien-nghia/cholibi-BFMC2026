import cv2
import numpy as np

class VisualOdometry2D:
    """
    2D Visual Odometry using Optical Flow on Birds-Eye View images.
    """
    def __init__(self, xm_per_pix=0.00135, ym_per_pix=0.00165):
        """
        xm_per_pix and ym_per_pix should be in METERS per pixel.
        The LaneDetector uses CM per pixel (0.135, 0.165).
        """
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        
        self.prev_gray = None
        self.prev_pts = None
        
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def process(self, warped_img):
        """
        Processes a new bird's-eye view image and estimates movement.
        Returns: dx (m), dy (m), dtheta (rad)
        """
        if len(warped_img.shape) == 3:
            gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped_img

        dx, dy, dtheta = 0.0, 0.0, 0.0

        if self.prev_gray is not None:
            # 1. Track points from previous frame
            if self.prev_pts is not None and len(self.prev_pts) > 10:
                new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
                
                # Filter valid points
                if new_pts is not None:
                    good_new = new_pts[st == 1]
                    good_old = self.prev_pts[st == 1]
                    
                    if len(good_new) > 5:
                        # 2. Estimate Affine Transformation (Partial = Rotation, Translation, Scale)
                        # Since we use BEV, we expect scale=1.0
                        matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                        
                        if matrix is not None:
                            # matrix is 2x3: 
                            # [ cos(th) -sin(th) tx ]
                            # [ sin(th)  cos(th) ty ]
                            
                            tx_pix = matrix[0, 2]
                            ty_pix = matrix[1, 2]
                            
                            # Rotation: atan2(sin, cos)
                            dtheta = -np.arctan2(matrix[1, 0], matrix[0, 0])
                            
                            # Convert pixel translation to physical meters
                            # Note: in BEV, dy is forward (Y axis in image usually points down)
                            dx = -tx_pix * self.xm_per_pix
                            dy = ty_pix * self.ym_per_pix
            
            # 3. Detect new features if tracking set is small
            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            else:
                # Update tracking points for next iteration
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        else:
            # First frame initialization
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        self.prev_gray = gray.copy()
        
        return dx, dy, dtheta
