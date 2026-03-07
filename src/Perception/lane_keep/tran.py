import cv2
import numpy as np
img_w = 640
img_h = 480
img = cv2.imread("output.jpg")

src = np.float32([
        [img_w * 0.2, img_h * 0.30],  # Top Left (Narrower)
        [img_w * 0.8, img_h * 0.30],  # Top Right
        [img_w * 1, img_h * 0.63],  # Bot Right (Wider)
        [img_w * 0, img_h * 0.63]   # Bot Left
])
        
dst = np.float32([
    [img_w * 0.2, 0],
    [img_w * 0.8, 0],
    [img_w * 0.8, img_h],
    [img_w * 0.2, img_h]
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(img, M, (img_w, img_h))

cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()