import cv2
import numpy as np
import sys

img1 = cv2.imread(str(sys.argv[1]), cv2.IMREAD_GRAYSCALE)

(minimg1, maximg1, tmp1, tmp2) = cv2.minMaxLoc(img1)

# convert image (uchar/ 1 byte) to 32bit float
imgf = np.float32(img1)
# apply log
imgf = cv2.log(imgf+1)
# multiply by the constant
imgf = cv2.multiply(imgf, (255/np.log(1+maximg1)))

# imgf is a 32 bit float matrix
# convertScaleAbs
imgf = cv2.convertScaleAbs(imgf)
# normalize
img1_log = cv2.normalize(imgf, imgf, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("input image",img1)
cv2.imshow("log function",img1_log)
cv2.waitKey(0)
cv2.destroyAllWindows()

