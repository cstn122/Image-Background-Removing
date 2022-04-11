import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import data, filters

# load images
imgL = cv2.imread('ycropped_0_l.jpg', 0)
imgR = cv2.imread('ycropped_0_r.jpg', 0)
stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=5)   #numDisparities 最大視差值與最小視差值的差，窗口大小必須是16的整數倍
                                                                #blockSIze 匹配的塊大小，必須是>=1的奇數
# generate disparity depth map
disparity = stereo.compute(imgL, imgR)

# convert the depth map to binary
# gray_dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(disparity, 230, 255, cv2.THRESH_BINARY)
cv2.imwrite('binaryThresholdSameSize.png', thresh1)

# erosion & dilation, to diminish noises
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(thresh1, kernel, iterations=1)
mask = cv2.dilate(erosion, kernel, iterations=1)

# convert sizes, to avoid incompatibility
min_16bit = np.min(mask)
max_16bit = np.max(mask)
mask = np.array(np.rint(255 * ((mask - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

# show plot
plt.subplot(231), plt.imshow(imgL, 'gray'), plt.title(
    'img_left'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(imgR, 'gray'), plt.title(
    'img_right'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(disparity, 'gray'), plt.title(
    'disparity'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(thresh1, 'gray'), plt.title(
    'binary'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(erosion, 'gray'), plt.title(
    'erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(mask, 'gray'), plt.title(
    'erosion+dilation'), plt.xticks([]), plt.yticks([])
plt.show()

# merge the mask and the original image to form a background-removed image
# save merged image
image = cv2.imread('ycropped_0_l.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
merged = np.multiply(mask, image)
merged_bwa = cv2.bitwise_and(mask, image)
cv2.imshow('merged.png', merged)
cv2.imshow('merged_bwa.png', merged_bwa)

mask = cv2.imread('binaryThresholdSameSize.png')
image = cv2.imread('ycropped_0_l.jpg')

#merged = np.multiply(mask, image)
merged = cv2.bitwise_and(mask, image)
cv2.imshow('mergedbwa.png', merged)

# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
