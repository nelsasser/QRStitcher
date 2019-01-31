import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print(cv.__version__)

img1 = cv.imread('tr_slice_small.png',0)          # queryImage
img1 = cv.resize(img1, None, fx=2, fy=2, interpolation = cv.INTER_NEAREST)
# img1 = cv.Canny(img1, 100, 150)


img2 = cv.imread('br_slice_small.png',0) # trainImage
img2 = cv.resize(img2, None, fx=2, fy=2, interpolation = cv.INTER_NEAREST)
# img2 = cv.Canny(img2, 100, 250)


kernal_size = 3;

# img2 = cv.GaussianBlur(img2, (kernal_size, kernal_size), 0)
#ret,img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
# img2 = cv.GaussianBlur(img2, (5, 5), 0)

# Initiate SIFT detector
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(	algorithm = FLANN_INDEX_LSH, 
						table_number = 6, # 12
						key_size = 12, # 20
						multi_probe_level = 1) # 2

search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,128,3, cv.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()