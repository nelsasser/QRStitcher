import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob


# ----Loads all of the slices in designated file path----

def loadImages(filepath):
	print('\n')
	print("OpenCV Version " + cv.__version__)
	print('\nLoading qr-slices...')

	# load all of the slices
	slices = []
	for filename in glob.glob(filepath + '/*.png'): #assuming slices are in png format
	    im = cv.imread(filename)
	    slices.append(im)
	    print("Loaded slice: " + filename)
	print('Done.\n\nCreating featues and key points for slices...')

	return slices


# ----Creates features and key points for each slice----

def initORB(slices):
	# create ORB detector to detect similarities between slices
	orb = cv.ORB_create()

	slice_kps = []
	slice_dsts = []

	# get features for each slice
	# store key points in kps array
	# store destinations in dsts array
	for slc in slices:
		# get output from ORB
		o = orb.detectAndCompute(slc, None)
		slice_kps.append(o[0])
		slice_dsts.append(o[1])

	print('Done.\n\nInitializing FLANN for matching...')

	return (orb, slice_kps, slice_dsts)

# ----Initializes FLANN and FLANN parameters for feature matching
def initFLANN():
	# initialize FLANN to find matches between images
	# FLANN parameters
	FLANN_INDEX_LSH = 6
	index_params = dict(	algorithm = FLANN_INDEX_LSH, 
							table_number = 6, # 12
							key_size = 12, # 20
							multi_probe_level = 1) # 2

	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv.FlannBasedMatcher(index_params, search_params)

	print('Done.\n\n')

	return flann

def findGoodMatches(des1, des2, flann):
	matches = flann.knnMatch(des1,des2,k=2)

	# Need to draw only good matches, so create a mask
	# matchesMask = [[0,0] for i in range(len(matches))]

	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)

	return good

def drawGoodMatches(img1, img2, kp1, kp2, good):
	MIN_MATCH_COUNT = 10

	if len(good)>=MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
	    matchesMask = mask.ravel().tolist()

	    dims = img1.shape
	    h = dims[0]
	    w = dims[1]
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

def main():
	filepath = 'slices/large'

	slices = loadImages(filepath)

	d = slices[1].shape

	rows = d[0]
	cols = d[1]

	M = cv.getRotationMatrix2D((cols/2,rows/2),45,1)
	slices[1] = cv.warpAffine(slices[1],M,(cols,rows))

	orb, slice_kps, slice_dsts = initORB(slices)

	flann = initFLANN()

	goodMatches = findGoodMatches(slice_dsts[0], slice_dsts[1], flann)

	drawGoodMatches(slices[0], slices[1], slice_kps[0], slice_kps[1], goodMatches)

main()
