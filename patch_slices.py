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

def main():
	filepath = 'slices/small'

	slices = loadImages(filepath)

	orb, slice_kps, slice_dsts = initORB(slices)

	flann = initFLANN()


main()
