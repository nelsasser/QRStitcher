import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

# helper function for summing square differences between query and ratio filter
# assumes lists are the same length
def sumSquareDifferences(query, filt):
	s = 0;

	# normalize our values in query
	# so that if the first set of pixels is, for example, 10 long
	# we can perserve that ratio and still test against our filter
	# for the correct ratio
	query = [x / query[0] for x in query]

	for i in range(0, len(query)):
		diff = query[i] - filt[i]
		s += diff * diff
	return s

'''
Compresses the black and white pixel vector into 
the number of each that are found in a sequence

@param <vector> - the black and white pixel vector to be compressed
@return - matrix with compressed vector in [0] and colors in [1]
'''
def getVectorRatios(vector):
	# keep track of number of black and white pixels in the
	# order they appear (compress vector)
	vec_r = [[0], [0]];
	i = 0
	for p in vector:
		# set color of first pixel in vector
		if(i == 0):
			vec_r[1][0] = int(p/255)
			i = 1

		# check if current pixel is same color as last pixel
		if(vec_r[1][len(vec_r[1]) - 1] == int(p/255)):
			# if it is same color, increase number
			# of pixels of that color found in a row
			vec_r[0][len(vec_r[0]) - 1] += 1
		else:
			# if not same color
			# switch to the new color
			vec_r[0].append(1)
			vec_r[1].append(int(p/255))

	return vec_r

'''
Calculates the sum of the squared differences for each convolution
of the ratio filter over the compressed ratio vector

@param <c_vec> - compressed ratio vector
@return - ratio filter to run over vector
'''
def getSqrdDiffs(c_vec, ratio):
	# if the number of color changes is >= than what our
	# ratio filter needs, find sqrd diffs
	if(len(c_vec[0]) >= len(ratio[0])):
		# convolute our ratio filter across compressed column

		# basically a CNN
		# google hire me

		# store sum of squared diffs here:
		sqr_diffs = []
		for i in range(0, (len(c_vec[0]) - len(ratio[0])) + 1):
			# if the pixel colors don't match then place a -1 for this
			# convoltion
			if(c_vec[1][i] != ratio[1][0]):
				sqr_diffs.append(-1.0)
				continue

			# otherwise, sum sqrd diffs
			query = c_vec[0][i: i + 4]
			sqr_diffs.append(sumSquareDifferences(query, ratio[0]))

		return sqr_diffs
	else:
		# if smaller thqn ratio filter
		# still append array so we keep columns in order
		# but fill with -1 since can't get negative from 
		# summing squared differences
		return [-1.0]


'''
Extracts key points from differences vector and gives position
in image based on index provided for row/column
'''
def getKeyPoints(c_vec, d_vec, index, threshold):
	key_points = []

	for i in range(0, len(d_vec)):
		# check if current score is less than threshold, and also not -1
		if(d_vec[i] < threshold and d_vec[i] != -1.0):
			# get the min of this hit (first part of filter)
			mn = 0
			for j in range(0, i):
				mn += c_vec[index][0][j]

			# get max of this hit (end part of filter)
			mx = mn
			for j in range(i, i + 5):
				mx += c_vec[index][0][j]

			mdpt = (mx+mn)/2
			key_points.append([index, mn, mx, mdpt])

	return key_points


def doColumns(img, ratio):
	# image dimensions
	height, width = img.shape

	# array to store our ratios for each column
	column_ratios = []

	# get column ratios
	for x in range(0, width):
		col = img[:,x]
		col_r = getVectorRatios(col)
		column_ratios.append(col_r)

	# get squared differences for the filter convolutions
	# store diffs in here:
	col_sqr_diffs = []

	for col in column_ratios:
		sdiffs = getSqrdDiffs(col, ratio)
		col_sqr_diffs.append(sdiffs)

	# threshold for key points
	KPT_THRESH = .5
	# keep track of what column we're at
	col_num = 0
	# store key points here
	key_points = []
	for col in col_sqr_diffs:
		key_points = key_points + getKeyPoints(	column_ratios, 
												col, 
												col_num, 
												KPT_THRESH)
		col_num += 1

	# clean up key points to get good key points
	#average midpoint
	avg = 0
	for pt in key_points:
		avg += pt[len(pt)-1]
	avg /= len(key_points)

	good_key_points = []
	# kinda arbitrary for right now
	DST_KPT_THRESH = width / 1
	for pt in key_points:
		if(math.sqrt(	(pt[len(pt)-1] - avg) * 
						(pt[len(pt)-1] - avg)) < DST_KPT_THRESH 
						and pt[len(pt) - 1] > 0):
			good_key_points.append(pt)

	return good_key_points

def doRows(img, ratio):
	# image dimensions
	height, width = img.shape

	# array to store our ratios for each column
	row_ratios = []

	# get column ratios
	for x in range(0, height):
		row = img[x,:]
		row_r = getVectorRatios(row)
		row_ratios.append(row_r)

	# get squared differences for the filter convolutions
	# store diffs in here:
	row_sqr_diffs = []

	for row in row_ratios:
		sdiffs = getSqrdDiffs(row, ratio)
		row_sqr_diffs.append(sdiffs)

	# threshold for key points
	KPT_THRESH = .5
	# keep track of what column we're at
	row_num = 0
	# store key points here
	key_points = []
	for row in row_sqr_diffs:
		key_points = key_points + getKeyPoints(	row_ratios, 
												row, 
												row_num, 
												KPT_THRESH)
		row_num += 1

	# clean up key points to get good key points
	#average midpoint
	avg = 0
	for pt in key_points:
		avg += pt[len(pt)-1]
	avg /= len(key_points)

	good_key_points = []
	# kinda arbitrary for right now
	DST_KPT_THRESH = width / 1
	for pt in key_points:
		if(math.sqrt(	(pt[len(pt)-1] - avg) * 
						(pt[len(pt)-1] - avg)) < DST_KPT_THRESH 
						and pt[len(pt) - 1] > 0):
			good_key_points.append(pt)

	return good_key_points

def main():
	file = 'test/worst_v2.png' 

	# image to search for qr code
	img = cv.imread(file, 0)
	height, width = img.shape
	M = cv.getRotationMatrix2D((width/2,height/2),10,1)
	img = cv.warpAffine(img,M,(width,height))

	# threshold image to get only white and black pixels
	ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)

	# ratios to test for
	ratio = np.array([[1, 1, 3, 1, 1], [0, 1, 0, 1, 0]])

	good_key_points_cols = doColumns(img, ratio)
	good_key_points_rows = doRows(img, ratio)


	#
	# debug stuff
	#

	img_c = cv.imread(file, 3)
	img_c = cv.warpAffine(img_c,M,(width,height))

	for pt in good_key_points_cols:
		img_c[pt[3], pt[0]] = [0, 255, 0]

	for pt in good_key_points_rows:
		img_c[pt[0], pt[3]] = [255, 0, 0]


	plt.imshow(img_c),plt.show()

main()