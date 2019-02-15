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


def doColumns(img, ratio, kpt_thresh, dst_thresh):
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

	# keep track of what column we're at
	col_num = 0
	# store key points here
	key_points = []
	for col in col_sqr_diffs:
		key_points = key_points + getKeyPoints(	column_ratios, 
												col, 
												col_num, 
												kpt_thresh)
		col_num += 1

	# clean up key points to get good key points
	#average midpoint
	avg = 0
	for pt in key_points:
		avg += pt[len(pt)-1]
	avg /= len(key_points)

	good_key_points = []

	for pt in key_points:
		if(math.sqrt(	(pt[len(pt)-1] - avg) * 
						(pt[len(pt)-1] - avg)) < dst_thresh 
						and pt[len(pt) - 1] > 0):
			good_key_points.append(pt)

	return good_key_points

def doRows(img, ratio, kpt_thresh, dst_thresh):
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

	# keep track of what column we're at
	row_num = 0
	# store key points here
	key_points = []
	for row in row_sqr_diffs:
		key_points = key_points + getKeyPoints(	row_ratios, 
												row, 
												row_num, 
												kpt_thresh)
		row_num += 1

	# clean up key points to get good key points
	#average midpoint
	avg = 0
	for pt in key_points:
		avg += pt[len(pt)-1]
	avg /= len(key_points)

	good_key_points = []
	# kinda arbitrary for right now
	for pt in key_points:
		if(math.sqrt(	(pt[len(pt)-1] - avg) * 
						(pt[len(pt)-1] - avg)) < dst_thresh
						and pt[len(pt) - 1] > 0):
			good_key_points.append(pt)

	return good_key_points

def getGoodKeyPoints(img, kpt_thresh, dst_thresh):
	# threshold image to get only white and black pixels
	ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)

	# ratios to test for
	ratio = np.array([[1, 1, 3, 1, 1], [0, 1, 0, 1, 0]])

	good_key_points_cols = doColumns(img, ratio, kpt_thresh, dst_thresh)
	good_key_points_rows = doRows(img, ratio, kpt_thresh, dst_thresh)

	gkp = []
	hor = []
	vert = []

	for pt in good_key_points_cols:
		gkp.append([pt[3], pt[0]])
		hor.append([pt[3], pt[0]])

	for pt in good_key_points_rows:
		gkp.append([pt[0], pt[3]])
		vert.append([pt[0], pt[3]])

	return (gkp, hor, vert)

def bestFit(pts):
	X = [p[0] for p in pts]
	Y = [p[1] for p in pts]

	xbar = sum(X)/len(X)
	ybar = sum(Y)/len(Y)
	n = len(X)

	numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
	denum = sum([xi**2 for xi in X]) - n * xbar**2

	if(denum == 0):
		b = 0.0000000000000001
	else:
		b = numer / denum
	a = ybar - b * xbar

	return a, b

def inRange(test, mn, mx):
	return test <= mx and test >= mn

def getIntersection(line1, line2):

	b2 = line2[1]
	a2 = line2[0]

	b1 = line1[1]
	a1 = line1[0]

	# b1*x + a1 = b2*x +a2
	# b1*x - b2*x = a2 - a1
	# c*x = d
	# x = d / c
	c = b1 - b2
	d = a2 - a1
	xf = d / c

	#if(inRange(xf, min(x1, x2), max(x1, x2))):
	return [xf, line1[0] + line1[1] * xf]
	#else:
	
	#	return False

def drawLine(image, color, line, width, height):
	for x in range(0, width):
		y = line[0] + line[1] * x
		if(inRange(y, 0, height-1)):
			image[x, y] = color

def getLine(p1, p2):
	x1, y1 = p1[0], p1[1]
	x2, y2 = p2[0], p2[1]

	m = (y2-y1)/(x2-x1)

	b = m * (-1 * x1) + y1

	return b, m
		
def iterate(img, KPT_THRESH, DST_KPT_THRESH, width, height):
	good_key_points, horiz, vert = getGoodKeyPoints(img, KPT_THRESH, DST_KPT_THRESH)

	img_r = img

	img_c = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

	line1 = bestFit(horiz)
	line2 = bestFit(vert)

	drawLine(img_c, [0, 255, 0], line1, width, height)
	drawLine(img_c, [255, 0, 0], line2, width, height)

	inter = getIntersection(line1, line2)

	try:
		cv.line(img_c, (int(inter[1]), 0), (int(inter[1]), height-1), (0, 0, 255), 1)
		cv.line(img_c, (0, int(inter[0])), (width - 1, int(inter[0])), (0, 0, 255), 1)

		cv.circle(img_c, (int(inter[1]), int(inter[0])), 3, (255, 0, 255), 2)
	except:
		print('Crash and burn baby')


	deg = math.degrees(math.atan(line1[1]))
	delta = np.sign(deg)*90 - deg

	while(delta < -360):
		delta += 360

	while(delta > 360):
		delta -= 360

	print(delta)

	M2 = cv.getRotationMatrix2D((width/2,height/2), delta, 1)
	img_r = cv.warpAffine(img_r, M2, (width, height))

	return img_r, delta, img_c

def main():
	file = 'slices/large/tl_slice_large.png' 

	# image to search for qr code
	img = cv.imread(file, 0)
	height, width = img.shape

	angle = 0

	M = cv.getRotationMatrix2D((width/2,height/2), angle, 1)
	img = cv.warpAffine(img,M,(width,height))

	KPT_THRESH = 0.1
	DST_KPT_THRESH = width / 20

	#
	# debug stuff
	#

	subplots = []
	subplots.append(cv.cvtColor(img, cv.COLOR_GRAY2RGB))

	nimg, delta, nimg_c = iterate(img, KPT_THRESH, DST_KPT_THRESH, width, height)
	subplots.append(nimg_c)
	#plt.imshow(nimg_c),plt.show()
	while not inRange(delta, -1, 1) and int(abs(delta)) != 90:
		nimg, delta, nimg_c = iterate(nimg, KPT_THRESH, DST_KPT_THRESH, width, height)
		subplots.append(nimg_c)
		#plt.imshow(nimg_c),plt.show()

	#M2 = cv.getRotationMatrix2D((width/2,height/2), delta, 1)
	#img_r = cv.warpAffine(img_r, M2, (width, height))

	f, axarr = plt.subplots(2, sharex = False)
	axarr[0].imshow(subplots[0])
	axarr[0].set_title("Original")

	axarr[1].imshow(subplots[len(subplots)-1])
	axarr[1].set_title("Iteration #" + str(len(subplots)))
	
	plt.show()

main()