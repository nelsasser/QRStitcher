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


file = 'test/test_qr.jpg' 

# image to search for qr code
img = cv.imread(file, 0)
# search image dimensions
height, width = img.shape
M = cv.getRotationMatrix2D((width/2,height/2),0,1)
img = cv.warpAffine(img,M,(width,height))

# threshold image to get only white and black pixels
ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)

# ratios to test for
ratio = np.array([[1, 1, 3, 1, 1], [0, 1, 0, 1, 0]])

# array to store our ratios for each column
column_ratios = []

# process each column
# dump each columns ratio and order into column_ratios
for x in range(0, width):
	# extract column
	col = img[:,x]

	# keep track of number of black and white pixels in the
	# order they appear (compress column)
	col_r = [[0], [0]];
	i = 0
	for p in col:
		# set color of first pixel in column
		if(i == 0):
			col_r[1][0] = int(p/255)
			i = 1

		# check if current pixel is same color as last pixel
		if(col_r[1][len(col_r[1]) - 1] == int(p/255)):
			# if it is same color, increase number
			# of pixels of that color found in a row
			col_r[0][len(col_r[0]) - 1] += 1
		else:
			# if not same color
			# switch to the new color
			col_r[0].append(1)
			col_r[1].append(int(p/255))

	column_ratios.append(col_r)
	

# get squared differences for the filter convolutions
# store diffs in here:
col_sqr_diffs = []
for col in column_ratios:
	# if the number of color changes is >= than what our
	# ratio filter needs, find sqrd diffs
	if(len(col[0]) >= len(ratio[0])):
		# convolute our ratio filter across compressed column

		# basically a CNN
		# google hire me

		# store sum of squared diffs here:
		sqr_diffs = []
		for i in range(0, (len(col[0]) - len(ratio[0])) + 1):
			# if the pixel colors don't match then place a -1 for this
			# convoltion
			if(col[1][i] != ratio[1][0]):
				sqr_diffs.append(-1.0)
				continue

			# otherwise, sum sqrd diffs
			query = col[0][i: i + 4]
			sqr_diffs.append(sumSquareDifferences(query, ratio[0]))

		col_sqr_diffs.append(sqr_diffs)
	else:
		# if smaller thqn ratio filter
		# still append array so we keep columns in order
		# but fill with -1 since can't get negative from 
		# summing squared differences
		col_sqr_diffs.append([-1.0])

# extract key points 
# (where the query scored a total difference below the threshold)
# sinced we used lossless compression and stored everything in order
# we can reverse the order of our steps
# keeping track of the index where the key point is
# until we get its exact 
# threshold for key points
KPT_THRESH = .5
# keep track of what column we're at
col_num = 0
# store key points here
key_points = []
for col in col_sqr_diffs:
	for i in range(0, len(col)):
		# check if current score is less than threshold, and also not -1
		if(col[i] < KPT_THRESH and col[i] != -1.0):
			# get the min of this hit (first part of filter)
			mn = 0
			for j in range(0, i):
				mn += column_ratios[col_num][0][j]

			# get max of this hit (end part of filter)
			mx = mn
			for j in range(i, i + 5):
				mx += column_ratios[col_num][0][j]

			mdpt = (mx+mn)/2
			key_points.append([col_num, mn, mx, mdpt])
	col_num += 1

# filter key points, get average midpoint, and if the distance between average and
# keypoint mid is to large, discard it

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


#
# debug stuff
#
print(good_key_points)

img_c = cv.imread(file, 3)
img_c = cv.warpAffine(img_c,M,(width,height))

for pt in good_key_points:
	img_c[pt[3], pt[0]] = [0, 255, 0]


plt.imshow(img_c),plt.show()