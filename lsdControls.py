import cv2
import numpy as np
from test import originalImageToGrayscale
from test import lsdWithParams
from test import layoutToDump
from test import printImageFromDump
from test import blankImage


def nothing(x):
	s = cv2.getTrackbarPos(switch, 'image')

	if s == 0:
		k = 2
	else:
		scale = cv2.getTrackbarPos('scale', 'image')
		sigma = cv2.getTrackbarPos('sigma', 'image')
		quant = cv2.getTrackbarPos('quant', 'image')
		ang   = cv2.getTrackbarPos('ang', 'image')
		density = cv2.getTrackbarPos('density', 'image')
		applyLSD(scale, sigma, quant, ang, density)


def applyLSD(scale, sigma_scale, quant, ang_th, density_th):

	scale = scale / 10
	sigma_scale = sigma_scale / 10
	density_th = density_th / 10
	print("scale = {}, sigma_scale = {}, quant = {}, ang_th = {}, density_th = {}".format(scale, sigma_scale, quant, ang_th, density_th))

	color, imGrayScale = originalImageToGrayscale("tests/small/k_2156.JPG")
	height, width = imGrayScale.shape
	dumpMatrix = np.zeros(shape = (height, width))
	lines, width, prec, nfa = lsdWithParams(imGrayScale, cv2.LSD_REFINE_ADV, scale, sigma_scale, quant, ang_th, 0, density_th, 1024)
	dumpMatrix = layoutToDump(dumpMatrix, lines)
	
	heightD, widthD = dumpMatrix.shape
	blank = blankImage(heightD, widthD, (255,255,255))

	for i in range(0, heightD):
		for j in range(0, widthD):
			if dumpMatrix[i,j] >= 1:
				blank[i,j][0] = 0
				blank[i,j][1] = 0
	

	cv2.imshow('image', blank)




print("trackbar!")

#create a blank image and window
color, img = originalImageToGrayscale("tests/small/k_2156.JPG")
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', img)

#create trackbars for color change
cv2.createTrackbar('scale', 'image', 1, 10, nothing)
cv2.createTrackbar('sigma', 'image', 1, 10, nothing)
cv2.createTrackbar('quant', 'image', 1, 10, nothing)
cv2.createTrackbar('ang', 'image', 1, 100, nothing)
cv2.createTrackbar('density', 'image', 1, 10, nothing)


#switch on off functionality
switch = 'OFF\ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

cv2.waitKey(0)
cv2.destroyAllWindows()