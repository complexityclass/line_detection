import cv2
import numpy as np

def originalImageToGrayscale(name):
	imgColor = cv2.imread(name)
	imgGrayScale = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

	return imgColor, imgGrayScale


def imageToIntencityMatr(img):
	height, width, channel = img.shape

	intensityMatr = np.zeros(shape = (height, width))

	for i in range(0, height):
		for j in range(0, width):
			intensityMatr[i, j] = (int(img[i,j][0]) + int(img[i,j][0]) + int(img[i,j][0])) // 3

	return intensityMatr


def blankImage(height, width, color):
	dump_image = np.zeros((height, width, 3), np.uint8)
	dump_image[:] = color

	return dump_image


def lsdDefault(img):
	LSD = cv2.createLineSegmentDetector(0)
	lines, width, prec, nfa = LSD.detect(img)

	return lines, width, prec, nfa


def lsdWithParams(img, refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins):
	LSD = cv2.createLineSegmentDetector(_refine = refine, _scale = scale, 
		_sigma_scale = sigma_scale, _quant = quant, _ang_th = ang_th, 
		_log_eps = log_eps, _density_th = density_th, _n_bins = n_bins)

	lines, width, prec, nfa = LSD.detect(img)

	return lines, width, prec, nfa



def layoutToDump(dumpMatr, lines):
	height, width = dumpMatr.shape
	blank = blankImage(height, width, (0,0,0))
	LSD = cv2.createLineSegmentDetector(0)
	layout = LSD.drawSegments(blank, lines)

	for i in range(0, height):
		for j in range(0, width):
			if layout[i][j][2] != 0 or layout[i][j][1] != 0 or layout[i][j][0] != 0:
				dumpMatr[i,j] = dumpMatr[i,j] + 1

	return dumpMatr


def printMatr(matr):
	height, width = matr.shape
	for i in range(0, height):
		raw = ""
		for j in range(0, width):
			raw = raw + " " + str(matr[i,j])
		print(raw)


def printImageLinesWithTreshold(dump, treshold):
	height, width = dump.shape
	blank = blankImage(height, width, (255,255,255))

	for i in range(0, height):
		for j in range(0, width):
			if dump[i,j] >= treshold:
				blank[i,j][0] = 0
				blank[i,j][1] = 0

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image',blank)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def test():

	imC, imG = originalImageToGrayscale("credit.png")
	height, width = imG.shape
	dumpMatrix = np.zeros(shape = (height, width))
	#lines, width, prec, nfa = lsdDefault(imG)
	lines, width, prec, nfa = lsdWithParams(imG, cv2.LSD_REFINE_ADV, 0.7, 0.6, 2.0, 22.5, 0, 0.6, 1024)
	dumpMatrix = layoutToDump(dumpMatrix, lines)

	printImageLinesWithTreshold(dumpMatrix, 1)


def completeTest(imageName):
	imColor, imgGrayScale = originalImageToGrayscale(imageName)
	height, width = imgGrayScale.shape
	dumpMatrix = np.zeros(shape = (height, width))
	
	lines1, width1, prec1, nfa1 = lsdWithParams(imgGrayScale, cv2.LSD_REFINE_ADV, 0.7, 0.6, 2.0, 22.5, 0, 0.6, 1024)
	lines2, width2, prec2, nfa2 = lsdDefault(imgGrayScale)

	dumpMatrix = layoutToDump(dumpMatrix, lines1)
	dumpMatrix = layoutToDump(dumpMatrix, lines2)

	printMatr(dumpMatrix)



#test()
completeTest("credit.png")

'''
imgColor = cv2.imread("card.png")
img = cv2.imread("card.png", cv2.IMREAD_GRAYSCALE)
height, width = img.shape
print(img.shape)

for i in range(0, height):
	st = ""
	for j in range(0, width):
		tmp = (int(imgColor[i,j][0]) + int(imgColor[i,j][0]) + int(imgColor[i,j][0])) // 3
		if tmp is not 1:
			imgColor.itemset((i,j,0),255)
			imgColor.itemset((i,j,1),255)
			imgColor.itemset((i,j,2),255)
		st = st + " " + str(tmp)
	#print(st)

from cv2 import __version__
print(__version__)

dump_image = np.zeros((height, width, 3), np.uint8)
rgb_color = (0,0,0)
dump_image[:] = rgb_color

LSD = cv2.createLineSegmentDetector(0)
lines, width, prec, nfa = LSD.detect(img)
img2 = LSD.drawSegments(imgColor, lines)

LSD2 = cv2.createLineSegmentDetector(0)
lines2, width2, prec2, nfa2 = LSD.detect(img)
img3 = LSD.drawSegments(imgColor, lines2)


for ind1 in range(0, 256):
	for ind2 in range(0, 256):
		avg = (int(img2[ind1, ind2][0]) + int(img2[ind1, ind2][0]) + int(img2[ind1, ind2][0]))

		if img2[ind1, ind2][1] != 255:
			dump_image[ind1, ind2][0] = dump_image[ind1, ind2][0] + 1 

		if img3[ind1, ind2][1] != 255:
			dump_image[ind1,ind2][0] = dump_image[ind1, ind2][0] + 1


total = 0
#print in dump
for ind1 in range(0, 256):
	for ind2 in range(0, 256):
		if dump_image[ind1, ind2][0] != 0:
			total = total + 1

print("total = ", total)





cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

