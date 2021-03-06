import cv2
import numpy as np

#
# Convert color image to grayscale
#
def originalImageToGrayscale(name):
	imgColor = cv2.imread(name)
	imgGrayScale = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

	return imgColor, imgGrayScale

def imageResize(imageName, fx, fy):
	img = cv2.imread(imageName)
	small = cv2.resize(img, (0,0), fx = fx, fy = fy)

	return small


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


def printMatrToFile(filename, matr):
	height, width = matr.shape

	with open(filename, "wt") as outfile:
		outfile.write("HEAD\n")
		for i in range(0, height):
			raw = ""
			for j in range(0, width):
				raw = raw + " " + str(matr[i,j])

			outfile.write(raw + "\n")
		outfile.write("END")



def printImageLinesWithTreshold(dump, treshold):
	height, width = dump.shape
	blank = blankImage(height, width, (255,255,255))

	for i in range(0, height):
		for j in range(0, width):
			if dump[i,j] >= treshold:
				blank[i,j][0] = 0
				blank[i,j][1] = 0

	cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
	cv2.resizeWindow('image',200,200)
	cv2.imshow('image',blank)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def printImageFromDump(dump, baseImage, k):

	height, width = dump.shape

	print("dump size = {}", dump.shape)
	print("baseImage size - {}", baseImage.shape)

	for i in range(0, height):
		for j in range(0, width):
			if dump[i,j] != 0:
				baseImage[i,j] = baseImage[i,j] - k * dump[i,j]

	return baseImage


def getAvailableTresholds(dump):
	height, width = dump.shape
	tresholds = []
	for i in range(0, height):
		for j in range(0, width):
			if dump[i,j] != 0 and not(dump[i,j] in tresholds):
				tresholds.append(dump[i,j])

	return tresholds


def getMatrFromTextFile(filename, height, width):
	matr = np.zeros(shape = (height, width))
	l = []
	with open(filename, 'r') as f:
		for line in f:
			if 'HEAD' in line:
				continue
			if 'END' in line:
				break
			line = line.strip()
			if len(line) > 0:
				lin = line.split(' ')
				l.append(lin)
	
	for i in range(0, height):
		for j in range(0, width):
			matr[i,j] = int(float(l[i][j]))

	return matr


def tokenizer(filename):
	with open(filename) as f:
		chunk = []
		for line in f:
			if 'HEAD' in line:
				continue
			if 'END' in line:
				yield chunk
				chunk = []
				continue
			chunk.append(line)



#TODO: imageNormalization
def imageNormalize(image, nMin, nMax):
	normalizedImage = cv2.normalize(image, image, nMin, nMax, cv2.NORM_MINMAX, cv2.CV_8UC1)
	return normalizedImage


def test():

	'''
	imC, imG = originalImageToGrayscale("credit.png")
	height, width = imG.shape
	dumpMatrix = np.zeros(shape = (height, width))
	#lines, width, prec, nfa = lsdDefault(imG)
	lines, width, prec, nfa = lsdWithParams(imG, cv2.LSD_REFINE_ADV, 0.7, 0.6, 2.0, 22.5, 0, 0.6, 1024)
	dumpMatrix = layoutToDump(dumpMatrix, lines)
	printImageLinesWithTreshold(dumpMatrix, 1)
	'''
	#matr = np.zeros(shape = (10, 10))
	#printMatrToFile("test.txt", matr)

	matr = getMatrFromTextFile("tests/IMG_2124.txt",1936, 2000)
	printImageLinesWithTreshold(matr, 10)

#600 - 400

def completeTest(imageName, filename):
	
	imColor, imgGrayScale = originalImageToGrayscale(imageName)
	height, width = imgGrayScale.shape
	dumpMatrix = np.zeros(shape = (height, width))

	#params = refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins
	counter = 0
	for scale in np.arange(0.5, 1.0, 0.1):
		for sigma_scale in np.arange(0.6, 0.7, 0.1):
			for quant in np.arange(1.0, 2.0, 1.0):
				for ang_th in np.arange(22, 24, 1.0):
					for density_th in np.arange(0.5, 0.7, 0.1):
						lines, width, prec, nfa = lsdWithParams(imgGrayScale, cv2.LSD_REFINE_ADV, scale, sigma_scale, quant, 
							ang_th, 0, density_th, 1024)
						dumpMatrix = layoutToDump(dumpMatrix, lines)
						counter = counter + 1
						print("test = {}", counter)

	filename = filename
	printMatrToFile(filename, dumpMatrix)


def completeTestHoughLines(imageName, fileName):

	imColor, imGrayScale = originalImageToGrayscale(imageName)
	height, width = imGrayScale.shape
	dumpMatrix = np.zeros(shape = (height, width))

	counter = 0
	angles = [np.pi/180, np.pi/140, np.pi/120, np.pi/100, np.pi/90];
	for rho in np.arange(0.5, 1.0, 0.1):
		for theta in angles:
			for threshold in np.arange(40, 100, 20):
				lines = cv2.HoughLinesP(image = imGrayScale, rho = rho, theta = theta, threshold = threshold)
				dumpMatrix = layoutToDump(dumpMatrix, lines)
				counter = counter + 1
				print("test = {}", counter)

	printMatrToFile(fileName, dumpMatrix)


#resize image
#smallImageGrayScale = imageResize("tests/IMG_2156.JPG", 0.3, 0.3)
#cv2.imwrite("tests/small/k_2156.JPG", smallImageGrayScale)

#run test
#completeTest("tests/small/sm_03_2156.JPG", "tests/data/small_03_2156.txt")
#completeTestHoughLines("tests/small/k_2156.JPG", "tests/data/small_hough_2156.txt")


#print image with layout
# color, grayscale = originalImageToGrayscale("tests/small/k_2156.JPG")
# w, h = grayscale.shape
# matr = getMatrFromTextFile("tests/data/small_hough_2156.txt",w, h)
# normalized = imageNormalize(grayscale, 200, 255)
# withLayout = printImageFromDump(matr, normalized, 3)


# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow('image',200,200)
# cv2.imshow('image', withLayout)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





