import cv2
import numpy as np

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

LSD2 = cv2.createLineSegmentDetector([])
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

