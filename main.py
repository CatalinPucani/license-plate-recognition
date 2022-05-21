import queue
from xml.etree.ElementTree import PI
import cv2 as cv
import imutils
import numpy as np
import pytesseract
import math

def filtruGaussian(img, w=5):
    dst = img

    sigma = w / 6.0

    x0 = y0 = w / 2

    k = (w - 1) / 2

    k = int(k)

    aux = (1.0) / (2 * math.pi * sigma * sigma)

    gauss = [[0] * w] * w

    for i in range(w):
        for j in range(w):
            expp = -((i - x0) * (i - x0) + (j - y0) * (j - y0))
            expp /= (2 * sigma * sigma)
            gauss[i][j] = math.exp(aux * expp)

    c = 0.0
    for i in range(int(2 * k + 1)):
        for j in range(int(2 * k + 1)):
            c += gauss[i][j]

    for i in range(img.shape[0]):
        print("Gaussian filter", (i*100)/(img.shape[0]), "%")
        for j in range(img.shape[1]):
            suma = 0.0
            for x in range(2 * k + 1):
                for y in range(2 * k + 1):
                    if i + x - k < img.shape[0] and j + y - k < img.shape[1]:
                        suma += gauss[x][y] * img[i + x - k][j + y - k]
                        dst[i][j] = (1.0 / c) * suma

    return dst

def connectEdges(src):
    height = src.shape[0]
    width = src.shape[1]

    di = [0, -1, -1, -1, 0, 1, 1, 1]
    dj = [1, 1, 0, -1, -1, -1, 0, 1]

    dst = src

    for i in range(1,height-1):
        print("Connecting edges", (i*100) / (height), "%")
        for j in range(1, width-1):
            if(dst[i][j] == 255):
                q = queue.LifoQueue()
                q.put((i,j))
                while(not q.empty()):
                    punct = q.get()

                    for k in range(8):
                        if(punct[0] + di[k] < height and punct[1] + dj[k] < width):
                            if(dst[punct[0] + di[k]][punct[1] + dj[k]] == 128):
                                dst[punct[0] + di[k]][punct[1] + dj[k]] = 255
                                q.put((punct[0] + di[k], punct[1]+ dj[k]))

    for i in range(height):
        for j in range(width):
            if (dst[i][j] == 128):
                dst[i][j] = 0
    return dst

def makeHistogram(src):
    histogram = [0]*256

    src = src.astype(np.uint8)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            histogram[src[i][j]]+=1.0

    return histogram

def binarizare(src, prag):
    dst = src

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] < prag:
                dst[i][j] = 0
            else: dst[i][j] = 255

    return dst

def Canny_detector(img, weak_th=None, strong_th=None):
    # conversion of image to grayscale

    img = filtruGaussian(img.copy())
    print("Filtru Gaussian - finished")

  

    # Calculating the gradients
    gx = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, 3)
    gy = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)
    
    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not strong_th: strong_th = mag_max * 0.5

    # getting the dimensions of the input image  
    height, width = img.shape

    # Looping through every pixel of the grayscale 
    # image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    print("Non-maximum suppresion - finished")

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)
    """
    histogram = makeHistogram(img.copy())

    nrPixelCuModulNul = histogram[0]
    nrPuncteMuchie = 0.1 * (height * width - nrPixelCuModulNul)
    nrNonMuchie = (1-0.1) * (height * width - nrPixelCuModulNul)

    sumHistogram = 0
    for i in range(256):
        sumHistogram+=histogram[i]
        if (sumHistogram>nrNonMuchie):
            pragAdaptiv = i
            break

    firstThreshold = 0.4*pragAdaptiv
    secondThreshold = pragAdaptiv

    weak_th = firstThreshold
    strong_th = secondThreshold
    """


    # double thresholding step
    for i_y in range(height):
        for i_x in range(width):

            if (int(mag[i_y][i_x]) < weak_th):
                mag[i_y, i_x] = 0
            elif (strong_th > int(mag[i_y][i_x]) >= weak_th):
                mag[i_y, i_x] = 128
            else:
                mag[i_y, i_x] = 255

    print("Double-tresholding - finished")

    # finally returning the magnitude of
    # gradients of edges

    mag = connectEdges(mag.copy())

    print("Connecting edges - finished")

    return mag


img = cv.imread('./images/numere - 7.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grey scale
print("Grayscale - finished")

edged = Canny_detector(gray, 30, 200)  # Perform Edge detection
print("Canny_detector - finished")

cv.imshow("edged", edged)

edged = edged.astype(np.uint8)



contours = cv.findContours(edged.copy(), cv.RETR_TREE,
                           cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.018 * peri, True)
    cv.drawContours(img, [approx], -1, (0, 0, 255), 3)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

cv.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(Cropped, config='--psm 7')
print("Detected license plate number is:",text)
img = cv.resize(img,(500,300))
Cropped = cv.resize(Cropped,(400,200))
cv.imshow('car',img)
cv.imshow('Cropped',Cropped)

cv.waitKey(0)