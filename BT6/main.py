import cv2
import numpy as np

image = cv2.imread('FeatureMap.png')
input_img = cv2.imread('Input.png')
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

width = input_img.shape[0]
height = input_img.shape[1]

thresh = cv2.threshold(gray, 100, 250, 0)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)

    # interpolate new x, y so as to draw bounding box in Input image

    x_new = int(width * (x / gray.shape[0]))
    y_new = int(height * (y / gray.shape[1]))

    h_new = int(height * h / gray.shape[1])
    w_new = int(width * w / gray.shape[0])

    print("{} {} {} {}".format(x_new, y_new, w_new, h_new))

    # cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    cv2.rectangle(input_img, (x_new, y_new), (x_new + w_new, y_new + h_new), (0,0,255), 2)


cv2.imshow('image', image)
cv2.imshow('box', input_img)
cv2.imwrite('result.png', input_img)
cv2.waitKey()
cv2.waitKey(0)
cv2.destroyAllWindows()