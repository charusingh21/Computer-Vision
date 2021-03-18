import cv2
import numpy as np
import argparse
image= cv2.imread("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/rdimage.000.ppm")
cv2.imshow("Edges", image)
k= cv2.waitKey(10000)
image2= cv2.imread("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/rdimage.001.ppm")
cv2.imshow("Edges", image2)
k= cv2.waitKey(10000)

image3= cv2.imread("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/rdimage.002.ppm")
cv2.imshow("Edges", image3)
k= cv2.waitKey(10000)
output_dir = 'C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/res'
cv2.imwrite("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/res/image.png",image)
cv2.imwrite("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/res/image2.png",image2)
cv2.imwrite("C:/Users/charu/Desktop/cv/HW3/HW3_Data/cs677_hw3_data/res/image3.png",image3)