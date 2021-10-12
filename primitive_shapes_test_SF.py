from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

#import image as grayscale
IMAGE_PATH = "./image5.png"
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

#get how many white pixels
area = len((binary_image == 255).nonzero()[0])

#use a Laplacian kernel operator to print out filter
kernel = np.array([
    [-1, -1, -1], 
    [-1, 8, -1], 
    [-1, -1, -1]
])

edges = cv2.filter2D(binary_image, -1, kernel)

#coordinates on edge
coordinates_of_edge = np.transpose(((edges == 255).nonzero()))

#find the maximum point between any point on the edge, note that this process can 
#be sped up using Rotating Caliper’s Method
max_distance = np.NINF
for coordinate_i in coordinates_of_edge:
    for coordinate_j in coordinates_of_edge:
        difference = coordinate_j - coordinate_i
        distance = np.sqrt(difference[0] ** 2 + difference[1] ** 2)
        max_distance = max(distance, max_distance)

#https://www.researchgate.net/publication/259949841_Automatic_Detection_and_Recognize_Different_Shapes_in_an_Image
print(area / max_distance ** 2)
