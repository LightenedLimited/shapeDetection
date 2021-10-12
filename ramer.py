from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
This algorithim works by first:
1) Getting an "contour" using the Square Trace algorithim
2) Eliminating similar points within this contour
3) Approximating contour using Douglas-Pecker to find amount of corners
4) Eliminating similar corner points
"""

#import image as grayscale
IMAGE_PATH = "./image6.png"
EPSILON_FACTOR = 0.01

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
#coordinates on edge

#https://en.wikipedia.org/wiki/Boundary_tracing
def GoRight(point):
    #point[0] = x, point[1] = y
    return (-point[1], point[0])

def GoLeft(point):
    return (point[1], -point[0])

def add(point_1, point_2):
    return (point_1[0] + point_2[0], point_1[1] + point_2[1])

def subtract(point_1, point_2):
    return (point_1[0] - point_2[0], point_1[1] - point_2[1])
#Algorithim is a simplifed version of https://www.mdpi.com/1424-8220/16/3/353/pdf
def SquareTrace(start):
    edge = set()
    edge.add(start)

    nextStep = GoLeft((1, 0))
    next_point = add(start, nextStep)

    #point[0] = x, point[1] = y
    while(next_point != start):
        
        if (binary_image[next_point[1]][next_point[0]] == 0):
            next_point = subtract(next_point, nextStep)
            nextStep = GoRight(nextStep)
            next_point = add(next_point, nextStep)
        else:
            edge.add(next_point)
            nextStep = GoLeft(nextStep)
            next_point = add(next_point, nextStep)
    return edge

edges = []

flag = False

for i in range(binary_image.shape[0]):
    for j in range(binary_image.shape[1]):
        if binary_image[i][j] == 255:
            edges = SquareTrace((j, i))
            flag = True
            break
    if flag:
        break
    
#to sort the list of edges, use a minimal forward pass

remaining_edges = list(edges)
sorted_edges = [remaining_edges[0]]
remaining_edges.pop(0)

#note that distance squared is only applicable for relative comparison
def distance_squared(point_1, point_2):
    return (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2 

total_distance = 0
DIST_THRESH = 10
#get rid of points that are too similar to each other
while len(remaining_edges) > 0:
    #find the minimum distance
    compare = sorted_edges[-1]
    min_distance = np.Inf
    min_index = -1
    for i in range(len(remaining_edges)):
        if distance_squared(compare, remaining_edges[i]) <= min_distance:
            min_distance = distance_squared(compare, remaining_edges[i])
            min_index = i
    if min_distance > DIST_THRESH:
        total_distance += np.sqrt(min_distance)
        sorted_edges.append(remaining_edges[min_index])
    remaining_edges.pop(min_index)

coordinates_of_edge = np.array(list(sorted_edges))
#print("??")

EPSILON = EPSILON_FACTOR * total_distance

#perpendicular distance
#https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
def perpendicularDistance(p1, l_p1, l_p2):
    # np.cross(p2-p1,p3-p1)/norm(p2-p1)
    perpendicular = np.abs(np.cross(l_p2 - l_p1, l_p1 - p1)/np.linalg.norm(l_p2 - l_p1))
    return perpendicular

#https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
def DouglasPeucker(coordinates):
    dmax = 0
    index = 0
    end = len(coordinates)
    for i in range(1, end-1):
        distance = perpendicularDistance(coordinates[i], coordinates[0], coordinates[-1])
        if(distance >= dmax):
            dmax = distance
            index = i
    results = []
    if(dmax > EPSILON):
        results_rec_1 = DouglasPeucker(coordinates[:index])
        results_rec_2 = DouglasPeucker(coordinates[index:])
        results = np.concatenate((results_rec_1, results_rec_2))
    else:
        results = [coordinates[0], coordinates[-1]]
    # if(len(coordinates) == len(results) and len(coordinates) != 2):
    #     print(coordinates)
    return results

corners = DouglasPeucker(coordinates_of_edge)

#resort list of corners
sorted_corners = [corners[0]]
corners = list(corners)
corners.pop(0)

DIST_THRESH_CORNERS = 100 ** 2
#foward pass, get rid of points that are two similar to each other
while len(corners) > 0:
    #find the minimum distance
    compare = sorted_corners[-1]
    min_distance = np.Inf
    min_index = -1
    for i in range(len(corners)):
        if distance_squared(compare, corners[i]) <= min_distance:
            min_distance = distance_squared(compare, corners[i])
            min_index = i
    if min_distance > DIST_THRESH_CORNERS:
        sorted_corners.append(corners[min_index])
    corners.pop(min_index)

sorted_corners.reverse()

#backward pass, get rid of points that are too similar to each other

backward_pass_sorted = [sorted_corners[0]]
for i in range(len(sorted_corners)-1):
    if(distance_squared(sorted_corners[i], sorted_corners[i+1]) > DIST_THRESH_CORNERS):
        backward_pass_sorted.append(sorted_corners[i+1])

sorted_corners = backward_pass_sorted

if(len(sorted_corners) == 11):
    print("Shape 3, rotated cross")
elif(len(sorted_corners) == 4):
    print("Shape 1, square")
elif(len(sorted_corners) == 3):
    print("Shape 2, triangle")
elif(len(sorted_corners) == 8):
    print("Shape 4, circle")
elif(len(sorted_corners) == 12):
    print("Shape 5, normal cross")
elif(len(sorted_corners) == 5):
    print("Shape 6, pentagon")

