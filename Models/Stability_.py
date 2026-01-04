import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
import imutils
import math
from engin import ImageMask
from skimage import img_as_ubyte
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from Bubble import Bubble
import time
import torch
import multiprocessing
# from numba import jit, roc

# @roc.jit
def watershedMethod(frame,cellnum = 3):
    img = frame

    #image mask 
    img, mask = ImageMask(img,cellnum)
    img = img_as_ubyte(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= gray

    edited_image=img.copy()
 
    # edited_image_area = np.sum(edited_image != 255)

    edited_image[edited_image == 255] = 0

    # Replacing the white spots with mean value 
    hist,bins = np.histogram(edited_image.flatten(),256,[0,256])
    #print(hist.argsort()[-2])
    edited_image[ edited_image > hist.argsort()[-2]] = int(hist.argsort()[-2]) #mean
    mean_image2 =  edited_image[edited_image != 0].mean()
    ret, thresh = cv2.threshold(edited_image, mean_image2, 255, cv2.THRESH_BINARY)
    #plt.figure(5)

    #Preprocessing - Histogram Equalization and Adaptive Histogram Equalization (CLAHE)
    # plt.imshow(edited_image, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equalized = clahe.apply(edited_image)
    # plt.show()

    # compute the exact Euclidean distance from every binary *Watershed*
    distance = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(distance, min_distance=1, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(localMax.T)] = True
    # plt.imshow(distance, cmap='gray')
    # plt.show()
    
    markers = ndimage.label(mask)[0]

    labels = watershed(-distance, markers, mask=thresh)

    # binary_labels = (labels > 0).astype(np.uint8) * 255
    # cnts, hier = cv2.findContours(binary_labels, cv2.RETR_TREE,
    # cv2.CHAIN_APPROX_SIMPLE)

    # cont_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(cont_image, cnts, -1, (0,255,0), 2)
    # plt.imshow(cont_image)
    # plt.show()
    # cnts = imutils.grab_contours(cnts)

    #print("[INFO] {} unique segments found in edited image".format(len(np.unique(labels)) - 1))
    #print("[INFO] {} # segments found in edited image".format(len((labels)) ))
    # plt.imshow(labels)
    # plt.show()
    
    ### CALCULATE AREA FOR EACH CIRCLE ###
    # loop over the unique labels returned by the Watershed
    # algorithm
    img_area = np.sum(frame != 255)

    tiny_commbined_list = []
    small_commbined_list = []
    mid_commbined_list = []
    large_commbined_list = []
    huge_commbined_list = []


    # cont_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((cx, cy), r) = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)
        cnt_area = cv2.contourArea(c)
        area_percent = cnt_area/img_area
        #X,Y,A,B = (x,y),(x,y+h),(x-w,y),(x-w,y-h)
        coordinates = [(x,y),(x,y+h),(x+w,y+h),(x+w,y)]

        #print("Coordinates :",coordinates)
        if area_percent > 0.00007:
            if area_percent < 0.00019:
                tiny_commbined_list.append(Bubble(cnt_area, coordinates))
            elif area_percent < 0.0014:
                small_commbined_list.append(Bubble(cnt_area, coordinates))
            elif area_percent < 0.04:
                mid_commbined_list.append(Bubble(cnt_area, coordinates))
            elif area_percent < 0.28:
                large_commbined_list.append(Bubble(cnt_area, coordinates))
                # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # cv2.imshow("Output", image)
                # cv2.waitKey(0)
            elif area_percent < 0.85:
                huge_commbined_list.append(Bubble(cnt_area, coordinates))
        # cv2.drawContours(cont_image, c, -1, (0, 255, 0), 3)
            # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # circle_areas.append(int(r))
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
        # print("x=",x)
        # print("y=",y)
        # print("r=",r)

    # all_images = np.hstack((org,tiny))
    # temp = np.hstack((small,mid))
    # all_images = np.vstack((all_images, temp))
    #cv2.imshow("Output", all_images)
    #cv2.waitKey(0)
    # plt.imshow(cont_image)
    # plt.show()

    # return meanAreaList
    allCombinedLists = [tiny_commbined_list,small_commbined_list,mid_commbined_list,large_commbined_list,huge_commbined_list]
    
    # print("allCombinedLists Shape:",allCombinedLists)
    # print(" Sizes for all Lists",allLists)
    return allCombinedLists

def findDifferenceSDC(frameList,prev_frameList):
    percentage = 0
    total = 0
    count = 0
    for i in range(5):
        
        minmimum = min(frameList[i],prev_frameList[i])
        maximum = max(frameList[i],prev_frameList[i])
        if not (minmimum == maximum == 0) :
            percentage = (minmimum/maximum+1e-20)*100
            #print("The presentage for the difference for : ", i ," is : " ,percentage, "% " )
            total = total + percentage
            count += 1

    avg = (total/ count)
    #print("The total Avg: ", avg)       

# @roc.jit
def findLifeCycle(allCombinedLists,prevallCombinedLists):
    remainingBubbles = [] 
    # Loop over each size (Tiny,small.. )
    for i in range(5):
        #print("###",i,"###")
        # SORTING
        # Search for each bubble in Frame#2, is it in Frame#1 "Previous"? Based on the size and location
        for size, location in allCombinedLists[i]:
            search_space = prevallCombinedLists[prevallCombinedLists[0] <= size+5 and prevallCombinedLists[0] >= size-5]
            for prevSize, prevLocation in prevallCombinedLists[i]:
                # print("Size : ", size, "location : ", location)
                # print("Prev-Size : ", prevSize, "Prev-location : ", prevLocation)
                lifeCycle = 0
                # if it exist add Lifcycle by 1
                if(size == prevSize and ( prevLocation[0]-5 < location[0]< prevLocation[0]+5) 
                                    and ( prevLocation[1]-5 < location[1]< prevLocation[1]+5)):
                                    lifeCycle += 1
                                    remainingBubbles.append((size,location,lifeCycle))

                    #print("location[0] :", location[0],"location[1] :", location[1],"Life Cycle :", lifeCycle)
                    
    #print("remainingBubbles",remainingBubbles)             
    return remainingBubbles 

# @roc.jit
def findIntersection(removed_bubbles, allCombinedLists, prevallCombinedLists, frame, lifeCycle):
    intesectionArea = [] 
    isIntersect = False
    survivedBubblesCounter = 0

    # Loop over each size (Tiny,small.. )
    for i in range(5):
        # print("###",i,"###")
        # Search for each bubble in Frame#2, is it in Frame#1 "Previous"? Based on the size and location
        for bubble in allCombinedLists[i]:
            for prev_bubble in prevallCombinedLists[i]:
                # compare two rectntgules
                # print(bubble.location)
                # print(prev_bubble.location)
                # current_polygon = Polygon(bubble.location)
                # previous_polygon = Polygon(prev_bubble.location)
                intersection = bubble.find_intersection(prev_bubble)
                if(intersection/prev_bubble.size > 0.5 and min(prev_bubble.size, bubble.size)/max(prev_bubble.size, bubble.size) > 0.5):
                    # print("Intersection Area:", intersection.area)
                    # print("Intersection presentage :", intersection.area/previous_polygon.area)
                    survivedBubblesCounter += 1
                    bubble.addlifecycle()
                    bubble.id = prev_bubble.id
                    # survivedBubbles.append()
                    lifeCycle += 1
                    # if intersection.area > 300:
                        # Create figure and axes
                    # fig, ax = plt.subplots()

                    # # Display the image
                    # #ax.imshow(frame)

                    # # Create a Rectangle patch
                    # rect = patches.Rectangle(bubble.location[0], bubble.location[3][0] - bubble.location[0][0], bubble.location[1][1] - bubble.location[0][1], linewidth=1, edgecolor='r', facecolor='none')

                    # # Add the patch to the Axes
                    # ax.add_patch(rect)

                    # rect = patches.Rectangle(prev_bubble.location[0], prev_bubble.location[3][0] - prev_bubble.location[0][0], prev_bubble.location[1][1] - prev_bubble.location[0][1], linewidth=1, edgecolor='g', facecolor='none')

                    # # Add the patch to the Axes
                    # ax.add_patch(rect)

        removed_bubbles += len(prevallCombinedLists) - survivedBubblesCounter
      
    #plt.show()
                # lifeCycle = 0
    #print("Number of Survived bubbles through these two frame is ",survivedBubblesCounter)
    return removed_bubbles, allCombinedLists, intesectionArea, lifeCycle


def stability_output(cap,fps):
    # main 
    # video_path = "C:\\Users\\USER\\Desktop\\Projects\\CPM\\bubbles.mp4"
    cap = cap
    # Read the first frame
    ret, prev_frame = cap.read()
    prevallCombinedLists = watershedMethod(prev_frame)

    c = 1
    #print("Figure: ", c)
    remainingBubbles = []
    removed_bubbles = 0
    lifeCycle = 0


    while cap.isOpened():
        start = time.time()
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Calling the watershed for all lists, watershedMethod returns all lists for each size "Current frame"
        allCombinedLists = watershedMethod(frame)

        # Searching based on .. Technique 
        # start = time.time()
        removed_bubbles, prevallCombinedLists, intersectionArea, lifeCycle = findIntersection(removed_bubbles, allCombinedLists, prevallCombinedLists, frame, lifeCycle)
        # end = time.time()
        # print("time : ", end -start)
        #print("Persentage for the intersection:", intersectionArea)
        # Calling the findLifeCycle to find the lifecycle for each bubble between two conscuitive frames function
        # remainingBubbles = findLifeCycle(allCombinedLists,prevallCombinedLists)
        # Update to the next frame
        c = c+1
        #print("Figure: ", c)
        prev_frame = frame
        
        end = time.time()
        time_ = end - start
        yield time_
        print("time : ", time_)
        # print("time : ", end -start)




        stability_index = lifeCycle/(removed_bubbles+len(prevallCombinedLists))
        yield stability_index
        print("stability_index : ", stability_index)

    x = 1
            # print("location", location[0])
            # search_space = prevallCombinedLists[prevallCombinedLists[0] <= size+5 and prevallCombinedLists[0] >= size-5]

                    # compare two rectntgules   ((),()) ((),()) "if ((A<X1) or (A1<X) or (B<Y1) or (B1<Y))"
                    # if (location[0] <  prevLocation[0] or location[1] <  prevLocation[1] or location[2] <  prevLocation[2] or location[3] <  prevLocation[3]):
                    #     isIntersect = True
                    #     print("not Intersected ")
                    #     lifeCycle += 1
                    # else: 
                    #     isIntersect = False 
                    #     print("Intersected ")

if __name__ == "__main__":
    stability_output()