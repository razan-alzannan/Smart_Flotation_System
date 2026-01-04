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
import multiprocessing
from functools import partial

def watershedMethod(frame,quarter):
    img = frame

    #image mask 
    img, mask = ImageMask(img,1,quarter)
    img = img_as_ubyte(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= gray

    edited_image=img.copy()
 
    edited_image_area = np.sum(edited_image != 255)

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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

    #print("[INFO] {} unique segments found in edited image".format(len(np.unique(labels)) - 1))
    #print("[INFO] {} # segments found in edited image".format(len((labels)) ))
    # plt.imshow(labels)
    # plt.show()

    ### CALCULATE AREA FOR EACH CIRCLE ###
    # loop over the unique labels returned by the Watershed
    # algorithm
    img_area = np.sum(frame != 255)
    circle_areas = []
    tiny_circles = 0
    small_circles = 0
    mid_circles = 0
    large_circles = 0
    huge_circles = 0
    tiny_circles_areas = []
    small_circles_areas = []
    mid_circles_areas = []
    large_circles_areas = []
    huge_circles_areas = []
    org = img.copy()
    tiny = img.copy()
    small = img.copy()
    mid = img.copy()

    tiny_circles_location = []
    small_circles_location = []
    mid_circles_location = []
    large_circles_location = []
    huge_circles_location = []

    tiny_commbined_list = []
    small_commbined_list = []
    mid_commbined_list = []
    large_commbined_list = []
    huge_commbined_list = []



    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
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
                tiny_circles += 1
                # tiny_circles_areas += cnt_area
                tiny_circles_areas.append(cnt_area)
                tiny_circles_location.append(cnts)
                tiny_commbined_list.append(Bubble(cnt_area, coordinates))
                cv2.circle(tiny, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            elif area_percent < 0.0014:
                small_circles += 1
                small_circles_areas.append(cnt_area)
                small_circles_location.append(cnts)
                small_commbined_list.append(Bubble(cnt_area, coordinates))
                cv2.circle(small, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
            elif area_percent < 0.04:
                mid_circles += 1
                mid_circles_areas.append(cnt_area)
                mid_circles_location.append(cnts)
                mid_commbined_list.append(Bubble(cnt_area, coordinates))
                cv2.circle(mid, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
            elif area_percent < 0.28:
                large_circles += 1
                large_circles_areas.append(cnt_area)
                large_circles_location.append(cnts)
                large_commbined_list.append(Bubble(cnt_area, coordinates))
                # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # cv2.imshow("Output", image)
                # cv2.waitKey(0)
            elif area_percent < 0.85:
                huge_circles += 1
                huge_circles_areas.append(cnt_area)
                huge_circles_location.append(cnts)
                huge_commbined_list.append(Bubble(cnt_area, coordinates))
        # cv2.drawContours(image, c, -1, (0, 255, 0), 3)
            # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # circle_areas.append(int(r))
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
        # print("x=",x)
        # print("y=",y)
        # print("r=",r)
    all_images = np.hstack((org,tiny))
    temp = np.hstack((small,mid))
    all_images = np.vstack((all_images, temp))
    #cv2.imshow("Output", all_images)
    #cv2.waitKey(0)
    total_bubbles = tiny_circles+small_circles+mid_circles+large_circles+huge_circles

    # return meanAreaList
    allCombinedLists = [tiny_commbined_list,small_commbined_list,mid_commbined_list,large_commbined_list,huge_commbined_list]
    allSizesLists = [tiny_circles_areas,small_circles_areas,mid_circles_areas,large_circles_areas,huge_circles_areas]
    allLocationLists = [tiny_circles_location, small_circles_location,mid_circles_location,large_circles_location,huge_circles_location]
    
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


def findIntersection(allCombinedLists, prevallCombinedLists):
    intesectionArea = [] 
    isIntersect = False
    survivedBubblesCounter = 0
    removed_bubbles = 0
    lifeCycle = 0

    # Loop over each size (Tiny,small.. )
    for i in range(5):
        # print("###",i,"###")
        # Search for each bubble in Frame#2, is it in Frame#1 "Previous"? Based on the size and location
        for prev_bubble in prevallCombinedLists[i]:
            for bubble in allCombinedLists[i]:
            # for prev_bubble in prevallCombinedLists[i]:
                # compare two rectntgules
                current_polygon = Polygon(bubble.location)
                previous_polygon = Polygon(prev_bubble.location)
                intersection = current_polygon.intersection(previous_polygon)
                if(intersection.area/previous_polygon.area > 0.5 and min(previous_polygon.area, current_polygon.area)/max(previous_polygon.area, current_polygon.area) > 0.5):
                    # print("Intersection Area:", intersection.area)
                    # print("Intersection presentage :", intersection.area/previous_polygon.area)
                    survivedBubblesCounter += 1
                    bubble.id = prev_bubble.id
                    bubble.lifecycle = prev_bubble.lifecycle
                    bubble.addlifecycle()
                    # survivedBubbles.append()
                    # lifeCycle += 1
                    continue
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
                    # ax.imshow(frame)
                    # plt.show()

            removed_bubbles += 1
            lifeCycle += prev_bubble.lifecycle
        # removed_bubbles += len(prevallCombinedLists[0]) + len(prevallCombinedLists[1]) + len(prevallCombinedLists[2]) + len(prevallCombinedLists[3]) + len(prevallCombinedLists[4]) - survivedBubblesCounter
    stability_index = lifeCycle/removed_bubbles
    #plt.show()
                # lifeCycle = 0
    #print("Number of Survived bubbles through these two frame is ",survivedBubblesCounter)
    return stability_index


# def process_data():
#     # Calling the watershed for all lists, watershedMethod returns all lists for each size "Current frame"
#     allCombinedLists = watershedMethod(frame, quarter)

#     # Searching based on .. Technique 
#     # removed_bubbles, prevallCombinedLists, intersectionArea, lifeCycle = findIntersection(removed_bubbles, allCombinedLists, prevallCombinedLists, frame, lifeCycle)
#     average_life_cylce = findIntersection(allCombinedLists, prevallCombinedLists)

def process_data(quarter, cap_fps):


    # video_path = "C:\\Users\\USER\\Desktop\\Projects\\CPM\\bubbles.mp4"
    cap = cap_fps[0]
    fps = cap_fps[1]
    quarter = [1,2,3,4]
    # Read the first frame
    ret, prev_frame = cap.read()
    prevallCombinedLists = watershedMethod(prev_frame, quarter)

    c = 1
    #print("Figure: ", c)
    remainingBubbles = []
    removed_bubbles = 0
    lifeCycle = 0
    seconds = 1

    while cap.isOpened():
        start = time.time()
        target_frame = int(seconds*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Calling the watershed for all lists, watershedMethod returns all lists for each size "Current frame"
        allCombinedLists = watershedMethod(frame, quarter)

        # Searching based on .. Technique 
        # removed_bubbles, prevallCombinedLists, intersectionArea, lifeCycle = findIntersection(removed_bubbles, allCombinedLists, prevallCombinedLists, frame, lifeCycle)
        stability_index = findIntersection(allCombinedLists, prevallCombinedLists) 
        prevallCombinedLists = allCombinedLists
        #print("Persentage for the intersection:", intersectionArea)
        # Calling the findLifeCycle to find the lifecycle for each bubble between two conscuitive frames function
        # remainingBubbles = findLifeCycle(allCombinedLists,prevallCombinedLists)
        # Update to the next frame
        seconds = seconds+1
        #print("Figure: ", c)
        prev_frame = frame
        end = time.time()
        # print("time : ", end -start)




    # stability_index = lifeCycle/(removed_bubbles+len(prevallCombinedLists[0]) + len(prevallCombinedLists[1]) + len(prevallCombinedLists[2]) + len(prevallCombinedLists[3]) + len(prevallCombinedLists[4]))
    print("stability_index : ", stability_index)
    return stability_index

def stability_output(cap,fps):
    quarters = [1,2,3,4]
    cap_fps = [cap, fps]
    print(type(cap))

    # with multiprocessing.Pool(processes=len(quarters)) as pool:
    #     stability_indicies = pool.map(partial(process_data, b = quarters), cap_fps )

    # Prepare list of argument tuples
    args = [(q, cap_fps) for q in quarters]

    with multiprocessing.Pool(processes=len(quarters)) as pool:
        stability_indicies = pool.starmap(process_data, args)

    stability_index = sum(stability_indicies)
    print('stability_index = ', stability_index)

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

#main
if __name__ == "__main__":
    video_path = "C:\\Users\\USER\\Desktop\\Projects\\CPM\\bubbles.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stability_output(cap,fps)

