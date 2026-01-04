import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
import imutils
import cv2
import math


def bubble_size_analyzer(cap, fps):
    while cap.isOpened():
        ret, img = cap.read()  # Unpack the tuple
        if not ret:  # Check if the frame was successfully read
            print("Failed to read frame from the video stream.")
            break
        image=img.copy()
        img_area = np.sum(image != 255)
        # img_area = 168100

        image[np.all(image == 255, axis=2)] = 0
        
        # Show output image
        #cv2.imshow('Black Background Image', image)
        # shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist,bins = np.histogram(gray.flatten(),256,[0,256])
        # print(hist.argsort()[-2])



        ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        # plt.imshow(thresh)
        # plt.show()

        # # noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0) # 0.1 much better !!, 0.5 \\ it is better , 0.7

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # plt.imshow(sure_fg)
        # plt.show()
        # plt.imshow(sure_bg)
        # plt.show()
        # plt.imshow(unknown)
        # plt.show()
        

        #distance map
        distance = ndimage.distance_transform_edt(thresh) ## TRY OTHER WAY to calculate the distance
        localMax = peak_local_max(distance, min_distance=1, labels=thresh)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(localMax.T)] = True
        # plt.imshow(distance, cmap='gray')
        # plt.show()

        ret, markers = cv2.connectedComponents(thresh)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        labels = cv2.watershed(img,markers)
        #img[markers == -1] = [255,0,0]

        # plt.imshow(labels)
        # plt.show()
        # loop over the unique labels returned by the Watershed
        # algorithm
        circle_areas = []
        tiny_circles = 0
        small_circles = 0
        mid_circles = 0
        large_circles = 0
        huge_circles = 0
        tiny_circles_areas = 0
        small_circles_areas = 0
        mid_circles_areas = 0
        large_circles_areas = 0
        huge_circles_areas = 0
        org = image.copy()
        tiny = image.copy()
        small = image.copy()
        mid = image.copy()
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
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cnt_area = cv2.contourArea(c)
            
            
            area_percent = cnt_area/img_area
            
            if area_percent > 0.00007:
                if area_percent < 0.00019:
                    tiny_circles += 1
                    tiny_circles_areas += cnt_area
                    cv2.circle(tiny, (int(x), int(y)), int(r), (0, 255, 0), 2)
                elif area_percent < 0.0014:
                    small_circles += 1
                    small_circles_areas += cnt_area
                    cv2.circle(small, (int(x), int(y)), int(r), (255, 0, 0), 2)
                elif area_percent < 0.04:
                    mid_circles += 1
                    mid_circles_areas += cnt_area
                    cv2.circle(mid, (int(x), int(y)), int(r), (0, 0, 255), 2)
                elif area_percent < 0.28:
                    large_circles += 1
                    large_circles_areas += cnt_area
                elif area_percent < 0.85:
                    huge_circles += 1
                    huge_circles_areas += cnt_area

        yield (tiny_circles, small_circles, mid_circles)
        
        # total_bubbles = tiny_circles+small_circles+mid_circles+large_circles+huge_circles
        total_bubbles = tiny_circles+small_circles+mid_circles
        # print(total_bubbles)

        bubbles_analyziz = [total_bubbles, tiny_circles, small_circles, mid_circles, large_circles,
                            huge_circles]

        
    return