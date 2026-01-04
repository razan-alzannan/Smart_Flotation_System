import os
# os.chdir("D:\\Cupper Process Monitoring\\image_analysis\\code\\CPM_v3(image size)")
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import the necessary packages specific to Computer vision
import cv2
from PIL import Image, ImageDraw
from skimage import img_as_ubyte
from cv2 import imshow
from skimage import io,  transform
import seaborn as sns
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation  import watershed
from scipy import ndimage

def ImageMask(img,cellnum, quarter=0):

    #img1 = Image.fromarray(img)
    #img1 = np.array(img)
    img = Image.fromarray(img) 
    img1 = img
    img2 = Image.new(img1.mode, img1.size, (255,255,255))
    mask = Image.new('1', img1.size, 0)
    draw = ImageDraw.Draw(mask)
    #ImageDraw.line(xy, fill=None, width=0, joint=None)
    # draw.polygon([(0,0),(0,200),(100,50),(400,250),(300,400)], fill=255)
    # #draw.line(((0,620),(650,140)), fill=255, width=200)
    # plt.imshow(img1)
    # plt.show()
    if cellnum==1:
        # draw.polygon([(0,250), (60,50), (400,220),(300,410),(0,410),(0,250)],fill=255)
        # draw.polygon([(0,300), (100,100), (350,250),(200,410),(0,410),(0,300)],fill=255)
        # draw.polygon([(0,500),(200,155),(500,310),(450,500),(0,500)],fill=255)
        if quarter == 1:
            draw.polygon([(100,300),(200,155),(300,200),(300,300),(100,300)],fill=255)
        if quarter == 2:
            draw.polygon([(300,200),(500,270),(500,300),(300,300),(300,200)],fill=255)
        if quarter == 3:
            draw.polygon([(0,500),(100,300),(300,300),(300,500),(0,500)],fill=255)
        if quarter == 4:
            draw.polygon([(300,500),(300,300),(500,300),(450,500),(300,500)],fill=255)

    elif cellnum==2:
       
       # draw.polygon([(15,410), (70,170), (410,280),(410,410),(15,410)],fill=255)
        draw.polygon([(30,410), (70,200), (410,300),(410,410),(30,410)],fill=255)
        draw.polygon([(410,0), (410,180), (120,80),(160,0),(410,0)],fill=255)

    elif cellnum==3:
        # draw.polygon([(0,0),(400,250),(400,100),(400,200),(400,360),(350,400),(0,400)], fill=255)
        # draw.polygon([(0,0), (0,190), (100,50)], fill=0)
        # draw.polygon([(400,235), (280,400), (400,400)], fill=0)
        draw.polygon([(0,400), (0,250), (80,100),(350,250),(260,400),(0,400)],fill=255)

    elif cellnum == 8:
        draw.polygon([(50, 300), (35, 35), (280, 35), (250, 300), (50, 300)], fill=255)
        
    # draw.line([(400,250), (300,0)])
    # plt.imshow(mask)
    # plt.show()
    im = Image.composite(img1, img2, mask)
    # plt.imshow(im)
    # plt.show()
    # cv2.imwrite('./image_masked.png', img_as_ubyte(im))

    return im ,mask

def Kernel(img,img_mask,kernel):
    # img = img.reshape((160000,3))
    # img_mask = img_mask.flatten()
    # temp = img.copy()
    # temp[img_mask] = cv2.filter2D(img[img_mask], -1, kernel)
    temp = cv2.filter2D(img, -1, kernel)
    img = img.reshape((400, 400, 3))
    image_kernelver = temp.reshape((400,400,3))
    list_my_image = np.hstack([img, image_kernelver])
    cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

    return image_kernelver

def contours(img,img_area):
    #Let us see the histogram to get cutoff value
   
    my_image = img.copy()
    hist = cv2.calcHist([img],[0], # channels
                        None,# mask - NOne means full image
                        [256], # bin count - for full scale [256]
                        [0,256]) # ranges - [0,256]
    plt.figure(0) 
    plt.plot(hist,color = 'b')
    plt.show()
    # Almost binary. Let us choose cutoff 200

    # Threshold the image to convert boolean image
    ret, im_th = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    # im_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    im_th = np.uint8(im_th)
    cv2.imshow("threshold",im_th)
    # plt.figure(1)
    # plt.imshow(im_th)
    # plt.show()

    # Find contours
    
    contours, hierarchy = cv2.findContours(im_th,cv2.RETR_CCOMP, #RETR_TREE related to hierarchy list. Not covering right now
                            cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE, all the boundary points
    #are stored. cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby
    #saving memory.

    # Let us draw the contours
    my_image_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    areas_df=pd.DataFrame(columns=['Rectarea','pixelArea','rate_pixelArea'])
    Rectarea=[]
    pixelArea=[]

    rate_pixelArea=[]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        Rectarea.append(w*h)
        area = cv2.contourArea(contour)
        pixelArea.append(area)
      
        rate_pixelArea.append(area/img_area)

    areas_df['Rectarea']=Rectarea
    areas_df['pixelArea']=pixelArea
    areas_df['rate_pixelArea']=rate_pixelArea
    areas_df['contours']=contours
    print(type(contours))
    print(type(contour))
    # cv2.drawContours(my_image_color, contours, -1, # all else provide index of contours
    #                 (255,0,0), # color - green
    #                 1) # width
    
    areas_df=areas_df.loc[(areas_df['Rectarea']>20) & (areas_df['rate_pixelArea']<.3)]

    data=areas_df.sort_values(by='rate_pixelArea') 
    sns.displot(data['rate_pixelArea'], kde=True)
    
    cv2.drawContours(my_image_color, tuple(areas_df['contours']), -1, # all else provide index of contours
                    (0,255,0), # color - green
                    1) # width
    
    from random import randint
    for i in tuple(areas_df['contours']):
        cv2.drawContours(my_image_color, i, -1, # all else provide index of contours
                    (randint(0,255),randint(0,255),randint(0,255)), # color - green
                    1) # width
    # cv2.drawContours(my_image_color, contours[592], -1, # all else provide index of contours
    #                 (0,0,255), # color - green
    #                 1) # width
    # cv2.drawContours(my_image_color, contours[631], -1, # all else provide index of contours
    #                 (0,255,255), # color - green
    #                 1) # width
    # cv2.drawContours(my_image_color, contours[389], -1, # all else provide index of contours
    #                 (255,0,255), # color - green
    #                 1) # width
    
    cv2.imshow("drawContours",my_image_color)
    # plt.figure(2)
    # plt.imshow(img)
    # plt.show()
    # list_my_image = np.hstack([my_image, im_th,my_image_color])
    # cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
    # Want to show you duplicate and hence making copy of original
    my_image_color_org = img.copy()
    return  my_image_color_org, contours # Image index just for readability

def bubble_size_calc(contours):
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0,0, 255),1)
    # cv2.putText(my_image_color,str(0),(int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
    plt.imshow(my_image_color, cmap='gray')
    plt.show()

    size = w*h

    return size

def contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0,
    tileGridSize=(8, 8))
    equalized = clahe.apply(img)

    imshow('original image', img)
    imshow('equalized image', equalized)

    return equalized
def Watershed(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold using OTSU
    ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    imshow( "thresh",thresh)

    # noise removal
    import numpy as np

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    imshow("sure_fg", sure_fg)
    imshow("sure_bg", sure_bg)
    imshow("unknown", unknown)
        # Marker labelling
    # Connected Components determines the connectivity of blob-like regions in a binary image.
    ret, markers = cv2.connectedComponents(thresh)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    imshow( "img",img)
def Watershed2(img):
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding ste
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    cv2.imshow("shifted", shifted)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D,  min_distance=20,
        labels=thresh)
    plt.imshow(D)
    plt.show()
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    # loop over the unique labels returned by the Watershed
    # algorithm
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
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# pil_image = Image.open('D:/Cupper Process Monitoring/Image_Processing/computer_vision/image_output/test_img.png')
# plt.imshow(pil_image)
# plt.show()
# # 1 resize image
# newsize = (400, 400)
# resized_image = pil_image.resize(newsize)
# plt.imshow(resized_image)
# plt.show()

# # 2 crop image
# my_image_color_masked ,mask = ImageMask(resized_image,3)
# plt.imshow(my_image_color_masked)
# plt.show()

# # 3 convert image from rgp to pgr
# my_image_color_masked = img_as_ubyte(my_image_color_masked)
# my_image_color=my_image_color_masked
# #cv2.imwrite('./my_image_color_masked.png', my_image_color_masked)
# # 4 use only the important part
# mask = img_as_ubyte(mask)
# img_mask = (mask==255)
# img_area=np.sum(img_mask) # find area for not wite area in image
# #kernel_ver = np.array([[0,1,-1,0], [1,3,-3,-1],[1,3,-3,-1],[0,1,-1,0]], np.float32)
# #kernel_ver = np.array([[-1,- 1,- 1], [-1, 8,-1], [-1, -1, -1]], dtype=np.float32)
# kernel_ver = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
# # temp = my_image_color[img_mask]
# # cv2.imshow("All", temp)
# # print(temp.shape)
# image_kernelver=Kernel(my_image_color,img_mask,kernel_ver)
# #Watershed(my_image_color_masked)

# my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)
# # # 5 apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
# my_image_gray = contrast(my_image_gray)


# my_image_color = cv2.cvtColor(my_image_gray,cv2.COLOR_GRAY2BGR)
# image = cv2.imread('watershed_coins_01.jpg')
# Watershed(my_image_color)

# image, img_contours = contours(my_image_gray,img_area)
# bubble_size_calc(img_contours)
# # 
# # kernel_ver = np.array([[0,1,-1,0], [1,3,-3,-1],[1,3,-3,-1],[0,1,-1,0]], np.float32)
# # image_kernelver=Kernel(my_image_color,kernel_ver)
# # my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)
# # my_image_gray = contrast(my_image_gray)
# # plt.imshow(my_image_gray, cmap='gray')
# # plt.show()
# # image, image_contours = contours(my_image_gray)
# # print(bubble_size_calc(image_contours))


