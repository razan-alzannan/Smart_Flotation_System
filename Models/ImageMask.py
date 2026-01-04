import os
# os.chdir("EC:\\Users\\USER\\Desktop\\Projects\\CPM\\FM\\Velocity\\vv")
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

def polygon_area(coords):
    """Calculate the area of the polygon using the shoelace formula."""
    n = len(coords)
    area = 0.5 * abs(sum(x0*y1 - x1*y0
                         for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1])))
    return area

def polygon_centroid(coords):
    """Calculate the centroid of the polygon."""
    n = len(coords)
    A = polygon_area(coords)  # Calculate the area of the polygon
    Cx = 1 / (6 * A) * sum((x0 + x1) * (x0 * y1 - x1 * y0)
                           for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]))
    Cy = 1 / (6 * A) * sum((y0 + y1) * (x0 * y1 - x1 * y0)
                           for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]))
    return Cx, Cy

def ImageMask(img,cellnum):
        #img1 = Image.fromarray(img)
    img = Image.fromarray(img) 
    cv2.imwrite('./image_cell03.png', img_as_ubyte(img))
    img1 = img
    img2 = Image.new(img1.mode, img1.size, (255,255,255))
    
    mask = Image.new('1', img1.size, 0)
    draw = ImageDraw.Draw(mask)
    angle_mask = Image.new('1', img1.size, 0)
    angle_draw = ImageDraw.Draw(angle_mask)
    #ImageDraw.line(xy, fill=None, width=0, joint=None)
    # draw.polygon([(0,0),(0,200),(100,50),(400,250),(300,400)], fill=255)
    # #draw.line(((0,620),(650,140)), fill=255, width=200)
    # print(img1.size)
    if cellnum==1:
        # draw.polygon([(0,0),(400,250),(400,100),(400,200),(400,360),(350,400),(0,400)], fill=255)
        # draw.polygon([(0,0), (0,190), (100,50)], fill=0)
        # draw.polygon([(400,235), (280,400), (400,400)], fill=0)
        draw.polygon([(0,400), (0,190), (100,50),(400,235),(280,400),(0,400)],fill=255)
    elif cellnum==3:
        coords = [(90,img1.size[1]), (190,110), (550,290),(430,img1.size[1]),(0,img1.size[1])]
        cx, cy = polygon_centroid(coords)
        center_point = (cx, cy)
        draw.polygon(coords, fill=255)
        # draw.polygon([(x,y), (x+10, y), (x+10, y+10), (x, y+10), (x,y)], fill=0)
        angle_draw.polygon([(90,img1.size[1]), (190,130), (450,250),(450,300),(200,200),(150,img1.size[1])],fill=255)

    elif cellnum == 8:
        coords = [(50, 300), (35, 35), (280, 35), (250, 300), (50, 300)]
        cx, cy = polygon_centroid(coords)
        center_point = (cx, cy)
        draw.polygon(coords, fill=255)
        angle_draw.polygon([(90,img1.size[1]), (190,130), (450,250),(450,300),(200,200),(150,img1.size[1])],fill=255)


        # draw.polygon([(50, 300), (35, 35), (280, 35), (250, 300), (50, 300)], fill=255)
    # draw.line([(400,250), (300,0)])
    # plt.imshow(mask)
    # plt.show()
    im = Image.composite(img1, img2, mask)
    # plt.imshow(im)
    # plt.show()
    my_image_color_masked = img_as_ubyte(im)
    angle_mask = img_as_ubyte(angle_mask)
    cv2.imwrite('./image_masked.png', img_as_ubyte(im))
    return my_image_color_masked, mask, angle_mask, center_point




# pil_image = cv2.imread('cell1.jpg')
# plt.imshow(pil_image)
# plt.show()
# # 1 resize image
# # newsize = (400, 400)
# # resized_image = cv2.resize(newsize)
# # plt.imshow(resized_image)
# # plt.show()

# # 2 crop image
# my_image_color_masked ,mask = ImageMask(pil_image,1)
# plt.imshow(my_image_color_masked)
# plt.show()




