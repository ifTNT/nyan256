#!/bin/python

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from math import ceil, sqrt

#============================================
#Extract color mapping from VGA color plate
#============================================

#Read image and convert into hsv color space
plate_bgr = cv2.imread('vga256color_plate_1px.png')
plate_hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)

vga_hue = np.transpose(plate_hsv[1])[0].tolist()
hue_sorted = vga_hue+[180]
hue_sorted.sort()

def h_(x):
    rtval = 0;
    for i in range(len(hue_sorted)-1):
        if(x>=hue_sorted[i] and x<hue_sorted[i+1]):
            #rtval = i+round((x-hue_sorted[i])/(hue_sorted[i+1]-hue_sorted[i]))
            rtval = np.array([i, i+1])
            break
    rtval = (rtval+8)%(len(hue_sorted)-1)
    return rtval.tolist()

#h_ = [(round(i/9)+8)%24 for i in range(180)] #Index mapping of hue

vga_sat = [255, 129, 73]
def s_(x): #Index mapping of saturation
    if(x>vga_sat[1]):
        return [0, 1]
        #return round((vga_sat[0]-x)/(vga_sat[0]-vga_sat[1]))
    else:
        return [1, 2]
        #return round((vga_sat[1]-x)/(vga_sat[1]-0))+1

vga_lightness = [255, 113, 65]
def v_(x): #Index mapping of lightness
    if(x>vga_lightness[1]):
        return [0, 1]
        #return round((vga_lightness[0]-x)/(vga_lightness[0]-vga_lightness[1]))
    else:
        return [1, 2]
        #return round((vga_lightness[1]-x)/(vga_lightness[1]-0))+1

#=========================
#Begin image proccessing
#=========================
img_bgra = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
if(img_bgra.shape[2]==4):
    img_alpha = img_bgra[:,:,3]
else:
    img_alpha = np.ones(shape=img_bgra.shape[0:2])*255
#img_blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)

#print(img_alpha.tolist())

img_hsv = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2HSV).astype(int)

grayscale_threshold = 60

#Convert to index of vga color plate in HSV color space
for i,row in enumerate(img_hsv):
    for j,v in enumerate(row):
        if(v[1]<=grayscale_threshold or v[2]<=grayscale_threshold): #grayscale
            img_hsv[i][j] = np.array([512,512,round(float(v[2])*15/255)])
        else:
            candidate_axis = [h_(v[0])]+[s_(v[1])]+[v_(v[2])]
            candidate = np.zeros((8,3), dtype=int)
            for k in range(8): #Enumerate candidates
                candidate[k][0] = candidate_axis[0][     k&1]
                candidate[k][1] = candidate_axis[1][(k>>1)&1]
                candidate[k][2] = candidate_axis[2][(k>>2)&1]
            
            #Calculate distance in HSV color space
            distance = []
            for k in candidate:
                dh = 1 * min(abs(v[0]-vga_hue[k[0]]), 180-abs(v[0]-vga_hue[k[0]]))
                ds = 1 * abs(v[1]-vga_sat[k[1]])
                dv = 1 * abs(v[2]-vga_lightness[k[2]])
                distance.append(sqrt(dh*dh+ds*ds+dv*dv)) #Euclidean distance
                #distance.append(max(dh, ds, dv)) #Chebyshev distance
            
            #Pick the cantidate that have minimal distance
            candidate_index = 0
            for k, v in enumerate(distance):
                if(v<distance[candidate_index]):
                    candidate_index = k
            
            #print(distance, candidate_index)
            img_hsv[i][j] = candidate[candidate_index]

#Map the binary value
img_bin = np.ndarray(shape=img_hsv.shape[0:2], dtype=np.uint8)
for i,row in enumerate(img_hsv):
    for j,v in enumerate(row):
        if(img_alpha[i][j]==0):
            img_bin[i][j] = 255 #Define transparent in unused value
        else:
            if(v[0]==512 and v[1]==512): #grayscale
                img_bin[i][j] = v[2]+0x10
            else:
                img_bin[i][j] = ((v[2]*3)+v[1])*24+v[0]+0x20
img_bin = img_bin.flatten()

#Convert to value of vga color plate in HSV color space
for i,row in enumerate(img_hsv):
    for j,v in enumerate(row):
        if(v[0]==512 and v[1]==512): #grayscale
            img_hsv[i][j] = np.array([0,0,v[2]*255/15])
        else:
            img_hsv[i][j] = np.array([vga_hue[v[0]], vga_sat[v[1]], vga_lightness[v[2]]])
img_hsv = img_hsv.astype(np.uint8)
img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
img_alpha = img_alpha.reshape(img_alpha.shape[0], img_alpha.shape[1], 1)
print(img_bgr.shape, img_alpha.shape)
img_bgra = np.concatenate((img_bgr, img_alpha), axis=2)
print(img_bin)

#Write output file
cv2.imwrite('output.png', img_bgra)

outFile = open(re.sub(r'\..+$', '.bin',sys.argv[1]), "wb")
outFile.write(bytearray(img_bin))
outFile.close()
