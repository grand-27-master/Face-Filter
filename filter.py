import cv2;
import numpy as np;

#steps to follow:
# 1) load both classifiers
# 2) load "before" image and optimize it accordingly
# 3) make an overlay image of "glasses" and load moustache image
# 4) call detectMultiScale (of eyes) function on "before" image
# 5) apply resize and loop
# 6) repeat this for glasses image


eyes_detect=cv2.CascadeClassifier('frontalEyes35x16.xml')
moustache_detect=cv2.CascadeClassifier('Nose18x15.xml')

img=cv2.imread('Jamie_Before')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Image overlay combines two existing NEF (RAW) photographs to create a single picture that is saved separately from the originals.
overlay_img=cv2.imread('glasses',IMREAD_UNCHANGED)
overlay_img=cv2.cvtColor(overlay_img,cv2.COLOR_BGRA2RGBA)

moustache_img=cv2.imread('mustache')
moustache_img=cv2.cvtColor(moustache_img,cv2.COLOR_BGRA2RGBA)

eyes=eyes_detect.detectMultiScale(img,1.3,4)

x,y,w,h=eyes[0]
overlay_img=cv2.resize(overlay_img,(w,h))

for i in range(overlay_img.shape[0]):
    for j in range(overlay_img.shape[1]):
        if(overlay_img[i,j,3]>0):
            img[y+i,x+j,:]=overlay_img[i,j,:-1]

cv2.imshow('image with glasses',img)   

moustache=moustache_detect.detectMultiScale(img,1.3,4)      

x,y,w,h=moustache[0]
moustache=cv2.resize(moustache,(w,h))

for i in range(moustache.shape[0]):
    for j in range(moustache.shape[1]):
        if(moustache[i,j,3]>0):
            img[y+i,x+j,:]=moustache[i,j,:-1]

cv2.imshow('image with moustache',img)

cap.release()        
cv2.destroyAllWindows()  