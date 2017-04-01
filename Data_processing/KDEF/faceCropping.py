'''
Sources:
http://opencv.willowgarage.com/documentation/python/cookbook.html
http://www.lucaamore.com/?p=638
'''
#Python 2.7.2
#Opencv 2.4.2
#PIL 1.1.7
EMODICT = {"AF":"3", "AN":"4","DI":"0", "HA":"1", "NE":"5", "SA":"6", "SU":"2"}
TARGET_SIZE = 158


import cv2
import numpy as np
import cv #Opencv
from PIL import Image #Image from PIL
import glob
import os

def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    #variables
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tobytes(), pil_im.size[0])
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.frombytes("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def faceCrop(imagePattern,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_default.xml')

    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print 'No Images Found'
        return
    n = 0
    for img in imgList:
        pil_im=Image.open(img)
        cv_im=pil2cvGrey(pil_im)
        faces=DetectFace(cv_im,faceCascade)
        if faces:
            for face in faces:
                croppedImage=imgCrop(pil_im, face[0], boxScale=boxScale)
                path = getResultPath(img)
                if (path != ""):
                    croppedImage.thumbnail([TARGET_SIZE, TARGET_SIZE], Image.ANTIALIAS)
                    croppedImage.save(path)
                    n+=1
        else:
            print 'No faces found:', img
    print n


def getResultPath(img):
    comp = img.split('/')
    path = comp[0] + '/' + comp[1] + '/' + comp[2] + '/KDEF_CROPPED/'
    if (comp[4][1] == 'M') :
      path = path + str(int(comp[4][2:]) - 1) + '-'
    else:
      path = path + str(int(comp[4][2:]) + 34) + '-'
    if (comp[5][6] == 'S'):
      path = path + EMODICT[comp[5][4:6]] + '-'
      path = path + ("0" if comp[4][0] == 'A' else "1") + ".JPG"
      return path
    return ""

def test(imageFilePath):
    pil_im=Image.open(imageFilePath)
    cv_im=pil2cvGrey(pil_im)
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_default.xml')
    face_im=DetectFace(cv_im,faceCascade, returnImage=True)
    img=cv2pil(face_im)
    img.show()
    img.save('test.png')


# Test the algorithm on an image
# test('../../KDEF_dataset/AF01/AF01AFS.jpg')

# Crop all jpegs in a folder. Note: the code uses glob which follows unix shell rules.
# Use the boxScale to scale the cropping area. 1=opencv box, 2=2x the width and height
faceCrop('../../KDEF_dataset/KDEF/*/*.JPG',boxScale=1)
