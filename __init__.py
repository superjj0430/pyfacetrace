#!/usr/bin/env python
# coding: utf-8
import sys,os,dlib,glob,numpy
#pip install scikit-image
from skimage import io
import cv2
import os
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from os.path import dirname
#Package_Dir = dirname(__file__)

_FONT = ImageFont.truetype("kaiu.ttf",20,index=0)

_detector = dlib.get_frontal_face_detector()

_sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

_facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#dlib_face_recognition_resnet_model_v1.dat  has to download by user


DB={}
 
def getFeatureVector(img,rect=None):
    if not rect:
        try:
            rect=_detector(img, 1)[0]
        except:return None
    shape = _sp(img,rect) 
    
    face_descriptor = _facerec.compute_face_descriptor(img, shape)
   
    return numpy.array(face_descriptor)

def _createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)
        
def predictFromDB(VTest,db=DB):
    minD = sys.float_info.max
    minK=''
    for k in DB:
        dist=numpy.linalg.norm(VTest-DB[k])
        if dist<minD:
            minK=k
            minD=dist
    return minK,minD

def addText2Img_cv2(img_cv2,text,font=_FONT,position=(20,20),fill=(255,0,0)):
    img_PIL = Image.fromarray(cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB))#cv2.COLOR_BGR2RGB cv2.COLOR_RGB2BGR
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fill)
    img_cv2 = cv2.cvtColor(numpy.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    img_PIL.close()
    return img_cv2

def LoadDB(db=DB,folder='train'):
    for D in os.listdir(folder):
        for r,d,f in os.walk(folder+"\\"+D):
            fname=folder+"\\"+D+"\\"+f[0]
            print("get Feature from:"+fname)
            img = io.imread(fname)
            DB[D]=getFeatureVector(img)
            break


def getPicFromCam(tag,folder="train"):
    ret = False
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    if not cap:return False
    cv2.startWindowThread()
    
    text="press p to take picture"
    num=0
    _createFolder(folder+"\\"+tag)

    while(True):
        if cap.isOpened():
            ret, frame = cap.read()
           
            if ret: 
                try:
                    rect=_detector(frame, 1)[0] 
                    if cv2.waitKey(1) & 0xFF == ord('p'):
                        im=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        io.imsave(folder+"\\"+tag+"\\"+str(num)+".jpg",im)
                        num+=1
                        text=str(num)+" pictures saved..."
                        ret = True
                    cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(255,0,0),3)
                    cv2.putText(frame, text,(rect.left()-80, rect.top()-20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)    
                    frame=addText2Img_cv2(frame,'tag='+tag,_FONT,(rect.left(), rect.top()-_FONT.size*2-5))   
                #except:pass 
                except IndexError:pass
                cv2.imshow('press esc to exit', frame)
        
            if cv2.waitKey(1) == 27: break
    cap.release()
    cv2.destroyAllWindows()
    return ret

def predictVedio(vedioPath,skipFranmes=50,db=DB):
    cv2.startWindowThread()
    cap = cv2.VideoCapture(vedioPath)
    success,image = cap.read()
    count = 0
    while success:
        success,frame = cap.read()
        count+=1
        if count%50!=0 :continue
        try:
            rects=_detector(frame, 1)
            for rect in rects:
                V=getFeatureVector(frame,rect)
                Tag,dist=predictFromDB(V,DB)
                cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(255,0,0),3)
                text=Tag+":"+str(dist)
                frame=addText2Img_cv2(frame,Tag+":"+str(round(dist,3)),_FONT,(rect.left(), rect.top()-_FONT.size-1))        
        except IndexError:pass    
        cv2.imshow('My Image', frame)
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
    cap.release()
    cv2.destroyAllWindows()
    
def getPicRawFeature(fname):
    try:
        img = io.imread(fname)
        #img.tofile('test.txt',',')
        return img.flatten()/255.0
    except Exception:print(Exception.args)
    return None