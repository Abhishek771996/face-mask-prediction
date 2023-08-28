#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import tensorflow as tf
import numpy as np
import os


# In[3]:


os.chdir(r'C:\Users\HP\Downloads\facemask machinlearning\mask')


# In[10]:


# use predicted model

model=tf.keras.models.load_model('mymodel.h5')


# In[11]:


# METHOD to detect face without using deep learning
cascade=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')


# In[13]:


cap=cv2.VideoCapture(0)

while(cap.isOpened()):
    b,frame=cap.read()
    faces=cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
    for x,y,w,h in faces:
        face=frame[y:y+h,x:x+w]
        cv2.imwrite("face.jpg",face)#Save image
        face=tf.keras.preprocessing.image.load_img("face.jpg",
                                                   target_size=(150,150,3))
        face=tf.keras.preprocessing.image.img_to_array(face)#Convert face to numpy array
        face=np.expand_dims(face,axis=0)#Convert to $D-batch size concept
        
        ans=model.predict(face)
        if ans<=0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,"With Mask",(x//2,y//2),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0))
            
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(frame,"Without Mask",(x//2,y//2),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255))
    
    
    
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



# In[14]:





# In[ ]:




