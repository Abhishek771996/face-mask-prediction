#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#cnn: #Convolutional neural network 


# In[46]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# In[47]:


# image data genration

train_datagen=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,horizontal_flip=True,rotation_range=0.2,rescale=1/255)


# In[48]:


test_datagen=ImageDataGenerator(rescale=1/255)


# In[49]:


# give path of images
train_datagen=train_datagen.flow_from_directory(r'C:\Users\HP\Downloads\facemask machinlearning\mask\train',
                                               class_mode='binary',
                                               target_size=(150,150),batch_size=16) # image size


# In[50]:


test_datagen=test_datagen.flow_from_directory(r'C:\Users\HP\Downloads\facemask machinlearning\mask\train',
                                               class_mode='binary',
                                               target_size=(150,150),batch_size=16)


# In[51]:


cnn=tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,
                               input_shape=(150,150,3),activation="relu"))


# In[52]:


#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)) # 4 BY 4 SHAPE CONVERTED INTO 2 BY 2


# In[53]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))


# In[54]:


#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Flatten())


# In[55]:


# hidden layer
cnn.add(tf.keras.layers.Dense(units=120,activation='relu'))


# In[56]:


cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[57]:


cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


cnn.fit(train_datagen,validation_data=test_datagen,epochs=100)  # dont load its take too much time


# In[ ]:


cnn.save("mymodel.h5")


# In[ ]:


cnn.save("mymodel.h5")


# In[ ]:




