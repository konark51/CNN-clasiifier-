#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import os
for dirname, _, filenames in os.walk('c:/ML/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import os
import zipfile
import random
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from shutil import copyfile
from os import getcwd


# In[5]:


😂


# In[3]:


base_dir = 'c:/ML/kaggle/working/'
train_dir = os.path.join(base_dir, 'train')
train_img_names = os.listdir(train_dir)
print(train_dir)


# In[4]:


train_img_names[:10]


# In[6]:


print('total training images :', len(train_img_names ))


# In[7]:


categories= list()
for image in train_img_names:
    category = image.split(".")[0]
    if category == "dog":
        categories.append("dog")
    else:
        categories.append("cat")
df= pd.DataFrame({"Image":train_img_names, "Category": categories})


# In[8]:


df.head()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import random
plt.figure(figsize=(12,10))
sns.countplot(data=df, x="Category",palette="magma")


# In[11]:


sample = random.choice(train_img_names)
plt.imshow(plt.imread(("c:/ML/kaggle/working/train/"+sample)))


# In[12]:


sample = random.choice(train_img_names)
plt.imshow(plt.imread(("c:/ML/kaggle/working/train/"+sample)))


# In[13]:


from sklearn.model_selection import train_test_split
train,validation= train_test_split(df, test_size=0.1)
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)


# In[14]:


train


# In[15]:


print(len(train))


# In[16]:


validation


# In[17]:


print(len(validation))


# In[18]:


plt.figure(figsize=(13,10))
sns.countplot(data=train, x="Category",palette="viridis")


# In[19]:


plt.figure(figsize=(13,10))
sns.countplot(data=validation, x="Category",palette="plasma")


# In[21]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )


# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_dataframe(train,
                                                    directory="c:/ML/kaggle/working/train/",
                                                    x_col='Image',
                                                    y_col='Category',
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))   


# In[22]:


validation_datagen  = ImageDataGenerator( rescale = 1.0/255.)
validation_generator =  validation_datagen.flow_from_dataframe(validation,
                                                            directory="c:/ML/kaggle/working/train/",
                                                              x_col='Image',
                                                             y_col='Category',
                                                              batch_size=20,
                                                              class_mode  = 'binary',
                                                              target_size = (150, 150))


# In[23]:


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# In[24]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3,activation="relu", input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))


# In[25]:


model.add(Conv2D(filters=64, kernel_size=3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=1024, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))


# In[26]:


model.summary()


# In[27]:


model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# In[28]:


callback=EarlyStopping(monitor="val_loss", patience=2)
callback_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=2, factor=0.5, min_lr=0.00001)


# In[29]:


history=model.fit(train_generator, validation_data=validation_generator, 
                  epochs=8, callbacks=[callback,callback_lr])


# In[30]:


pd.DataFrame(model.history.history)


# In[31]:


sns.set_style("darkgrid")
pd.DataFrame(model.history.history).plot(figsize=(15,10))


# In[37]:


acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs
print(epochs)
print(acc)
print(len(acc))
print(val_acc)


# In[38]:


plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()


# In[39]:


plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )


# In[40]:


test_dir = 'c:/ML/kaggle/working/test/'
test_images = os.listdir(os.path.join(test_dir))
test_images[:10]


# In[55]:


test_dir = 'c:/ML/kaggle/working/test/'
test_images = os.listdir(os.path.join(test_dir))
test_images[:10]
sample = test_images[1]
plt.imshow(plt.imread("c:/ML/kaggle/working/test/"+sample))


# In[56]:


test_df = pd.DataFrame({'Image': test_images})
test_df.head()


# In[58]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  directory="c:/ML/kaggle/working/test/",
                                                 x_col="Image",
                                                 y_col=None,
                                                  class_mode  = None,
                                                 target_size=(150,150),
                                                shuffle = True,
                                                batch_size=20)


# In[59]:


predictions = model.predict(test_generator,steps = np.ceil(12500/20))
predictions


# In[60]:


test_df["category"]=pd.DataFrame(predictions, columns=["category"])
test_df


# In[61]:


def labelizor(prediction):
    if prediction > 0.5:
        return 1
    else:
        return 0


# In[62]:


test_df["category"] = test_df["category"].apply(labelizor)
test_df


# In[63]:


plt.figure(figsize=(13,10))
sns.countplot(data=test_df, x="category",palette="magma")


# In[64]:


test_df=test_df.reset_index()
test_df


# In[65]:


test_df=test_df.rename(columns={"index": "id"})
test_df


# In[66]:


submission_df=test_df.copy()
submission_df.drop("Image", axis=1, inplace=True)
submission_df


# In[67]:


submission_df.to_csv('submission.csv', index=False)


# In[68]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:




