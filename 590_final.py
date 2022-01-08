#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os, shutil
import pandas as pd
original_dataset_dir = 'dataset'
base_dir = os. getcwd()
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#--------------------training---------------------------------
# Directories for the training, validation, and test splits
# Directory with training cheese cake pictures
train_cheese_cake_dir = os.path.join(train_dir, 'cheese cake')
os.mkdir(train_cheese_cake_dir)

# Directory with training chicken wings pictures
train_chicken_wings_dir = os.path.join(train_dir, 'chicken wings')
os.mkdir(train_chicken_wings_dir)

# Directory with training donuts pictures
train_donuts_dir = os.path.join(train_dir, 'donuts')
os.mkdir(train_donuts_dir)

# Directory with training fried rice pictures
train_fried_rice_dir = os.path.join(train_dir, 'fried rice')
os.mkdir(train_fried_rice_dir)

# Directory with training french fries pictures
train_french_fries_dir = os.path.join(train_dir, 'french fries')
os.mkdir(train_french_fries_dir)

# Directory with training gyoza pictures
train_gyoza_dir = os.path.join(train_dir, 'gyoza')
os.mkdir(train_gyoza_dir)

# Directory with training ice-cream pictures
train_ice_cream_dir = os.path.join(train_dir, 'ice-cream')
os.mkdir(train_ice_cream_dir)

# Directory with training oyster pictures
train_oyster_dir = os.path.join(train_dir, 'oyster')
os.mkdir(train_oyster_dir)

# Directory with training scallops pictures
train_scallops_dir = os.path.join(train_dir, 'scallops')
os.mkdir(train_scallops_dir)

# Directory with training waffle pictures
train_waffle_dir = os.path.join(train_dir, 'waffle')
os.mkdir(train_waffle_dir)

#--------------------validation---------------------------------

# Directory with validation cheese cake pictures
validation_cheese_cake_dir = os.path.join(validation_dir, 'cheese cake')
os.mkdir(validation_cheese_cake_dir)

# Directory with validation chicken wings pictures
validation_chicken_wings_dir = os.path.join(validation_dir, 'chicken wings')
os.mkdir(validation_chicken_wings_dir)

# Directory with validation donuts pictures
validation_donuts_dir = os.path.join(validation_dir, 'donuts')
os.mkdir(validation_donuts_dir)

# Directory with validation fried rice pictures
validation_fried_rice_dir = os.path.join(validation_dir, 'fried rice')
os.mkdir(validation_fried_rice_dir)

# Directory with validation french fries pictures
validation_french_fries_dir = os.path.join(validation_dir, 'french fries')
os.mkdir(validation_french_fries_dir)

# Directory with validation gyoza pictures
validation_gyoza_dir = os.path.join(validation_dir, 'gyoza')
os.mkdir(validation_gyoza_dir)

# Directory with validation ice-cream pictures
validation_ice_cream_dir = os.path.join(validation_dir, 'ice-cream')
os.mkdir(validation_ice_cream_dir)

# Directory with validation oyster pictures
validation_oyster_dir = os.path.join(validation_dir, 'oyster')
os.mkdir(validation_oyster_dir)

# Directory with validation scallops pictures
validation_scallops_dir = os.path.join(validation_dir, 'scallops')
os.mkdir(validation_scallops_dir)

# Directory with validation waffle pictures
validation_waffle_dir = os.path.join(validation_dir, 'waffle')
os.mkdir(validation_waffle_dir)

#--------------------test---------------------------------
# Directory with test cheese cake pictures
test_cheese_cake_dir = os.path.join(test_dir, 'cheese cake')
os.mkdir(test_cheese_cake_dir)

# Directory with test chicken wings pictures
test_chicken_wings_dir = os.path.join(test_dir, 'chicken wings')
os.mkdir(test_chicken_wings_dir)

# Directory with test donuts pictures
test_donuts_dir = os.path.join(test_dir, 'donuts')
os.mkdir(test_donuts_dir)

# Directory with test fried rice pictures
test_fried_rice_dir = os.path.join(test_dir, 'fried rice')
os.mkdir(test_fried_rice_dir)

# Directory with test french fries pictures
test_french_fries_dir = os.path.join(test_dir, 'french fries')
os.mkdir(test_french_fries_dir)

# Directory with test gyoza pictures
test_gyoza_dir = os.path.join(test_dir, 'gyoza')
os.mkdir(test_gyoza_dir)

# Directory with test ice-cream pictures
test_ice_cream_dir = os.path.join(test_dir, 'ice-cream')
os.mkdir(test_ice_cream_dir)

# Directory with test oyster pictures
test_oyster_dir = os.path.join(test_dir, 'oyster')
os.mkdir(test_oyster_dir)

# Directory with test scallops pictures
test_scallops_dir = os.path.join(test_dir, 'scallops')
os.mkdir(test_scallops_dir)

# Directory with test waffle pictures
test_waffle_dir = os.path.join(test_dir, 'waffle')
os.mkdir(test_waffle_dir)


#--------------------copy pictures to according folders---------------------------------

# Copies the first 500 cheese cake images to train_cheese_cake_dir
fnames = ['cheesecake.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/cheesecake', fname)
    dst = os.path.join(train_cheese_cake_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 cheese cake images to validation_cheese_cake_dir    
fnames = ['cheesecake.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/cheesecake', fname)
    dst = os.path.join(validation_cheese_cake_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 cheese cake images to test_cheese_cake_dir
fnames = ['cheesecake.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/cheesecake', fname)
    dst = os.path.join(test_cheese_cake_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 chicken wings images to train_chicken_wings_dir
fnames = ['chicken_wings.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/chicken_wings', fname)
    dst = os.path.join(train_chicken_wings_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 chicken wings images to validation_chicken_wings_dir    
fnames = ['chicken_wings.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/chicken_wings', fname)
    dst = os.path.join(validation_chicken_wings_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 chicken wings images to test_chicken_wings_dir
fnames = ['chicken_wings.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/chicken_wings', fname)
    dst = os.path.join(test_chicken_wings_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 donuts images to train_donuts_dir
fnames = ['donuts.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/donuts', fname)
    dst = os.path.join(train_donuts_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 donuts images to validation_donuts_dir    
fnames = ['donuts.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/donuts', fname)
    dst = os.path.join(validation_donuts_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 donuts images to test_donuts_dir
fnames = ['donuts.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/donuts', fname)
    dst = os.path.join(test_donuts_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 french fries images to train_french_fries_dir
fnames = ['french_fries.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/french_fries', fname)
    dst = os.path.join(train_french_fries_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 french fries images to validation_french_fries_dir    
fnames = ['french_fries.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/french_fries', fname)
    dst = os.path.join(validation_french_fries_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 french fries images to test_french_fries_dir
fnames = ['french_fries.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/french_fries', fname)
    dst = os.path.join(test_french_fries_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 fried rice images to train_fried_rice_dir
fnames = ['fried_rice.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/fried_rice', fname)
    dst = os.path.join(train_fried_rice_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 fried rice images to validation_fried_rice_dir    
fnames = ['fried_rice.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/fried_rice', fname)
    dst = os.path.join(validation_fried_rice_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 fried rice images to test_fried_rice_dir
fnames = ['fried_rice.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/fried_rice', fname)
    dst = os.path.join(test_fried_rice_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 gyoza images to train_gyoza_dir
fnames = ['gyoza.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/gyoza', fname)
    dst = os.path.join(train_gyoza_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 gyoza images to validation_gyoza_dir    
fnames = ['gyoza.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/gyoza', fname)
    dst = os.path.join(validation_gyoza_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 gyoza images to test_gyoza_dir
fnames = ['gyoza.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/gyoza', fname)
    dst = os.path.join(test_gyoza_dir, fname)
    shutil.copyfile(src, dst)
    
# Copies the first 500 ice cream images to train_ice_cream_dir
fnames = ['ice_cream.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/ice_cream', fname)
    dst = os.path.join(train_ice_cream_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 ice cream images to validation_ice_cream_dir    
fnames = ['ice_cream.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/ice_cream', fname)
    dst = os.path.join(validation_ice_cream_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 ice cream images to test_ice_cream_dir
fnames = ['ice_cream.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/ice_cream', fname)
    dst = os.path.join(test_ice_cream_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 500 oysters images to train_oysters_dir
fnames = ['oysters.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/oysters', fname)
    dst = os.path.join(train_oyster_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 oysters images to validation_oysters_dir    
fnames = ['oysters.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/oysters', fname)
    dst = os.path.join(validation_oyster_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 oysters images to test_oysters_dir
fnames = ['oysters.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/oysters', fname)
    dst = os.path.join(test_oyster_dir, fname)
    shutil.copyfile(src, dst)
    
# Copies the first 500 scallops images to train_scallops_dir
fnames = ['scallops.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/scallops', fname)
    dst = os.path.join(train_scallops_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 scallops images to validation_scallops_dir    
fnames = ['scallops.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/scallops', fname)
    dst = os.path.join(validation_scallops_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 scallops images to test_scallops_dir
fnames = ['scallops.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/scallops', fname)
    dst = os.path.join(test_scallops_dir, fname)
    shutil.copyfile(src, dst)
    
# Copies the first 500 waffles images to train_waffles_dir
fnames = ['waffle.{}.jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/waffles', fname)
    dst = os.path.join(train_waffle_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 200 waffles images to validation_waffles_dir    
fnames = ['waffle.{}.jpg'.format(i) for i in range(500, 700)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/waffles', fname)
    dst = os.path.join(validation_waffle_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 300 waffles images to test_waffles_dir
fnames = ['waffle.{}.jpg'.format(i) for i in range(700, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+'/waffles', fname)
    dst = os.path.join(test_waffle_dir, fname)
    shutil.copyfile(src, dst)

   
print('total training cheese_cake images:', len(os.listdir(train_cheese_cake_dir)))
print('total training chicken_wings images:', len(os.listdir(train_chicken_wings_dir)))
print('total training donuts images:', len(os.listdir(train_donuts_dir)))
print('total training fried_rice images:', len(os.listdir(train_fried_rice_dir)))
print('total training gyoza images:', len(os.listdir(train_gyoza_dir)))
print('total training french_fries images:', len(os.listdir(train_french_fries_dir)))
print('total training ice_cream images:', len(os.listdir(train_ice_cream_dir)))
print('total training oyster images:', len(os.listdir(train_oyster_dir)))
print('total training scallops images:', len(os.listdir(train_scallops_dir)))
print('total training waffle images:', len(os.listdir(train_waffle_dir)))

print('total validation cheese_cake images:', len(os.listdir(validation_cheese_cake_dir)))
print('total validation chicken_wings images:', len(os.listdir(validation_chicken_wings_dir)))
print('total validation donuts images:', len(os.listdir(validation_donuts_dir)))
print('total validation fried_rice images:', len(os.listdir(validation_fried_rice_dir)))
print('total validation gyoza images:', len(os.listdir(validation_gyoza_dir)))
print('total validation french_fries images:', len(os.listdir(validation_french_fries_dir)))
print('total validation ice_cream images:', len(os.listdir(validation_ice_cream_dir)))
print('total validation oyster images:', len(os.listdir(validation_oyster_dir)))
print('total validation scallops images:', len(os.listdir(validation_scallops_dir)))
print('total validation waffle images:', len(os.listdir(validation_waffle_dir)))

print('total test cheese_cake images:', len(os.listdir(test_cheese_cake_dir)))
print('total test chicken_wings images:', len(os.listdir(test_chicken_wings_dir)))
print('total test donuts images:', len(os.listdir(test_donuts_dir)))
print('total test fried_rice images:', len(os.listdir(test_fried_rice_dir)))
print('total test gyoza images:', len(os.listdir(test_gyoza_dir)))
print('total test french_fries images:', len(os.listdir(test_french_fries_dir)))
print('total test ice_cream images:', len(os.listdir(test_ice_cream_dir)))
print('total test oyster images:', len(os.listdir(test_oyster_dir)))
print('total test scallops images:', len(os.listdir(test_scallops_dir)))
print('total test waffle images:', len(os.listdir(test_waffle_dir)))

#-------------------------------------
#Build the network
#-------------------------------------
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

# the dimensions of the feature maps change with every successive layer
model.summary()

#-------------------------------------
#Configuring the model for training
#-------------------------------------
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

#-------------------------------------
#Using ImageDataGenerator to read images from directories
#-------------------------------------
from keras.preprocessing.image import ImageDataGenerator

# Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, #Target directory
        target_size=(150, 150), #Resizes all images to 150 × 150
        batch_size=20, 
        class_mode='binary') 

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# Fitting the model using a batch generator
history = model.fit_generator(
      train_generator,
      steps_per_epoch=len(train_generator)//20,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=len(validation_generator)//20)

# Saving the model
model.save('food_classification.h5')

# Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#-------------------------------------
#Setting up a data augmentation configuration via ImageDataGenerator
#-------------------------------------
datagen = ImageDataGenerator(
      # rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
      rotation_range=40, 
      # width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
      width_shift_range=0.2,
      height_shift_range=0.2,
      # shear_range is for randomly applying shearing transformations.
      shear_range=0.2,
      # zoom_range is for randomly zooming inside pictures.
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#-------------------------------------
#Displaying some randomly augmented training images
#-------------------------------------
from keras.preprocessing import image #Module with image- preprocessing utilities
fnames = [os.path.join(train_waffle_dir, fname) for
     fname in os.listdir(train_waffle_dir)]

img_path = fnames[3] # Chooses one image to augment

img = image.load_img(img_path, target_size=(150, 150))

#Converts it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

#Reshapes it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

#Generates batches of randomly transformed images. Loops indefinitely, so you need to break the loop at some point
i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc = model.evaluate(test_generator, steps=3, verbose=0)
print('test_acc:', test_acc)

#-------------------------------------
#Defining a new convnet that includes dropout
#-------------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

#Configuring the model for training
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

'''

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam

base = MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
base.trainable = True
model = models.Sequential()
model.add(base)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))
# opt = SGD(lr=0.001, momentum=0.9)
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

'''
#-------------------------------------
#Training the convnet using data-augmentation generators
#-------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=len(train_generator)//32,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=len(validation_generator)//32)

# Saving the model
model.save('food_classification_2.h5')

# Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy with Dropout layer')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss with Dropout layer')
plt.legend()
plt.show()

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc = model.evaluate(test_generator, steps=3, verbose=0)
print('test_acc with Dropout layer:', test_acc)

pred=model.predict(test_generator)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filename=test_generator.filenames


calorie = []
for i in predictions:
    if i == 'cheese cake':
        calorie.append('321 cal')
    if i == 'chicken wings':
        calorie.append('203 cal')
    if i == 'donuts':
        calorie.append('452 cal')
    if i == 'french fries':
        calorie.append('164 cal')
    if i == 'fried rice':
        calorie.append('312 cal')
    if i == 'gyoza':
        calorie.append('146 cal')
    if i == 'ice-cream':
        calorie.append('201 cal')
    if i == 'oyster':
        calorie.append('199 cal')
    if i == 'scallops':
        calorie.append('111 cal')
    if i == 'waffle':
        calorie.append('291 cal')
        
results=pd.DataFrame({"Filename":filename,
                  "Predictions":predictions,
                  "Calories per 100 g": calorie})   


results

a = results.sample(frac=1).reset_index(drop=True)
a.to_csv('final_results.csv',index=False)

