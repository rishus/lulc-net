#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:32:47 2017

@author: rishu
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
# dimensions of our images.
img_width, img_height = 204, 153

train_data_dir = 'rs_data/train/'  #'cats_dogs_data_small/train/'
validation_data_dir = 'rs_data/validation'  #cats_dogs_data_small/validation/'
test_data_dir = 'rs_data/test'
nb_train_samples = 100   #200  #2000
nb_validation_samples = 50  #100  # 800
epochs = 3  #50
batch_size = 1   #8  #

K.set_image_data_format('channels_first')
if K.image_data_format() == 'channels_first':
    #input_shape = (3, img_width, img_height)
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
convout_l1_act = Activation('relu') 
model.add(convout_l1_act)
convout_l1_mp = MaxPooling2D()  #(pool_size=(2, 2))
model.add(convout_l1_mp)


model.add(Conv2D(32, (3, 3)))
convout_l2_act = Activation('relu')
model.add(convout_l2_act)
convout_l2_mp = MaxPooling2D()
model.add(convout_l2_mp)  #MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
convout_l3_act = Activation('relu')
model.add(convout_l3_act)
convout_l3_mp = MaxPooling2D()  #pool_size=(2,2)
model.add(convout_l3_mp)

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

# this is the augmentation configuration we will use for trainin  
#width_shift_range=0.2, height_shift_range=0.2,g
# train_datagen = ImageDataGenerator(rescale=1./255, \
#                                    shear_range=0.2,zoom_range=0.2, horizontal_flip=True, fill_mode='Nearest')

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255, \
                                   shear_range=0.2,zoom_range=0.2, horizontal_flip=False)


# this is the augmentation configuration we will use for validation and testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


# train_generator = train_datagen.flow_from_directory(train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size, shuffle=True, class_mode='binary')
train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, shuffle=False, class_mode='binary')
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
figname = 'inputs.png'
for img_num in range(8):
	img_to_visualize = np.array(train_generator[img_num][0])
	for_plot = np.zeros((img_width, img_height))
	ax = fig.add_subplot(5, 5, img_num+1)
	for col in range(img_width):
	    for row in range(img_height):
		for_plot[col, row] = sum(img_to_visualize[0, :, col, row])
	ax.imshow(for_plot, cmap='gray')
	ax.set_title(str(img_num))
	plt.axis('off')
fig.savefig(figname, format='png')


train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, shuffle=False, class_mode='binary')

img_to_visualize = np.array(train_generator[7][0])

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

model.fit_generator(train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs, validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


#img_to_visualize = np.expand_dims(img_to_visualize, axis=0)

# model.save_weights( 'first_try.h5')  # ALWAYS SAVE weights after training!
# #model.load_weights('first_try.h5')

test_generator = test_datagen.flow_from_directory(
         test_data_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode=None,  # only data, no labels
         shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(test_generator, 500)
#print probabilities

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report

y_true = np.array([0] * 70 + [1] * 70)
y_pred = probabilities > 0.5
print(classification_report(y_true, y_pred))
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print tn, fp
print fn, tp

# #score = model.evaluate()

# retrive hidden layer information:
#(1) outputs:
inp = model.input  
outputs = [layer.output for layer in model.layers]   # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs] #evaluation functions
print functors[0]
# Testing
#test = np.random.random(input_shape)[np.newaxis,...]
#layer_outs = [func([test]) for func in functors]
#layer_outs = functor([test, 1.])
#print layer_outs

#(2) filters:




