import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle


BATCH_SIZE = 64
TARGET_SIZE = (224, 224)

def data_loading(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_datagen = ImageDataGenerator(
        rescale = 1./255., 
        rotation_range = 40, 
        width_shift_range = 0.2, 
        height_shift_range = 0.2, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True)

    test_datagen = ImageDataGenerator( rescale = 1.0/255. )

    train_generator = train_datagen.flow_from_directory(
        train_dir, 
        batch_size = BATCH_SIZE, 
        class_mode = 'categorical', 
        target_size = TARGET_SIZE,
        subset = 'training')

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        target_size = TARGET_SIZE)

    return train_generator, validation_generator


def model_building(base_dir, epochs, steps_per_epoch):
    train_generator, validation_generator = data_loading(base_dir)
    base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])

    filepath="weights-improvement-{epoch:02d}-{acc:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=callbacks_list, verbose=1)

    with open('/trainHistoryDictNew', 'wb') as file_pi:
        pickle.dump(vgghist.history, file_pi)


if __name__ == "__main__":
    base_dir = "/dataset"
    epochs = 10
    steps_per_epoch = 100
    
    model_building(base_dir, epochs, steps_per_epoch)