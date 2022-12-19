import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import numpy as np
import os



def model_init():
    base_model = VGG16(input_shape = (224, 224, 3), include_top = False)
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)

    return model


def infer(weight_path, folder_path):
    class_name = ['berry', 'bird', 'dog', 'flower']
    model = model_init()

    model.load_weights(weight_path)

    image_path = os.listdir(folder_path)

    images_path = [os.path.join(folder_path, filepath) for filepath in image_path]

    images = []

    for path in images_path:
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))

        # image_expand = np.expand_dims(image, axis=0)
        images.append(image)

    images = np.array(images)
    print(images.shape)

    prediction = model.predict(images)
    # print("prediction: ", prediction)
    score = tf.nn.softmax(prediction)
    # print("score: ", score)
    # class_id = np.argmax(score)
    class_ids = np.argmax(score, axis=1)
    # print("class_id", class_ids)

    for i in range(len(image_path)):
        print(image_path[i], ": ", class_name[class_ids[i]])





if __name__=="__main__":
    image_path = './image'
    weight_path = './weight/weight.hdf5'

    infer(weight_path, image_path)
