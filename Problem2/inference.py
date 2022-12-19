import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
import numpy as np
import os



def model_init(model_name="vgg16"):
    if model_name=="vgg16":
        base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights=None)
    else:
        base_model = VGG19(input_shape = (224, 224, 3), include_top = False, weights=None)
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)
    


    return model



def infer():
    class_name = ['berry', 'bird', 'dog', 'flower']
    model = model_init()
    model2 = model_init("vgg19")

    model.load_weights(os.getcwd() + 'weight/weight.hdf5')
    model2.load_weights(os.getcwd() + 'weight/weights2.hdf5')

    # model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001))
    # model2.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001))

    # model.compile(..., run_eagerly=True)
    # model2.compile(..., run_eagerly=True)

    # model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
    # model2.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
    
    folder_path = '/image'

    image_path = os.listdir(folder_path)

    # print("image_path: ", image_path)

    images_path = [os.path.join(folder_path, filepath) for filepath in image_path]

    # print("images_path")

    images = []

    for path in images_path:
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224))

        # image_expand = np.expand_dims(image, axis=0)
        images.append(image)

    images = np.array(images)
    print(images.shape)

    #######################################
    # VGG 16 prediciton
    #######################################
    prediction = model.predict(images)
    # print("prediction: ", prediction)
    score = tf.nn.softmax(prediction)
    # print("score: ", score)
    weighted_score = score*0.60
    # class_id = np.argmax(score)
    class_ids = np.argmax(score, axis=1)
    # print("class_id", class_ids)

    #########################################
    # VGG 19
    #########################################
    prediction2 = model2.predict(images)
    # print("prediction: ", prediction)
    score2 = tf.nn.softmax(prediction2)
    # print("score2: ", score2)
    weighted_score2 = score2*0.40
    # class_id = np.argmax(score)
    class_ids2 = np.argmax(score2, axis=1)
    # print("class_id2", class_ids2)

    score_final = np.add(np.array(weighted_score), np.array(weighted_score2) )
    score_final = score_final/2
    # print("score final", score_final)

    class_id_final = np.argmax(score_final, axis=1)
    # print("class id final", class_id_final)

    for i in range(len(image_path)):
        print(image_path[i], ": ", class_name[class_id_final[i]])





if __name__=="__main__":
    # image_path = os.getcwd() + '/image'
    # weight_path = os.getcwd() + '/weight/weight.hdf5'

    infer()
