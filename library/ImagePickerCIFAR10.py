import keras
import keras.models
import os
import numpy as np
from keras.datasets import cifar10


def get_cifar10_images(model
                       , num_images: int = 500
                       , num_classes: int = 10):

    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_test = x_test.astype('float32')
    x_test /= 255

    picked_image_indices = []
    picked_classes_counter = np.zeros(num_classes)

    while picked_image_indices.__len__() < num_images:
        i = np.random.randint(x_test.shape[0])
        prediction = model.predict(np.expand_dims(x_test[i],axis=0))
        prediction = np.argmax(prediction)
        true_label = np.argmax(y_test[i])

        if prediction == true_label and picked_classes_counter[true_label] < num_images/num_classes:
            picked_image_indices.append(i)
            picked_classes_counter[true_label] += 1

    print('%d images picked' % picked_image_indices.__len__())
    # for j in picked_image_indices:
    #    print('Index: %5d, Class: %2d' % (j, np.argmax(y_test[j])))

    return np.array(picked_image_indices)
