#https://keras.io/examples/mnist_cnn/

from __future__ import print_function
import keras
from keras.datasets import mnist
import os
import numpy as np
from library import attack_evaluation as atk
from library import differential_evolution as de

batch_size = 128
num_classes = 10
epochs = 12

perturbation1 = np.array([15, 10, 45])
#perturbation2 = np.array([0, 0, 0])
indexOfImage = 92

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model.h5'

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model(save_dir + '\\' + model_name)

test_image = np.expand_dims((np.expand_dims(x_test[indexOfImage]
                                            , axis=0)), axis=3)
#print(np.argmax(model.predict(test_image)))

atk.print_attack_result(model, x_test[indexOfImage]
                        , y_test[indexOfImage]
                        , perturbation1
                        #, perturbation2
                        , dataset_name='MNIST')

print("Done!")
