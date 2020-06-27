import os
import keras
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.metrics import classification_report
import numpy as np

#use cpu only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_classes = 10
multi_class_evaluation = True

save_dir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
model_name = 'improved_allconv_cifar10_trained_model.h5'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


if multi_class_evaluation:
    Y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    y_pred = model.predict_classes(x_test)
    print(classification_report(Y_test, y_pred))
else:
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

#print(model.metrics_names)
#print(model.summary())
