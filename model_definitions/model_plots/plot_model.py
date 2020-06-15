import tensorflow as tf
import os

#path to saved models
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'tests')
save_dir = os.path.join(save_dir, 'saved_models')

#model name to plot & file name
model_name = 'keras_cifar10_trained_model.h5'
file_name = 'keras_CIFAR10.png'

model = tf.keras.models.load_model(save_dir + '\\' + model_name)

tf.keras.utils.plot_model(model
                          , to_file=file_name
                          , show_shapes=True
                          , show_layer_names=True)
