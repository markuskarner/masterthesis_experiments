import tensorflow as tf
import keras
import os
import time
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model
from library import CustomStopper
from sklearn.model_selection import train_test_split

batch_size = 256
num_classes = 10
epochs = 100
validation_split = 0.1
input_shape = (32, 32, 3)
multi_gpus = False
add_dropout = True
less_neurons_fc_layers = True   #only if dropout true
dropout_percent = 0.6
data_augmentation = False
save_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
save_dir = os.path.join(save_dir, 'saved_models')
model_name = 'VGG16_CIFAR10_trained_model.h5'

# initialize optimizer & early stopping
opt = keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2 / epochs)
#callback = CustomStopper.CustomStopper(monitor='val_loss', patience=2, verbose=1, start_epoch=10)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# define model
def _create_model():
    model = VGG16(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=input_shape
    )

    if add_dropout:
        # dropout https://stackoverflow.com/questions/42475381/add-dropout-layers-between-pretrained-dense-layers-in-keras
        # reduce neurons if parameter is set https://github.com/keras-team/keras/issues/4465
        # Store the fully connected layers, reduce neurons if parameter is set

        if less_neurons_fc_layers:
            x = Flatten(name='flatten')(model.layers[-5].output)
            x = Dense(2048, activation='relu', name='fc1')(x)
            x = Dropout(dropout_percent, name='dropout_1')(x)
            x = Dense(2048, activation='relu', name='fc2')(x)
            x = Dropout(dropout_percent, name='dropout_2')(x)
            predictors = Dense(10, activation='softmax', name='predictions')(x)
        else:   # only add dropout to exiting layers
            # get existing layers
            fc1 = model.layers[-3]
            fc2 = model.layers[-2]
            predictions = model.layers[-1]
            # Create the dropout layers
            dropout1 = Dropout(dropout_percent)
            dropout2 = Dropout(dropout_percent)
            # Reconnect the layers
            x = dropout1(fc1.output)
            x = fc2(x)
            x = dropout2(x)
            predictors = predictions(x)

        # Create a new model
        updated_model = Model(input=model.input, output=predictors)

        return updated_model
    else:
        return model


if multi_gpus:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = _create_model()

        # Compile the model
        model.compile(loss=keras.losses.categorical_crossentropy
                      , optimizer=opt
                      , metrics=["accuracy"])

else:
    model = _create_model()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy
                  , optimizer=opt
                  , metrics=["accuracy"])


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(model.summary())

#exit() # uncomment if you want to print the model summary only before the start of the training

start_time = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        shuffle=True,
                        callbacks=[callback],
                        verbose=1)
else:
    print('Using real-time data augmentation.')

    # Split train data into train and validation data, as model_fit_generator does not support
    # the validation_split parameter
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train
                                     , y_train
                                     , batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_valid, y_valid),
                                    workers=4,
                                    shuffle=True,
                                    callbacks=[callback],
                                    verbose=1
    )

print(time.time()-start_time)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join('model_history', 'VGG16_CIFAR10_model_accuracy.png'))
plt.clf() # clear plot

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join('model_history', 'VGG16_CIFAR10_model_loss.png'))