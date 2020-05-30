import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
import os
import time
import matplotlib.pyplot as plt

batch_size = 256
num_classes = 10
epochs = 150
num_predictions = 20
input_shape = (32, 32, 3)
multi_gpus = False
save_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
save_dir = os.path.join(save_dir, 'saved_models')
model_name = 'VGG16_CIFAR10_trained_model.h5'

# initialize optimizer & early stopping
opt = keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2 / epochs)
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)

if multi_gpus:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = VGG16(
            weights=None,
            include_top=True,
            classes=num_classes,
            input_shape=input_shape
        )

        # Compile the model
        model.compile(loss=keras.losses.categorical_crossentropy
                      , optimizer=opt
                      , metrics=["accuracy"])

else:
    model = VGG16(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=input_shape
    )
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

start_time = time.time()

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    shuffle=True,
                    callbacks=[callback],
                    verbose=1)

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
plt.savefig('VGG16_CIFAR10_model_accuracy.png')
plt.clf() # clear plot

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('VGG16_CIFAR10_model_loss.png')