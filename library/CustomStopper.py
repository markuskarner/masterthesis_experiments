import keras

# https://stackoverflow.com/questions/46287403/is-there-a-way-to-implement-early-stopping-in-keras-only-after-the-first-say-1


class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self
                 , monitor='val_loss'
                 , min_delta=0
                 , patience=0
                 , verbose=0
                 , mode='auto'
                 , start_epoch=10):
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
