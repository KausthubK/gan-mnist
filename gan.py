import sys
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam

class GAN(object):
    def __init__(self, discriminator,generator):
        # Initialize Variables
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.Generator = generator
        
        self.Discriminator = discriminator
        self.Discriminator.trainable = False

        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.gan_model.summary()

    def model(self):
        # Build the adversarial model and return it
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        # Prints the Model Summary to the Screen
        return gan_model.summary()

    def save_model(self):
        # Saves the model structure to a file in the data folder
        plot_model(self.gan_model.model, to_file='/data/GAN_Model.png')