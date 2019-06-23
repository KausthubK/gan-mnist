###################
#    imports
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Discriminator(object):
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        #initialize variables
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        self.Discriminator.summary()
    
    def model(self):
        # build binary classifier and return it
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        # Layer 1
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE)) #FC1
        model.add(LeakyReLU(alpha=0.2))                         #FC1 Activation Function
        # Layer 2
        model.add(Dense(int(self.CAPACITY/2)))                  #FC2
        model.add(LeakyReLU(alpha=0.2))                         #FC2 Activation Function
        # Output Layer
        model.add(Dense(1, activation='sigmoid'))               # Out - probability of class or not
               
        return model
    
    def summary(self):
        # prints the model summary to the screen
        return self.Discriminator.summary()

    def save_model(self):
        # saves the model structure to a file in the data folder
        plot_model(self.Discriminator.model, to_file='/data/Discriminator_Model.png')


