import sys
import numpy as np
from keras.layers import Dense, Reshape
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1, 
                 latent_size=100):
        # Initialize Variables
        self.W = width
        self.H = height
        self.C = channels
        
        #instance of optimizer 
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        
        #latent space definition
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0,1,self.LATENT_SPACE_SIZE)

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.Generator.summary()

    def model(self, block_starting_size=128, num_blocks=4):
        # Build the generator model and returns it
        model = Sequential()
        block_size = block_starting_size
        # Layer 1
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE,)))  # FC1
        model.add(LeakyReLU(alpha=0.2))                                     # FC1 Activation Fn
        model.add(BatchNormalization(momentum=0.8))                         # FC1 Norm

        for i in range(num_blocks-1):
            block_size = block_size*2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
        
        # restructures output to be the same shape as the input image
        model.add(Dense(self.W * self.H * self.C, activation='tanh'))
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        # Prints the Model Summary to the screen
        return self.Generator.summary()

    def save_model(self):
        # Saves the model structure to a file in the data folder
        plot_model(self.Generator.model, to_file='/data/Generator_Model.png')

