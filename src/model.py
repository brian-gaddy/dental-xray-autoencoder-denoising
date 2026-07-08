"""Convolutional denoising autoencoder for panoramic dental X-rays."""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose

class DenoiseAutoencoder(Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            Input(shape=input_shape),
            Conv2D(64, 3, activation="relu", padding="same", strides=2),
            Conv2D(32, 3, activation="relu", padding="same", strides=2),
        ], name="encoder")
        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2),
            Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2),
            Conv2D(3, 3, activation="sigmoid", padding="same"),
        ], name="decoder")

    def call(self, inputs, training=False):
        return self.decoder(self.encoder(inputs, training=training), training=training)
