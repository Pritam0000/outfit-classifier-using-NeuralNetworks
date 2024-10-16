import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_autoencoder(input_shape):
    input_img = keras.Input(shape=input_shape)
    x = layers.Flatten()(input_img)
    
    # Encoder
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(decoded)
    
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder, encoder

def load_trained_autoencoder(model_path):
    return tf.keras.models.load_model(model_path)