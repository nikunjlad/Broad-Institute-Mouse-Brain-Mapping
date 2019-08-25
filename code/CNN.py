"""
Created by nikunjlad on 2019-08-23

"""
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import  Sequential


class CNN:

    def __init__(self):
        pass

    @staticmethod
    def create_model(input_shape, final_activation, layer_activation, conv_layers, dense_layers, no_of_filters,
                     filter_size, pool_size, final_dense_neurons, kernel_initializer):
        model = Sequential()

        activations_dict = {"RELU": 'relu',
                            "LEAKY_RELU": 'tanh',
                            "TANH": 'tanh',
                            "SOFTMAX": 'softmax'}

        initializer_dict = {"ZEROS": "zeros",
                            "UNIFORM": "uniform",
                            "RAND_NORM": "random_normal",
                            "RAND_UNIF": "random_uniform"}

        for i in range(conv_layers):
            model.add(Conv2D(no_of_filters[i], (filter_size[i], filter_size[i]), input_shape=input_shape,
                                       activation=activations_dict[layer_activation]))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size[i], pool_size[i])))
            model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

        hidden_units = 64

        for i in range(dense_layers):
            model.add(tf.keras.layers.Dense(hidden_units, kernel_initializer=initializer_dict[kernel_initializer]))
            model.add(tf.keras.layers.BatchNormalization(axis=1))
            model.add(tf.keras.layers.Activation(activations_dict[final_activation]))

        model.add(tf.keras.layers.Dense(final_dense_neurons))
        model.add(tf.keras.layers.Activation(activations_dict[final_activation]))

        return model
