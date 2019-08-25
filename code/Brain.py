"""
Created by nikunjlad on 2019-08-24

"""

import numpy as np
import os, sys, datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate, Input
from keras.models import Sequential, Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from DataGenerator import *
from keras.utils.vis_utils import plot_model


class Brain(DataGenerator):

    def __init__(self, debug):
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")
        super().__init__(debug)
        self.debug = debug

    def main(self):
        Img_Size = [512, 512]

        paths, binaries = self.get_data()
        train_data = np.load(binaries["train_data"])
        valid_data = np.load(binaries["valid_data"])
        test_data = np.load(binaries["test_data"])
        train_labels = np.load(binaries["train_labels"])
        valid_labels = np.load(binaries["valid_labels"])
        test_labels = np.load(binaries["test_labels"])

        print(valid_data.shape[1:])

        proc = Processing()
        colormap = "BGR2GRAY"
        # reshape data in order to make it convolution ready
        input_shape, train_data, __, _ = proc.data_reshape(train_data, train_labels, colormap, 1)
        __, valid_data, __, ___ = proc.data_reshape(valid_data, valid_labels, colormap, 1)
        ___, test_data, classes, nClasses = proc.data_reshape(test_data, test_labels, colormap, 1)

        # scale images
        train_data = proc.scale_images(train_data, 255)
        valid_data = proc.scale_images(valid_data, 255)
        test_data = proc.scale_images(test_data, 255)

        train_labels = proc.do_one_hot_encoding(train_labels)
        print(train_labels)
        valid_labels = proc.do_one_hot_encoding(valid_labels)
        test_labels = proc.do_one_hot_encoding(test_labels)

        # cnn = CNN(train_matrix, train_labels, test_matrix, test_labels)

        print("Training data: ", train_data.shape)
        print("Validation data: ", valid_data.shape)
        print("Testing data: ", test_data.shape)
        print("Training labels: ", train_labels.shape)
        print("Validation labels: ", valid_labels.shape)
        print("Testing labels: ", test_labels.shape)

        # model = Sequential()
        # model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        # model.add(Conv2D(8, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(16, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(3, activation='softmax'))

        # model.compile(loss=categorical_crossentropy,
        #               optimizer=Adadelta(),
        #               metrics=['accuracy'])

        shape_x = 512
        shape_y = 512
        input_img = Input(shape=(shape_x, shape_y, 1))

        ### 1st layer
        layer_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
        layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_1)

        layer_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
        layer_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(layer_2)

        layer_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
        layer_3 = Conv2D(10, (1, 1), padding='same', activation='relu')(layer_3)

        mid_1 = concatenate([layer_1, layer_2, layer_3], axis=3)

        flat_1 = Flatten()(mid_1)

        dense_1 = Dense(1200, activation='relu')(flat_1)
        dense_2 = Dense(600, activation='relu')(dense_1)
        dense_3 = Dense(150, activation='relu')(dense_2)
        output = Dense(nClasses, activation='softmax')(dense_3)

        model = Model([input_img], output)

        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        aug.fit(train_data)
        batch_size = 16
        history = model.fit_generator(aug.flow(train_data, train_labels, batch_size=batch_size),
                                      steps_per_epoch=train_data.shape[0], epochs=20,
                                      validation_data=(valid_data, valid_labels))
        print("History:", history)

        scores = model.evaluate(x=valid_data, y=valid_labels)
        print("Scores:", scores)
        sys.exit(1)


if __name__ == "__main__":
    b = Brain(debug=False)
    b.main()
