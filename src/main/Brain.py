"""
Created by nikunjlad on 2019-08-24

"""

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from .DataGenerator import *
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils.vis_utils import plot_model
from .ResNet import *


class Brain(DataGenerator):

    # we inherit the DataGenerator class into the main class for we will be using it's properties there
    def __init__(self, debug):
        """
        :param debug: debug variable just checks if we need to debug the application so that DEBUG messages can be
        printed.
        """
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")
        super().__init__(debug)   # calling the Data Generator class with debug parameter as well.
        self.debug = debug

    def main(self):
        """
        :return: just completes training a network on ResNet architecture.
        """
        # setting environment variable to 2
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # when the flag is debug
        if self.debug:
            print(tf.version.VERSION)

        Img_Size = [512, 512]  # make size of the image as 512 x 512

        # get the paths and binaries, binaries are in .npy format and are like reusable pickle files
        paths, binaries = self.get_data()   # with the current object access the Data Generator functions.

        # load the data from the binaries dictionary.
        train_data = np.load(binaries["train_data"])
        valid_data = np.load(binaries["valid_data"])
        test_data = np.load(binaries["test_data"])
        train_labels = np.load(binaries["train_labels"])
        valid_labels = np.load(binaries["valid_labels"])
        test_labels = np.load(binaries["test_labels"])

        if self.debug:
            print(valid_data.shape[1:])

        # once data is loaded resize it to 256x256 for network training. This is specific to this network since ResNet
        # is to be trained
        train_data = np.resize(train_data, (train_data.shape[0], 256, 256))
        valid_data = np.resize(valid_data, (valid_data.shape[0], 256, 256))
        test_data = np.resize(test_data, (test_data.shape[0], 256, 256))

        # change the datatype to float32
        train_data = train_data.astype('float32')
        valid_data = valid_data.astype('float32')
        test_data = test_data.astype('float32')

        # a wrangler object to do data wrangling, i.e to reshape data for training
        proc = Wrangler()
        colormap = "BGR2GRAY" # we would like to change the colormap of data from BGR to GRAY
        # reshape data in order to make it convolution ready
        input_shape, train_data, __, _ = proc.data_reshape(train_data, train_labels, colormap, 1)
        __, valid_data, __, ___ = proc.data_reshape(valid_data, valid_labels, colormap, 1)
        ___, test_data, classes, nClasses = proc.data_reshape(test_data, test_labels, colormap, 1)

        # scale images from 255 range to 0-1 for reducing pixel variation.
        train_data = proc.scale_images(train_data, 255)
        valid_data = proc.scale_images(valid_data, 255)
        test_data = proc.scale_images(test_data, 255)

        # do one hot encoding of the data labels so as to convert from categorical to binary format per class
        train_labels = proc.do_one_hot_encoding(train_labels)
        valid_labels = proc.do_one_hot_encoding(valid_labels)
        test_labels = proc.do_one_hot_encoding(test_labels)

        if self.debug:
            print("Training data: ", train_data.shape)
            print("Validation data: ", valid_data.shape)
            print("Testing data: ", test_data.shape)
            print("Training labels: ", train_labels.shape)
            print("Validation labels: ", valid_labels.shape)
            print("Testing labels: ", test_labels.shape)

        # define ResNet object
        rn = Resnet()
        model = rn.build_resnet_50(input_shape=input_shape, num_outputs=nClasses)
        batch_size = 64
        nb_epoch = 200
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        csv_logger = CSVLogger('resnet18_brain.csv')
        augmentation = True

        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if not augmentation:
            model.fit(train_data, train_labels,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(test_data, test_labels),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper, csv_logger])
        else:
            aug = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            aug.fit(train_data)

            history = model.fit_generator(aug.flow(train_data, train_labels, batch_size=batch_size),
                                          steps_per_epoch=train_data.shape[0] // batch_size, epochs=nb_epoch,
                                          validation_data=(valid_data, valid_labels),
                                          callbacks=[lr_reducer, early_stopper, csv_logger])
            print("History:", history)

        scores = model.evaluate(x=valid_data, y=valid_labels)
        print("Scores:", scores)
        sys.exit(1)


if __name__ == "__main__":
    """ """
    b = Brain(debug=False)
    b.main()
