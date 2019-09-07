"""
Created by nikunjlad on 2019-08-24

"""

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from main.DataGenerator import *
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils.vis_utils import plot_model
from main.ResNet import *
from main.Metrics import *
import logging
from configparser import ConfigParser


class Brain(DataGenerator):

    # we inherit the DataGenerator class into the main class for we will be using it's properties there
    def __init__(self, debug):
        """
        Constructor to initialize variables to be used class wide

        :param debug: debug variable just checks if we need to debug the application so that DEBUG messages can be
        printed.
        """
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")
        # getting the custom logger
        self.logger_name = "mouse_brain_" + self.current_time + "_.log"
        self.logger = self.get_loggers(self.logger_name)
        self.logger.info("Mouse Brain Image Logs Initiated!")
        self.logger.info("Current time: " + str(self.current_time))

        super().__init__(debug, self.logger)  # calling the Data Generator class with debug parameter as well.
        self.debug = debug

    @staticmethod
    def get_loggers(name):
        logger = logging.getLogger("brain")  # name the logger as squark
        logger.setLevel(logging.INFO)
        f_hand = logging.FileHandler(name)  # file where the custom logs needs to be handled
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        return logger

    def main(self):
        """
        Main function from where program execution starts
        :return: just completes training a network on ResNet architecture.
        """

        # setting environment variable to 2
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # when the flag is debug
        if self.debug:
            print(tf.version.VERSION)

        # get the paths and binaries, binaries are in .npy format and are like reusable pickle files
        paths = self.get_directory_paths()

        # removing any existing log files if present directory
        if os.path.exists(paths["code_path"] + "/" + self.logger_name):
            os.remove(paths["code_path"] + "/" + self.logger_name)

        # parsing the configurations file
        parser = ConfigParser()  # defining the parser
        parser.read(paths["config_path"] + "/config.ini")  # reading the config file
        df_obj = self.parse_json(paths["config_path"] + "/default_config.json")  # parse the json
        procs = self.parse_configurations(parser, df_obj, 'processing')  # get the meta data as well as the latency info
        resnets = self.parse_configurations(parser, df_obj, 'resnet')  # get the meta data of the resnet network

        # get the data binaries
        binaries = self.get_data(paths, procs)  # with the current object access the Data Generator functions.

        # load the data from the binaries dictionary.
        train_data = np.load(binaries["train_data"])
        valid_data = np.load(binaries["valid_data"])
        test_data = np.load(binaries["test_data"])
        train_labels = np.load(binaries["train_labels"])
        valid_labels = np.load(binaries["valid_labels"])
        test_labels = np.load(binaries["test_labels"])
        self.logger.info("Binary data files loaded successfully!")

        if self.debug:
            print(valid_data.shape[1:])

        # once data is loaded resize it to 256x256 for network training. This is specific to this network since ResNet
        # is to be trained
        train_data = np.resize(train_data, (train_data.shape[0], resnets["inputSize"][0], resnets["inputSize"][1]))
        valid_data = np.resize(valid_data, (valid_data.shape[0], resnets["inputSize"][0], resnets["inputSize"][1]))
        test_data = np.resize(test_data, (test_data.shape[0], resnets["inputSize"][0], resnets["inputSize"][1]))
        self.logger.info("Data resized as per ResNet architecture requirement")

        # change the datatype to float32
        train_data = train_data.astype(procs["dataType"])
        valid_data = valid_data.astype(procs["dataType"])
        test_data = test_data.astype(procs["dataType"])
        self.logger.info("Data type changed successfully!")

        # a wrangler object to do data wrangling, i.e to reshape data for training
        proc = Wrangler(self.logger)
        colormap = procs["colorMap"]  # we would like to change the colormap of data from BGR to GRAY
        # reshape data in order to make it convolution ready
        input_shape, train_data, __, _ = proc.data_reshape(train_data, train_labels, colormap, procs["imageDepth"])
        __, valid_data, __, ___ = proc.data_reshape(valid_data, valid_labels, colormap, procs["imageDepth"])
        ___, test_data, classes, nClasses = proc.data_reshape(test_data, test_labels, colormap, procs["imageDepth"])
        self.logger.info("Data reshaped for training successfully!")

        # scale images from 255 range to 0-1 for reducing pixel variation.
        train_data = proc.scale_images(train_data, procs["scaling"])
        valid_data = proc.scale_images(valid_data, procs["scaling"])
        test_data = proc.scale_images(test_data, procs["scaling"])
        self.logger.info("Data scaled successfully!")

        # do one hot encoding of the data labels so as to convert from categorical to binary format per class
        train_labels = proc.do_one_hot_encoding(train_labels)
        valid_labels = proc.do_one_hot_encoding(valid_labels)
        test_labels = proc.do_one_hot_encoding(test_labels)
        self.logger.info("One hot encoding of labels done successfully!")

        if self.debug:
            print("Training data: ", train_data.shape)
            print("Validation data: ", valid_data.shape)
            print("Testing data: ", test_data.shape)
            print("Training labels: ", train_labels.shape)
            print("Validation labels: ", valid_labels.shape)
            print("Testing labels: ", test_labels.shape)

        # define ResNet object
        rn = Resnet(self.logger)
        model = rn.build_resnet_50(input_shape=input_shape, num_outputs=nClasses)
        batch_size = resnets["batchSize"]
        nb_epoch = resnets["epochs"]
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(resnets["reduceFactor"]), cooldown=resnets["coolDown"],
                                       patience=resnets["patience"], min_lr=resnets["minRate"])
        early_stopper = EarlyStopping(min_delta=resnets["minDelta"], patience=resnets["patience"])
        csv_logger = CSVLogger(paths["logs_path"]+'/resnet18_brain_' + self.current_time + '_.csv')
        augmentation = resnets["augmentation"]

        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.compile(optimizer=resnets["optimizer"], loss=resnets["loss"], metrics=resnets["metrics"])

        if not augmentation:
            history = model.fit(train_data, train_labels,
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

        # evaluate the model
        scores = model.evaluate(x=valid_data, y=valid_labels)

        # predict on unknown data, in our case the test data
        predicted_values = model.predict(x=test_data, verbose=resnets["verbose"])

        # getting the performace evaluation of the models based on certain metrics
        metric = Metrics(self.logger, paths["plots_path"])

        # getting the prediction accuracy
        _, __, accuracy = metric.get_accuracy(predicted_values, test_labels)
        print("Accuracy: ", accuracy)

        # getting the MSE
        mse = metric.mean_squared_error(history, True)
        print("Mean Squared error: ", mse)

        # getting the train and validation accuracy
        train_acc, val_acc = metric.train_validation_accuracy(history, True)
        print("Training accuracy: ", train_acc)
        print("Validation accuracy: ", val_acc)

        # getting the train and validation loss
        train_loss, val_loss = metric.train_validation_loss(history, True)
        print("Training loss: ", train_loss)
        print("Validation loss: ", val_loss)

        # getting the confusion matrix
        metric.confusion(test_labels.argmax(axis=1), predicted_values.round().argmax(axis=1), True)

        if os.path.isfile(paths['code_path'] + "/" + self.logger_name):
            os.rename(paths['code_path'] + "/" + self.logger_name, paths['logs_path'] + "/" + self.logger_name)

        print("Scores:", scores)
        sys.exit(1)


if __name__ == "__main__":
    """ """
    b = Brain(debug=False)
    b.main()
