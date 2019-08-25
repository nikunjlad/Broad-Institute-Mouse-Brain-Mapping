"""
Created by nikunjlad on 2019-08-20

"""
import os, shutil, sys, datetime
import numpy as np
from DataImport import *
from Processing import *
from sklearn.preprocessing import LabelEncoder


class DataGenerator:

    def __init__(self, debug):
        self.debug = debug

    def get_directory_paths(self):

        paths = dict()  # defining a dictionary to hold all the paths
        current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")

        # getting the code path, which is the current directory
        code_path = os.path.dirname(os.path.abspath(__file__))
        paths["code_path"] = code_path  # getting the code_path

        # we move one directory up to get the root directory, essentially our repository directory
        os.chdir("..")  # changing one directory up
        paths["root_path"] = os.getcwd()  # getting the root_path

        # checking if the data_path exists or not
        if os.path.exists("data"):
            print("Data Directory exists!")

            # acquire the data path and store it in the dictionary
            paths["data_path"] = os.path.sep.join([os.getcwd(), "data/stitched"])
        else:
            print("Data Directory does not exist!")
            sys.exit(1)  # exit the program since no data directory exists

        # check if the configuration directory exist for the code to run
        if os.path.exists("configurations"):
            print("Configurations exist!")
        else:
            print("Configurations don't exist!")
            os.mkdir(os.path.sep.join([os.getcwd(), "configurations"]))
        paths["config_path"] = os.path.sep.join([os.getcwd(), "configurations"])

        # check if the temporary directory exists for the runtime files to be stored
        if os.path.exists("temp"):
            print("Temporary directory exists!")
        else:
            print("Temporary Directory don't exist!")
            os.mkdir(os.path.sep.join([os.getcwd(), "temp"]))
        paths["temp_path"] = os.path.sep.join([os.getcwd(), "temp"])

        # create a run_dir for the current file states and changes
        os.chdir(paths["temp_path"])
        run_dir = "temp__" + current_time

        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        paths["run_path"] = os.path.sep.join([os.getcwd(), run_dir])
        os.chdir(paths["root_path"])

        # if binaries directory does not exist, then create one.
        if os.path.exists("binaries"):
            print("Binaries directory exists!")
        else:
            print("Binaries Directory don't exist!")
            os.mkdir(os.path.sep.join([os.getcwd(), "binaries"]))
        paths["binary_path"] = os.path.sep.join([os.getcwd(), "binaries"])

        # if output directory does not exist, then create one.
        if os.path.exists("output"):
            print("Output directory exists!")
            os.chdir("output")

            # check if models path exists
            if not os.path.exists(os.path.sep.join([os.getcwd(), "models"])):
                os.mkdir(os.path.sep.join([os.getcwd(), "models"]))

            # check if plots path exists
            if not os.path.exists(os.path.sep.join([os.getcwd(), "plots"])):
                os.mkdir(os.path.sep.join([os.getcwd(), "plots"]))

            os.chdir(paths["root_path"])
        else:
            print("Output Directory don't exist!")
            os.mkdir(os.path.sep.join([os.getcwd(), "output"]))
            os.mkdir(os.path.sep.join([os.getcwd(), "output/models"]))
            os.mkdir(os.path.sep.join([os.getcwd(), "output/plots"]))
        paths["output_path"] = os.path.sep.join([os.getcwd(), "output"])
        paths["models_path"] = os.path.sep.join([os.getcwd(), "output/models"])
        paths["plots_path"] = os.path.sep.join([os.getcwd(), "output/plots"])

        # if logs directory does not exist, then create one.
        if os.path.exists("logs"):
            print("Logs directory exists!")
        else:
            print("Logs Directory don't exist!")
            os.mkdir(os.path.sep.join([os.getcwd(), "logs"]))
        paths["logs_path"] = os.path.sep.join([os.getcwd(), "logs"])

        # if in DEBUG mode, print this out to the console
        if self.debug:
            print("Code path: ", paths["code_path"])
            print("Temp path: ", paths["temp_path"])
            print("Data path: ", paths["data_path"])
            print("Configuration path: ", paths["config_path"])
            print("Binaries path: ", paths["binary_path"])
            print("Output path: ", paths["output_path"])
            print("Models path: ", paths["models_path"])
            print("Plots path: ", paths["plots_path"])
            print("Logs path: ", paths["logs_path"])
            print("Root path: ", paths["root_path"])

        return paths

    def get_data(self):

        # get the directory paths for use throughout the code
        paths = self.get_directory_paths()
        binaries = dict()

        # get the classes
        categories = [folder for folder in os.listdir(paths["data_path"]) if not folder.startswith('.')]

        if self.debug:
            print("Categories :", categories)

        Img_Size = [512, 512]

        di = DataImport()
        split_data = True

        try:
            if split_data:
                # takes folder of images and splits them into train, test, valid and returns those paths
                data = di.create_train_test_valid(categories, paths)

                # pass the training, validation, test dir paths and get the train, test, validation matrices
                colormap = 'BGR2GRAY'
                train_data, train_labels = di.create_data_matrices(data["train_labels"], data["train_data"], colormap)
                valid_data, valid_labels = di.create_data_matrices(data["valid_labels"], data["valid_data"], colormap)
                test_data, test_labels = di.create_data_matrices(data["test_labels"], data["test_data"], colormap)

                # resize images to a predefined size
                proc = Processing()
                train_matrix = proc.resize_images(Img_Size[0], Img_Size[1], train_data, colormap, paths["run_path"])
                valid_matrix = proc.resize_images(Img_Size[0], Img_Size[1], valid_data, colormap, paths["run_path"])
                test_matrix = proc.resize_images(Img_Size[0], Img_Size[1], test_data, colormap, paths["run_path"])

                # process the data to right format
                train_data = proc.create_numpy_data(train_matrix)
                valid_data = proc.create_numpy_data(valid_matrix)
                test_data = proc.create_numpy_data(test_matrix)

                print("Writing data binaries to disk...")
                binaries["train_data"] = os.path.sep.join([paths["binary_path"], "train_data.npy"])
                binaries["train_labels"] = os.path.sep.join([paths["binary_path"], "train_labels.npy"])
                binaries["valid_data"] = os.path.sep.join([paths["binary_path"], "valid_data.npy"])
                binaries["valid_labels"] = os.path.sep.join([paths["binary_path"], "valid_labels.npy"])
                binaries["test_data"] = os.path.sep.join([paths["binary_path"], "test_data.npy"])
                binaries["test_labels"] = os.path.sep.join([paths["binary_path"], "test_labels.npy"])

                np.save(binaries["train_data"], train_data)
                np.save(binaries["valid_data"], valid_data)
                np.save(binaries["test_data"], test_data)
                np.save(binaries["train_labels"], train_labels)
                np.save(binaries["valid_labels"], valid_labels)
                np.save(binaries["test_labels"], test_labels)
                print("Binaries writing successfully to disk!")

            elif os.path.exists(paths["binary_path"]) and os.listdir(paths["binary_path"]) != []:

                binaries["train_data"] = os.path.sep.join([paths["binary_path"], "train_data.npy"])
                binaries["train_labels"] = os.path.sep.join([paths["binary_path"], "train_labels.npy"])
                binaries["valid_data"] = os.path.sep.join([paths["binary_path"], "valid_data.npy"])
                binaries["valid_labels"] = os.path.sep.join([paths["binary_path"], "valid_labels.npy"])
                binaries["test_data"] = os.path.sep.join([paths["binary_path"], "test_data.npy"])
                binaries["test_labels"] = os.path.sep.join([paths["binary_path"], "test_labels.npy"])
        except Exception as e:
            print(e)
            print("No binaries present!")
            sys.exit(1)

        return paths, binaries

        # # reshape data in order to make it convolution ready
        # input_shape, train_data, __, _ = proc.data_reshape(train_data, train_labels, colormap, 1)
        # __, valid_data, __, ___ = proc.data_reshape(valid_data, valid_labels, colormap, 1)
        # ___, test_data, classes, nClasses = proc.data_reshape(test_data, test_labels, colormap, 1)
        #
        # # scale images
        # train_data = proc.scale_images(train_data, 255)
        # valid_data = proc.scale_images(valid_data, 255)
        # test_data = proc.scale_images(test_data, 255)
        #
        # train_labels = proc.do_one_hot_encoding(train_labels)
        # valid_labels = proc.do_one_hot_encoding(valid_labels)
        # test_labels = proc.do_one_hot_encoding(test_labels)
        #
        # # cnn = CNN(train_matrix, train_labels, test_matrix, test_labels)
        # print(train_data.shape)
        # print(valid_data.shape)
        # print(test_data.shape)
        # print(train_labels.shape)
        # print(valid_labels.shape)
        # print(test_labels.shape)
        #
        # # np.save(os.path.sep.join([self.model_path, "train_data.npy"]), train_data)
        # # np.save(os.path.sep.join([self.model_path, "valid_data.npy"]), valid_data)
        # # np.save(os.path.sep.join([self.model_path, "test_data.npy"]), test_data)
        # # np.save(os.path.sep.join([self.model_path, "train_labels.npy"]), train_labels)
        # # np.save(os.path.sep.join([self.model_path, "valid_labels.npy"]), valid_labels)
        # # np.save(os.path.sep.join([self.model_path, "test_labels.npy"]), test_labels)
        #
        # plt.imshow(train_matrix[0])
        # cv2.imwrite(self.temp_path + "/sample.png", train_matrix[0])
        # plt.show()
        # sys.exit(1)


if __name__ == '__main__':
    b = DataGenerator(debug=True)
    paths, binaries = b.get_data()
