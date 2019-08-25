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

    # def get_directory_paths(self, file_name):
    #
    #     # getting the code path, which is the current directory
    #     code_path = os.path.dirname(os.path.abspath(file_name))
    #     os.chdir("..")   # changing one directory up
    #     root_path = os.getcwd()
    #     temp_path = os.getcwd() + "/temp"   # creating the temp path for storing temporary runtime files
    #
    #     # if a temporary directory exists, then remove it
    #     if os.path.exists(temp_path):
    #         print("It exists")
    #         shutil.rmtree(temp_path)
    #     os.mkdir(temp_path)   # create a new temporary directory for the current run
    #     os.chdir("data/stitched")   # change to the data directory where all the class folders lie
    #     orig_data_path = os.getcwd()  # get the data directory path
    #     os.chdir("../")
    #     print(os.getcwd())
    #     data_path = os.getcwd()
    #     os.chdir(root_path)
    #
    #     # if in DEBUG mode, print this out to the console
    #     if self.debug:
    #         print("Code path: ", code_path)
    #         print("Data path: ", orig_data_path)
    #         print("Temp path:", temp_path)
    #
    #     return code_path, orig_data_path, temp_path, data_path, root_path
    current_time = datetime.datetime.now().strftime("%Y.%m.%d__%H-%M-%S")
    code_path = "/Users/nikunjlad/RA/broad/Broad-Institute-Mouse-Brain-Mapping/code/"
    orig_data_path = "/Users/nikunjlad/RA/broad/Broad-Institute-Mouse-Brain-Mapping/data/stitched/"
    data_path = "/Users/nikunjlad/RA/broad/Broad-Institute-Mouse-Brain-Mapping/data/"
    temp_path = "/Users/nikunjlad/RA/broad/Broad-Institute-Mouse-Brain-Mapping/temp_" + current_time
    model_path = "/Users/nikunjlad/RA/broad/Broad-Institute-Mouse-Brain-Mapping/models"

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    def main(self):

        # get the directory paths for use throughout the code
        # code_path, orig_data_path, temp_path, data_path, root_path = self.get_directory_paths(__file__)

        # get the classes
        categories = [folder for folder in os.listdir(self.orig_data_path) if not folder.startswith('.')]

        if self.debug:
            print("Categories :", categories)

        Img_Size = [512, 512]

        di = DataImport()
        train_test_split = False

        if not train_test_split:
            # takes folder of images and splits them into train, test, valid and returns those paths
            train_path, valid_path, test_path = di.create_train_test_valid(categories, self.orig_data_path,
                                                                           self.data_path)
        else:
            train_path = self.data_path + "/training"
            valid_path = self.data_path + "/validation"
            test_path = self.data_path + "/testing"

        # pass the training, validation, test dir paths and get the train, test, validation matrices
        colormap = 'BGR2GRAY'
        train_data, train_labels, tr_labels = di.create_data_matrices(categories, train_path, colormap)
        valid_data, valid_labels, vd_labels = di.create_data_matrices(categories, valid_path, colormap)
        test_data, test_labels, tst_labels = di.create_data_matrices(categories, test_path, colormap)

        # resize images to a predefined size
        proc = Processing()
        train_matrix = proc.resize_images(Img_Size[0], Img_Size[1], train_data, colormap, self.temp_path)
        valid_matrix = proc.resize_images(Img_Size[0], Img_Size[1], valid_data, colormap, self.temp_path)
        test_matrix = proc.resize_images(Img_Size[0], Img_Size[1], test_data, colormap, self.temp_path)

        # process the data to right format
        train_data = proc.create_numpy_data(train_matrix)
        valid_data = proc.create_numpy_data(valid_matrix)
        test_data = proc.create_numpy_data(test_matrix)

        np.save(os.path.sep.join([self.model_path, "train_data.npy"]), train_data)
        np.save(os.path.sep.join([self.model_path, "valid_data.npy"]), valid_data)
        np.save(os.path.sep.join([self.model_path, "test_data.npy"]), test_data)
        np.save(os.path.sep.join([self.model_path, "train_labels.npy"]), train_labels)
        np.save(os.path.sep.join([self.model_path, "valid_labels.npy"]), valid_labels)
        np.save(os.path.sep.join([self.model_path, "test_labels.npy"]), test_labels)

        # reshape data in order to make it convolution ready
        input_shape, train_data, __, _ = proc.data_reshape(train_data, train_labels, colormap, 1)
        __, valid_data, __, ___ = proc.data_reshape(valid_data, valid_labels, colormap, 1)
        ___, test_data, classes, nClasses = proc.data_reshape(test_data, test_labels, colormap, 1)

        # scale images
        train_data = proc.scale_images(train_data, 255)
        valid_data = proc.scale_images(valid_data, 255)
        test_data = proc.scale_images(test_data, 255)

        train_labels = proc.do_one_hot_encoding(train_labels)
        valid_labels = proc.do_one_hot_encoding(valid_labels)
        test_labels = proc.do_one_hot_encoding(test_labels)

        # cnn = CNN(train_matrix, train_labels, test_matrix, test_labels)
        print(train_data.shape)
        print(valid_data.shape)
        print(test_data.shape)
        print(train_labels.shape)
        print(valid_labels.shape)
        print(test_labels.shape)

        # np.save(os.path.sep.join([self.model_path, "train_data.npy"]), train_data)
        # np.save(os.path.sep.join([self.model_path, "valid_data.npy"]), valid_data)
        # np.save(os.path.sep.join([self.model_path, "test_data.npy"]), test_data)
        # np.save(os.path.sep.join([self.model_path, "train_labels.npy"]), train_labels)
        # np.save(os.path.sep.join([self.model_path, "valid_labels.npy"]), valid_labels)
        # np.save(os.path.sep.join([self.model_path, "test_labels.npy"]), test_labels)

        plt.imshow(train_matrix[0])
        cv2.imwrite(self.temp_path + "/sample.png", train_matrix[0])
        plt.show()
        sys.exit(1)


if __name__ == '__main__':
    b = DataGenerator(debug=True)
    b.main()
