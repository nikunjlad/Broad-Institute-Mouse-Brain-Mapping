"""
Created by nikunjlad on 2019-08-20

"""
from random import shuffle
import os, shutil, cv2
from Processing import *

class DataImport:

    def __init__(self):
        self.process = Processing()

    def read_image(self):
        pass

    def save_image(self, path, image):

        if isinstance(image, str):
            pass
        elif isinstance(image, list):
            for im in image:
                img = cv2.imread(im, cv2.IMREAD_COLOR)
                cv2.imwrite(path + "/" + im.split("/")[-1], img)

    def manage_dirs(self, path, categories=None):
        """
        This function takes in any directory path and checks if it exists. If it does exits, then it deletes it and creates a new
        one. Once the directory is created, we go about creating categories into it, if specified
        :param path:
        :param categories:
        :return:
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        if categories is not None:
            if isinstance(categories, list):
                for cat in categories:
                    os.mkdir(path + "/" + cat)
            else:
                os.mkdir(path + "/" + categories)

    def create_train_test_valid(self, categories, orig_data_path, data_path, split_ratio=(0.85,0.10,0.05)):
        """
        This function is used to split data lying in different categorical folders into training, testing and validation
        Essentially our directory structure is transformed from: data
                                                                  |--> coronal
                                                                  |--> horizontal
                                                                  |--> sagittal
        to the following structure: training
                                      |--> coronal
                                      |--> horizontal
                                      |--> sagittal
                                    validation
                                      |--> coronal
                                      |--> horizontal
                                      |--> sagittal
                                    testing
                                      |--> coronal
                                      |--> horizontal
                                      |--> sagittal

        :param categories: list of output classes / categories to be predicted by the model
        :param orig_data_path: the original data path where all the data lies
        :param data_path: the path where we would create train, test and valid directories and essentially save our
        splitted data
        :param split_ratio: a tuple of 3 numbers, depicting percentage split needed in training, validation and testing
        respectively, default is (0.70, 0.20, 0.10) i.e 70%, 20% and 10%
        :return: None
        """
        # a data dictionary holds categories and a list of all the image paths
        data_dict = dict()

        # training directory path
        train_path = data_path + "/training"
        self.manage_dirs(train_path, categories)

        # validation directory path
        valid_path = data_path + "/validation"
        self.manage_dirs(valid_path, categories)

        # testing directory path
        test_path = data_path + "/testing"
        self.manage_dirs(test_path, categories)

        # loop through all the categories, make a list of all images lying in each categorical folder and append their
        # paths into a list. This list of paths corresponds to a categorical key in the dictionary
        for category in categories:
            category_dir = orig_data_path + "/" + category   # the original data path corresponding to each category

            # populate the dictionary with list of image paths belonging to a particular class
            data_dict[category] = [category_dir + "/" + file for file in os.listdir(category_dir) if not file.startswith('.')]
            shuffle(data_dict[category])   # randomly shuffle the list of image paths

            # based on user provided percentage split list, split the given data for training, validation and testing
            training = data_dict[category][:int(len(data_dict[category]) * split_ratio[0])]
            validation = data_dict[category][int(len(data_dict[category]) * split_ratio[0]):
                                             int(len(data_dict[category]) * sum(split_ratio[:2]))]
            testing = data_dict[category][int(len(data_dict[category]) * sum(split_ratio[:2])):]

            # once the randomly shuffled data is split based on percentage we would want to read them from their
            # original directory and write them to a new location based on whether they are training, validation or
            # testing and further based on whether they are coronal, horizontal or sagittal.
            self.save_image(train_path + "/" + category, training)
            self.save_image(valid_path + "/" + category, validation)
            self.save_image(test_path + "/" + category, testing)

        return train_path, valid_path, test_path

    def create_data_matrices(self, categories, data_path, colormap):

        data_matrix = list()
        label_matrix = list()
        cat_matrix = list()
        for index, category in enumerate(categories):
            path = os.path.join(data_path, category)

            for img in os.listdir(path):
                try:
                    image = self.process.read_image(os.path.join(path, img), colormap)

                    data_matrix.append(image)
                    label_matrix.append(index)
                    cat_matrix.append(category)
                except Exception as e:
                    print(e)

        combined = list(zip(data_matrix,label_matrix, cat_matrix))
        shuffle(combined)
        data_matrix[:], label_matrix[:], cat_matrix[:] = zip(*combined)

        return data_matrix, label_matrix, cat_matrix


