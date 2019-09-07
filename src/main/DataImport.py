"""
Created by nikunjlad on 2019-08-20

"""
from random import shuffle
import os, shutil, cv2
from main.Wrangler import *
from sklearn.model_selection import train_test_split


class DataImport:

    def __init__(self, logger):
        self.logger = logger
        self.wrangler = Wrangler(self.logger)

    def save_image(self, path, image):
        """
        This function is used to help us save an image given a path
        :param path: path where the image needs to be saved
        :param image: the image to be saved
        :return: None
        """
        try:
            if isinstance(image, str):
                pass
            elif isinstance(image, list):
                for im in image:
                    img = cv2.imread(im, cv2.IMREAD_COLOR)
                    cv2.imwrite(path + "/" + im.split("/")[-1], img)
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def manage_dirs(self, path, categories=None):
        """
        This function takes in any directory path and checks if it exists. If it does exits, then it deletes it and creates
        a new one. Once the directory is created, we go about creating categories into it, if specified

        :param path: path which needs to be inspected
        :param categories: the categories to be looked into
        :return: None
        """
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

            if categories is not None:
                if isinstance(categories, list):
                    for cat in categories:
                        os.mkdir(path + "/" + cat)
                else:
                    os.mkdir(path + "/" + categories)
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def create_train_test_valid(self, categories, paths, split_ratio=(0.85, 0.10, 0.05)):
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
        :return: a data dictionary containing the training, validation and testing data as well as labels
        """
        # # a data dictionary holds categories and a list of all the image paths
        try:
            data_dict = dict()

            data_list = list()
            label_list = list()
            # loop through all the categories, make a list of all images lying in each categorical folder and append their
            # paths into a list. This list of paths corresponds to a categorical key in the dictionary
            for category in categories:
                category_dir = os.path.sep.join([paths["data_path"], category])

                dl = [category_dir + "/" + file for file in os.listdir(category_dir) if not file.startswith('.')]
                data_list.extend(dl)
                label_list.extend([category] * len(dl))

            combined = list(zip(data_list, label_list))
            shuffle(combined)
            data_list[:], label_list[:] = zip(*combined)
            tr_data, data_dict["test_data"], tr_labels, data_dict["test_labels"] = train_test_split(data_list, label_list,
                                                                                                    test_size=split_ratio[2], random_state=42)
            data_dict["train_data"], data_dict["valid_data"], data_dict["train_labels"], data_dict["valid_labels"] = \
                train_test_split(tr_data, tr_labels, test_size=split_ratio[1], random_state=42)

            return data_dict
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def create_data_matrices(self, labels, image_path_lst, colormap):
        """
        This function is used to create the data matrices given the list of image paths, labels and colormap
        :param labels: list of labels
        :param image_path_lst: list of images
        :param colormap: the colormap to which the images needs to be loaded into
        :return: returns the data matrix and the corresponding labels matrix
        """
        try:
            data_matrix = list()
            label_matrix = list()
            for index, image in enumerate(image_path_lst):
                # path = os.path.join(data_path, category)

                try:
                    imag = self.wrangler.read_image(image_path=image, color_space=colormap)

                    data_matrix.append(imag)
                    label_matrix.append(labels[index])
                except Exception as e:
                    print(e)

            combined = list(zip(data_matrix, label_matrix))
            shuffle(combined)
            data_matrix[:], label_matrix[:] = zip(*combined)

            return data_matrix, label_matrix
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)
