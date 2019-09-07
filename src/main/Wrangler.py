"""
Created by nikunjlad on 2019-08-21

"""
import cv2, sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class Wrangler:

    def __init__(self, logger):
        self.logger = logger

    def display_image(self, image):
        """
        This method displays an image

        :param image: the image to be displayed.
        :return: nothing, just plots the image to the screen
        """
        try:
            plt.imshow(image)
            plt.show()
            self.logger.info("Image Displayed successfully!")
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def create_numpy_data(self, data):
        """
        The function to convert data to numpy format

        :param data: any data which is not in numpy format but should be converted to numpy format
        :return: returns data which is in numpy format
        """
        try:
            numpy_data = np.array(data)
            return numpy_data
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def data_reshape(self, data, labels, colormap, image_depth):
        """
        This function reshapes the data which is provided to it. It returns a reshaped data in
        (no_of_data, rows, columns, image_depth) format

        :param data: takes the input data which is color or grayscale images
        :param labels: labels corresponding to the input data provided.
        :param colormap: colormap can be any color space in which the data exists
        :param image_depth: the depth of the image which can be 1 in case of grayscale of 3 in case of anything else
        :return: returns data shape in (rows, cols, depth) format, the new reshaped data for training in
        (no_of_data, rows, columns, image_depth) format, name of the classes and number of unique classes.
        """
        try:
            images = data

            # Find the unique numbers from the train labels
            classes = np.unique(labels)
            n_classes = len(classes)
            print('Total number of outputs : ', n_classes)
            print('Output classes : ', classes)

            if "GRAY" in colormap:
                n_rows, n_cols = images.shape[1:]
                depth = 1
            else:
                n_rows, n_cols, depth = images.shape[1:]

            print(n_rows, n_cols, images.shape[0], depth)
            images = images.reshape(images.shape[0], n_rows, n_cols, image_depth)
            input_shape = (n_rows, n_cols, image_depth)

            return input_shape, images, classes, n_classes
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def do_one_hot_encoding(self, just_labels):
        """
        This function is used to one hot encode the labels, i.e. convert them from categorical to binary encoded format

        :param just_labels: the label vector to be one hot encoded
        :return: a binary encoded label matrix
        """
        try:
            le = LabelEncoder()
            labels = le.fit_transform(just_labels)
            one_hot_encoded_labels = to_categorical(labels, len(set(just_labels)))

            return one_hot_encoded_labels
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def read_image(self, image_path, color_space=None):
        """
        This method is used to read the image in the way provided by the user. It can read the image as it is or can
        read it in a grayscale or any color format with or without alpha channel

        :param image_path: the path where the image exists on the disk
        :param color_space: the color space in which the image needs to be read
        :return:
        """
        try:
            color_map = {'BGR2RGBA': cv2.COLOR_BGR2RGBA,
                         'BGR2GRAY': cv2.COLOR_BGR2GRAY,
                         'BGR2RGB': cv2.COLOR_BGR2RGB,
                         'BGR2BGRA': cv2.COLOR_BGR2BGRA,
                         'ORIGINAL': cv2.IMREAD_COLOR
                         }

            # read the input image and convert it from BGR to BGRA
            img = cv2.imread(image_path, color_map['ORIGINAL'])  # read the color image

            if color_space != 'ORIGINAL' and color_space is not None:
                # convert from 3 channel BGR to RGBA taking into consideration alpha
                img = cv2.cvtColor(img, color_map[color_space])

            return img
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def image_resize_with_aspect(self, height, width, img_path, fit_style, temp_path, color_map, padding="white",
                                 debug=False):
        """
        The image needs to be resized keeping aspect ratio in mind, instead of mere resizing to avoid stretching or
        compressing the images if they are not in a standaraized shapes

        :param height: desired image height
        :param width: desired image width
        :param img_path: path from where the image needs to be read
        :param fit_style: the style in which the image needs to be fit into, it can be square, rectangle
        :param temp_path: the temporary path where the data needs to be stored during runtime execution
        :param color_map: the color map which needs to be used for reading the images
        :param padding: what is the padding which needs to be kept, it can be any color, or border replication
        :param debug: debug mode to be enabled or disabled
        :return: returns the resized image maintaining the aspect ratio
        """
        if isinstance(img_path, np.ndarray):
            img = img_path
        else:
            img = self.read_image(img_path, color_map)

        if debug:
            self.display_image(img)

        # get the shape of the image
        if "GRAY" in color_map:
            img_h, img_w = img.shape
            depth = 1
        else:
            img_h, img_w, depth = img.shape  # returns (height, width, depth)

        # find the aspect ratio of the actual image and the desired image
        image_aspect = float(img_w / img_h)
        desired_aspect = float(width / height)
        print("original image height, width and depth:", img_h, img_w, depth)

        # resizing the image maintaining the aspect ratio. This is a function to resize it to a square
        if fit_style == "square":

            if image_aspect > desired_aspect:
                img_h = int(width / image_aspect)
                img_w = width
            elif image_aspect < desired_aspect:
                img_w = int(height * image_aspect)
                img_h = height
            else:
                img_w = width
                img_h = height

        print("new width and height: ", img_w, img_h)

        # resizing the image and displaying it.
        try:
            new_img = cv2.resize(img, (int(img_w), int(img_h)), interpolation=cv2.INTER_AREA)

            if debug:
                self.display_image(new_img)

            if padding is not None:
                # TIP: 0, 1 and border extension logic
                # create a black image using numpy and the alpha channel
                val = 0
                if padding == "extend":
                    if img_w == width:
                        val = int((np.mean(img[0:5, :]) + np.mean(img[-5:, :])) / 2)
                    elif img_h == height:
                        val = int((np.mean(img[:, 0:5]) + np.mean(img[:, -5:])) / 2)
                    print("Val :", val)
                elif padding == "white":
                    val = 255
                elif padding == "black":
                    val = 0

                back_img = np.ones((height, width, 3), np.float32) * val
                cv2.imwrite(temp_path + "/image.png", back_img)
                black_img = self.read_image(temp_path + "/image.png", color_map)

                # based on the resized image shapes, paste the original image on the  canvas with relevant offsets
                if img_w == width:
                    mid_y = int(height / 2) - int(img_h / 2)
                    black_img[mid_y:mid_y + img_h, :] = new_img
                elif img_h == height:
                    mid_x = int(width / 2) - int(img_w / 2)
                    black_img[:, mid_x:mid_x + img_w] = new_img
                else:
                    black_img = new_img
            else:
                black_img = new_img

            if debug:
                self.display_image(black_img)

            return black_img
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def resize_images(self, x_size, y_size, image, colormap, temp_path, procs):
        """
        wrapper function to resize the image based on sizes provided. It handles either single images or a list of
        images to be iterated over and worked with.

        :param x_size: the desired width of the image
        :param y_size: the desired height of the image
        :param image: the image to be resized. It can be either a single image or a list of images
        :param colormap: the colormap to be used to read the image
        :param temp_path: path to store temporarily generated runtime files.
        :return: returns resized image maintaining the aspect ratio
        """
        try:
            if isinstance(image, list):
                image_matrix = list()
                for img in image:
                    resized_image = self.image_resize_with_aspect(y_size, x_size, img, procs["fitStyle"], temp_path,
                                                                  colormap,procs["padding"], False)
                    # resized_image = cv2.resize(img, (x_size, y_size))
                    image_matrix.append(resized_image)
                return image_matrix
            else:
                resized_image = self.image_resize_with_aspect(y_size, x_size, image, procs["fitStyle"], temp_path, colormap,
                                                              procs["padding"], False)
                return resized_image
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def change_datatype(self, data, dataType):
        """
        This function is used to change the datatype of the provided data to float32, int64 or any other type

        :param data: input data
        :param dataType: datatype to be converted to
        :return: type casted data is returned
        """
        try:
            new_data = data.astype(dataType)
            return new_data
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def scale_images(self, data, factor):
        """
        This function is used to scale the images or normalize them. Usually image data consists of pixels from 0-255
        so the factor value usually lies in this range.

        :param data: the data to be scaled. can be since image or list of images
        :param factor: the factor by which to scale
        :return: scaled images
        """
        try:
            data = data / factor
            return data
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

