"""
Created by nikunjlad on 2019-08-21

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class Wrangler:

    def __init__(self):
        pass

    @staticmethod
    def display_image(image):
        plt.imshow(image)
        plt.show()

    @staticmethod
    def create_numpy_data(data):
        numpy_data = np.array(data)
        return numpy_data

    @staticmethod
    def data_reshape(data, labels, colormap, image_depth):
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

    @staticmethod
    def do_one_hot_encoding(just_labels):
        le = LabelEncoder()
        labels = le.fit_transform(just_labels)
        one_hot_encoded_labels = to_categorical(labels, 3)

        return one_hot_encoded_labels

    def read_image(self, image_path, color_space=None):
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

    def image_resize_with_aspect(self, height, width, img_path, fit_style, temp_path, color_map, padding="white",
                                 debug=False):

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
        except Exception as e:
            print(e)

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

    def resize_images(self, x_size, y_size, image, colormap, temp_path):

        if isinstance(image, list):
            image_matrix = list()
            for img in image:
                resized_image = self.image_resize_with_aspect(y_size, x_size, img, "square", temp_path, colormap,
                                                              "extend", False)
                # resized_image = cv2.resize(img, (x_size, y_size))
                image_matrix.append(resized_image)
            return image_matrix
        else:
            resized_image = self.image_resize_with_aspect(y_size, x_size, image, "square", temp_path, colormap,
                                                          "extend", False)
            return resized_image

    @staticmethod
    def change_datatype(data, dataType):
        new_data = data.astype(dataType)
        return new_data

    @staticmethod
    def scale_images(data, factor):
        data = data / factor
        return data
