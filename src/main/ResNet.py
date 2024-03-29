"""
Created by nikunjlad on 2019-08-27

"""

import six, sys
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


class Resnet:

    def __init__(self, logger):
        self.logger = logger

    def _get_block(self, identifier):

        try:
            if isinstance(identifier, six.string_types):
                res = globals().get(identifier)
                if not res:
                    raise ValueError('Invalid {}'.format(identifier))
                return res
            return identifier
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _handle_dim_ordering(self):
        try:
            global ROW_AXIS
            global COL_AXIS
            global CHANNEL_AXIS
            if K.backend() == 'tensorflow':
                ROW_AXIS = 1
                COL_AXIS = 2
                CHANNEL_AXIS = 3
            else:
                CHANNEL_AXIS = 1
                ROW_AXIS = 2
                COL_AXIS = 3
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _bn_relu(self, input):
        """Helper to build a BN -> relu block
        """
        try:
            norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
            return Activation("relu")(norm)
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _shortcut(self, input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        try:
            input_shape = K.int_shape(input)
            residual_shape = K.int_shape(residual)
            stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
            stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
            equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

            shortcut = input
            # 1 X 1 conv if shape is different. Else identity.
            if stride_width > 1 or stride_height > 1 or not equal_channels:
                shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                  kernel_size=(1, 1),
                                  strides=(stride_width, stride_height),
                                  padding="valid",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(0.0001))(input)

            return add([shortcut, residual])
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _conv_bn_relu(self, **conv_params):
        """Helper to build a conv -> BN -> relu block
        """
        try:
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(input):
                conv = Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(input)
                return self._bn_relu(conv)
            return f
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _bn_relu_conv(self, **conv_params):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        try:
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(input):
                activation = self._bn_relu(input)
                return Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(activation)

            return f
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def bottleneck(self, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        Returns:
            A final conv layer of filters * 4
        """
        try:
            def f(input):

                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                      strides=init_strides,
                                      padding="same",
                                      kernel_initializer="he_normal",
                                      kernel_regularizer=l2(1e-4))(input)
                else:
                    conv_1_1 = self._bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                                  strides=init_strides)(input)

                conv_3_3 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
                residual = self._bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
                return self._shortcut(input, residual)

            return f
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def basic_block(self, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        try:
            def f(input):

                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                                   strides=init_strides,
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=l2(1e-4))(input)
                else:
                    conv1 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                               strides=init_strides)(input)

                residual = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
                return self._shortcut(input, residual)

            return f
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def _residual_block(self, block_function, filters, repetitions, is_first_layer=False):
        """Builds a residual block with repeating bottleneck blocks.
        """
        try:
            def f(input):
                for i in range(repetitions):
                    init_strides = (1, 1)
                    if i == 0 and not is_first_layer:
                        init_strides = (2, 2)
                    input = block_function(filters=filters, init_strides=init_strides,
                                           is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
                return input

            return f
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def build(self, input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        try:
            self._handle_dim_ordering()
            if len(input_shape) != 3:
                raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

            # Permute dimension order if necessary
            if K.backend() == 'theano':
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

            # Load function from str if needed.
            block_fn = self._get_block(block_fn)

            input = Input(shape=input_shape)
            conv1 = self._conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
            pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

            block = pool1
            filters = 64
            for i, r in enumerate(repetitions):
                block = self._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
                filters *= 2

            # Last activation
            block = self._bn_relu(block)

            # Classifier block
            block_shape = K.int_shape(block)
            pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                     strides=(1, 1))(block)
            flatten1 = Flatten()(pool2)
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                          activation="softmax")(flatten1)

            model = Model(inputs=input, outputs=dense)
            return model
        except Exception as e:
            self.logger.exception(e)
            sys.exit(1)

    def build_resnet_18(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs, self.basic_block, [2, 2, 2, 2])

    def build_resnet_34(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs, self.basic_block, [3, 4, 6, 3])

    def build_resnet_50(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs, self.bottleneck, [3, 4, 6, 3])

    def build_resnet_101(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs, self.bottleneck, [3, 4, 23, 3])

    def build_resnet_152(self, input_shape, num_outputs):
        return self.build(input_shape, num_outputs, self.bottleneck, [3, 8, 36, 3])
